"""Run an Optuna study: one worker process trains trials one after another.

Several workers (for example a Slurm array) share one study through an
Optuna journal file. The first worker creates the study and its MLflow
parent run; every trial is a normal training run, logged as a child run.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import cast

import matplotlib.pyplot as plt
import mlflow
import optuna
from lightning.pytorch.callbacks import Callback
from mlflow import MlflowClient
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState
from optuna.visualization import matplotlib as optuna_plots
from optuna_integration import PyTorchLightningPruningCallback
from pydantic import JsonValue

from seisbench_trainer.config import TrainConfig
from seisbench_trainer.data.loaders import Splits, load_splits
from seisbench_trainer.hpo.config import HPOConfig
from seisbench_trainer.tracking.run import experiment_name, run_tags, set_experiment
from seisbench_trainer.training.train import train

logger = logging.getLogger(__name__)

PARENT_RUN = "mlflow_parent_run_id"
TRIAL_RUN = "mlflow_run_id"
FINISHED = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)


def _hpo(config: TrainConfig) -> HPOConfig:
    if config.hpo is None:
        raise ValueError("the config has no hpo section")
    return config.hpo


def _open_study(config: TrainConfig) -> tuple[optuna.Study, str]:
    """Create or load the study; create the MLflow parent run exactly once."""
    hpo = _hpo(config)
    hpo.storage.parent.mkdir(parents=True, exist_ok=True)
    path = str(hpo.storage)
    # O_EXCL lock files work on NFS (the cluster's shared filesystem).
    storage = JournalStorage(JournalFileBackend(path, JournalFileOpenLock(path)))
    experiment_id = set_experiment(experiment_name(config))
    minimize = config.training.monitor_mode == "min"
    setup = JournalFileOpenLock(f"{path}.setup")
    setup.acquire()
    try:
        study = optuna.create_study(
            study_name=hpo.study_name,
            storage=storage,
            sampler=hpo.sampler.build(),
            pruner=hpo.pruner.build() if hpo.pruner else optuna.pruners.NopPruner(),
            direction="minimize" if minimize else "maximize",
            load_if_exists=True,
        )
        parent = study.user_attrs.get(PARENT_RUN)
        if parent is None:
            parent = _create_parent_run(config, experiment_id)
            study.set_user_attr(PARENT_RUN, parent)
    finally:
        setup.release()
    return study, str(parent)


def _create_parent_run(config: TrainConfig, experiment_id: str) -> str:
    hpo = _hpo(config)
    client = MlflowClient()
    tags = run_tags(config) | {"purpose": "hpo", "hpo.study": hpo.study_name}
    run = client.create_run(experiment_id, run_name=f"hpo-{hpo.study_name}", tags=tags)
    client.log_dict(run.info.run_id, config.model_dump(mode="json"), "config.json")
    return run.info.run_id


def _objective(
    config: TrainConfig, splits: Splits, parent: str
) -> Callable[[optuna.Trial], float]:
    hpo = _hpo(config)
    metric = config.training.monitor

    def objective(trial: optuna.Trial) -> float:
        values: dict[str, JsonValue] = {p.path: p.suggest(trial) for p in hpo.params}
        trial_config = config.with_values(values)
        tracking = trial_config.tracking.model_copy(
            update={
                "run_name": f"{hpo.study_name}-trial-{trial.number}",
                "tags": trial_config.tracking.tags
                | {"hpo.study": hpo.study_name, "hpo.trial": str(trial.number)},
            }
        )
        # The test split is never used for model selection.
        evaluation = trial_config.evaluation.model_copy(update={"test_at_end": False})
        trial_config = trial_config.model_copy(
            update={"tracking": tracking, "evaluation": evaluation}
        )
        callbacks: list[Callback] = []
        if hpo.pruner is not None:
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor=metric))
        try:
            run_id = train(trial_config, splits, parent, callbacks)
        except optuna.TrialPruned:
            last = mlflow.last_active_run()
            if last is not None:
                trial.set_user_attr(TRIAL_RUN, last.info.run_id)
                MlflowClient().set_terminated(last.info.run_id, "KILLED")
            raise
        trial.set_user_attr(TRIAL_RUN, run_id)
        return mlflow.get_run(run_id).data.metrics[f"best_{metric}"]

    return objective


def _finished(study: optuna.Study) -> int:
    return len(study.get_trials(deepcopy=False, states=FINISHED))


def summarize(study: optuna.Study, config: TrainConfig) -> None:
    """Log the best trial and the Optuna plots to the parent run."""
    hpo = _hpo(config)
    parent = str(study.user_attrs[PARENT_RUN])
    client = MlflowClient()
    complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    if complete:
        best = study.best_trial
        client.log_dict(
            parent,
            {
                "number": best.number,
                "value": best.value,
                "params": best.params,
                "mlflow_run_id": best.user_attrs.get(TRIAL_RUN),
            },
            "best_trial.json",
        )
        client.set_tag(parent, "hpo.best_trial", str(best.number))
        metric = f"best_{config.training.monitor}"
        client.log_metric(parent, metric, cast(float, best.value), step=len(complete))
    plots = {
        "optimization_history": optuna_plots.plot_optimization_history,
        "intermediate_values": optuna_plots.plot_intermediate_values,
        "param_importances": optuna_plots.plot_param_importances,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        for name, plot in plots.items():
            try:
                axes = plot(study)
            except (ValueError, RuntimeError) as error:  # e.g. too few trials
                logger.info("Skipping %s plot: %s", name, error)
                continue
            figure = axes.get_figure()
            if figure is not None:
                # mlflow leaves the figure parameter untyped
                client.log_figure(parent, figure, f"optuna/{name}.png")
            plt.close("all")
    if _finished(study) >= hpo.n_trials:
        client.set_terminated(parent, "FINISHED")


def run_worker(config: TrainConfig) -> optuna.Study:
    """Join the study and run trials until it has ``n_trials`` finished trials."""
    hpo = _hpo(config)
    study, parent = _open_study(config)
    if _finished(study) < hpo.n_trials:
        splits = load_splits(config.data, label_events=config.augmentation.label_events)
        stop = optuna.study.MaxTrialsCallback(hpo.n_trials, states=FINISHED)
        study.optimize(
            _objective(config, splits, parent),
            callbacks=[stop],
            catch=(Exception,),  # a failed trial is recorded; the worker goes on
            gc_after_trial=True,
        )
    summarize(study, config)
    return study
