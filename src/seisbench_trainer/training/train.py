"""Train one model from a run config and track it in MLflow."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import lightning.pytorch as pl
import mlflow
import torch
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger

from seisbench_trainer.config import TrainConfig
from seisbench_trainer.data.augmentations import (
    eval_augmentations,
    train_augmentations,
)
from seisbench_trainer.data.loaders import Splits, load_splits, make_loader
from seisbench_trainer.data.steps import StepContext
from seisbench_trainer.evaluate import LoadedModel, evaluate
from seisbench_trainer.tracking.callbacks import EvaluationCallback
from seisbench_trainer.tracking.run import (
    experiment_name,
    log_run_metadata,
    log_seisbench_model,
    model_name,
    set_experiment,
)
from seisbench_trainer.training.module import PickerModule
from seisbench_trainer.units import seconds_to_samples

logger = logging.getLogger(__name__)


def _callbacks(
    config: TrainConfig, evaluation: EvaluationCallback, checkpoint: ModelCheckpoint
) -> list[Callback]:
    training = config.training
    callbacks: list[Callback] = [
        evaluation,
        checkpoint,
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]
    if training.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor=training.monitor,
                mode=training.monitor_mode,
                patience=training.early_stopping_patience,
                min_delta=training.min_delta,
            )
        )
    return callbacks


def _load_best(module: PickerModule, checkpoint_path: str) -> int:
    """Load the best checkpoint's weights into the module; return its epoch."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    module.load_state_dict(checkpoint["state_dict"])
    return int(checkpoint["epoch"])


def _test(
    config: TrainConfig, module: PickerModule, model_id: str, splits: Splits
) -> None:
    """Test-set evaluation of the best model and the baseline."""
    trained = LoadedModel(config.model, module.model, model_name(config), model_id)
    evaluate(trained, config, "test", splits)
    baseline = config.evaluation.baseline
    if baseline is not None:
        loaded = LoadedModel(
            baseline, baseline.load_pretrained(), f"baseline {baseline.variant()}"
        )
        evaluate(loaded, config, "test", splits, prefix="baseline/")


def train(
    config: TrainConfig,
    splits: Splits | None = None,
    parent_run_id: str | None = None,
    callbacks: Sequence[Callback] = (),
) -> str:
    """Train, log to MLflow and export the best model. Returns the run ID.

    ``splits`` reuses already loaded data (HPO trials); they must match
    ``config.data``. ``parent_run_id`` makes the run a child run, and
    ``callbacks`` are added to the Lightning trainer.
    """
    pl.seed_everything(config.training.seed, workers=True)
    module = PickerModule(config)

    augmentation = config.augmentation
    sampling_rate = config.data.sampling_rate
    window = module.window_samples
    labels = config.model.label_augmentations(
        module.model,
        seconds_to_samples(augmentation.label_sigma_s, sampling_rate),
        augmentation.label_shape,
    )
    if splits is None:
        splits = load_splits(config.data)
    train_loader = make_loader(
        splits.train,
        [
            *train_augmentations(
                augmentation,
                StepContext(
                    sampling_rate, window, config.data.component_order, splits.train
                ),
            ),
            *labels,
        ],
        config.data,
        shuffle=True,
    )
    dev_loader = make_loader(
        splits.dev,
        [
            *eval_augmentations(
                augmentation, window, config.training.seed, sampling_rate
            ),
            *labels,
        ],
        config.data,
        shuffle=False,
    )

    dev_loaders = [dev_loader]
    noisy = config.evaluation.noisy_validation
    noise = (
        None
        if noisy is None
        else noisy.build(splits.dev, window, config.data.component_order)
    )
    if noise is not None:
        dev_loaders.append(
            make_loader(
                splits.dev,
                [
                    *eval_augmentations(
                        augmentation, window, config.training.seed, sampling_rate, noise
                    ),
                    *labels,
                ],
                config.data,
                shuffle=False,
            )
        )

    set_experiment(experiment_name(config))
    with mlflow.start_run(
        run_name=config.tracking.run_name,
        parent_run_id=parent_run_id,
        log_system_metrics=True,
    ) as run:
        run_id = run.info.run_id
        log_run_metadata(config)
        evaluation = EvaluationCallback(config)
        checkpoint = ModelCheckpoint(
            dirpath=Path(config.training.checkpoint_dir) / run_id,
            filename="best",
            monitor=config.training.monitor,
            mode=config.training.monitor_mode,
            save_weights_only=True,
        )
        trainer = pl.Trainer(
            max_epochs=config.training.epochs,
            accelerator=config.training.accelerator,
            devices=1,
            precision=config.training.precision,
            # MLFlowLogger's default URI ignores mlflow.set_tracking_uri().
            logger=MLFlowLogger(
                tracking_uri=mlflow.get_tracking_uri(),
                run_id=run_id,
                synchronous=False,
            ),
            callbacks=[*_callbacks(config, evaluation, checkpoint), *callbacks],
            log_every_n_steps=min(50, len(train_loader)),
        )
        trainer.fit(module, train_loader, dev_loaders)
        mlflow.flush_async_logging()

        # Lightning types this attribute loosely; it is the checkpoint path.
        best_path = str(checkpoint.best_model_path)
        if not best_path:
            raise RuntimeError("training produced no checkpoint")
        epoch = _load_best(module, best_path)
        best_score = checkpoint.best_model_score
        score = None if best_score is None else float(best_score)
        thresholds = evaluation.thresholds.get(epoch, {})
        model_id = log_seisbench_model(
            module.model,
            config,
            window,
            epoch,
            score,
            thresholds,
            evaluation.figures.get(epoch, {}),
        )
        if score is not None:
            mlflow.log_metric(
                f"best_{config.training.monitor}",
                score,
                step=epoch,
                model_id=model_id,
            )
        if config.evaluation.test_at_end:
            _test(config, module, model_id, splits)
        logger.info("Run %s finished; best epoch %d, model %s", run_id, epoch, model_id)
    return run_id
