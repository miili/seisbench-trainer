"""Command-line interface."""

from __future__ import annotations

import logging
import signal
from pathlib import Path
from types import FrameType
from typing import Annotated, Literal

import typer

from seisbench_trainer import __version__
from seisbench_trainer.config import TrainConfig, dump_config, load_config
from seisbench_trainer.console import (
    console,
    err_console,
    evaluation_table,
    metrics_table,
    setup_logging,
)
from seisbench_trainer.data.phases import PHASES

app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich")


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show debug log messages.")
    ] = False,
) -> None:
    """Train, evaluate and optimise SeisBench phase pickers."""
    setup_logging(logging.DEBUG if verbose else logging.INFO)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"seisbench-trainer {__version__}")


@app.command("config")
def config_command(
    schema: Annotated[
        bool, typer.Option("--schema", help="Print the JSON schema instead.")
    ] = False,
) -> None:
    """Print the default run config as JSON, e.g. [bold]config > run.json[/]."""
    if schema:
        console.print_json(data=TrainConfig.model_json_schema())
    else:
        console.print_json(dump_config(TrainConfig()))


def _interrupt_on_sigterm(signum: int, frame: FrameType | None) -> None:
    # Slurm sends SIGTERM on cancellation and timeout. As an exception it
    # unwinds the MLflow run context, so the run is not left RUNNING.
    raise KeyboardInterrupt(f"received signal {signum}")


@app.command("train")
def train_command(
    config_path: Annotated[
        Path,
        typer.Argument(
            metavar="CONFIG", exists=True, dir_okay=False, help="Run config (JSON)."
        ),
    ],
) -> None:
    """Train one model from a JSON run config."""
    import mlflow

    from seisbench_trainer.training.train import train

    config = load_config(config_path)
    signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    run_id = train(config)

    metrics = mlflow.get_run(run_id).data.metrics
    selection = f"best_{config.training.monitor}"
    console.print(f"Run [bold]{run_id}[/] ({mlflow.get_tracking_uri()})")
    if selection in metrics:
        console.print(f"{selection} = {metrics[selection]:.4g}")
    console.print(metrics_table(metrics, PHASES))


@app.command("evaluate")
def evaluate_command(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="models:/<id>, models:/<name>@<alias>, a training run ID, "
            "or <Type>:<weights> (e.g. PhaseNet:original).",
        ),
    ],
    datasets: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Dataset to evaluate on (repeatable): a seisbench.data class "
            "name, a local directory, or a dataset config as JSON. Default: the "
            "training datasets of the model, or those of --config.",
        ),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            help="Run config for data, augmentation and evaluation settings.",
        ),
    ] = None,
    split: Annotated[
        Literal["train", "dev", "test"], typer.Option(help="Split to evaluate.")
    ] = "test",
    experiment: Annotated[str | None, typer.Option(help="MLflow experiment.")] = None,
) -> None:
    """Evaluate a model on a dataset split; thresholds come from the dev split."""
    import mlflow

    from seisbench_trainer.evaluate import evaluate, load_model, parse_dataset
    from seisbench_trainer.tracking.run import experiment_name, set_experiment

    loaded = load_model(model)
    if config_path is not None:
        settings = load_config(config_path)
    else:
        settings = loaded.source or TrainConfig()
    if datasets:
        sources = [parse_dataset(spec) for spec in datasets]
        settings = settings.model_copy(
            update={"data": settings.data.model_copy(update={"datasets": sources})}
        )
    set_experiment(experiment or experiment_name(settings))
    tags = {"purpose": "evaluation", "model_ref": model, "split": split}
    with mlflow.start_run(run_name=f"eval-{loaded.name}", tags=tags) as run:
        rows = evaluate(loaded, settings, split)
    console.print(f"Evaluation run [bold]{run.info.run_id}[/]")
    console.print(evaluation_table([vars(row) for row in rows]))


@app.command("hpo")
def hpo_command(
    config_path: Annotated[
        Path,
        typer.Argument(
            metavar="CONFIG",
            exists=True,
            dir_okay=False,
            help="Run config (JSON) with an hpo section.",
        ),
    ],
) -> None:
    """Run HPO trials as one worker; start several workers to share the study."""
    config = load_config(config_path)
    if config.hpo is None:
        raise typer.BadParameter("the config has no hpo section", param_hint="CONFIG")
    try:
        from seisbench_trainer.hpo.study import run_worker
    except ModuleNotFoundError as error:
        if error.name not in ("optuna", "optuna_integration"):
            raise
        err_console.print(
            "[red]HPO needs the optional extra:[/] pip install 'seisbench-trainer[hpo]'"
        )
        raise typer.Exit(1) from error
    signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    study = run_worker(config)

    rows: list[dict[str, str | float | int]] = []
    for trial in study.trials:
        row: dict[str, str | float | int] = {
            "trial": trial.number,
            "state": trial.state.name,
            config.training.monitor: trial.value or float("nan"),
        }
        params = trial.params.items()
        rows.append(
            row | {k: f"{v:.4g}" if isinstance(v, float) else str(v) for k, v in params}
        )
    reverse = config.training.monitor_mode == "max"
    rows.sort(key=lambda r: _sort_value(r, config.training.monitor, reverse))
    console.print(f"Study [bold]{config.hpo.study_name}[/]: {len(study.trials)} trials")
    console.print(evaluation_table(rows[:10]))


def _sort_value(row: dict[str, str | float | int], key: str, reverse: bool) -> float:
    value = float(row[key])
    if value != value:  # NaN: unfinished or failed trials last
        return float("inf")
    return -value if reverse else value
