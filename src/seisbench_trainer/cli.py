"""Command-line interface.

Results go to stdout; logs, and anything a library prints during a run, go
to stderr. With ``--json`` a command prints one JSON object and nothing else
on stdout, for scripts and agents. Exit codes:
0 success, 1 a failed run, 2 invalid input (arguments or config).
"""

from __future__ import annotations

import json
import logging
import math
import signal
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Annotated, Any

import typer
from pydantic import ValidationError
from rich.markup import escape

from seisbench_trainer import __version__
from seisbench_trainer.config import TrainConfig, dump_config, load_config
from seisbench_trainer.console import (
    console,
    err_console,
    metrics_table,
    rows_table,
    setup_logging,
)
from seisbench_trainer.data.phases import PHASES
from seisbench_trainer.data.preprocessing import Split
from seisbench_trainer.training.optim import Mode

if TYPE_CHECKING:  # optuna is the optional hpo extra
    from optuna.trial import FrozenTrial

app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich")

ConfigArgument = Annotated[
    Path,
    typer.Argument(
        metavar="CONFIG", exists=True, dir_okay=False, help="Run config (JSON)."
    ),
]
JsonOption = Annotated[
    bool,
    typer.Option(
        "--json", help="Print one JSON object with the results on stdout instead."
    ),
]


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show debug log messages.")
    ] = False,
) -> None:
    """Train, evaluate and optimise SeisBench phase pickers.

    One JSON file describes a run; [bold]config[/] prints the defaults and
    checks a file. Every run is tracked in MLflow (MLFLOW_TRACKING_URI).
    """
    setup_logging(logging.DEBUG if verbose else logging.INFO)


def _load(path: Path) -> TrainConfig:
    """Read a run config; if it is invalid, list the errors and exit with 2."""
    try:
        return load_config(path)
    except ValidationError as error:
        err_console.print(f"[red]{escape(str(path))}: invalid run config[/]")
        for problem in error.errors():
            where = ".".join(str(part) for part in problem["loc"]) or "(file)"
            # one line per problem, never wrapped: easy to read for scripts
            err_console.print(
                escape(f"  {where}: {problem['msg']}"), soft_wrap=True, highlight=False
            )
        raise typer.Exit(2) from None


def _print_json(data: Any) -> None:  # noqa: ANN401 - any JSON value
    """Plain JSON on stdout; NaN and infinity become null."""

    def clean(value: Any) -> Any:  # noqa: ANN401
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [clean(item) for item in value]
        return value

    typer.echo(json.dumps(clean(data), indent=2))


@app.command()
def version() -> None:
    """Print the package version (it names the git commit)."""
    typer.echo(f"seisbench-trainer {__version__}")


@app.command("config")
def config_command(
    config_path: Annotated[
        Path | None,
        typer.Argument(
            metavar="[CONFIG]",
            exists=True,
            dir_okay=False,
            help="Check this run config and print it with every default filled in.",
        ),
    ] = None,
    schema: Annotated[
        bool,
        typer.Option(
            "--schema", help="Print the JSON schema with all fields described."
        ),
    ] = False,
) -> None:
    """Print the default run config, or check one, as JSON.

    [bold]config > run.json[/] writes the defaults to edit.
    [bold]config run.json[/] validates a config (exit code 2 and a list of the
    problems if it is invalid) and prints every value it resolves to.
    """
    if schema:
        _print_json(TrainConfig.model_json_schema())
        return
    config = TrainConfig() if config_path is None else _load(config_path)
    typer.echo(dump_config(config))


def _interrupt_on_sigterm(signum: int, frame: FrameType | None) -> None:
    # Slurm sends SIGTERM on cancellation and timeout. As an exception it
    # unwinds the MLflow run context, so the run is not left RUNNING.
    raise KeyboardInterrupt(f"received signal {signum}")


@app.command("train")
def train_command(config_path: ConfigArgument, as_json: JsonOption = False) -> None:
    """Train one model from a JSON run config.

    Logs the run to MLflow and exports the best model (SeisBench .json/.pt
    pair). Prints the run ID and the validation scores of the best epoch.
    """
    import mlflow

    from seisbench_trainer.tracking.run import logged_model
    from seisbench_trainer.training.train import train

    config = _load(config_path)
    signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    with redirect_stdout(sys.stderr):  # what libraries print is log, not result
        run_id = train(config)

    run = mlflow.get_run(run_id)
    metrics = run.data.metrics
    selection = f"best_{config.training.monitor}"
    if as_json:
        model = logged_model(run_id)
        _print_json(
            {
                "run_id": run_id,
                "experiment_id": run.info.experiment_id,
                "tracking_uri": mlflow.get_tracking_uri(),
                "model_id": model.model_id,
                "model_name": model.name,
                "monitor": config.training.monitor,
                "best": metrics.get(selection),
                "metrics": metrics,
            }
        )
        return
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
            help="Run config for data, augmentation and evaluation settings. "
            "Default: the model's training config.",
        ),
    ] = None,
    split: Annotated[Split, typer.Option(help="Split to evaluate.")] = "test",
    experiment: Annotated[
        str | None,
        typer.Option(help="MLflow experiment. Default: <datasets>/<model>."),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Evaluate a model on a dataset split.

    Detection thresholds are selected on the dev split and applied unchanged
    to the evaluated split, at each test tolerance. Logs an MLflow run linked
    to the model and the dataset.
    """
    import mlflow

    from seisbench_trainer.evaluate import evaluate, load_model, parse_dataset
    from seisbench_trainer.tracking.run import experiment_name, set_experiment

    loaded = load_model(model)
    if config_path is not None:
        settings = _load(config_path)
    else:
        settings = loaded.source or TrainConfig()
    if datasets:
        sources = [parse_dataset(spec) for spec in datasets]
        settings = settings.model_copy(
            update={"data": settings.data.model_copy(update={"datasets": sources})}
        )
    set_experiment(experiment or experiment_name(settings))
    tags = {"purpose": "evaluation", "model_ref": model, "split": split}
    with (
        redirect_stdout(sys.stderr),
        mlflow.start_run(run_name=f"eval-{loaded.name}", tags=tags) as run,
    ):
        rows = evaluate(loaded, settings, split)
    if as_json:
        _print_json(
            {
                "run_id": run.info.run_id,
                "model": loaded.name,
                "model_id": loaded.model_id,
                "split": split,
                "rows": [asdict(row) for row in rows],
            }
        )
        return
    console.print(f"Evaluation run [bold]{run.info.run_id}[/]")
    console.print(rows_table([asdict(row) for row in rows]))


@app.command("hpo")
def hpo_command(config_path: ConfigArgument, as_json: JsonOption = False) -> None:
    """Run HPO trials as one worker; start several workers to share the study.

    Needs the config's hpo section and the optional extra
    [bold]pip install 'seisbench-trainer\\[hpo]'[/]. Every trial is an MLflow
    child run of the study's parent run.
    """
    config = _load(config_path)
    if config.hpo is None:
        err_console.print(f"[red]{config_path}: the config has no hpo section[/]")
        raise typer.Exit(2)
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
    with redirect_stdout(sys.stderr):
        study = run_worker(config)

    training = config.training
    rows = _trial_rows(study.trials, training.monitor, training.monitor_mode)
    if as_json:
        _print_json(
            {
                "study": config.hpo.study_name,
                "monitor": training.monitor,
                "trials": rows,
            }
        )
        return
    console.print(f"Study [bold]{config.hpo.study_name}[/]: {len(study.trials)} trials")
    console.print(rows_table(rows[:10]))


def _trial_rows(
    trials: list[FrozenTrial], monitor: str, mode: Mode
) -> list[dict[str, str | float | int]]:
    """One row per trial, the best first; unfinished and failed trials last."""
    rows: list[dict[str, str | float | int]] = []
    for trial in trials:
        row: dict[str, str | float | int] = {
            "trial": trial.number,
            "state": trial.state.name,
            monitor: float("nan") if trial.value is None else trial.value,
        }
        params = trial.params.items()
        rows.append(
            row | {k: f"{v:.4g}" if isinstance(v, float) else str(v) for k, v in params}
        )
    rows.sort(key=lambda r: _sort_value(r, monitor, mode == "max"))
    return rows


def _sort_value(row: dict[str, str | float | int], key: str, reverse: bool) -> float:
    value = float(row[key])
    if value != value:  # NaN: unfinished or failed trials last
        return float("inf")
    return -value if reverse else value
