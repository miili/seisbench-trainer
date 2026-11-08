"""Command-line interface."""

import logging
import signal
from pathlib import Path
from types import FrameType
from typing import Annotated

import typer
from rich.table import Table

from seisbench_trainer import __version__
from seisbench_trainer.config import TrainConfig, dump_config, load_config
from seisbench_trainer.console import console, setup_logging

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

    run = mlflow.get_run(run_id)
    console.print(f"Run [bold]{run_id}[/] ({mlflow.get_tracking_uri()})")
    table = Table()
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in sorted(run.data.metrics.items()):
        if not key.startswith("system/"):
            table.add_row(key, f"{value:.4g}")
    console.print(table)
