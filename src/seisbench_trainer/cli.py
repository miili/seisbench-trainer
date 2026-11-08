"""Command-line interface."""

import logging
from typing import Annotated

import typer

from seisbench_trainer import __version__
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
