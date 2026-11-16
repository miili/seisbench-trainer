"""Rich console output and logging setup.

Results go to stdout (``console``), so they can be piped or redirected.
Logs go to stderr (``err_console``).
"""

from __future__ import annotations

import logging
import math

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from seisbench_trainer.metrics.names import BASELINE

console = Console()
err_console = Console(stderr=True)


def setup_logging(level: int = logging.INFO) -> None:
    """Send all log records through a Rich handler on stderr."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=err_console, rich_tracebacks=True)],
        force=True,
    )


SUMMARY_COLUMNS = (
    ("F1", "det_{phase}/f1"),
    ("prec.", "det_{phase}/precision"),
    ("recall", "det_{phase}/recall"),
    ("thr.", "det_{phase}/threshold"),
    ("MAE s", "pick_{phase}/mae_s"),
    ("AUC", "pick_{phase}/auc"),
)
"""Detection metrics (F1 to threshold), then pick accuracy (MAE, AUC)."""


def metrics_table(metrics: dict[str, float], phases: tuple[str, ...]) -> Table:
    """Evaluation summary: one row per model and phase (latest epoch)."""
    table = Table(caption="detection: F1 to thr.; pick window: MAE, AUC")
    table.add_column("model")
    table.add_column("phase")
    for header, _ in SUMMARY_COLUMNS:
        table.add_column(header, justify="right")
    for prefix, name in (("", "trained"), (BASELINE, "baseline")):
        for phase in phases:
            keys = [prefix + key.format(phase=phase) for _, key in SUMMARY_COLUMNS]
            if not any(key in metrics for key in keys):
                continue
            values = [metrics.get(key, math.nan) for key in keys]
            table.add_row(name, phase, *(f"{value:.3f}" for value in values))
    return table


def rows_table(rows: list[dict[str, str | float | int]]) -> Table:
    """A table of rows with the same keys: evaluation results, HPO trials."""
    table = Table()
    if not rows:
        return table
    for column in rows[0]:
        justify = "left" if column in ("model", "phase") else "right"
        table.add_column(column, justify=justify)
    for row in rows:
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values()]
        table.add_row(*cells)
    return table
