from __future__ import annotations

from typer.testing import CliRunner

from seisbench_trainer import __version__
from seisbench_trainer.cli import app
from seisbench_trainer.console import metrics_table

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.output


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_metrics_table_rows() -> None:
    metrics = {"det_P/f1": 0.8, "pick_P/mae_s": 0.02, "baseline/det_P/f1": 0.7}
    table = metrics_table(metrics, ("P", "S"))
    assert table.row_count == 2  # trained P and baseline P; no S metrics
