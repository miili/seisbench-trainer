from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import small_config
from typer.testing import CliRunner

from seisbench_trainer import __version__
from seisbench_trainer.cli import app
from seisbench_trainer.config import TrainConfig
from seisbench_trainer.console import metrics_table

runner = CliRunner()


def test_config_prints_the_defaults_as_json() -> None:
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert TrainConfig.model_validate_json(result.stdout) == TrainConfig()


def test_config_resolves_a_file(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text('{"training": {"epochs": 3}}')
    result = runner.invoke(app, ["config", str(path)])
    assert result.exit_code == 0
    resolved = json.loads(result.stdout)
    assert resolved["training"]["epochs"] == 3
    assert resolved["augmentation"]["label_events"] == "all_events"  # a default


def test_invalid_config_lists_the_problems(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"training": {"epochs": -1, "lr": 3}}')
    for command in ("config", "train", "hpo"):
        result = runner.invoke(app, [command, str(path)])
        assert result.exit_code == 2
        assert "training.epochs: Input should be greater than 0" in result.stderr
        assert "training.lr: Extra inputs are not permitted" in result.stderr
        assert "Traceback" not in result.output


def test_train_json_summary(
    synthetic_path: Path, tmp_path: Path, tracking_uri: str
) -> None:
    config = small_config(synthetic_path, tmp_path, epochs=1)
    path = tmp_path / "run.json"
    path.write_text(config.model_dump_json())
    result = runner.invoke(app, ["train", "--json", str(path)])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)  # nothing but the JSON on stdout
    assert summary["monitor"] == "det/f1_mean"
    assert summary["model_id"].startswith("m-")
    assert summary["best"] == summary["metrics"]["best_det/f1_mean"]


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


def test_trial_rows_keep_a_zero_score() -> None:
    optuna = pytest.importorskip("optuna")
    from seisbench_trainer.cli import _trial_rows

    lr = optuna.distributions.FloatDistribution(1e-4, 1e-2)
    trials = [
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.RUNNING, params={}, distributions={}
        ),
        optuna.trial.create_trial(
            value=0.0, params={"lr": 1e-3}, distributions={"lr": lr}
        ),
    ]
    rows = _trial_rows(trials, "det/f1_mean", "max")
    # a real score of 0, ranked before the unfinished trial
    assert rows[0]["det/f1_mean"] == 0.0
    assert rows[0]["lr"] == "0.001"
