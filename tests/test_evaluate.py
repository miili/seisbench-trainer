from __future__ import annotations

import io
from pathlib import Path

import mlflow
import pandas as pd
import pytest
import seisbench.models as sbm
from conftest import small_config
from mlflow.artifacts import load_text
from pydantic import ValidationError
from typer.testing import CliRunner

from seisbench_trainer.cli import app
from seisbench_trainer.data import LocalDataset, SeisBenchDataset, SyntheticDataset
from seisbench_trainer.evaluate import evaluate, load_model, parse_dataset
from seisbench_trainer.training.train import train

TOLERANCES = (0.1, 0.2, 0.5)


@pytest.fixture
def exported_model(tmp_path: Path) -> Path:
    """A scratch PhaseNet saved as a SeisBench .json/.pt pair."""
    path = tmp_path / "export" / "PhaseNet-test"
    path.parent.mkdir()
    sbm.PhaseNet(component_order="ZNE", sampling_rate=100).save(path)
    return path


def test_load_model_from_local_path(exported_model: Path) -> None:
    loaded = load_model(f"PhaseNet:{exported_model}")
    assert loaded.model.component_order == "ZNE"
    assert loaded.model_id is None
    assert loaded.config.variant() == "from-PhaseNet-test"


@pytest.mark.parametrize(
    ("ref", "error"),
    [("PhaseNet", ValueError), ("EQT:original", ValidationError)],
)
def test_load_model_rejects_bad_references(ref: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        load_model(ref)


def test_parse_dataset(synthetic_path: Path) -> None:
    assert parse_dataset("ETHZ") == SeisBenchDataset(name="ETHZ")
    assert parse_dataset(str(synthetic_path)) == LocalDataset(path=synthetic_path)
    assert parse_dataset('{"type": "synthetic", "n_traces": 50}') == SyntheticDataset(
        n_traces=50
    )


def test_evaluate_logs_metrics_table_and_dataset(
    synthetic_path: Path, tmp_path: Path, tracking_uri: str, exported_model: Path
) -> None:
    settings = small_config(synthetic_path, tmp_path, epochs=1)
    loaded = load_model(f"PhaseNet:{exported_model}")
    with mlflow.start_run() as run:
        rows = evaluate(loaded, settings, "test")
    assert [(row.phase, row.tolerance_s) for row in rows] == [
        (phase, tolerance) for phase in ("P", "S") for tolerance in TOLERANCES
    ]
    assert all(0.0 <= row.f1 <= 1.0 for row in rows)
    assert len({row.threshold for row in rows if row.phase == "P"}) == 1

    finished = mlflow.get_run(run.info.run_id)
    metrics = finished.data.metrics
    assert {
        "test_det_P/threshold",
        "test_det_P/tol0.1s/f1",
        "test_det_S/tol0.5s/mae_s",
        "test_pick_S/mae_s",
    } <= set(metrics)
    csv = load_text(f"runs:/{run.info.run_id}/evaluation/test_summary.csv")
    table = pd.read_csv(io.StringIO(csv))
    assert len(table) == len(rows)
    assert list(table.columns[:3]) == ["model", "phase", "tolerance_s"]
    assert finished.inputs is not None
    inputs = finished.inputs.dataset_inputs
    assert [i.dataset.name for i in inputs] == [f"{synthetic_path.name}/test"]


def test_train_evaluates_test_split_at_end(
    synthetic_path: Path, tmp_path: Path, tracking_uri: str
) -> None:
    config = small_config(
        synthetic_path,
        tmp_path,
        epochs=1,
        evaluation={"baseline": None, "test_at_end": True, "plot_every_n_epochs": 0},
    )
    run_id = train(config)
    metrics = mlflow.get_run(run_id).data.metrics
    assert "test_det_P/tol0.1s/f1" in metrics
    model = mlflow.search_logged_models(
        filter_string=f"source_run_id = '{run_id}'", output_format="list"
    )[0]
    linked = {metric.key for metric in model.metrics or []}
    assert "test_det_P/tol0.1s/f1" in linked


def test_cli_evaluates_a_training_run(
    synthetic_path: Path, tmp_path: Path, tracking_uri: str
) -> None:
    config = small_config(
        synthetic_path,
        tmp_path,
        epochs=1,
        evaluation={"baseline": None, "plot_every_n_epochs": 0},
    )
    run_id = train(config)
    loaded = load_model(run_id)
    assert loaded.source == config
    assert loaded.model_id is not None

    result = CliRunner().invoke(app, ["evaluate", "--model", run_id, "--split", "dev"])
    assert result.exit_code == 0, result.output
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string="tags.purpose = 'evaluation'",
        output_format="list",
    )
    assert len(runs) == 1
    assert runs[0].data.tags["model_ref"] == run_id
    assert "dev_det_S/tol0.2s/recall" in runs[0].data.metrics
