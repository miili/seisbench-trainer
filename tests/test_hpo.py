from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mlflow
import pytest
from conftest import small_config
from mlflow.artifacts import load_dict
from pydantic import JsonValue, ValidationError

from seisbench_trainer.config import TrainConfig
from seisbench_trainer.hpo.config import set_path

pytest.importorskip("optuna")
pytest.importorskip("optuna_integration")

LR: dict[str, JsonValue] = {
    "type": "float",
    "path": "training.optimizer.lr",
    "low": 1e-4,
    "high": 1e-2,
}


def with_hpo(params: list[JsonValue], **hpo: JsonValue) -> dict[str, JsonValue]:
    section: dict[str, JsonValue] = {"study_name": "s", "params": params}
    return {"hpo": section | hpo}


def test_set_path() -> None:
    data: dict[str, JsonValue] = {"a": {"b": 1}, "l": [{"c": 2}]}
    set_path(data, "a.b", 3)
    set_path(data, "l.0.c", 4)
    assert data == {"a": {"b": 3}, "l": [{"c": 4}]}
    for path in ("a.x", "l.1.c", "a.b.c"):
        with pytest.raises(ValueError, match="unknown config path"):
            set_path(data, path, 0)


def test_with_values_applies_and_validates() -> None:
    config = TrainConfig.model_validate(with_hpo([LR]))
    trial = config.with_values({"training.optimizer.lr": 0.005, "data.batch_size": 64})
    assert trial.training.optimizer.lr == 0.005
    assert trial.data.batch_size == 64
    assert trial.hpo is None
    with pytest.raises(ValidationError):
        config.with_values({"data.batch_size": 0})


SGD: JsonValue = {
    "type": "categorical",
    "path": "training.optimizer.type",
    "choices": ["sgd"],
}


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ([LR | {"path": "training.optimizer.lrr"}], "unknown config path"),
        ([LR | {"path": "data.component_order"}], "cannot be tuned"),
        ([LR | {"path": "data.datasets.0.name"}], "cannot be tuned"),
        ([LR, LR], "only once"),
        ([LR | {"low": 1.0, "high": 0.1}], "smaller than high"),
        ([{"type": "int", "path": "data.batch_size", "low": 0, "high": 8}], "greater"),
        ([SGD], "invalid config"),
    ],
)
def test_invalid_search_spaces(params: list[JsonValue], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        TrainConfig.model_validate(with_hpo(params))


def test_two_workers_share_one_study(
    synthetic_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three trials over two worker processes: one parent run, child trial runs."""
    storage = str(tmp_path / "optuna" / "study.journal")
    config = small_config(
        synthetic_path,
        tmp_path,
        epochs=1,
        evaluation={"baseline": None, "plot_every_n_epochs": 0},
        **with_hpo([LR], n_trials=3, pruner=None, storage=storage),
    )
    path = tmp_path / "hpo.json"
    path.write_text(config.model_dump_json())
    uri = f"file://{tmp_path / 'mlruns'}"
    env = os.environ | {"MLFLOW_TRACKING_URI": uri, "MLFLOW_ALLOW_FILE_STORE": "true"}
    script = Path(sys.executable).parent / "seisbench-trainer"
    workers = [
        subprocess.Popen(
            [str(script), "hpo", str(path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for _ in range(2)
    ]
    outputs = [worker.communicate(timeout=600)[0].decode() for worker in workers]
    for worker, output in zip(workers, outputs, strict=True):
        assert worker.returncode == 0, output[-3000:]

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(search_all_experiments=True, output_format="list")
    parents = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    children = [r for r in runs if "mlflow.parentRunId" in r.data.tags]
    assert [p.data.tags["purpose"] for p in parents] == ["hpo"]
    parent = parents[0]
    # each worker may start one trial just before the target is reached
    assert 3 <= len(children) <= 4
    assert {c.data.tags["mlflow.parentRunId"] for c in children} == {parent.info.run_id}
    assert all(c.data.tags["hpo.study"] == "s" for c in children)
    assert parent.info.status == "FINISHED"
    assert "best_val_loss" in parent.data.metrics
    best = load_dict(f"runs:/{parent.info.run_id}/best_trial.json")
    assert set(best["params"]) == {"training.optimizer.lr"}
