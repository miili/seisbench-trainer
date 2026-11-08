from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest
import seisbench.models as sbm
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts, load_dict
from pydantic import JsonValue

from seisbench_trainer.config import TrainConfig
from seisbench_trainer.tracking import run as tracking_run
from seisbench_trainer.tracking.run import flatten, model_name
from seisbench_trainer.training import train as train_module
from seisbench_trainer.training.train import train


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A fresh SQLite MLflow store for each test."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield uri


def small_config(synthetic_path: Path, tmp_path: Path, epochs: int) -> TrainConfig:
    return TrainConfig.model_validate(
        {
            "data": {
                "datasets": [{"type": "local", "path": str(synthetic_path)}],
                "batch_size": 16,
                "num_workers": 1,
            },
            "training": {
                "epochs": epochs,
                "accelerator": "cpu",
                "checkpoint_dir": str(tmp_path / "checkpoints"),
            },
            "tracking": {"tags": {"purpose": "test"}},
        }
    )


def test_flatten() -> None:
    nested: JsonValue = {"a": {"b": 1, "c": [{"d": None}, "x"]}, "e": True}
    assert flatten(nested) == {"a.b": "1", "a.c.0.d": "None", "a.c.1": "x", "e": "True"}


def test_model_name(synthetic_path: Path, tmp_path: Path) -> None:
    config = small_config(synthetic_path, tmp_path, epochs=1)
    assert model_name(config) == f"PhaseNet-{synthetic_path.name}-scratch"


def test_train_end_to_end(
    synthetic_path: Path, tmp_path: Path, tracking_uri: str
) -> None:
    config = small_config(synthetic_path, tmp_path, epochs=2)
    run_id = train(config)

    client = MlflowClient()
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    experiment = client.get_experiment(run.info.experiment_id)
    assert experiment.name == f"{synthetic_path.name}/PhaseNet"

    # metrics per epoch, the best value, and system metrics
    assert len(client.get_metric_history(run_id, "val_loss")) == 2
    assert client.get_metric_history(run_id, "train_loss")
    assert "best_val_loss" in run.data.metrics

    # config artifact, params and tags reproduce the run
    logged = load_dict(f"runs:/{run_id}/config.json")
    assert TrainConfig.model_validate(logged) == config
    assert run.data.params["training.epochs"] == "2"
    assert run.data.params["data.datasets.0.type"] == "local"
    assert run.data.tags["model"] == "PhaseNet"
    assert run.data.tags["purpose"] == "test"
    assert "version.seisbench" in run.data.tags

    # checkpoint stays local
    assert (tmp_path / "checkpoints" / run_id / "best.ckpt").exists()

    # LoggedModel with the SeisBench export attached
    models = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id], output_format="list"
    )
    assert len(models) == 1
    assert models[0].source_run_id == run_id
    download = Path(
        download_artifacts(
            f"models:/{models[0].model_id}", dst_path=str(tmp_path / "download")
        )
    )
    exported = sorted(download.rglob("*.json"))
    exported = [path for path in exported if path.parent.name == "extra_files"]
    assert [path.stem for path in exported] == [model_name(config)]
    model = sbm.PhaseNet.load(exported[0].with_suffix(""))
    assert model.component_order == config.data.component_order


def test_failed_run_is_marked_failed(
    synthetic_path: Path,
    tmp_path: Path,
    tracking_uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_export(*args: object, **kwargs: object) -> str:
        raise RuntimeError("export failed")

    monkeypatch.setattr(train_module, "log_seisbench_model", broken_export)
    assert tracking_run.log_seisbench_model is not broken_export
    with pytest.raises(RuntimeError, match="export failed"):
        train(small_config(synthetic_path, tmp_path, epochs=1))
    runs = mlflow.search_runs(search_all_experiments=True, output_format="list")
    assert [run.info.status for run in runs] == ["FAILED"]
