from pathlib import Path

import mlflow
import pytest
import seisbench.models as sbm
from conftest import small_config
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts, list_artifacts, load_dict
from pydantic import JsonValue

from seisbench_trainer.config import TrainConfig
from seisbench_trainer.tracking import run as tracking_run
from seisbench_trainer.tracking.run import flatten, model_name, read_git_head
from seisbench_trainer.training import train as train_module
from seisbench_trainer.training.train import train


def test_flatten() -> None:
    nested: JsonValue = {"a": {"b": 1, "c": [{"d": None}, "x"]}, "e": True}
    assert flatten(nested) == {"a.b": "1", "a.c.0.d": "None", "a.c.1": "x", "e": "True"}


def test_read_git_head(tmp_path: Path) -> None:
    git_dir = tmp_path / "repo" / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "refs" / "heads" / "main").write_text("abc123\n")
    nested = tmp_path / "repo" / "src" / "pkg"
    nested.mkdir(parents=True)
    assert read_git_head(nested) == "abc123"

    (git_dir / "refs" / "heads" / "main").unlink()
    (git_dir / "packed-refs").write_text("# pack-refs\ndef456 refs/heads/main\n")
    assert read_git_head(nested) == "def456"

    (git_dir / "HEAD").write_text("0123abcd\n")
    assert read_git_head(nested) == "0123abcd"
    assert read_git_head(tmp_path) is None


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
    expected = ("pick_P/mae_s", "pick_S/auc", "det_P/f1", "det_S/recall", "det/f1_mean")
    for key in expected:
        assert len(client.get_metric_history(run_id, key)) == 2, key
    assert not any(key.startswith("baseline_") for key in run.data.metrics)
    images = list_artifacts(run_id=run_id, artifact_path="images")
    assert images, "no evaluation figures logged"

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


def test_train_with_baseline_and_detection_selection(
    synthetic_path: Path, tmp_path: Path, tracking_uri: str
) -> None:
    try:
        sbm.PhaseNet.from_pretrained("original")
    except Exception as error:  # download failures come in many types
        pytest.skip(f"pretrained weights not available: {error}")
    config = small_config(
        synthetic_path,
        tmp_path,
        epochs=1,
        evaluation={"plot_every_n_epochs": 0},
    )
    config = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={"selection_metric": "det/f1_mean", "selection_mode": "max"}
            ),
            "evaluation": config.evaluation.model_copy(
                update={"baseline": TrainConfig().evaluation.baseline}
            ),
        }
    )
    run = MlflowClient().get_run(train(config))
    metrics = run.data.metrics
    assert "best_det/f1_mean" in metrics
    baseline = {"baseline_det_P/f1", "baseline_pick_S/mae_s", "baseline_det/f1_mean"}
    assert baseline <= set(metrics)
    assert not list_artifacts(run_id=run.info.run_id, artifact_path="images")
