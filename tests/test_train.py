from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
import seisbench.models as sbm
from conftest import small_config
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts, list_artifacts, load_dict, load_text
from pydantic import JsonValue

from seisbench_trainer.config import TrainConfig
from seisbench_trainer.tracking import run as tracking_run
from seisbench_trainer.tracking.run import flatten, model_name
from seisbench_trainer.training import train as train_module
from seisbench_trainer.training.train import train


def test_flatten() -> None:
    nested: JsonValue = {"a": {"b": 1, "c": [{"d": None}, "x"]}, "e": True}
    assert flatten(nested) == {"a.b": "1", "a.c.0.d": "None", "a.c.1": "x", "e": "True"}


def test_model_name(synthetic_path: Path, tmp_path: Path) -> None:
    config = small_config(synthetic_path, tmp_path, epochs=1)
    assert model_name(config) == f"PhaseNet-{synthetic_path.name}-scratch"
    weights = config.model.model_copy(update={"pretrained": "/x/my.model:v1"})
    dotted = config.model_copy(update={"model": weights})
    assert model_name(dotted) == f"PhaseNet-{synthetic_path.name}-from--x-my-model-v1"


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
    assert "best_det/f1_mean" in run.data.metrics
    expected = (
        "pick_P/mae_s",
        "pick_S/auc",
        "pick_P/ap",
        "det_P/f1",
        "det_S/recall",
        "det_S/ap",
        "det/f1_mean",
    )
    for key in expected:
        assert len(client.get_metric_history(run_id, key)) == 2, key
    assert not any(key.startswith("baseline") for key in run.data.metrics)
    assert len(client.get_metric_history(run_id, "pick_P/n_picks")) == 1
    assert len(client.get_metric_history(run_id, "det_P/n_negatives")) == 2
    assert "det_P/within_0.5s" not in run.data.metrics
    assert "pick_P/within_0.5s" in run.data.metrics
    tags = client.get_experiment(run.info.experiment_id).tags
    assert tags["mlflow.experimentKind"] == "custom_model_development"
    pages = {a.path for a in list_artifacts(run_id=run_id, artifact_path="evaluation")}
    assert pages == {"evaluation/validation_P.html", "evaluation/validation_S.html"}
    page = load_text(f"runs:/{run_id}/evaluation/validation_P.html")
    assert page.count('"epoch ') == 2

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
    assert models[0].name == model_name(config)
    description = models[0].tags["mlflow.note.content"]
    assert "trained from scratch" in description
    assert "Best epoch" in description and "det/f1_mean" in description
    download = Path(
        download_artifacts(
            f"models:/{models[0].model_id}", dst_path=str(tmp_path / "download")
        )
    )
    extra = next(download.rglob("extra_files"))
    assert {path.name for path in extra.glob("*.json")} == {
        f"{model_name(config)}.json",
        tracking_run.TRAIN_CONFIG,
    }
    stored = (extra / tracking_run.TRAIN_CONFIG).read_text()
    assert TrainConfig.model_validate_json(stored) == config
    model = sbm.PhaseNet.load(extra / model_name(config))
    assert model.component_order == config.data.component_order

    # thresholds selected on the validation set (logged as float32)
    for phase in ("P", "S"):
        value = model.default_args[f"{phase}_threshold"]
        logged = client.get_metric_history(run_id, f"det_{phase}/threshold")
        assert any(abs(value - m.value) < 1e-6 for m in logged)
    figures = sorted(p.name for p in extra.glob("*.png"))
    assert figures and all(name.startswith("validation_") for name in figures)


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
        epochs=2,
        evaluation={"plot_every_n_epochs": 0},
    )
    config = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(
                update={"baseline": TrainConfig().evaluation.baseline}
            ),
        }
    )
    run_id = train(config)
    client = MlflowClient()
    metrics = client.get_run(run_id).data.metrics
    assert "best_det/f1_mean" in metrics
    baseline = {"baseline/det_P/f1", "baseline/pick_S/mae_s", "baseline/det/f1_mean"}
    assert baseline <= set(metrics)
    # the fixed baseline is evaluated and logged once; the model every epoch
    assert len(client.get_metric_history(run_id, "baseline/det_P/f1")) == 1
    assert len(client.get_metric_history(run_id, "det_P/f1")) == 2
    assert not list_artifacts(run_id=run_id, artifact_path="evaluation")


def test_early_stopping_uses_min_delta() -> None:
    from typing import Any

    from lightning.pytorch.callbacks import EarlyStopping

    placeholder: Any = object()  # evaluation and checkpoint are passed through
    callbacks = train_module._callbacks(TrainConfig(), placeholder, placeholder)
    (stopping,) = [c for c in callbacks if isinstance(c, EarlyStopping)]
    assert stopping.monitor == "det/f1_mean"
    assert stopping.patience == 15
    assert abs(stopping.min_delta) == 0.002
    no_stop = TrainConfig.model_validate(
        {"training": {"early_stopping_patience": None}}
    )
    callbacks = train_module._callbacks(no_stop, placeholder, placeholder)
    assert not any(isinstance(c, EarlyStopping) for c in callbacks)
