"""Training at another rate than the data or the pretrained weights have.

The synthetic data is written at 100 Hz and loaded at 50 Hz; the local
"pretrained" PhaseNet is saved at 100 Hz and run at 50 Hz (time-stretched).
"""

from __future__ import annotations

import re
from pathlib import Path

import mlflow
import numpy as np
import pytest
import seisbench.models as sbm
from conftest import small_config
from mlflow.artifacts import download_artifacts
from pydantic import ValidationError

from seisbench_trainer.config import TrainConfig
from seisbench_trainer.data import LocalDataset
from seisbench_trainer.models import PhaseNetConfig
from seisbench_trainer.training.train import train

NATIVE = 100.0
STRETCHED = 50.0


@pytest.fixture
def weights_100hz(tmp_path: Path) -> Path:
    """A local PhaseNet saved at 100 Hz (stands in for pretrained weights)."""
    path = tmp_path / "weights" / "PhaseNet-100hz"
    path.parent.mkdir()
    sbm.PhaseNet(component_order="ZNE", sampling_rate=NATIVE).save(path)
    return path


def test_dataset_is_resampled_with_its_picks(synthetic_path: Path) -> None:
    native = LocalDataset(path=synthetic_path).load("ZNE", NATIVE)
    half = LocalDataset(path=synthetic_path).load("ZNE", STRETCHED)
    assert half.resample_zerophase
    index = int(np.flatnonzero(native.metadata["trace_category"] == "earthquake")[0])
    x_native, meta_native = native.get_sample(index)
    x_half, meta_half = half.get_sample(index)
    assert x_half.shape[-1] == x_native.shape[-1] // 2
    assert meta_half["trace_sampling_rate_hz"] == STRETCHED
    for column in ("trace_p_arrival_sample", "trace_s_arrival_sample"):
        assert meta_half[column] == pytest.approx(meta_native[column] / 2)


def test_zero_phase_decimation_keeps_picks_aligned(synthetic_path: Path) -> None:
    """The causal anti-alias filter delays the waveforms against the picks."""
    native = LocalDataset(path=synthetic_path).load("ZNE", NATIVE)
    events = native.metadata["trace_category"] == "earthquake"
    index = int(np.flatnonzero(events)[0])
    x_native, _ = native.get_sample(index)
    reference = x_native[0, ::2]  # Z at 50 Hz, no filter: aligned with the picks
    lags = {}
    for zerophase in (True, False):
        dataset = LocalDataset(path=synthetic_path).load(
            "ZNE", STRETCHED, resample_zerophase=zerophase
        )
        z = dataset.get_sample(index)[0][0]
        correlation = np.correlate(z, reference, mode="full")
        lags[zerophase] = int(np.argmax(correlation)) - (len(reference) - 1)
    assert lags[True] == 0
    assert lags[False] == 2  # about 45 ms at 50 Hz


def test_pretrained_runs_at_the_configured_rate(weights_100hz: Path) -> None:
    config = PhaseNetConfig(pretrained=str(weights_100hz), sampling_rate=STRETCHED)
    model = config.build("ZNE", STRETCHED)
    assert model.sampling_rate == STRETCHED
    assert model.in_samples == 3001  # 60 s at 50 Hz
    assert config.variant() == "from-PhaseNet-100hz-50hz"
    native = PhaseNetConfig(pretrained=str(weights_100hz))
    assert native.load_pretrained().sampling_rate == NATIVE


def test_rate_mismatch_names_the_setting(weights_100hz: Path) -> None:
    with pytest.raises(ValueError, match=re.escape('set "model.sampling_rate" to 50')):
        PhaseNetConfig(pretrained=str(weights_100hz)).build("ZNE", STRETCHED)


@pytest.mark.parametrize("section", ["model", "baseline"])
def test_model_rate_must_equal_data_rate(section: str) -> None:
    model = {"type": "PhaseNet", "pretrained": "original", "sampling_rate": 50.0}
    raw: dict[str, object] = {"data": {"sampling_rate": 100.0}}
    raw |= (
        {"model": model} if section == "model" else {"evaluation": {"baseline": model}}
    )
    with pytest.raises(
        ValidationError, match=re.escape("must equal data.sampling_rate")
    ):
        TrainConfig.model_validate(raw)


def test_fine_tuning_at_50hz_exports_a_50hz_model(
    synthetic_path: Path, tmp_path: Path, weights_100hz: Path, tracking_uri: str
) -> None:
    stretched = {
        "type": "PhaseNet",
        "pretrained": str(weights_100hz),
        "sampling_rate": STRETCHED,
    }
    config = small_config(synthetic_path, tmp_path, epochs=1)
    config = TrainConfig.model_validate(
        config.model_dump(mode="json")
        | {
            "model": stretched,
            "data": config.data.model_dump(mode="json") | {"sampling_rate": STRETCHED},
            # twice the 60 s model input, as 60 s / 30 s is at 100 Hz
            "augmentation": {
                "train": [
                    {
                        "type": "pick_window",
                        "probability": 0.667,
                        "length_s": 120.0,
                        "before_s": 60.0,
                    },
                    {"type": "random_window"},
                ]
            },
            "evaluation": config.evaluation.model_dump(mode="json")
            | {"baseline": stretched, "plot_every_n_epochs": 0},
        }
    )
    run_id = train(config)
    run = mlflow.get_run(run_id)
    assert any(key.startswith("baseline/") for key in run.data.metrics)
    (logged,) = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id], output_format="list"
    )
    assert "time-stretched" in logged.tags["mlflow.note.content"]
    download = Path(
        download_artifacts(f"models:/{logged.model_id}", dst_path=str(tmp_path / "dl"))
    )
    (exported,) = [p for p in download.rglob("*.pt") if p.parent.name == "extra_files"]
    model = sbm.PhaseNet.load(exported.with_suffix(""))
    assert model.sampling_rate == STRETCHED


def test_pick_window_must_exceed_the_model_input(synthetic_path: Path) -> None:
    """At 50 Hz the 60 s default context is shorter than PhaseNet's 60.02 s."""
    from seisbench_trainer.data.steps import PickWindow, StepContext

    dataset = LocalDataset(path=synthetic_path).load("ZNE", STRETCHED)
    context = StepContext(STRETCHED, 3001, "ZNE", dataset)
    with pytest.raises(ValueError, match="must exceed the model input"):
        PickWindow(probability=0.5).build(context)
    PickWindow(probability=0.5, length_s=120.0, before_s=60.0).build(context)
