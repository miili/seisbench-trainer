"""The reviewed perturbations, including where they deviate from seisbench."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import seisbench.data as sbd
from numpy.typing import NDArray

from seisbench_trainer.data import LocalDataset
from seisbench_trainer.data.perturbations import (
    AddGap,
    ChannelDropout,
    GaussianNoise,
    RealNoise,
    RotateHorizontal,
    Sometimes,
)
from seisbench_trainer.data.phases import Augmentation, StateDict
from seisbench_trainer.data.steps import RealNoise as RealNoiseStep
from seisbench_trainer.data.steps import StepContext

N_SAMPLES = 3001


def waveforms(seed: int = 0, offset: float = 0.0) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return (offset + rng.normal(0.0, 1.0, (3, N_SAMPLES))).astype(np.float32)


def apply(
    augmentation: Augmentation, x: NDArray[Any], metadata: dict[str, Any] | None = None
) -> tuple[NDArray[Any], dict[str, Any] | None]:
    state: StateDict = {"X": (x, metadata)}
    augmentation(state)
    return state["X"]


def test_sometimes_applies_with_probability() -> None:
    calls: list[int] = []

    def count(state: StateDict) -> None:
        calls.append(1)

    np.random.seed(0)
    for probability, expected in ((0.0, 0), (1.0, 1000)):
        calls.clear()
        gate = Sometimes(count, probability)
        for _ in range(1000):
            gate({})
        assert len(calls) == expected
    calls.clear()
    gate = Sometimes(count, 0.3)
    for _ in range(1000):
        gate({})
    assert 250 < len(calls) < 350


def test_channel_dropout_drops_at_least_one_never_all() -> None:
    x = waveforms()
    np.random.seed(0)
    dropped_counts: set[int] = set()
    for _ in range(200):
        out, _ = apply(ChannelDropout(), x)
        dropped = int((~out.any(axis=-1)).sum())
        dropped_counts.add(dropped)
        kept = out.any(axis=-1)
        np.testing.assert_array_equal(out[kept], x[kept])  # no rescaling
    assert dropped_counts == {1, 2}
    assert x.all()  # input untouched


def test_add_gap_length_and_picks_in_gap() -> None:
    x = waveforms()
    metadata = {"trace_p_arrival_sample": 1000, "trace_s_arrival_sample": 2500.0}
    np.random.seed(0)
    removed = kept = 0
    for _ in range(300):
        out, meta = apply(AddGap(50, 500), x, metadata)
        assert meta is not None
        zero = np.flatnonzero(~out.any(axis=0))
        assert 50 <= zero.size <= 500
        assert np.all(np.diff(zero) == 1)  # one contiguous gap
        for key, pick in metadata.items():
            if zero[0] <= pick <= zero[-1]:
                assert np.isnan(meta[key])
                removed += 1
            else:
                assert meta[key] == pick
                kept += 1
    assert removed > 0 and kept > 0
    assert metadata["trace_p_arrival_sample"] == 1000  # input untouched
    assert x.all()


def test_gaussian_noise_ignores_dc_offset() -> None:
    """seisbench scales by np.max(x): with an offset the noise would be huge."""
    x = waveforms(offset=1e6)
    peak = float(np.abs(x - x.mean(axis=-1, keepdims=True)).max())
    np.random.seed(0)
    out, _ = apply(GaussianNoise(0.1, 0.1), x)
    added = (out - x).astype(np.float64)
    assert added.std() == pytest.approx(0.1 * peak, rel=0.1)


def test_rotation_in_loaded_order_keeps_z() -> None:
    x = waveforms()  # loaded as ENZ: Z is the last channel
    np.random.seed(0)
    out, _ = apply(RotateHorizontal("ENZ"), x, {"trace_component_order": "ZNE"})
    np.testing.assert_array_equal(out[2], x[2])
    assert not np.allclose(out[:2], x[:2])
    np.testing.assert_allclose(
        (out[:2] ** 2).sum(axis=0), (x[:2] ** 2).sum(axis=0), rtol=1e-4
    )


def test_rotation_needs_horizontals() -> None:
    with pytest.raises(ValueError, match="needs components"):
        RotateHorizontal("Z")


def train_split(synthetic_path: Path) -> sbd.WaveformDataset:
    dataset = LocalDataset(path=synthetic_path).load("ZNE", 100.0)
    return dataset.train_dev_test()[0]


def test_real_noise_uses_noise_traces_of_the_train_split(synthetic_path: Path) -> None:
    train = train_split(synthetic_path)
    context = StepContext(100.0, N_SAMPLES, "ZNE", train)
    gate = RealNoiseStep(probability=1.0, scale=(0.5, 0.5)).build(context)
    assert isinstance(gate, Sometimes)
    noise = gate.augmentation
    assert isinstance(noise, RealNoise)
    metadata = noise.noise.metadata
    assert len(metadata) > 0
    assert set(metadata["trace_category"]) == {"noise"}
    assert set(metadata["split"]) == {"train"}

    x = waveforms()
    peak = float(np.abs(x - x.mean(axis=-1, keepdims=True)).max())
    np.random.seed(0)
    out, _ = apply(noise, x)
    added = out - x
    assert np.abs(added).max() == pytest.approx(0.5 * peak, rel=1e-3)
    assert np.isfinite(out).all()


def test_real_noise_excludes_short_traces(synthetic_path: Path) -> None:
    train = train_split(synthetic_path)
    too_long = 10 * 6000  # longer than every synthetic trace (60 s)
    with pytest.raises(ValueError, match="no noise trace"):
        RealNoise(train, too_long, 0.0, 1.0)


def test_real_noise_query_must_select_traces(synthetic_path: Path) -> None:
    context = StepContext(100.0, N_SAMPLES, "ZNE", train_split(synthetic_path))
    step = RealNoiseStep(probability=1.0, query='trace_category == "explosion"')
    with pytest.raises(ValueError, match="selects no traces"):
        step.build(context)
