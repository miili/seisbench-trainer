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
    PeakLevel,
    RealNoise,
    RotateHorizontal,
    Sometimes,
    StdLevel,
)
from seisbench_trainer.data.phases import Augmentation, StateDict
from seisbench_trainer.data.steps import PeakRatio, StdRatio, StepContext
from seisbench_trainer.data.steps import RealNoise as RealNoiseStep

N_SAMPLES = 3001
PICK = {"trace_P_arrival_sample": 1000.0}


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
    level = PeakRatio(ratio=(0.5, 0.5))
    gate = RealNoiseStep(probability=1.0, level=level).build(context)
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
    out, _ = apply(noise, x, PICK)
    added = out - x
    assert np.abs(added).max() == pytest.approx(0.5 * peak, rel=1e-3)
    assert np.isfinite(out).all()


def test_real_noise_rotates_the_noise_horizontals(synthetic_path: Path) -> None:
    train = train_split(synthetic_path)
    rotated = RealNoise(train, N_SAMPLES, PeakLevel(0.5, 0.5), "ZNE")
    plain = RealNoise(train, N_SAMPLES, PeakLevel(0.5, 0.5), "Z")  # not rotated
    x = waveforms()
    for seed in range(5):
        # the noise trace and offset are drawn before the angle: same window
        np.random.seed(seed)
        a = apply(rotated, x, PICK)[0] - x
        np.random.seed(seed)
        b = apply(plain, x, PICK)[0] - x
        factor = np.abs(a[0]).max() / np.abs(b[0]).max()  # peaks differ: rescaled
        np.testing.assert_allclose(a[0], factor * b[0], rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(
            (a[1:] ** 2).sum(axis=0),
            factor**2 * (b[1:] ** 2).sum(axis=0),
            rtol=1e-3,
            atol=1e-6,
        )
        assert not np.allclose(a[1:], factor * b[1:], rtol=1e-2)


def test_real_noise_leaves_windows_without_picks(synthetic_path: Path) -> None:
    noise = RealNoise(
        train_split(synthetic_path), N_SAMPLES, PeakLevel(0.5, 0.5), "ZNE"
    )
    x = waveforms()
    outside = {"trace_P_arrival_sample": -5.0, "trace_S_arrival_sample": np.nan}
    for metadata in (None, {}, outside, {"trace_S_arrival_sample": N_SAMPLES}):
        out, _ = apply(noise, x, metadata)
        np.testing.assert_array_equal(out, x)
    out, _ = apply(noise, x, {"trace_S_arrival_sample": N_SAMPLES - 1})
    assert not np.array_equal(out, x)


def test_real_noise_excludes_short_traces(synthetic_path: Path) -> None:
    train = train_split(synthetic_path)
    too_long = 10 * 6000  # longer than every synthetic trace (60 s)
    with pytest.raises(ValueError, match="no noise trace"):
        RealNoise(train, too_long, PeakLevel(0.0, 1.0), "ZNE")


@pytest.mark.parametrize(
    "query", ['trace_category == "explosion"', 'no_such_column == "noise"']
)
def test_real_noise_without_noise_traces_adds_nothing(
    synthetic_path: Path, query: str, caplog: pytest.LogCaptureFixture
) -> None:
    context = StepContext(100.0, N_SAMPLES, "ZNE", train_split(synthetic_path))
    step = RealNoiseStep(probability=1.0, query=query).build(context)
    assert "selects no training traces" in caplog.text
    x = waveforms()
    np.testing.assert_array_equal(apply(step, x, PICK)[0], x)


def test_std_level_sets_the_std_ratio_without_zero_channels(
    synthetic_path: Path,
) -> None:
    train = train_split(synthetic_path)
    x = waveforms()
    x[1] = 0  # a missing component
    noise = RealNoise(train, N_SAMPLES, StdLevel(0.7, 0.7), "ZNE")
    out, _ = apply(noise, x, PICK)
    added = out - x
    assert not added[1].any()  # the missing component stays missing
    live = [0, 2]
    ratio = added[live].std() / x[live].std()
    assert ratio == pytest.approx(0.7, rel=1e-3)


def test_peak_level_leaves_zero_channels_alone(synthetic_path: Path) -> None:
    noise = RealNoise(
        train_split(synthetic_path), N_SAMPLES, PeakLevel(0.5, 0.5), "ZNE"
    )
    x = waveforms()
    x[2] = 0
    out, _ = apply(noise, x, PICK)
    assert not out[2].any()


def test_std_ratio_is_log_uniform() -> None:
    level = StdRatio().build()
    rng = np.random.RandomState(0)
    draws = np.array(
        [
            np.exp(rng.uniform(np.log(level.low), np.log(level.high)))
            for _ in range(4000)
        ]
    )
    assert draws.min() >= 0.1 and draws.max() <= 2.0
    # log-uniform: the median is the geometric mean, sqrt(0.1 * 2) = 0.447
    assert np.median(draws) == pytest.approx(np.sqrt(0.2), rel=0.05)
    with pytest.raises(ValueError, match="low > 0"):
        StdRatio(ratio=(0.0, 1.0))


def test_real_noise_without_long_enough_traces_adds_nothing(
    synthetic_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """60 s noise traces are 3000 samples at 50 Hz; a stretched model needs 3001."""
    too_long = 10 * 6000
    context = StepContext(100.0, too_long, "ZNE", train_split(synthetic_path))
    step = RealNoiseStep(probability=1.0).build(context)
    assert "no real noise is added to training windows" in caplog.text
    x = np.zeros((3, too_long), dtype=np.float32)
    x[0, 100] = 1.0
    np.testing.assert_array_equal(apply(step, x, PICK)[0], x)
