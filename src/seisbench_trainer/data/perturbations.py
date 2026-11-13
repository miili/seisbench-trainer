"""Training perturbations: reviewed re-implementations of seisbench augmentations.

Each perturbation is a callable on the generator state dict. It acts on
``state_dict["X"]``: waveforms of shape (channels, samples) in the run's
component order, and their metadata. They run after the windowing steps and
before normalization and labelling, so picks removed from the metadata are
not labelled. Input arrays and metadata are never modified in place.

Random numbers come from ``np.random``, which ``seisbench.util.worker_seeding``
seeds per DataLoader worker. Each class lists where it deviates from its
``seisbench.generate`` counterpart (seisbench 0.12).
"""

from __future__ import annotations

from numbers import Real
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from seisbench_trainer.data.phases import PHASE_COLUMNS, Augmentation, StateDict

if TYPE_CHECKING:
    from seisbench_trainer.data.loaders import AnyDataset

Waveforms = NDArray[Any]
Random = ModuleType | np.random.RandomState
"""``np.random`` (seeded per DataLoader worker) or a seeded ``RandomState``."""


def _peak(x: Waveforms) -> float:
    """Absolute peak of the demeaned waveforms, over all channels."""
    return float(np.max(np.abs(x - x.mean(axis=-1, keepdims=True)), initial=0.0))


def _horizontals(component_order: str) -> tuple[int, int]:
    """Channel indices of N and E in ``component_order``."""
    missing = set("NE") - set(component_order)
    if missing:
        raise ValueError(f"rotation needs components {sorted(missing)}")
    return component_order.index("N"), component_order.index("E")


def _rotate(x: Waveforms, north: int, east: int, rng: Random = np.random) -> Waveforms:
    """A copy of ``x`` with N and E rotated by a random angle in [-pi, pi)."""
    alpha = rng.uniform(-np.pi, np.pi)
    n, e = x[north], x[east]
    x = x.copy()
    x[north] = n * np.cos(alpha) - e * np.sin(alpha)
    x[east] = n * np.sin(alpha) + e * np.cos(alpha)
    return x


def _has_pick(metadata: dict[str, Any] | None, n_samples: int) -> bool:
    """Whether a pick column of ``metadata`` lies inside the window."""
    for column in PHASE_COLUMNS:
        value = (metadata or {}).get(column)
        if isinstance(value, Real) and 0 <= float(value) < n_samples:
            return True
    return False


class Sometimes:
    """Apply ``augmentation`` with ``probability``, else leave the sample as is.

    Replaces ``OneOf([augmentation, NullAugmentation()], [p, 1 - p])``.
    """

    def __init__(self, augmentation: Augmentation, probability: float) -> None:
        self.augmentation = augmentation
        self.probability = probability

    def __call__(self, state_dict: StateDict) -> None:
        if np.random.random() < self.probability:
            self.augmentation(state_dict)


class ChannelDropout:
    """Zero a random subset of channels: at least one, never all.

    seisbench draws 0 to c - 1 channels, so on 3C data a third of its
    "dropouts" change nothing, and its docstring promises a rescaling of the
    remaining channels that the code does not do. Here at least one channel
    is dropped, and nothing is rescaled: normalization is per channel.
    """

    def __call__(self, state_dict: StateDict) -> None:
        x, metadata = state_dict["X"]
        n_channels = x.shape[0]
        if n_channels < 2:
            return
        n_drop = np.random.randint(1, n_channels)
        dropped = np.random.permutation(n_channels)[:n_drop]
        x = x.copy()
        x[dropped] = 0
        state_dict["X"] = (x, metadata)


class AddGap:
    """Zero all channels over a stretch of ``min_samples`` to ``max_samples``.

    seisbench draws the gap end uniformly up to the window end, so gaps range
    from empty to almost the whole window, and picks inside the gap stay in
    the metadata, which then labels picks on zeros. Here the gap length is
    drawn from an explicit range and picks inside the gap are set to NaN.
    """

    def __init__(self, min_samples: int, max_samples: int) -> None:
        self.min_samples = min_samples
        self.max_samples = max_samples

    def __call__(self, state_dict: StateDict) -> None:
        x, metadata = state_dict["X"]
        n_samples = x.shape[-1]
        length = min(
            np.random.randint(self.min_samples, self.max_samples + 1), n_samples
        )
        start = np.random.randint(0, n_samples - length + 1)
        end = start + length
        x = x.copy()
        x[..., start:end] = 0
        updated = dict(metadata or {})
        for key, value in updated.items():
            if (
                key.endswith("_arrival_sample")
                and isinstance(value, Real)
                and start <= float(value) < end
            ):
                updated[key] = np.nan
        state_dict["X"] = (x, updated)


class GaussianNoise:
    """Add white noise with std ``scale`` times the waveform peak.

    ``scale`` is drawn uniformly from (``low``, ``high``) per sample. seisbench
    uses the signed maximum ``np.max(x)``, which is the DC offset for raw
    counts and too small for traces with a larger negative swing. Here the
    peak is the absolute maximum of the demeaned waveforms.
    """

    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high

    def __call__(self, state_dict: StateDict) -> None:
        x, metadata = state_dict["X"]
        std = np.random.uniform(self.low, self.high) * _peak(x)
        noise = np.random.standard_normal(x.shape).astype(x.dtype)
        state_dict["X"] = (x + std * noise, metadata)


class RotateHorizontal:
    """Rotate the two horizontal channels by a random angle in [-pi, pi).

    seisbench finds N and E through the metadata ``trace_component_order``,
    which is the order on disk, not the loaded order: with ``ENZ`` data from a
    ``ZNE`` file it rotates N with Z. Here the channel indices come from the
    run's component order.
    """

    def __init__(self, component_order: str) -> None:
        self.north, self.east = _horizontals(component_order)

    def __call__(self, state_dict: StateDict) -> None:
        x, metadata = state_dict["X"]
        state_dict["X"] = (_rotate(x, self.north, self.east), metadata)


class NoUsableNoise(ValueError):
    """No noise trace is long enough for the window, finite and not flat."""


class PeakLevel:
    """Noise peak = ``ratio`` times the waveform peak; ``ratio`` uniform.

    seisbench's ``scaling_type="peak"``. Peaks are over all channels.
    """

    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high

    def factor(self, x: Waveforms, noise: Waveforms, rng: Random) -> float:
        """Scale for the demeaned ``noise`` window added to ``x``."""
        noise_peak = _peak(noise)
        if noise_peak == 0:
            return 0.0
        return rng.uniform(self.low, self.high) * _peak(x) / noise_peak


def _std(x: Waveforms) -> float:
    """Standard deviation of the demeaned channels, pooled over channels."""
    return float(np.sqrt(np.mean((x - x.mean(axis=-1, keepdims=True)) ** 2)))


class StdLevel:
    """Noise std = ``ratio`` times the waveform std; ``ratio`` log-uniform.

    The dlf-catalog reference recipe (ratio 0.1 to 2). The std is pooled over
    the channels that are not all zero (missing components).
    """

    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high

    def factor(self, x: Waveforms, noise: Waveforms, rng: Random) -> float:
        """Scale for the demeaned ``noise`` window added to ``x``."""
        live = x.any(axis=-1)
        if not live.any():
            return 0.0
        noise_std = _std(noise[live])
        if noise_std == 0:
            return 0.0
        ratio = float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        return ratio * _std(x[live]) / noise_std


NoiseLevel = PeakLevel | StdLevel


class RealNoise:
    """Add a random window of a recorded noise trace, horizontals rotated.

    The noise window's N and E channels are rotated by a random angle (as
    ``RotateHorizontal``), so one noise recording gives many horizontal
    noise fields; without N and E in ``component_order`` it is not rotated.
    The window is then demeaned and scaled by ``level`` (``PeakLevel``:
    seisbench's ``scaling_type="peak"``; ``StdLevel``: the dlf-catalog
    reference). Channels of the waveform that are all zero (missing
    components, dropped channels) get no noise: they stay missing. Differences:
    noise traces shorter than the window, without signal or with non-finite
    samples are excluded once,
    here, where seisbench raises in the middle of training or divides by zero
    (NaN waveforms); a noise window that is all zeros is skipped. The noise
    traces must have the run's component order and sampling rate, which holds
    when they come from a split of the run.

    Windows without a pick are left as they are (seisbench adds noise to
    every window): the noise is there to lower the SNR of picks, and noise
    added to noise (noise traces, pick-free crops) teaches nothing the noise
    traces do not. ``probability`` is then the chance per window with a pick.

    ``rng`` is ``np.random`` in training; the noisy validation passes a
    ``RandomState`` seeded per trace, so its windows are the same every epoch.
    """

    def __init__(
        self,
        noise: AnyDataset,
        window_samples: int,
        level: NoiseLevel,
        component_order: str,
    ) -> None:
        self.noise = noise
        self.level = level
        self.horizontals = (
            _horizontals(component_order) if set("NE") <= set(component_order) else None
        )
        self.window_samples = window_samples
        self.indices = [
            index
            for index in range(len(noise))
            if self._usable(noise.get_sample(index)[0])
        ]
        if not self.indices:
            raise NoUsableNoise(
                f"no noise trace with at least {window_samples} finite samples "
                "and signal"
            )

    def _usable(self, waveforms: Waveforms) -> bool:
        return (
            waveforms.shape[-1] >= self.window_samples
            and bool(np.isfinite(waveforms).all())
            and _peak(waveforms) > 0
        )

    def __call__(self, state_dict: StateDict, rng: Random = np.random) -> None:
        x, metadata = state_dict["X"]
        if not _has_pick(metadata, x.shape[-1]):
            return
        trace, _ = self.noise.get_sample(int(rng.choice(self.indices)))
        start = rng.randint(0, trace.shape[-1] - x.shape[-1] + 1)
        window = trace[:, start : start + x.shape[-1]]
        if self.horizontals is not None:
            window = _rotate(window, *self.horizontals, rng)
        window = window - window.mean(axis=-1, keepdims=True)
        window[~x.any(axis=-1)] = 0  # missing components stay missing
        factor = self.level.factor(x, window, rng)
        if factor == 0:
            return
        state_dict["X"] = (x + (window * factor).astype(x.dtype), metadata)
