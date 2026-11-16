"""Augmentation pipelines for training and evaluation.

Label augmentations depend on the model and are appended by the caller.
"""

from __future__ import annotations

import zlib

import numpy as np
import seisbench.generate as sbg

from seisbench_trainer.data.arrivals import Arrivals, StationPicks
from seisbench_trainer.data.config import AugmentationConfig
from seisbench_trainer.data.perturbations import RealNoise
from seisbench_trainer.data.phases import (
    PHASES,
    Augmentation,
    StateDict,
    own_picks,
    phase_arrivals,
)
from seisbench_trainer.data.steps import RandomWindow, StepContext
from seisbench_trainer.units import seconds_to_samples


class SeededPickWindow(sbg.FixedWindow):
    """Cut a window with the first pick at a random but per-trace fixed offset.

    The offset depends only on the seed and the trace name. Validation windows
    are then identical in every epoch and run, but the pick is not always at
    the same position. Pick-free traces get a seeded random window. Only the
    trace's own event places the window.
    """

    def __init__(self, windowlen: int, seed: int) -> None:
        super().__init__(windowlen=windowlen, strategy="pad")
        self.length = windowlen
        self.seed = seed

    def __call__(
        self, state_dict: StateDict, p0: int | None = None, windowlen: int | None = None
    ) -> None:
        waveforms, metadata = state_dict["X"]
        metadata = metadata or {}
        name = str(metadata.get("trace_name", ""))
        rng = np.random.default_rng([self.seed, zlib.crc32(name.encode())])
        picks = [*own_picks(metadata, "P"), *own_picks(metadata, "S")]
        if picks:
            start = int(min(picks)) - int(rng.integers(0, self.length))
        else:
            last_start = max(0, waveforms.shape[-1] - self.length)
            start = int(rng.integers(0, last_start + 1))
        super().__call__(state_dict, p0=start)


class ZeroNonFinite:
    """Set NaN and infinite samples to zero; the first step of every pipeline.

    Datasets store missing components and gaps as NaN (deeplowfrequency:
    whole NaN channels of stations without that component). One NaN sample
    makes the loss of its whole batch NaN, and training never recovers. Zero
    is what seisbench's ``annotate`` feeds for a missing component, so the
    model sees the same at inference. Picks in a NaN stretch stay labelled.
    """

    def __call__(self, state_dict: StateDict) -> None:
        x, metadata = state_dict["X"]
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            state_dict["X"] = (x, metadata)


class SeededRealNoise:
    """``RealNoise`` with a random state that depends only on seed and trace.

    The noisy validation windows are then identical in every epoch and run.
    """

    def __init__(self, noise: RealNoise, seed: int) -> None:
        self.noise = noise
        self.seed = seed

    def __call__(self, state_dict: StateDict) -> None:
        metadata = state_dict["X"][1] or {}
        name = str(metadata.get("trace_name", ""))
        rng = np.random.RandomState([self.seed, zlib.crc32(name.encode()), 1])
        self.noise(state_dict, rng)


class UnreadPhaseMask:
    """Add key ``label_mask``: 0 where a phase may arrive unread, else 1.

    Shape ``(len(PHASES), samples)``, float32, row order ``PHASES``. Each
    event of the window's arrivals counts on its own: if it has S but no P,
    its P lies before its first S, and row P is 0 up to that S minus
    ``margin``. If it has P but no S, row S is 0 from its first P plus
    ``margin``. Arrival positions are window-relative, so this runs right
    after the windows, before perturbations that remove arrivals
    (``add_gap``). With ``enabled=False`` (``missing_picks="absent"``) the
    mask is all ones.
    """

    def __init__(self, margin: int, enabled: bool = True) -> None:
        self.margin = margin
        self.enabled = enabled

    def __call__(self, state_dict: StateDict) -> None:
        x, metadata = state_dict["X"]
        n_samples = x.shape[-1]
        mask = np.ones((len(PHASES), n_samples), dtype=np.float32)
        if self.enabled:
            p_picks, p_events = phase_arrivals(metadata, "P")
            s_picks, s_events = phase_arrivals(metadata, "S")
            p_index, s_index = PHASES.index("P"), PHASES.index("S")
            for event in np.union1d(p_events, s_events):
                p = p_picks[p_events == event]
                s = s_picks[s_events == event]
                if s.size and not p.size:
                    end = int(np.clip(s.min() - self.margin, 0, n_samples))
                    mask[p_index, :end] = 0
                elif p.size and not s.size:
                    start = int(np.clip(p.min() + self.margin, 0, n_samples))
                    mask[s_index, start:] = 0
        state_dict["label_mask"] = (mask, None)


class PickSamples:
    """Add key ``pick_samples``: the arrival positions in the window per phase.

    Shape ``(len(PHASES), max_picks)``, float32, sorted, NaN-padded. Arrivals
    outside the window are dropped. Must run after all windowing steps.
    """

    def __init__(self, windowlen: int, max_picks: int = 16) -> None:
        self.windowlen = windowlen
        self.max_picks = max_picks

    def __call__(self, state_dict: StateDict) -> None:
        metadata = state_dict["X"][1] or {}
        out = np.full((len(PHASES), self.max_picks), np.nan, dtype=np.float32)
        for row, phase in enumerate(PHASES):
            picks, _ = phase_arrivals(metadata, phase)
            inside = np.unique(picks[(picks >= 0) & (picks < self.windowlen)])
            picks = inside[: self.max_picks]
            out[row, : len(picks)] = picks
        state_dict["pick_samples"] = (out, None)


def _normalize(config: AugmentationConfig) -> list[Augmentation]:
    return [
        sbg.Normalize(
            demean_axis=-1, amp_norm_axis=-1, amp_norm_type=config.normalization
        ),
        sbg.ChangeDtype(np.float32),
    ]


def _unread_mask(config: AugmentationConfig, sampling_rate: float) -> UnreadPhaseMask:
    sigma = seconds_to_samples(config.label_sigma_s, sampling_rate)
    return UnreadPhaseMask(3 * sigma, enabled=config.missing_picks == "unread")


def train_augmentations(
    config: AugmentationConfig, context: StepContext
) -> list[Augmentation]:
    """The arrivals, the configured training steps, then normalization.

    The ``label_mask`` is computed right after the ``random_window`` crop.
    """
    steps: list[Augmentation] = [ZeroNonFinite(), Arrivals(context.station_picks)]
    for step in config.train:
        steps.append(step.build(context))
        if isinstance(step, RandomWindow):
            steps.append(_unread_mask(config, context.sampling_rate))
    return [*steps, *_normalize(config)]


def eval_augmentations(
    config: AugmentationConfig,
    window_samples: int,
    seed: int,
    sampling_rate: float,
    noise: RealNoise | None = None,
    station_picks: StationPicks | None = None,
) -> list[Augmentation]:
    """Deterministic windows for validation and test, plus ``pick_samples``
    and ``label_mask``.

    With ``noise``, real noise is added to each window, seeded per trace.
    With ``station_picks``, the arrivals include other events at the station.
    """
    return [
        ZeroNonFinite(),
        Arrivals(station_picks),
        SeededPickWindow(window_samples, seed),
        _unread_mask(config, sampling_rate),
        *([SeededRealNoise(noise, seed)] if noise is not None else []),
        *_normalize(config),
        PickSamples(window_samples),
    ]
