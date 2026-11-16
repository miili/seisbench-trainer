"""Augmentation pipelines for training and evaluation.

Label augmentations depend on the model and are appended by the caller.
"""

from __future__ import annotations

import math
import zlib
from typing import Any

import numpy as np
import seisbench.generate as sbg

from seisbench_trainer.data.config import AugmentationConfig
from seisbench_trainer.data.perturbations import RealNoise
from seisbench_trainer.data.phases import (
    PHASE_COLUMNS,
    PHASES,
    Augmentation,
    StateDict,
    phase_columns,
)
from seisbench_trainer.data.steps import RandomWindow, StepContext
from seisbench_trainer.units import seconds_to_samples


def _pick_values(metadata: dict[str, Any], columns: list[str]) -> list[float]:
    values: list[float] = []
    for column in columns:
        value = metadata.get(column)
        if isinstance(value, int | float) and math.isfinite(value):
            values.append(float(value))
    return values


class SeededPickWindow(sbg.FixedWindow):
    """Cut a window with the first pick at a random but per-trace fixed offset.

    The offset depends only on the seed and the trace name. Validation windows
    are then identical in every epoch and run, but the pick is not always at
    the same position. Pick-free traces get a seeded random window.
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
        picks = _pick_values(metadata, list(PHASE_COLUMNS))
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

    Shape ``(len(PHASES), samples)``, float32, row order ``PHASES``. If a trace
    has S but no P pick, P lies before its first S: row P is 0 up to that S
    minus ``margin``. If it has P but no S, row S is 0 from its first P plus
    ``margin``. Pick positions are window-relative, so this runs right after
    the windows, before perturbations that remove picks (``add_gap``). With
    ``enabled=False`` (``missing_picks="absent"``) the mask is all ones.
    """

    def __init__(self, margin: int, enabled: bool = True) -> None:
        self.margin = margin
        self.enabled = enabled

    def __call__(self, state_dict: StateDict) -> None:
        x, metadata = state_dict["X"]
        n_samples = x.shape[-1]
        mask = np.ones((len(PHASES), n_samples), dtype=np.float32)
        if self.enabled:
            first: dict[str, float] = {}
            for phase in PHASES:
                picks = _pick_values(metadata or {}, phase_columns(phase))
                if picks:
                    first[phase] = min(picks)
            p_index, s_index = PHASES.index("P"), PHASES.index("S")
            if "S" in first and "P" not in first:
                end = int(np.clip(first["S"] - self.margin, 0, n_samples))
                mask[p_index, :end] = 0
            elif "P" in first and "S" not in first:
                start = int(np.clip(first["P"] + self.margin, 0, n_samples))
                mask[s_index, start:] = 0
        state_dict["label_mask"] = (mask, None)


class PickSamples:
    """Add key ``pick_samples``: the pick positions in the window per phase.

    Shape ``(len(PHASES), max_picks)``, float32, sorted, NaN-padded. Picks
    outside the window are dropped. Must run after all windowing steps.
    """

    def __init__(self, windowlen: int, max_picks: int = 4) -> None:
        self.windowlen = windowlen
        self.max_picks = max_picks

    def __call__(self, state_dict: StateDict) -> None:
        metadata = state_dict["X"][1] or {}
        out = np.full((len(PHASES), self.max_picks), np.nan, dtype=np.float32)
        for row, phase in enumerate(PHASES):
            picks = _pick_values(metadata, phase_columns(phase))
            inside = {v for v in picks if 0 <= v < self.windowlen}
            picks = sorted(inside)[: self.max_picks]
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
    """The configured training steps, then normalization.

    The ``label_mask`` is computed right after the ``random_window`` crop.
    """
    steps: list[Augmentation] = [ZeroNonFinite()]
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
) -> list[Augmentation]:
    """Deterministic windows for validation and test, plus ``pick_samples``
    and ``label_mask``.

    With ``noise``, real noise is added to each window, seeded per trace.
    """
    return [
        ZeroNonFinite(),
        SeededPickWindow(window_samples, seed),
        _unread_mask(config, sampling_rate),
        *([SeededRealNoise(noise, seed)] if noise is not None else []),
        *_normalize(config),
        PickSamples(window_samples),
    ]
