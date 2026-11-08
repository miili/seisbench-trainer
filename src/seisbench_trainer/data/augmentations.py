"""Augmentation pipelines for training and evaluation.

Label augmentations depend on the model and are appended by the caller.
"""

import math
import zlib
from typing import Any

import numpy as np
import seisbench.generate as sbg

from seisbench_trainer.data.config import AugmentationConfig
from seisbench_trainer.data.phases import (
    PHASE_COLUMNS,
    PHASES,
    Augmentation,
    StateDict,
)
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
            columns = [c for c, p in PHASE_COLUMNS.items() if p == phase]
            inside = {
                v for v in _pick_values(metadata, columns) if 0 <= v < self.windowlen
            }
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


def train_augmentations(
    config: AugmentationConfig, sampling_rate: float, window_samples: int
) -> list[Augmentation]:
    """Random context window around a pick (or not), then a random crop."""
    probability = config.pick_context_probability
    return [
        sbg.OneOf(
            [
                sbg.WindowAroundSample(
                    list(PHASE_COLUMNS),
                    samples_before=seconds_to_samples(
                        config.pick_context_before_s, sampling_rate
                    ),
                    windowlen=seconds_to_samples(config.pick_context_s, sampling_rate),
                    selection="random",
                    strategy="variable",
                ),
                sbg.NullAugmentation(),
            ],
            probabilities=[probability, 1.0 - probability],
        ),
        sbg.RandomWindow(windowlen=window_samples, strategy="pad"),
        *_normalize(config),
    ]


def eval_augmentations(
    config: AugmentationConfig, window_samples: int, seed: int
) -> list[Augmentation]:
    """Deterministic windows for validation and test, plus ``pick_samples``."""
    return [
        SeededPickWindow(window_samples, seed),
        *_normalize(config),
        PickSamples(window_samples),
    ]
