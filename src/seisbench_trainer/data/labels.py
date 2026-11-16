"""Training targets from picks.

Replaces seisbench's ``ProbabilisticLabeller``, which also handles polarity
labels, multi-window batches, column auto-detection and dimension orders, and
reads only the picks of the trace's own metadata row. This labeller does one
thing: one curve per pick, the maximum per phase, and noise as the rest. The
curves and the arithmetic are seisbench's, so the targets are the same for
the same picks.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from seisbench_trainer.data.phases import PHASES, StateDict, phase_picks

LabelShape = Literal["gaussian", "triangle", "box"]
"""Shape of the label curve around a pick."""


def _curve(shape: LabelShape, onset: float, length: int, sigma: float) -> NDArray[Any]:
    """The label curve of one pick; seisbench's formulas."""
    x = np.arange(length)
    if shape == "gaussian":
        return np.exp(-np.power(x - onset, 2.0) / (2 * np.power(sigma, 2.0)))
    if shape == "triangle":
        rising = -(x - onset) / sigma + 1
        rising *= (rising >= 0) & (rising <= 1)
        falling = (x - onset) / sigma + 1
        falling *= (falling >= 0) & (falling < 1)
        return rising + falling
    return ((x - onset) <= sigma) & (-(x - onset + 1) < sigma)


class PickLabeller:
    """Write the target ``y``: shape ``(len(labels), samples)``, float64.

    Every finite pick of a phase (``phase_picks``), inside the window or
    not, adds its curve; a phase row is the maximum of its curves. Where the
    phase rows sum to more than 1 they are scaled to sum to 1, and the noise
    row (``N`` in ``labels``, any case) is 1 minus their sum.
    """

    def __init__(self, labels: str, sigma: float, shape: LabelShape) -> None:
        self.rows = {phase: labels.index(phase) for phase in PHASES}
        self.noise = labels.lower().index("n")
        self.n_labels = len(labels)
        self.sigma = sigma
        self.shape: LabelShape = shape

    def __call__(self, state_dict: StateDict) -> None:
        x, metadata = state_dict["X"]
        length = x.shape[-1]
        y = np.zeros((self.n_labels, length))
        for phase, row in self.rows.items():
            for onset in phase_picks(metadata, phase):
                y[row] = np.maximum(
                    y[row], _curve(self.shape, onset, length, self.sigma)
                )
        y /= np.maximum(1, np.nansum(y, axis=0, keepdims=True))
        y[self.noise] = 1 - np.nansum(y, axis=0)
        state_dict["y"] = (y, None)
