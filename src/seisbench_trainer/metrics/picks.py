"""Pick extraction for one phase. Pure NumPy; all positions in samples.

Inputs are the predicted probability curves ``probs`` of shape
(traces, samples) and the true pick positions ``picks`` of shape
(traces, max_picks), NaN-padded and sorted per trace.

Both functions return ``PickScores``: one residual and one score per true
pick, plus the scores of the negative examples.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PickScores:
    residuals: FloatArray
    """Predicted minus true pick sample, per true pick; NaN if not found."""
    scores: FloatArray
    """Probability assigned to each true pick; 0 if not found."""
    negative_scores: FloatArray
    """Scores of the negative examples (noise windows or false picks)."""

    @classmethod
    def concatenate(cls, parts: Sequence[Self]) -> Self:
        return cls(
            residuals=np.concatenate([p.residuals for p in parts]),
            scores=np.concatenate([p.scores for p in parts]),
            negative_scores=np.concatenate([p.negative_scores for p in parts]),
        )


def _flat_picks(picks: NDArray[np.floating]) -> tuple[NDArray[np.intp], FloatArray]:
    """Trace index and position of every finite pick."""
    traces, slots = np.nonzero(np.isfinite(picks))
    return traces, picks[traces, slots].astype(np.float64)


def pick_accuracy(
    probs: NDArray[np.floating],
    picks: NDArray[np.floating],
    half_window: int,
    edge: int,
) -> PickScores:
    """Window-based pick accuracy around the expected picks.

    For each true pick, only the samples within ``half_window`` of it count.
    Where the windows of two picks in one trace overlap, each sample belongs
    to the nearer pick. The predicted pick is the highest point in the
    window, its value is the score.

    Negatives: the maximum probability of each trace without any pick of this
    phase, ignoring ``edge`` samples at both ends.
    """
    n_traces, n_samples = probs.shape
    traces, positions = _flat_picks(picks)

    lower = positions - half_window
    upper = positions + half_window
    # Neighbouring picks in the same trace split their windows at the midpoint.
    next_is_same = traces[1:] == traces[:-1]
    same_trace_before = np.zeros(traces.size, dtype=bool)
    same_trace_before[1:] = next_is_same
    same_trace_after = np.zeros(traces.size, dtype=bool)
    same_trace_after[:-1] = next_is_same
    midpoints = (positions[1:] + positions[:-1]) / 2
    lower[same_trace_before] = np.maximum(
        lower[same_trace_before], midpoints[same_trace_before[1:]]
    )
    upper[same_trace_after] = np.minimum(
        upper[same_trace_after], midpoints[same_trace_after[:-1]]
    )

    index: NDArray[np.intp] = np.arange(n_samples, dtype=np.intp)
    inside: NDArray[np.bool_] = (index >= lower[:, None]) & (index <= upper[:, None])
    window_probs = np.where(inside, probs[traces], -np.inf)
    predicted = np.argmax(window_probs, axis=1)
    scores = window_probs[np.arange(traces.size), predicted]

    pick_free = np.ones(n_traces, dtype=bool)
    pick_free[traces] = False
    interior = probs[pick_free, edge : n_samples - edge]
    negative = interior.max(axis=1) if interior.shape[1] else np.zeros(0)

    return PickScores(
        residuals=predicted - positions,
        scores=scores.astype(np.float64),
        negative_scores=negative.astype(np.float64),
    )


def detect_picks(
    probs: NDArray[np.floating],
    picks: NDArray[np.floating],
    min_height: float,
    min_distance: int,
    tolerance: int,
) -> PickScores:
    """Label-free pick detection with one-to-one matching.

    Peaks are found over the whole trace (``find_peaks``). Each true pick, in
    order, takes the highest unused peak within ``tolerance``. Unmatched
    peaks are false picks (negatives); unmatched true picks score 0.
    """
    residuals: list[float] = []
    scores: list[float] = []
    false_scores: list[float] = []
    for trace, trace_picks in zip(probs, picks, strict=True):
        peaks, properties = find_peaks(
            trace, height=min_height, distance=max(1, min_distance)
        )
        heights: FloatArray = properties["peak_heights"]
        used = np.zeros(peaks.size, dtype=bool)
        for pick in trace_picks[np.isfinite(trace_picks)]:
            near = np.flatnonzero((np.abs(peaks - pick) <= tolerance) & ~used)
            if near.size:
                best = near[np.argmax(heights[near])]
                used[best] = True
                residuals.append(float(peaks[best] - pick))
                scores.append(float(heights[best]))
            else:
                residuals.append(np.nan)
                scores.append(0.0)
        false_scores.extend(heights[~used].tolist())
    return PickScores(
        residuals=np.asarray(residuals, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float64),
        negative_scores=np.asarray(false_scores, dtype=np.float64),
    )
