"""Scores from ``PickScores``: threshold curves, ROC AUC, average precision,
residual statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from seisbench_trainer.metrics.picks import FloatArray, PickScores

DEFAULT_THRESHOLDS: FloatArray = np.linspace(0.025, 1.0, 40)


@dataclass(frozen=True)
class ThresholdCurve:
    thresholds: FloatArray
    precision: FloatArray
    recall: FloatArray
    f1: FloatArray

    def best(self) -> int:
        """Index of the threshold with the highest F1."""
        return int(np.argmax(self.f1))


def _ratio(numerator: NDArray[np.int_], denominator: NDArray[np.int_]) -> FloatArray:
    """numerator / denominator, 0 where the denominator is 0."""
    out = np.zeros(numerator.shape, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def threshold_curve(
    scores: PickScores, thresholds: FloatArray = DEFAULT_THRESHOLDS
) -> ThresholdCurve:
    """Precision, recall and F1 when a pick counts as found at score >= t.

    True picks scoring >= t are true positives, the others false negatives.
    Negatives scoring >= t are false positives. Undefined ratios are 0.
    """
    true_positive = (scores.scores[None, :] >= thresholds[:, None]).sum(axis=1)
    false_positive = (scores.negative_scores[None, :] >= thresholds[:, None]).sum(
        axis=1
    )
    n_true = np.full_like(true_positive, scores.scores.size)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, n_true)
    f1 = np.zeros_like(precision)
    np.divide(
        2 * precision * recall,
        precision + recall,
        out=f1,
        where=(precision + recall) > 0,
    )
    return ThresholdCurve(thresholds, precision, recall, f1)


def best_threshold(scores: PickScores) -> float:
    """The threshold with the highest F1."""
    curve = threshold_curve(scores)
    return float(curve.thresholds[curve.best()])


def at_threshold(
    scores: PickScores, threshold: float, sampling_rate: float
) -> dict[str, float]:
    """Precision, recall and F1 at ``threshold``, and the residual statistics
    (in seconds) of the true picks found there."""
    curve = threshold_curve(scores, np.array([threshold]))
    found = scores.residuals[scores.scores >= threshold] / sampling_rate
    return {
        "precision": float(curve.precision[0]),
        "recall": float(curve.recall[0]),
        "f1": float(curve.f1[0]),
        **residual_summary(found, []),
    }


def _ranked_counts(
    positives: FloatArray, negatives: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """True and false positives at each distinct score, from the highest
    score down. Needs at least one score."""
    values = np.r_[positives, negatives]
    is_true = np.r_[np.ones(positives.size), np.zeros(negatives.size)]
    order = np.argsort(-values, kind="stable")
    true_positive = np.cumsum(is_true[order])
    false_positive = np.cumsum(1 - is_true[order])
    # last position of each distinct score: ties move together
    distinct = np.r_[np.flatnonzero(np.diff(values[order])), values.size - 1]
    return true_positive[distinct], false_positive[distinct]


def roc_curve(scores: PickScores) -> tuple[FloatArray, FloatArray]:
    """False and true positive rates over all distinct score thresholds."""
    n_true, n_negative = scores.scores.size, scores.negative_scores.size
    true_positive, false_positive = _ranked_counts(
        scores.scores, scores.negative_scores
    )
    false_rate = np.r_[0.0, false_positive / max(n_negative, 1)]
    true_rate = np.r_[0.0, true_positive / max(n_true, 1)]
    return false_rate, true_rate


def roc_auc(scores: PickScores) -> float:
    """Area under the ROC curve of true picks against negatives; NaN if a
    class is empty."""
    if not (scores.scores.size and scores.negative_scores.size):
        return float("nan")
    false_rate, true_rate = roc_curve(scores)
    return float(np.trapezoid(true_rate, false_rate))


def average_precision(scores: PickScores) -> float:
    """Area under the precision-recall curve (step-wise, as scikit-learn).

    Unlike the ROC AUC it counts false picks against the true ones, so it
    falls when a model fires on noise, even if it ranks picks above noise.
    True picks with score 0 were not found and are misses at every
    threshold. NaN without true picks.
    """
    n_true = scores.scores.size
    if not n_true:
        return float("nan")
    found = scores.scores[scores.scores > 0]
    if not (found.size or scores.negative_scores.size):
        return 0.0
    true_positive, false_positive = _ranked_counts(found, scores.negative_scores)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / n_true
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def residual_summary(
    residuals_s: FloatArray, tolerances_s: list[float]
) -> dict[str, float]:
    """Statistics of finite residuals in seconds; NaN when there are none."""
    finite = residuals_s[np.isfinite(residuals_s)]
    if not finite.size:
        nan = float("nan")
        stats = dict.fromkeys(
            ("mean_s", "median_s", "mae_s", "rms_s", "p05_s", "p95_s"), nan
        )
        return stats | {f"within_{t:g}s": nan for t in tolerances_s}
    median = float(np.median(finite))
    stats = {
        "mean_s": float(np.mean(finite)),
        "median_s": median,
        "mae_s": float(np.mean(np.abs(finite))),
        "rms_s": float(np.sqrt(np.mean(finite**2))),
        "p05_s": float(np.percentile(finite, 5)),
        "p95_s": float(np.percentile(finite, 95)),
    }
    return stats | {
        f"within_{t:g}s": float(np.mean(np.abs(finite) <= t)) for t in tolerances_s
    }


def summarize(
    scores: PickScores, sampling_rate: float, tolerances_s: list[float]
) -> dict[str, float]:
    """Scalar metrics of one family and phase, for logging.

    Detection quality at the F1-optimal threshold, ROC AUC, average
    precision (``ap``), counts, and the
    residual statistics of all true picks (``*_s``) and of the picks found at
    the selected threshold (``*_s_at_threshold``).
    """
    threshold = best_threshold(scores)
    found = at_threshold(scores, threshold, sampling_rate)
    return {
        "threshold": threshold,
        "precision": found["precision"],
        "recall": found["recall"],
        "f1": found["f1"],
        "auc": roc_auc(scores),
        "ap": average_precision(scores),
        "n_picks": float(scores.scores.size),
        "n_negatives": float(scores.negative_scores.size),
        **residual_summary(scores.residuals / sampling_rate, tolerances_s),
        "mae_s_at_threshold": found["mae_s"],
        "rms_s_at_threshold": found["rms_s"],
    }
