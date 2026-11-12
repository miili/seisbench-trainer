"""Pure evaluation functions: pick accuracy, detection, scores and plots."""

from __future__ import annotations

from seisbench_trainer.metrics.picks import PickScores, detect_picks, pick_accuracy
from seisbench_trainer.metrics.scores import (
    ThresholdCurve,
    residual_summary,
    roc_auc,
    summarize,
    threshold_curve,
)

__all__ = [
    "PickScores",
    "ThresholdCurve",
    "detect_picks",
    "pick_accuracy",
    "residual_summary",
    "roc_auc",
    "summarize",
    "threshold_curve",
]
