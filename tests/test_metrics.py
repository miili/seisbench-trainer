from __future__ import annotations

import base64
import json
import warnings

import numpy as np
import pytest
from numpy.typing import NDArray

from seisbench_trainer.metrics import (
    PickScores,
    average_precision,
    detect_picks,
    pick_accuracy,
    residual_summary,
    roc_auc,
    summarize,
    threshold_curve,
)
from seisbench_trainer.metrics.plots import (
    PhaseResult,
    figure_to_png,
    phase_figure,
    slider_page,
)

NAN = np.nan
SAMPLES = 1000


def curve(*peaks: tuple[int, float], sigma: float = 5.0) -> NDArray[np.float64]:
    """A probability curve with Gaussian peaks at (position, height)."""
    t = np.arange(SAMPLES, dtype=np.float64)
    out = np.zeros(SAMPLES)
    for position, height in peaks:
        out = np.maximum(out, height * np.exp(-0.5 * ((t - position) / sigma) ** 2))
    return out


def picks(*rows: list[float]) -> NDArray[np.float64]:
    out = np.full((len(rows), 4), NAN)
    for index, row in enumerate(rows):
        out[index, : len(row)] = row
    return out


def scores(
    true: list[float], negative: list[float], residuals: list[float] | None = None
) -> PickScores:
    return PickScores(
        residuals=np.asarray(residuals if residuals is not None else [0.0] * len(true)),
        scores=np.asarray(true, dtype=np.float64),
        negative_scores=np.asarray(negative, dtype=np.float64),
    )


# pick accuracy (window-based)


def test_pick_accuracy_perfect_and_shifted() -> None:
    probs = np.stack([curve((300, 1.0)), curve((505, 0.8))])
    result = pick_accuracy(probs, picks([300], [500]), half_window=100, edge=10)
    np.testing.assert_array_equal(result.residuals, [0, 5])
    np.testing.assert_allclose(result.scores, [1.0, 0.8])
    assert result.negative_scores.size == 0


def test_pick_accuracy_ignores_peaks_outside_the_window() -> None:
    probs = curve((700, 1.0), (310, 0.3))[None]
    result = pick_accuracy(probs, picks([300]), half_window=100, edge=10)
    np.testing.assert_array_equal(result.residuals, [10])
    np.testing.assert_allclose(result.scores, [0.3])


def test_window_edge_is_not_a_pick() -> None:
    """A maximum on the window boundary means no peak: no residual, score kept."""
    rising = np.linspace(0.0, 0.5, SAMPLES)[None]
    result = pick_accuracy(rising, picks([300]), half_window=100, edge=10)
    assert np.isnan(result.residuals).all()
    np.testing.assert_allclose(result.scores, [rising[0, 400]])


def test_pick_at_the_trace_edge() -> None:
    probs = curve((0, 0.9))[None]
    result = pick_accuracy(probs, picks([3]), half_window=100, edge=10)
    np.testing.assert_array_equal(result.residuals, [-3])


def test_two_picks_split_their_windows() -> None:
    """Without the split, the pick at 360 would take the higher peak at 290."""
    probs = curve((290, 0.9), (365, 0.8))[None]
    result = pick_accuracy(probs, picks([300, 360]), half_window=100, edge=10)
    np.testing.assert_array_equal(result.residuals, [-10, 5])
    np.testing.assert_allclose(result.scores, [0.9, 0.8])


def test_noise_scores_use_every_pick_free_trace() -> None:
    """Regression: the reference code dropped the first traces of each batch."""
    noise = curve((500, 0.4))
    noise[2] = 0.99  # inside the edge mask: ignored
    probs = np.stack([noise] * 5 + [curve((300, 1.0))])
    result = pick_accuracy(
        probs, picks([], [], [], [], [], [300]), half_window=100, edge=10
    )
    np.testing.assert_allclose(result.negative_scores, [0.4] * 5)
    assert result.scores.size == 1


def test_pick_accuracy_without_picks() -> None:
    result = pick_accuracy(np.zeros((3, SAMPLES)), picks([], [], []), 100, 10)
    assert result.residuals.size == 0
    assert result.negative_scores.size == 3


# detection (label-free)


def test_detection_matches_misses_and_false_picks() -> None:
    probs = curve((302, 0.9), (800, 0.7))[None]
    result = detect_picks(
        probs, picks([300, 600]), min_height=0.05, min_distance=100, tolerance=50
    )
    np.testing.assert_array_equal(result.residuals, [2, NAN])
    np.testing.assert_allclose(result.scores, [0.9, 0.0])
    np.testing.assert_allclose(result.negative_scores, [0.7])


def test_detection_is_one_to_one() -> None:
    probs = curve((310, 0.9))[None]
    result = detect_picks(probs, picks([300, 320]), 0.05, 100, 50)
    np.testing.assert_array_equal(result.residuals, [10, NAN])
    assert result.negative_scores.size == 0


def test_detection_ignores_low_peaks() -> None:
    probs = curve((300, 0.03))[None]
    result = detect_picks(probs, picks([300]), 0.05, 100, 50)
    np.testing.assert_allclose(result.scores, [0.0])


# scores


def test_threshold_curve_known_values() -> None:
    curve_ = threshold_curve(
        scores([0.9, 0.6, 0.0], [0.7]), thresholds=np.array([0.5, 0.8])
    )
    np.testing.assert_allclose(curve_.precision, [2 / 3, 1.0])
    np.testing.assert_allclose(curve_.recall, [2 / 3, 1 / 3])
    np.testing.assert_allclose(curve_.f1, [2 / 3, 0.5])
    assert curve_.best() == 0


def test_threshold_curve_without_data_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        curve_ = threshold_curve(scores([], []))
    assert not curve_.f1.any()


@pytest.mark.parametrize(
    ("true", "negative", "expected"),
    [([0.9, 0.8], [0.1, 0.2], 1.0), ([0.1], [0.9], 0.0), ([0.5], [0.5], 0.5)],
)
def test_roc_auc_known_values(
    true: list[float], negative: list[float], expected: float
) -> None:
    assert roc_auc(scores(true, negative)) == pytest.approx(expected)


def test_roc_auc_matches_pairwise_definition() -> None:
    rng = np.random.default_rng(0)
    true = np.round(rng.uniform(0.3, 1.0, 50), 1)
    negative = np.round(rng.uniform(0.0, 0.7, 40), 1)
    greater = (true[:, None] > negative[None, :]).mean()
    ties = (true[:, None] == negative[None, :]).mean()
    result = roc_auc(scores(true.tolist(), negative.tolist()))
    assert result == pytest.approx(greater + 0.5 * ties)


def test_roc_auc_needs_both_classes() -> None:
    assert np.isnan(roc_auc(scores([0.9], [])))


@pytest.mark.parametrize(
    ("true", "negative", "expected"),
    [
        ([0.9, 0.8], [0.1, 0.2], 1.0),
        ([0.9, 0.4], [0.6], 0.5 * 1.0 + 0.5 * 2 / 3),
        ([0.9, 0.0], [], 0.5),  # the missed pick is never found
        ([0.0], [0.3], 0.0),
    ],
)
def test_average_precision_known_values(
    true: list[float], negative: list[float], expected: float
) -> None:
    assert average_precision(scores(true, negative)) == pytest.approx(expected)


def test_average_precision_counts_false_picks() -> None:
    """More false picks at the same scores: ROC AUC unchanged, AP falls."""
    few, many = scores([0.9, 0.5], [0.6]), scores([0.9, 0.5], [0.6] * 10)
    assert roc_auc(few) == roc_auc(many) == pytest.approx(0.5)
    assert average_precision(many) < average_precision(few)


def test_average_precision_needs_true_picks() -> None:
    assert np.isnan(average_precision(scores([], [0.5])))


def test_residual_summary_known_values() -> None:
    stats = residual_summary(np.array([-0.1, 0.0, 0.1, 0.3, NAN]), [0.1, 0.2])
    assert stats["mean_s"] == pytest.approx(0.075)
    assert stats["median_s"] == pytest.approx(0.05)
    assert stats["mae_s"] == pytest.approx(0.125)
    assert stats["rms_s"] == pytest.approx(np.sqrt(0.11 / 4))
    assert stats["within_0.1s"] == pytest.approx(0.75)
    assert stats["within_0.2s"] == pytest.approx(0.75)


def test_residual_summary_empty_is_nan() -> None:
    stats = residual_summary(np.array([NAN]), [0.1])
    assert all(np.isnan(value) for value in stats.values())


def test_summarize_uses_found_picks_for_threshold_residuals() -> None:
    """Best F1 (0.8) is reached above 0.3, which excludes the weak pick."""
    result = summarize(
        scores([0.9, 0.8, 0.2], [0.3, 0.25], residuals=[1.0, 1.0, 50.0]),
        sampling_rate=100.0,
        tolerances_s=[0.1],
    )
    assert result["f1"] == pytest.approx(0.8)
    assert 0.3 < result["threshold"] <= 0.8
    assert result["mae_s"] == pytest.approx((0.01 + 0.01 + 0.5) / 3)
    assert result["mae_s_at_threshold"] == pytest.approx(0.01)
    assert result["n_picks"] == 3
    assert result["n_negatives"] == 2


def test_phase_figure_renders() -> None:
    result = PhaseResult(
        pick=scores([0.9, 0.5], [0.2], residuals=[1.0, -3.0]),
        detection=scores([0.8, 0.0], [0.3], residuals=[2.0, NAN]),
    )
    results = {"PhaseNet-Test": result, "PhaseNet (original)": result}
    figure = phase_figure("P", results, 100.0, 2.0, 0.1, 0.5)
    assert len(figure.axes) == 4
    assert len(figure.legends) == 1  # one model key for the whole figure
    png = figure_to_png(figure)
    assert png.startswith(b"\x89PNG")


def test_slider_page_embeds_all_frames() -> None:
    frames = [("epoch 0", b"\x89PNG-a"), ("epoch 1", b"\x89PNG-b")]
    page = slider_page("P <phase>", frames)
    assert "P &lt;phase&gt;" in page
    assert 'max="1"' in page and 'value="1"' in page
    payload = page.split("const frames = ")[1].split(";\n")[0]
    embedded = json.loads(payload)
    assert [label for label, _ in embedded] == ["epoch 0", "epoch 1"]
    assert base64.b64decode(embedded[1][1]) == b"\x89PNG-b"


def test_unread_region_gives_no_negatives() -> None:
    from seisbench_trainer.metrics.picks import detect_picks, pick_accuracy

    probs = np.zeros((2, 300))
    probs[0, 50] = 0.9  # an unread P before the S of trace 0
    probs[0, 200] = 0.8  # a false P after it
    probs[1, 100] = 0.7  # a false P on a trace without any P
    picks = np.full((2, 1), np.nan)
    ignore = np.zeros((2, 300), dtype=bool)
    ignore[0, :150] = True

    found = detect_picks(probs, picks, 0.1, 10, 5)
    assert sorted(found.negative_scores) == [0.7, 0.8, 0.9]
    found = detect_picks(probs, picks, 0.1, 10, 5, ignore)
    assert sorted(found.negative_scores) == [0.7, 0.8]

    accuracy = pick_accuracy(probs, picks, 20, 0, ignore)
    assert sorted(accuracy.negative_scores) == [0.7, 0.8]
    ignore[1] = True  # nothing left of trace 1: no negative from it
    accuracy = pick_accuracy(probs, picks, 20, 0, ignore)
    assert accuracy.negative_scores.tolist() == [0.8]
