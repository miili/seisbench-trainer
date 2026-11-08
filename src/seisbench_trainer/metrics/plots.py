"""Evaluation figures. Uses the object-oriented matplotlib API (no pyplot)."""

import base64
import html
import io
import json
from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from seisbench_trainer.metrics.picks import PickScores
from seisbench_trainer.metrics.scores import roc_auc, roc_curve, threshold_curve

COLORS = ("#2a78d6", "#6f6f6f", "#eb6834", "#1baf7a")


@dataclass(frozen=True)
class PhaseResult:
    """The two metric families of one model and phase."""

    pick: PickScores
    detection: PickScores


def phase_figure(
    title: str,
    results: dict[str, PhaseResult],
    sampling_rate: float,
    half_window_s: float,
) -> Figure:
    """Residual histogram, P/R/F1 against threshold and ROC, one line per model."""
    figure = Figure(figsize=(15, 4.5), layout="constrained")
    histogram, curves, roc = figure.subplots(1, 3)
    bins = np.linspace(-half_window_s, half_window_s, 101)

    for (name, result), color in zip(results.items(), COLORS, strict=False):
        residuals_s = result.pick.residuals / sampling_rate
        residuals_s = residuals_s[np.isfinite(residuals_s)]
        mae = float(np.mean(np.abs(residuals_s))) if residuals_s.size else np.nan
        histogram.hist(
            residuals_s,
            bins=bins,
            histtype="step",
            linewidth=1.5,
            color=color,
            label=f"{name} (MAE {mae:.3f} s)",
        )

        curve = threshold_curve(result.detection)
        curves.plot(curve.thresholds, curve.f1, color=color, label=f"{name} F1")
        curves.plot(curve.thresholds, curve.precision, color=color, linestyle=":")
        curves.plot(curve.thresholds, curve.recall, color=color, linestyle="--")

        pick = result.pick
        if pick.scores.size and pick.negative_scores.size:
            false_rate, true_rate = roc_curve(pick)
            roc.plot(
                false_rate,
                true_rate,
                color=color,
                label=f"{name} (AUC {roc_auc(pick):.3f})",
            )

    histogram.set(
        title="Pick residuals (window)",
        xlabel="Predicted - true (s)",
        ylabel="Count",
        xlim=(-half_window_s, half_window_s),
    )
    curves.set(
        title="Detection: F1 (solid), precision (:), recall (--)",
        xlabel="Threshold",
        ylim=(0, 1.02),
        xlim=(0, 1),
    )
    roc.plot([0, 1], [0, 1], color="0.7", linestyle=":")
    roc.set(
        title="ROC (window)",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    for axes in (histogram, curves, roc):
        axes.grid(alpha=0.25)
        if axes.get_legend_handles_labels()[0]:
            axes.legend(fontsize=8, frameon=False)
    figure.suptitle(title)  # pyright: ignore[reportUnknownMemberType]
    return figure


def figure_to_png(figure: Figure) -> bytes:
    """Render a figure to PNG bytes."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")  # pyright: ignore[reportUnknownMemberType]
    return buffer.getvalue()


SLIDER_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font: 14px sans-serif; margin: 12px; background: #fff; color: #222; }}
.bar {{ display: flex; gap: 12px; align-items: center; margin-bottom: 8px; }}
input[type=range] {{ flex: 1; }}
img {{ max-width: 100%; }}
</style></head><body>
<div class="bar"><b>{title}</b><input id="s" type="range" min="0" max="{last}"
value="{last}"><span id="l"></span></div>
<img id="i" alt="evaluation figure">
<script>
const frames = {frames};
const s = document.getElementById("s"), i = document.getElementById("i"),
  l = document.getElementById("l");
function show() {{
  const f = frames[s.value];
  i.src = "data:image/png;base64," + f[1];
  l.textContent = f[0];
}}
s.oninput = show;
document.onkeydown = (e) => {{
  if (e.key === "ArrowLeft" || e.key === "ArrowRight") {{
    s.value = +s.value + (e.key === "ArrowRight" ? 1 : -1); show();
  }}
}};
show();
</script></body></html>
"""


def slider_page(title: str, frames: list[tuple[str, bytes]]) -> str:
    """Self-contained HTML page showing PNG frames with a slider.

    ``frames`` are (label, PNG) pairs, e.g. one per epoch. The images are
    embedded, so the page works in MLflow's sandboxed artifact viewer.
    """
    encoded = [(label, base64.b64encode(png).decode()) for label, png in frames]
    return SLIDER_PAGE.format(
        title=html.escape(title), last=len(frames) - 1, frames=json.dumps(encoded)
    )
