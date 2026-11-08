"""Evaluation figures. Uses the object-oriented matplotlib API (no pyplot)."""

import base64
import html
import io
import json
from dataclasses import dataclass

import numpy as np
from matplotlib import rc_context  # pyright: ignore[reportUnknownVariableType]
from matplotlib.figure import Figure
from matplotlib.typing import RcKeyType

from seisbench_trainer.metrics.picks import PickScores
from seisbench_trainer.metrics.scores import (
    residual_summary,
    roc_auc,
    roc_curve,
    threshold_curve,
)

COLORS = ("#2a78d6", "#6f6f6f", "#eb6834", "#1baf7a")
MEAN_COLOR, MEDIAN_COLOR = "#eb6834", "#1baf7a"
STYLE: dict[RcKeyType, object] = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.8,
}


@dataclass(frozen=True)
class PhaseResult:
    """The two metric families of one model and phase."""

    pick: PickScores
    detection: PickScores


def _stats_table(stats: dict[str, dict[str, float]], band_s: float) -> str:
    """Monospace table of residual statistics, one column per model."""
    names = list(stats)
    rows = [
        ("Picks", "n", "{:.0f}"),
        ("Median", "median_s", "{:+.3f}"),
        ("Mean", "mean_s", "{:+.3f}"),
        ("MAE", "mae_s", "{:.3f}"),
        ("RMS", "rms_s", "{:.3f}"),
        ("P05", "p05_s", "{:+.3f}"),
        ("P95", "p95_s", "{:+.3f}"),
        (f"\u00b1{band_s:g} s", f"within_{band_s:g}s", "{:.1%}"),
    ]
    width = max(9, *(len(name) for name in names))
    lines = [" " * 8 + "".join(f"{name[:width]:>{width + 1}}" for name in names)]
    for label, key, fmt in rows:
        cells = "".join(f"{fmt.format(stats[n][key]):>{width + 1}}" for n in names)
        lines.append(f"{label:<8}{cells}")
    return "\n".join(lines)


def phase_figure(
    title: str,
    results: dict[str, PhaseResult],
    sampling_rate: float,
    half_window_s: float,
    band_s: float,
) -> Figure:
    """Residual histogram, P/R/F1 against threshold and ROC, one line per model.

    Legends sit at fixed positions, so they do not move between epochs.
    ``band_s`` is the shaded tolerance band of the histogram.
    """
    with rc_context(STYLE):
        figure = Figure(figsize=(18, 5), layout="constrained")
        table, histogram, curves, roc = figure.subplots(
            1, 4, gridspec_kw={"width_ratios": [0.6, 1, 1, 1]}
        )
        bins = np.linspace(-half_window_s, half_window_s, 101)
        histogram.axvspan(
            -band_s,
            band_s,
            color="0.93",
            linewidth=0,
            zorder=0,
            label=f"\u00b1{band_s:g} s",
        )
        stats: dict[str, dict[str, float]] = {}

        for index, ((name, result), color) in enumerate(
            zip(results.items(), COLORS, strict=False)
        ):
            residuals_s = result.pick.residuals / sampling_rate
            residuals_s = residuals_s[np.isfinite(residuals_s)]
            summary = residual_summary(residuals_s, [band_s])
            stats[name] = summary | {"n": float(residuals_s.size)}
            histogram.hist(
                residuals_s, bins=bins, histtype="stepfilled", alpha=0.15, color=color
            )
            histogram.hist(
                residuals_s,
                bins=bins,
                histtype="step",
                linewidth=1.6,
                color=color,
                label=name,
            )
            if index == 0 and residuals_s.size:
                histogram.axvline(
                    summary["mean_s"],
                    color=MEAN_COLOR,
                    linestyle="--",
                    label=f"{name} mean",
                )
                histogram.axvline(
                    summary["median_s"],
                    color=MEDIAN_COLOR,
                    linestyle=":",
                    label=f"{name} median",
                )

            curve = threshold_curve(result.detection)
            best = curve.best()
            label = (
                f"{name}: F1 {curve.f1[best]:.3f} at {curve.thresholds[best]:.2f} "
                f"(P {curve.precision[best]:.3f}, R {curve.recall[best]:.3f})"
            )
            curves.plot(curve.thresholds, curve.f1, color=color, label=label)
            curves.plot(curve.thresholds, curve.precision, color=color, linestyle=":")
            curves.plot(curve.thresholds, curve.recall, color=color, linestyle="--")
            curves.axvline(
                curve.thresholds[best], color=color, linewidth=0.8, alpha=0.6
            )

            pick = result.pick
            if pick.scores.size and pick.negative_scores.size:
                false_rate, true_rate = roc_curve(pick)
                roc.plot(
                    false_rate,
                    true_rate,
                    color=color,
                    label=f"{name} (AUC {roc_auc(pick):.3f})",
                )

        table.axis("off")
        table.set_title("Residual statistics (s)")
        table.text(
            0.0,
            1.0,
            _stats_table(stats, band_s),
            transform=table.transAxes,
            va="top",
            fontsize=8.5,
            family="monospace",
        )
        histogram.set(
            title="Pick residuals (window)",
            xlabel=r"$t_\mathrm{pred} - t_\mathrm{true}$ (s)",
            ylabel="Count",
            xlim=(-half_window_s, half_window_s),
        )
        histogram.legend(loc="upper right")
        curves.set(
            title="Detection: F1 (solid), precision (:), recall (--)",
            xlabel="Threshold",
            ylabel="Score",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        curves.legend(loc="lower left")
        roc.plot([0, 1], [0, 1], color="0.6", linestyle=":", linewidth=1)
        roc.set(
            title="ROC (window)",
            xlabel="False positive rate",
            ylabel="True positive rate",
            xlim=(0, 1),
            ylim=(0, 1.01),
            aspect="equal",
        )
        roc.legend(loc="lower right")
        figure.suptitle(title, fontweight="bold")  # pyright: ignore[reportUnknownMemberType]
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
