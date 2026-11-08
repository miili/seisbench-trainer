"""Evaluation figures. Uses the object-oriented matplotlib API (no pyplot)."""

# matplotlib leaves the **kwargs of its plotting methods untyped.
# pyright: reportUnknownMemberType=false

import base64
import html
import io
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib import rc_context  # pyright: ignore[reportUnknownVariableType]
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.typing import LineStyleType, RcKeyType

from seisbench_trainer.metrics.picks import PickScores
from seisbench_trainer.metrics.scores import (
    residual_summary,
    roc_auc,
    roc_curve,
    threshold_curve,
)

# Okabe-Ito (colour-blind safe); the reference model is grey.
COLORS = ("#0072B2", "#7A7A7A", "#D55E00", "#009E73", "#CC79A7")
STYLE: dict[RcKeyType, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "axes.titlelocation": "left",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.2,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "mathtext.default": "regular",
}
METRIC_STYLES: tuple[tuple[str, LineStyleType], ...] = (
    ("F1", "-"),
    ("Precision", "--"),
    ("Recall", ":"),
)


@dataclass(frozen=True)
class PhaseResult:
    """The two metric families of one model and phase."""

    pick: PickScores
    detection: PickScores
    description: str = ""
    """Shown next to the model name in the figure key, e.g. 'trained from scratch'."""


Better = Literal["max", "min", "absmin"] | None


@dataclass(frozen=True)
class _Row:
    label: str
    values: list[float]
    fmt: str
    better: Better = None
    text: list[str] | None = None


def _bold_index(row: _Row) -> int | None:
    """Index of the best value of a row, if the row has a direction."""
    values = np.asarray(row.values, dtype=np.float64)
    if row.better is None or len(values) < 2 or not np.isfinite(values).all():
        return None
    if row.better == "absmin":
        values = np.abs(values)
    best = np.max(values) if row.better == "max" else np.min(values)
    winners = np.flatnonzero(values == best)
    return int(winners[0]) if winners.size == 1 else None


def _panel_label(axes: Axes, letter: str) -> None:
    axes.text(
        -0.16,
        1.04,
        letter,
        transform=axes.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )


def _two_lines(name: str) -> str:
    """'PhaseNet-BohemiaSaxony' -> 'PhaseNet-\nBohemiaSaxony' for table headers."""
    head, found, tail = name.partition("-")
    if found:
        return f"{head}-\n{tail}"
    head, found, tail = name.partition(" ")
    if found:
        return f"{head}\n{tail}"
    return name


def _table(
    axes: Axes,
    names: list[str],
    rows: list[_Row | str],
) -> None:
    """Booktabs-style table: section titles (str) and value rows."""
    axes.axis("off")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    n_lines = (
        2
        + sum(1 for row in rows if isinstance(row, _Row))
        + 1.4 * sum(1 for row in rows if isinstance(row, str))
    )
    step = 0.98 / (n_lines + 1)
    width = 0.58 / len(names)
    columns = [0.42 + width * (index + 1) for index in range(len(names))]

    def rule(y: float, weight: float) -> None:
        axes.plot(
            [0, 1], [y, y], color="black", linewidth=weight, solid_capstyle="butt"
        )

    y = 0.98
    rule(y, 0.9)
    y -= 0.25 * step
    for x, name, color in zip(columns, names, COLORS, strict=False):
        axes.text(
            x, y, _two_lines(name), ha="right", va="top", color=color, fontweight="bold"
        )
    y -= 2.1 * step
    rule(y, 0.5)
    for row in rows:
        if isinstance(row, str):
            y -= 1.15 * step
            axes.text(0, y, row, fontstyle="italic", va="center", color="0.25")
            y -= 0.25 * step
            continue
        y -= step
        axes.text(0, y, row.label, va="center")
        bold = _bold_index(row)
        cells = row.text or [row.fmt.format(value) for value in row.values]
        for index, (x, cell) in enumerate(zip(columns, cells, strict=True)):
            weight = "bold" if index == bold else "normal"
            axes.text(x, y, cell, ha="right", va="center", fontweight=weight)
    y -= 0.7 * step
    rule(y, 0.9)


def phase_figure(
    title: str,
    results: dict[str, PhaseResult],
    sampling_rate: float,
    half_window_s: float,
    band_s: float,
    tolerance_s: float,
) -> Figure:
    """Publication figure of one phase: residuals, detection, ROC and statistics.

    a: residual histogram (log counts, bins on the sample grid) with the
    ``band_s`` tolerance band; b: F1, precision and recall against the
    threshold of the detection metrics (matching tolerance ``tolerance_s``),
    with the F1-optimal threshold marked; c: ROC of the pick-window scores;
    d: summary table, the better value per row in bold. Colours identify
    the models (key at the top), line styles the metrics.
    """
    names = list(results)
    colors = dict(zip(names, COLORS, strict=False))
    half = round(half_window_s * sampling_rate)
    step = max(1, half // 200)
    edges = (np.arange(-half, half + 2 * step, step) - step / 2) / sampling_rate

    with rc_context(STYLE):  # pyright: ignore[reportUnknownArgumentType]
        figure = Figure(figsize=(7.2, 6.2), layout="constrained")
        (histogram, curves), (roc, table) = figure.subplots(2, 2)
        histogram.axvspan(-band_s, band_s, color="0.92", linewidth=0, zorder=0)

        stats: dict[str, dict[str, float]] = {}
        detection: dict[str, tuple[float, float, float, float]] = {}
        aucs: list[float] = []
        for index, (name, result) in enumerate(results.items()):
            color = colors[name]
            residuals_s = result.pick.residuals / sampling_rate
            residuals_s = residuals_s[np.isfinite(residuals_s)]
            stats[name] = residual_summary(residuals_s, [band_s]) | {
                "n": float(residuals_s.size)
            }
            histogram.hist(
                residuals_s,
                bins=edges,
                histtype="step",
                color=color,
                linewidth=1.1 if index == 0 else 0.8,
                zorder=3 - index,
            )

            curve = threshold_curve(result.detection)
            best = curve.best()
            detection[name] = (
                float(curve.thresholds[best]),
                float(curve.f1[best]),
                float(curve.precision[best]),
                float(curve.recall[best]),
            )
            for (_, style), values in zip(
                METRIC_STYLES, (curve.f1, curve.precision, curve.recall), strict=True
            ):
                curves.plot(curve.thresholds, values, color=color, linestyle=style)
            curves.plot(
                curve.thresholds[best],
                curve.f1[best],
                "o",
                color=color,
                markersize=4,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=5,
            )

            pick = result.pick
            aucs.append(roc_auc(pick))
            if pick.scores.size and pick.negative_scores.size:
                false_rate, true_rate = roc_curve(pick)
                roc.plot(false_rate, true_rate, color=color)

        histogram.set_yscale("log")
        histogram.set(
            xlim=(-half_window_s, half_window_s),
            xlabel="Pick residual, $t_{pred} - t_{true}$ (s)",
            ylabel="Picks",
            title="Pick residuals",
        )
        histogram.text(
            band_s,
            0.97,
            f" ±{band_s:g} s",
            transform=histogram.get_xaxis_transform(),
            va="top",
            fontsize=6.5,
            color="0.4",
        )
        curves.set(
            xlim=(0, 1),
            ylim=(0, 1.02),
            xlabel="Detection threshold",
            ylabel="Score",
            title=f"Detection (matching ±{tolerance_s:g} s)",
        )
        style_key = [
            Line2D([], [], color="0.2", linestyle=style, label=label)
            for label, style in METRIC_STYLES
        ] + [
            Line2D(
                [],
                [],
                color="0.2",
                marker="o",
                linestyle="",
                markersize=4,
                label="Best F1",
            )
        ]
        curves.legend(handles=style_key, loc="lower left", handlelength=2.2)
        roc.plot([0, 1], [0, 1], color="0.6", linestyle=":", linewidth=0.8)
        roc.set(
            xlim=(0, 1),
            ylim=(0, 1.01),
            aspect="equal",
            xlabel="False positive rate",
            ylabel="True positive rate",
            title="Pick-window ROC",
        )

        rows: list[_Row | str] = [
            f"Detection (±{tolerance_s:g} s)",
            _Row("Threshold", [detection[n][0] for n in names], "{:.2f}"),
            _Row("F1", [detection[n][1] for n in names], "{:.3f}", "max"),
            _Row("Precision", [detection[n][2] for n in names], "{:.3f}", "max"),
            _Row("Recall", [detection[n][3] for n in names], "{:.3f}", "max"),
            f"Pick window (±{half_window_s:g} s)",
            _Row("Picks", [stats[n]["n"] for n in names], "{:,.0f}"),
            _Row("ROC AUC", aucs, "{:.3f}", "max"),
            _Row(
                "Median (s)", [stats[n]["median_s"] for n in names], "{:+.3f}", "absmin"
            ),
            _Row("Mean (s)", [stats[n]["mean_s"] for n in names], "{:+.3f}", "absmin"),
            _Row("MAE (s)", [stats[n]["mae_s"] for n in names], "{:.3f}", "min"),
            _Row("RMS (s)", [stats[n]["rms_s"] for n in names], "{:.3f}", "min"),
            _Row(
                "5th–95th pct. (s)",
                [stats[n]["p95_s"] - stats[n]["p05_s"] for n in names],
                "",
                "min",
                [f"{stats[n]['p05_s']:+.2f} / {stats[n]['p95_s']:+.2f}" for n in names],
            ),
            _Row(
                f"Within ±{band_s:g} s",
                [stats[n][f"within_{band_s:g}s"] for n in names],
                "{:.1%}",
                "max",
            ),
        ]
        _table(table, names, rows)
        table.set_title("Summary")

        for axes, letter in zip((histogram, curves, roc, table), "abcd", strict=True):
            _panel_label(axes, letter)
        key = [
            Line2D(
                [],
                [],
                color=colors[name],
                linewidth=2,
                label=f"{name}, {result.description}" if result.description else name,
            )
            for name, result in results.items()
        ]
        figure.legend(handles=key, loc="outside lower center", ncols=len(key))
        figure.suptitle(title, fontsize=9, fontweight="bold")
    return figure


def figure_to_png(figure: Figure) -> bytes:
    """Render a figure to PNG bytes."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=200)
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
