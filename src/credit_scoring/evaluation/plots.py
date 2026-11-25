"""Static figures for the model card (PNG, light surface, rendered on GitHub).

Styling follows a validated palette (blue / orange pass colour-vision-deficiency
separation checks). Text uses ink tones rather than series colours, grids and axes
recede, lines are 2px, and every multi-series chart has a legend.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: CI, Docker, servers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#9a9893"  # neutral reference lines (diagonals, thresholds)
SERIES = ("#2a78d6", "#eb6834")  # blue, orange

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2,
        "font.family": "sans-serif",
    }
)


def _save(fig: Figure, path: Path, *, layout: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if layout:
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _diagonal(ax: Axes) -> None:
    ax.plot([0, 1], [0, 1], color=REFERENCE, lw=1, ls="--", zorder=1)


def roc_chart(y: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    _diagonal(ax)
    for (name, p), color in zip(scores.items(), SERIES, strict=False):
        fpr, tpr, _ = roc_curve(y, p)
        ks_i = int(np.argmax(tpr - fpr))
        ax.plot(fpr, tpr, color=color, label=name, zorder=3)
        if name == next(iter(scores)):  # mark KS on the champion only
            ax.vlines(fpr[ks_i], fpr[ks_i], tpr[ks_i], color=INK_2, lw=1, zorder=2)
            ax.annotate(
                f"KS {tpr[ks_i] - fpr[ks_i]:.2f}",
                (fpr[ks_i], (fpr[ks_i] + tpr[ks_i]) / 2),
                xytext=(6, 0),
                textcoords="offset points",
                color=INK_2,
                fontsize=9,
                va="center",
            )
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title("ROC curve (test)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def calibration_chart(table: pd.DataFrame, path: Path) -> Path:
    """Predicted vs observed per PD decile, on log axes: PDs are heavily skewed, so a
    linear scale would crowd most deciles into one corner."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    vals = pd.concat([table["predicted_pd"], table["observed_rate"]])
    lo, hi = float(vals[vals > 0].min() * 0.7), float(vals.max() * 1.4)
    ax.plot([lo, hi], [lo, hi], color=REFERENCE, lw=1, ls="--", zorder=1)
    ax.plot(
        table["predicted_pd"],
        table["observed_rate"],
        color=SERIES[0],
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Predicted PD", ylabel="Observed default rate")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}" if v >= 0.01 else f"{v:.1%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}" if v >= 0.01 else f"{v:.1%}")
    ax.xaxis.set_minor_formatter(lambda v, _: "")
    ax.yaxis.set_minor_formatter(lambda v, _: "")
    ax.annotate(
        "On the dashed line = perfectly calibrated",
        (0.03, 0.95),
        xycoords="axes fraction",
        color=INK_2,
        fontsize=9,
        va="top",
    )
    ax.set_title("Calibration by PD decile (test, log scale)")
    return _save(fig, path)


def gains_chart(deciles: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    _diagonal(ax)
    x = np.r_[0, deciles["cum_population"]]
    y = np.r_[0, deciles["cum_capture"]]
    ax.plot(x, y, color=SERIES[0], marker="o", markersize=5, markeredgecolor=SURFACE, zorder=3)
    top = deciles.iloc[0]
    ax.annotate(
        f"Riskiest 10% hold {top['cum_capture']:.0%} of defaults",
        (top["cum_population"], top["cum_capture"]),
        xytext=(12, -4),
        textcoords="offset points",
        color=INK_2,
        fontsize=9,
    )
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Share of applicants (riskiest first)")
    ax.set_ylabel("Share of defaults captured")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Cumulative gains (test)")
    return _save(fig, path)


def importance_chart(importance: pd.DataFrame, path: Path, top: int = 12) -> Path:
    data = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.36 * len(data) + 1.0))
    ax.barh(
        data["feature"],
        data["mean_abs_contribution"],
        height=0.62,
        color=SERIES[0],
        edgecolor=SURFACE,
        linewidth=2,
        zorder=3,
    )
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP| (log-odds of default)")
    ax.set_title("What drives the model (test)")
    return _save(fig, path)


def score_distribution_chart(
    scores: np.ndarray, y: np.ndarray, cutoff_score: float, path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bins = [float(b) for b in range(300, 861, 15)]
    for label, mask, color in (("Repaid", y == 0, SERIES[0]), ("Defaulted", y == 1, SERIES[1])):
        ax.hist(
            scores[mask],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            label=label,
            zorder=3,
        )
    ax.axvline(cutoff_score, color=INK_2, lw=1, ls="--", zorder=2)
    ax.annotate(
        f"Decline below {cutoff_score:.0f}",
        (cutoff_score, ax.get_ylim()[1] * 0.92),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        color=INK_2,
        fontsize=9,
    )
    ax.set(xlabel="Credit score", ylabel="Share of applicants")
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_title("Score distribution by outcome (test)")
    ax.legend(loc="upper left")
    return _save(fig, path)


def fairness_chart(with_age: pd.DataFrame, without_age: pd.DataFrame, path: Path) -> Path:
    """Two panels on separate axes: approval rate, and calibration gap, by age band."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    groups = with_age["group"].tolist()
    x = np.arange(len(groups))
    w = 0.38
    panels = (
        (left, "approval_rate", "Approval rate", "{:.0%}"),
        (right, "calibration_gap", "Predicted PD minus actual default rate", "{:+.1%}"),
    )
    for ax, col, title, fmt in panels:
        for offset, (name, frame), color in zip(
            (-w / 2, w / 2),
            (("With age (champion)", with_age), ("Without age", without_age)),
            SERIES,
            strict=True,
        ):
            ax.bar(
                x + offset,
                frame[col],
                width=w,
                color=color,
                edgecolor=SURFACE,
                linewidth=2,
                label=name,
                zorder=3,
            )
        ax.set_xticks(x, groups)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(lambda v, _, f=fmt: f.format(v))
        ax.set_title(title)
        ax.set_xlabel("Age band")
    left.set_ylim(0, 1.0)  # bars start at zero
    right.axhline(0, color=INK_2, lw=1, zorder=4)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))  # leave room for the legend row
    return _save(fig, path, layout=False)


def stability_chart(csi: dict[str, float], path: Path, monitor: float, action: float) -> Path:
    """PSI / CSI per variable, with the monitor and action thresholds as reference lines."""
    items = sorted(csi.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(items) + 1.2))
    ax.barh(
        [k for k, _ in items],
        [v for _, v in items],
        height=0.62,
        color=SERIES[0],
        edgecolor=SURFACE,
        linewidth=2,
        zorder=3,
    )
    top = max(action * 1.3, max(v for _, v in items) * 1.1)
    for x, label in ((monitor, "monitor"), (action, "action")):
        ax.axvline(x, color=INK_2, lw=1, ls="--", zorder=2)
        ax.annotate(
            f"{label} {x:.2f}",
            (x, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -2),
            textcoords="offset points",
            va="top",
            color=INK_2,
            fontsize=9,
        )
    ax.set_xlim(0, top)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Population stability index vs training")
    ax.set_title("Drift by variable")
    return _save(fig, path)


def score_shift_chart(
    reference: list[float], current: list[float], edges: list[float], path: Path
) -> Path:
    """Share of applicants per reference score band: training vs current traffic."""
    n = len(edges) + 1  # numeric bins only (the score is never missing)
    bounds = [300.0, *edges, 850.0]
    labels = [f"{bounds[i]:.0f}-{bounds[i + 1]:.0f}" for i in range(n)]
    x = np.arange(n)
    w = 0.4
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for offset, (name, shares), color in zip(
        (-w / 2, w / 2),
        (("Training", reference[:n]), ("Current", current[:n])),
        SERIES,
        strict=True,
    ):
        ax.bar(x + offset, shares, width=w, color=color, edgecolor=SURFACE, linewidth=2,
               label=name, zorder=3)  # fmt: skip
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Credit score band (training deciles)")
    ax.set_ylabel("Share of applicants")
    ax.set_title("Score distribution shift", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    return _save(fig, path)
