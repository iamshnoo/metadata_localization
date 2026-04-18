#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


CSV_PATH = Path("/scratch/amukher6/metacul/results/significance/plot7.csv")
RESULTS_DIR = Path("/scratch/amukher6/metacul/results/plots/plot7")
LATEX_FIG = Path("/scratch/amukher6/metacul/latex/figs/main/7_leave_one_out_with_metadata.pdf")

STEPS = [2000, 4000, 8000, 10000]
STEP_LABELS = ["2k", "4k", "8k", "10k"]
CONTINENTS = ["africa", "america", "asia", "europe"]
COLORS = {
    "africa": "#fbc4a9",
    "america": "#f7b6b2",
    "asia": "#b2df8a",
    "europe": "#cbb7e5",
}
MARKERS = {"africa": "o", "america": "s", "asia": "^", "europe": "D"}


def _style_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)
    ax.set_axisbelow(True)


def main():
    df = pd.read_csv(CSV_PATH)
    df = df[(df["plot"] == "plot7") & (df["meta"] == "with_metadata")].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.8), sharey=True)
    bbox_props = dict(facecolor="lightgrey", edgecolor="black", linewidth=1.0, alpha=0.82, boxstyle="round,pad=0.3")

    panels = [("left_out", "Δ on held-out Local test"), ("all", "Δ on Global test")]
    all_vals = []

    for ax, (scope, title) in zip(axes, panels):
        sub = df[df["test_scope"] == scope]
        for continent in CONTINENTS:
            cont = sub[sub["continent"] == continent].sort_values("step")
            y = cont["delta_ppl"].to_numpy(dtype=float)
            lo = cont["delta_ppl_ci_low"].to_numpy(dtype=float)
            hi = cont["delta_ppl_ci_high"].to_numpy(dtype=float)
            x = np.arange(len(y))
            color = COLORS[continent]
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, zorder=1)
            ax.plot(
                x,
                y,
                marker=MARKERS[continent],
                color=color,
                linestyle="-",
                linewidth=2.8,
                markersize=6.8,
                markeredgecolor="black",
                markeredgewidth=0.95,
                zorder=3,
                label=f"[No{continent.capitalize()}] - [ALL]",
            )
            all_vals.extend([v for v in y if np.isfinite(v)])

        _style_axes(ax)
        ax.set_xticks(np.arange(len(STEP_LABELS)))
        ax.set_xticklabels(STEP_LABELS, fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.text(
            0.5,
            0.90,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=16.5,
            fontweight="bold",
            bbox=bbox_props,
            zorder=10,
            clip_on=False,
        )

    if all_vals:
        pad = 0.45
        lo, hi = min(all_vals) - pad, max(all_vals) + pad
        axes[0].set_ylim(lo, hi)
        axes[1].set_ylim(lo, hi)

    axes[0].set_ylabel("Δ Perplexity", fontsize=22)
    fig.text(0.5, 0.04, "Training steps", ha="center", fontsize=22)

    legend_handles = [
        Line2D([], [], color=COLORS[cont], marker=MARKERS[cont], linestyle="-", linewidth=2.6,
               markersize=7.2, markeredgecolor="black", markeredgewidth=0.95,
               label=f"[No{cont.capitalize()}] - [ALL]")
        for cont in CONTINENTS
    ]
    fig.legend(
        handles=legend_handles,
        frameon=True,
        fancybox=True,
        framealpha=0.93,
        edgecolor="black",
        fontsize=15,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.78, bottom=0.18)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIG.parent.mkdir(parents=True, exist_ok=True)
    png_path = RESULTS_DIR / "leave_one_out_with_metadata.png"
    pdf_path = RESULTS_DIR / "leave_one_out_with_metadata.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(LATEX_FIG, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {LATEX_FIG}")


if __name__ == "__main__":
    main()
