#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


CSV_PATH = Path("/scratch/amukher6/metacul/results/fp16_old/plots_fp16/plot2/plot_2.csv")
RESULTS_DIR = Path("/scratch/amukher6/metacul/results/plots/plot2")
LATEX_FIG = Path("/scratch/amukher6/metacul/latex/figs/main/2_perplexity_local_vs_global_1b.pdf")


REGIONS = ["Africa", "America", "Asia", "Europe", "All"]
PLOT_ORDER = ["Global(T+,I+)", "Global(T-,I-)", "Local(T+,I+)", "Local(T-,I-)"]
COMBO_STYLES = {
    "Global(T+,I+)": {"color": "#a6cee3", "hatch": "o"},
    "Global(T-,I-)": {"color": "#7f7f7f", "hatch": ""},
    "Local(T+,I+)": {"color": "#9ad1a6", "hatch": "\\"},
    "Local(T-,I-)": {"color": "#d9d9d9", "hatch": ""},
}


def _row_to_values(row):
    return float(row["mean_ppl"]), float(row["ci_low"]), float(row["ci_high"])


def _aggregate_rows(rows):
    means, lows, highs = [], [], []
    for _, row in rows.iterrows():
        m, lo, hi = _row_to_values(row)
        means.append(m)
        lows.append(lo)
        highs.append(hi)
    if not means:
        return np.nan, np.nan, np.nan
    return float(np.mean(means)), float(np.mean(lows)), float(np.mean(highs))


def _build_plot_df(df):
    continents = ["africa", "america", "asia", "europe"]
    size = "1b"
    records = []
    combos = [
        {"label": "Global(T+,I+)", "scope": "global", "meta": "with_metadata"},
        {"label": "Local(T+,I+)", "scope": "local", "meta": "with_metadata"},
        {"label": "Global(T-,I-)", "scope": "global", "meta": "without_metadata"},
        {"label": "Local(T-,I-)", "scope": "local", "meta": "without_metadata"},
    ]

    for cont in continents:
        for combo in combos:
            meta = combo["meta"]
            if combo["scope"] == "local":
                model_path = f"/scratch/amukher6/metacul/models/{cont}_{meta}_{size}"
            else:
                model_path = f"/scratch/amukher6/metacul/models/combined_{meta}_{size}"
            test_path = f"/scratch/amukher6/metacul/training_data/meco_datasets/continents/{cont}/{meta}/"
            row = df[(df["model_path"] == model_path) & (df["test_set_path"] == test_path)]
            if row.empty:
                continue
            mean, ci_low, ci_high = _row_to_values(row.iloc[0])
            records.append(
                {
                    "region": cont.capitalize(),
                    "combo": combo["label"],
                    "mean_ppl": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    for combo in combos:
        meta = combo["meta"]
        if combo["scope"] == "global":
            model_path = f"/scratch/amukher6/metacul/models/combined_{meta}_{size}"
            test_path = f"/scratch/amukher6/metacul/training_data/meco_datasets/combined/{meta}/"
            row = df[(df["model_path"] == model_path) & (df["test_set_path"] == test_path)]
            if row.empty:
                continue
            mean, ci_low, ci_high = _row_to_values(row.iloc[0])
        else:
            test_path = f"/scratch/amukher6/metacul/training_data/meco_datasets/combined/{meta}/"
            rows = []
            for cont in continents:
                model_path = f"/scratch/amukher6/metacul/models/{cont}_{meta}_1b"
                row = df[(df["model_path"] == model_path) & (df["test_set_path"] == test_path)]
                if not row.empty:
                    rows.append(row.iloc[0])
            if not rows:
                continue
            mean, ci_low, ci_high = _aggregate_rows(pd.DataFrame(rows))
        records.append(
            {
                "region": "All",
                "combo": combo["label"],
                "mean_ppl": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(records)


def _style_axes(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.28)
    ax.set_axisbelow(True)


def _plot_panel(ax, subset, regions, labels, width=0.18, gap=0.06):
    x_pos = np.arange(len(labels))
    for i, combo in enumerate(PLOT_ORDER):
        combo_subset = subset[subset["combo"] == combo]
        means, yerr_lo, yerr_hi = [], [], []
        for region in regions:
            row = combo_subset[combo_subset["region"] == region]
            if row.empty:
                means.append(np.nan)
                yerr_lo.append(0.0)
                yerr_hi.append(0.0)
                continue
            mean = float(row.iloc[0]["mean_ppl"])
            ci_low = float(row.iloc[0]["ci_low"])
            ci_high = float(row.iloc[0]["ci_high"])
            means.append(mean)
            yerr_lo.append(mean - ci_low)
            yerr_hi.append(ci_high - mean)

        offset = i * width + (gap if i >= 2 else 0.0)
        style = COMBO_STYLES[combo]
        ax.bar(
            x_pos + offset,
            means,
            width,
            yerr=np.array([yerr_lo, yerr_hi]),
            capsize=4.5,
            color=style["color"],
            hatch=style["hatch"],
            edgecolor="black",
            linewidth=1.3,
            error_kw=dict(ecolor="#2b2b2b", lw=2.0, capthick=2.0),
            label=combo,
            zorder=3,
        )

    ax.set_xticks(x_pos + (3 * width + gap) / 2)
    ax.set_xticklabels(labels, fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    _style_axes(ax)


def main():
    df = pd.read_csv(CSV_PATH)
    plot_df = _build_plot_df(df)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 6.1),
        sharey=True,
        gridspec_kw={"width_ratios": [4, 1], "wspace": 0.18},
    )

    local_regions = REGIONS[:-1]
    _plot_panel(axes[0], plot_df, local_regions, local_regions)
    _plot_panel(axes[1], plot_df, ["All"], ["All"])

    bbox_props = dict(
        facecolor="lightgrey",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.82,
        boxstyle="round,pad=0.3",
    )
    axes[0].set_title(
        "Local test sets",
        fontsize=17,
        fontweight="bold",
        pad=6,
        y=0.62,
        bbox=bbox_props,
    )
    axes[1].set_title(
        "Global test set",
        fontsize=17,
        fontweight="bold",
        pad=6,
        y=0.62,
        bbox=bbox_props,
    )
    axes[0].set_ylabel("Perplexity (↓ better)", fontsize=22)

    y_top = float(plot_df["mean_ppl"].max()) + 1.0
    axes[0].set_ylim(6, y_top)
    axes[1].set_ylim(6, y_top)

    legend_handles = [
        Line2D(
            [],
            [],
            color=COMBO_STYLES[k]["color"],
            marker="s",
            linestyle="None",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=1.2,
            label=k,
        )
        for k in PLOT_ORDER
    ]
    axes[0].legend(
        handles=legend_handles,
        frameon=True,
        fancybox=True,
        framealpha=0.93,
        edgecolor="black",
        fontsize=16,
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.98, 0.99),
    )

    fig.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIG.parent.mkdir(parents=True, exist_ok=True)
    png_path = RESULTS_DIR / "perplexity_local_vs_global_1b.png"
    pdf_path = RESULTS_DIR / "perplexity_local_vs_global_1b.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(LATEX_FIG, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {LATEX_FIG}")


if __name__ == "__main__":
    main()
