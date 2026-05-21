#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_PATH = Path("/path/to/metacul/results/fp16_old/plots_fp16/plot2/plot_2.csv")
RESULTS_DIR = Path("/path/to/metacul/results/plots/plot2")
LATEX_FIG = Path("/path/to/metacul/latex/figs/main/2_perplexity_local_vs_global_1b.pdf")


LOCAL_REGIONS = ["Africa", "America", "Asia", "Europe"]
PLOT_ORDER = ["Global (T+, I+)", "Global (T-, I-)", "Local (T+, I+)", "Local (T-, I-)"]
COMBO_STYLES = {
    "Global (T+, I+)": {"color": "#a6cee3", "hatch": "o"},
    "Global (T-, I-)": {"color": "#7f7f7f", "hatch": ""},
    "Local (T+, I+)": {"color": "#9ad1a6", "hatch": "\\"},
    "Local (T-, I-)": {"color": "#d9d9d9", "hatch": ""},
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
        {"label": "Global (T+, I+)", "scope": "global", "meta": "with_metadata"},
        {"label": "Local (T+, I+)", "scope": "local", "meta": "with_metadata"},
        {"label": "Global (T-, I-)", "scope": "global", "meta": "without_metadata"},
        {"label": "Local (T-, I-)", "scope": "local", "meta": "without_metadata"},
    ]

    for cont in continents:
        for combo in combos:
            meta = combo["meta"]
            if combo["scope"] == "local":
                model_path = f"/path/to/metacul/models/{cont}_{meta}_{size}"
            else:
                model_path = f"/path/to/metacul/models/combined_{meta}_{size}"
            test_path = f"/path/to/metacul/training_data/meco_datasets/continents/{cont}/{meta}/"
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
            model_path = f"/path/to/metacul/models/combined_{meta}_{size}"
            test_path = f"/path/to/metacul/training_data/meco_datasets/combined/{meta}/"
            row = df[(df["model_path"] == model_path) & (df["test_set_path"] == test_path)]
            if row.empty:
                continue
            mean, ci_low, ci_high = _row_to_values(row.iloc[0])
        else:
            test_path = f"/path/to/metacul/training_data/meco_datasets/combined/{meta}/"
            rows = []
            for cont in continents:
                model_path = f"/path/to/metacul/models/{cont}_{meta}_1b"
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


def _style_axes(ax, show_ylabels=True):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.25)
        spine.set_color("black")
    ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.26)
    ax.set_axisbelow(True)
    ax.set_ylim(6.0, 16.7)
    ax.set_yticks([6, 8, 10, 12, 14, 16])
    ax.tick_params(axis="both", width=1.0, length=3.5)
    ax.tick_params(axis="y", labelleft=show_ylabels, labelsize=8.5)


def _plot_panel(ax, subset, regions, width=0.16):
    x_pos = np.arange(len(regions))
    offsets = np.array([-1.65, -0.55, 0.75, 1.85]) * width
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

        style = COMBO_STYLES[combo]
        ax.bar(
            x_pos + offsets[i],
            means,
            width,
            yerr=np.array([yerr_lo, yerr_hi]),
            capsize=2.8,
            color=style["color"],
            hatch=style["hatch"],
            edgecolor="black",
            linewidth=1.0,
            error_kw=dict(ecolor="#2b2b2b", lw=1.0, capthick=1.0),
            label=combo,
            zorder=3,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(regions, fontsize=8.6)


def main():
    df = pd.read_csv(CSV_PATH)
    plot_df = _build_plot_df(df)

    fig, (ax_local, ax_global) = plt.subplots(
        1,
        2,
        figsize=(3.46, 2.08),
        gridspec_kw={"width_ratios": [4.0, 1.1], "wspace": 0.16},
        sharey=True,
    )

    _plot_panel(ax_local, plot_df, LOCAL_REGIONS, width=0.15)
    _plot_panel(ax_global, plot_df, ["All"], width=0.15)

    _style_axes(ax_local, show_ylabels=True)
    _style_axes(ax_global, show_ylabels=False)
    ax_local.set_ylabel("Perplexity (↓ better)", fontsize=9.2, labelpad=2.0)
    ax_global.set_title("Global", fontsize=10.5, fontweight="bold", pad=1.5)
    ax_local.text(
        1.5,
        13.35,
        "Local test sets",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        bbox=dict(facecolor="#eeeeee", edgecolor="#444444", boxstyle="round,pad=0.22"),
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COMBO_STYLES["Global (T+, I+)"]["color"], edgecolor="black", hatch="o", label="Global (T+, I+)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COMBO_STYLES["Global (T-, I-)"]["color"], edgecolor="black", label="Global (T-, I-)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COMBO_STYLES["Local (T+, I+)"]["color"], edgecolor="black", hatch="\\", label="Local (T+, I+)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COMBO_STYLES["Local (T-, I-)"]["color"], edgecolor="black", label="Local (T-, I-)"),
    ]
    ax_local.legend(
        handles=legend_handles,
        frameon=True,
        fancybox=False,
        framealpha=0.93,
        edgecolor="black",
        fontsize=7.0,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.55, 0.98),
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.65,
        borderpad=0.30,
        labelspacing=0.22,
    )

    fig.subplots_adjust(left=0.125, right=0.99, bottom=0.18, top=0.96)

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
