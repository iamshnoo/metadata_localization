#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


CSV_PATH = Path("/path/to/metacul/results/fp16_old/plots_fp16/plot2/plot_2.csv")
RESULTS_DIR = Path("/path/to/metacul/results/plots/plot2")
LATEX_FIG = Path("/path/to/metacul/latex/figs/appendix/2_perplexity_local_vs_global_500m.pdf")


REGIONS = ["Africa", "America", "Asia", "Europe", "All"]
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
        mean, ci_low, ci_high = _row_to_values(row)
        means.append(mean)
        lows.append(ci_low)
        highs.append(ci_high)
    if not means:
        return np.nan, np.nan, np.nan
    return float(np.mean(means)), float(np.mean(lows)), float(np.mean(highs))


def _build_plot_df(df):
    continents = ["africa", "america", "asia", "europe"]
    size = "500m"
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
        test_path = f"/path/to/metacul/training_data/meco_datasets/combined/{meta}/"
        if combo["scope"] == "global":
            model_path = f"/path/to/metacul/models/combined_{meta}_{size}"
            row = df[(df["model_path"] == model_path) & (df["test_set_path"] == test_path)]
            if row.empty:
                continue
            mean, ci_low, ci_high = _row_to_values(row.iloc[0])
        else:
            rows = []
            for cont in continents:
                model_path = f"/path/to/metacul/models/{cont}_{meta}_{size}"
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


def main():
    df = pd.read_csv(CSV_PATH)
    plot_df = _build_plot_df(df)
    if plot_df.empty:
        raise SystemExit("no local-vs-global 500M data found")

    fig, ax = plt.subplots(figsize=(12.2, 4.9))

    width = 0.18
    inner_gap = 0.06
    x_pos = np.array([0.0, 1.0, 2.0, 3.0, 4.35])
    tick_centers = x_pos + (3 * width + inner_gap) / 2

    for i, combo in enumerate(PLOT_ORDER):
        combo_subset = plot_df[plot_df["combo"] == combo]
        means, yerr_lo, yerr_hi = [], [], []
        for region in REGIONS:
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

        offset = i * width + (inner_gap if i >= 2 else 0.0)
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

    ax.set_xlim(-0.25, 5.05)
    ax.set_ylim(6, float(plot_df["mean_ppl"].max()) + 1.0)
    ax.set_xticks(tick_centers)
    ax.set_xticklabels(REGIONS, fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylabel("Perplexity (↓ better)", fontsize=22)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.28)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")

    separator_x = (tick_centers[3] + tick_centers[4]) / 2
    ax.axvline(separator_x, color="#7f7f7f", linewidth=1.2, linestyle="--", alpha=0.6, zorder=1)

    bbox_props = dict(
        facecolor="lightgrey",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.82,
        boxstyle="round,pad=0.3",
    )
    local_mid = (tick_centers[0] + tick_centers[3]) / 2
    global_mid = tick_centers[4]
    y_annot = ax.get_ylim()[0] + 0.76 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.text(local_mid, y_annot, "Local test sets", ha="center", va="center", fontsize=17, fontweight="bold", bbox=bbox_props)
    ax.text(global_mid, y_annot, "Global test set", ha="center", va="center", fontsize=15, fontweight="bold", bbox=bbox_props)

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
    ax.legend(
        handles=legend_handles,
        frameon=True,
        fancybox=True,
        framealpha=0.93,
        edgecolor="black",
        fontsize=16,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.99),
    )

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.16, top=0.96)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIG.parent.mkdir(parents=True, exist_ok=True)
    png_path = RESULTS_DIR / "perplexity_local_vs_global_500m_compact.png"
    pdf_path = RESULTS_DIR / "perplexity_local_vs_global_500m_compact.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(LATEX_FIG, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {LATEX_FIG}")


if __name__ == "__main__":
    main()
