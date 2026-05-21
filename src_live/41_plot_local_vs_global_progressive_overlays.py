#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


CSV_PATH = Path("/path/to/metacul/results/fp16_old/plots_fp16/plot2/plot_2.csv")
OUT_DIR = Path("/path/to/metacul/slides")

CONTINENT_KEYS = ["africa", "america", "asia", "europe"]
GROUP_LABEL = "All"
PLOT_ORDER = ["Global (T+, I+)", "Global (T-, I-)", "Local (T+, I+)", "Local (T-, I-)"]
LEGEND_ORDER = ["Global (T-, I-)", "Global (T+, I+)", "Local (T-, I-)", "Local (T+, I+)"]
COMBO_STYLES = {
    "Global (T+, I+)": {"color": "#a6cee3", "hatch": "o"},
    "Global (T-, I-)": {"color": "#7f7f7f", "hatch": ""},
    "Local (T+, I+)": {"color": "#9ad1a6", "hatch": "\\"},
    "Local (T-, I-)": {"color": "#d9d9d9", "hatch": ""},
}
STAGES = [
    ("local_vs_global_localtests_overlay_step1_baselines", {"Global (T-, I-)", "Local (T-, I-)"}),
    ("local_vs_global_localtests_overlay_step2_add_global_ours", {"Global (T+, I+)", "Global (T-, I-)", "Local (T-, I-)"}),
    ("local_vs_global_localtests_overlay_step3_add_local_ours", set(PLOT_ORDER)),
]


def _row_to_values(row):
    return float(row["mean_ppl"]), float(row["ci_low"]), float(row["ci_high"])


def _aggregate_rows(rows):
    means, lows, highs = [], [], []
    for _, row in rows.iterrows():
        mean, ci_low, ci_high = _row_to_values(row)
        means.append(mean)
        lows.append(ci_low)
        highs.append(ci_high)
    return float(np.mean(means)), float(np.mean(lows)), float(np.mean(highs))


def load_records():
    df = pd.read_csv(CSV_PATH)
    size = "1b"
    combos = [
        {"label": "Global (T+, I+)", "scope": "global", "meta": "with_metadata"},
        {"label": "Local (T+, I+)", "scope": "local", "meta": "with_metadata"},
        {"label": "Global (T-, I-)", "scope": "global", "meta": "without_metadata"},
        {"label": "Local (T-, I-)", "scope": "local", "meta": "without_metadata"},
    ]

    records = {}
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
            rows = []
            test_path = f"/path/to/metacul/training_data/meco_datasets/combined/{meta}/"
            for continent in CONTINENT_KEYS:
                model_path = f"/path/to/metacul/models/{continent}_{meta}_{size}"
                row = df[(df["model_path"] == model_path) & (df["test_set_path"] == test_path)]
                if not row.empty:
                    rows.append(row.iloc[0])
            if not rows:
                continue
            mean, ci_low, ci_high = _aggregate_rows(pd.DataFrame(rows))
        records[combo["label"]] = (mean, ci_low, ci_high)
    return records


def build_plot(records, visible_labels, out_stem):
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    center = np.array([0.0])
    width = 0.18
    gap = 0.06
    offsets = np.array([0.0, width, 2 * width + gap, 3 * width + gap])
    offsets = offsets - offsets.mean()

    all_means = [records[label][0] for label in PLOT_ORDER]
    y_min = min(all_means)
    y_max = max(all_means)

    for i, label in enumerate(PLOT_ORDER):
        if label not in visible_labels:
            continue
        mean, ci_low, ci_high = records[label]
        style = COMBO_STYLES[label]
        ax.bar(
            center + offsets[i],
            [mean],
            width,
            yerr=np.array([[mean - ci_low], [ci_high - mean]]),
            capsize=4.2,
            color=style["color"],
            hatch=style["hatch"],
            edgecolor="black",
            linewidth=1.3,
            error_kw=dict(ecolor="#2b2b2b", lw=1.9, capthick=1.9),
            zorder=3,
        )

    ax.set_xticks(center)
    ax.set_xticklabels([GROUP_LABEL], fontsize=18)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel("Perplexity (↓ better)", fontsize=20)
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(y_min - 0.55, y_max + 1.05)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.28)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)
        spine.set_color("black")

    legend_handles = [
        Patch(
            facecolor=COMBO_STYLES[label]["color"],
            hatch=COMBO_STYLES[label]["hatch"],
            edgecolor="black",
            linewidth=1.2,
            label=label,
        )
        for label in LEGEND_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        frameon=True,
        fancybox=True,
        framealpha=0.93,
        edgecolor="black",
        fontsize=14,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.03),
        borderpad=0.5,
        handlelength=1.6,
        columnspacing=1.2,
    )

    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.83)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{out_stem}.png"
    pdf_path = OUT_DIR / f"{out_stem}.pdf"
    fig.savefig(png_path, dpi=220, pad_inches=0.0)
    fig.savefig(pdf_path, dpi=600, pad_inches=0.0)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


def main():
    records = load_records()
    for out_stem, visible_labels in STAGES:
        build_plot(records, visible_labels, out_stem)


if __name__ == "__main__":
    main()
