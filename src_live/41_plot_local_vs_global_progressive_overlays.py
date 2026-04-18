#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


CSV_PATH = Path("/scratch/amukher6/metacul/results/fp16_old/plots_fp16/plot2/plot_2.csv")
OUT_DIR = Path("/scratch/amukher6/metacul/slides")

REGIONS = ["Africa", "America", "Asia", "Europe"]
REGION_KEYS = ["africa", "america", "asia", "europe"]
PLOT_ORDER = ["Global(T+,I+)", "Global(T-,I-)", "Local(T+,I+)", "Local(T-,I-)"]
LEGEND_ORDER = ["Global(T-,I-)", "Global(T+,I+)", "Local(T-,I-)", "Local(T+,I+)"]
COMBO_STYLES = {
    "Global(T+,I+)": {"color": "#a6cee3", "hatch": "o"},
    "Global(T-,I-)": {"color": "#7f7f7f", "hatch": ""},
    "Local(T+,I+)": {"color": "#9ad1a6", "hatch": "\\"},
    "Local(T-,I-)": {"color": "#d9d9d9", "hatch": ""},
}
STAGES = [
    ("local_vs_global_localtests_overlay_step1_baselines", {"Global(T-,I-)", "Local(T-,I-)"}),
    ("local_vs_global_localtests_overlay_step2_add_global_ours", {"Global(T+,I+)", "Global(T-,I-)", "Local(T-,I-)"}),
    ("local_vs_global_localtests_overlay_step3_add_local_ours", set(PLOT_ORDER)),
]


def load_records():
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    combos = [
        ("Global(T+,I+)", "global", "with_metadata"),
        ("Global(T-,I-)", "global", "without_metadata"),
        ("Local(T+,I+)", "local", "with_metadata"),
        ("Local(T-,I-)", "local", "without_metadata"),
    ]

    records = {}
    for region in REGION_KEYS:
        for label, scope, meta in combos:
            if scope == "local":
                model_path = f"/scratch/amukher6/metacul/models/{region}_{meta}_1b"
            else:
                model_path = f"/scratch/amukher6/metacul/models/combined_{meta}_1b"
            test_path = f"/scratch/amukher6/metacul/training_data/meco_datasets/continents/{region}/{meta}/"
            match = next(
                row for row in rows
                if row["model_path"] == model_path and row["test_set_path"] == test_path
            )
            mean = float(match["mean_ppl"])
            ci_low = float(match["ci_low"])
            ci_high = float(match["ci_high"])
            records[(region.capitalize(), label)] = (mean, ci_low, ci_high)
    return records


def build_plot(records, visible_labels, out_stem):
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    x_pos = np.arange(len(REGIONS))
    width = 0.18
    gap = 0.06

    all_means = [records[(region, label)][0] for region in REGIONS for label in PLOT_ORDER]
    y_min = min(all_means)
    y_max = max(all_means)

    for i, label in enumerate(PLOT_ORDER):
        offset = i * width + (gap if i >= 2 else 0.0)
        means, yerr_lo, yerr_hi = [], [], []
        for region in REGIONS:
            mean, ci_low, ci_high = records[(region, label)]
            means.append(mean)
            yerr_lo.append(mean - ci_low)
            yerr_hi.append(ci_high - mean)

        if label in visible_labels:
            style = COMBO_STYLES[label]
            ax.bar(
                x_pos + offset,
                means,
                width,
                yerr=np.array([yerr_lo, yerr_hi]),
                capsize=4.2,
                color=style["color"],
                hatch=style["hatch"],
                edgecolor="black",
                linewidth=1.3,
                error_kw=dict(ecolor="#2b2b2b", lw=1.9, capthick=1.9),
                zorder=3,
            )

    ax.set_xticks(x_pos + (3 * width + gap) / 2)
    ax.set_xticklabels(REGIONS, fontsize=17)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel("Perplexity (↓ better)", fontsize=20)
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
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.985, 0.985),
        borderpad=0.5,
        handlelength=1.6,
        columnspacing=1.2,
    )

    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.12, top=0.96)
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
