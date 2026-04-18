#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt


CSV_PATH = Path("/scratch/amukher6/metacul/results/plots/plot1/plot_1.csv")
OUT_DIR = Path("/scratch/amukher6/metacul/slides")

CONTINENTS = ["Africa", "America", "Asia", "Europe"]
TRAIN_ORDER = ["T-", "T+"]
INFER_ORDER = ["I-", "I+"]

COMBO_STYLES = {
    ("T-", "I-"): {"color": "#d9d9d9", "marker": "s", "label": "T-/I-"},
    ("T-", "I+"): {"color": "#fad9b7", "marker": "o", "label": "T-/I+"},
    ("T+", "I-"): {"color": "#eca7a4", "marker": "s", "label": "T+/I-"},
    ("T+", "I+"): {"color": "#a3cea8", "marker": "o", "label": "T+/I+"},
}
LABEL_COLORS = {
    ("T-", "I-"): "#666666",
    ("T-", "I+"): "#b67c16",
    ("T+", "I-"): "#d26b68",
    ("T+", "I+"): "#5f9e65",
}
X_OFFSET = {"I-": -0.035, "I+": 0.035}
LINESTYLE = {"I-": "--", "I+": "-"}


def parse_model_path(path):
    name = Path(path).name
    if "_with_metadata_" in name:
        continent, size = name.split("_with_metadata_")
        train = "T+"
    elif "_without_metadata_" in name:
        continent, size = name.split("_without_metadata_")
        train = "T-"
    else:
        raise ValueError(f"Unexpected model path: {path}")
    return continent.capitalize(), train, size


def parse_test_path(path):
    parts = [p for p in Path(path).parts if p]
    continent = parts[-2].capitalize()
    infer = "I+" if parts[-1] == "with_metadata" else "I-"
    return continent, infer


def load_plot_rows():
    rows = []
    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            continent, train_meta, size = parse_model_path(row["model_path"])
            test_continent, infer_meta = parse_test_path(row["test_set_path"])
            if continent != test_continent:
                continue
            rows.append(
                {
                    "continent": continent,
                    "train_meta": train_meta,
                    "infer_meta": infer_meta,
                    "size": size,
                    "mean_ppl": float(row["mean_ppl"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                }
            )
    return rows


def build_plot(visible_combos, out_stem):
    subset = [row for row in load_plot_rows() if row["size"] == "1b"]

    fig, axes = plt.subplots(2, 2, figsize=(8.35, 6.9), sharex=True, sharey=True)
    axes = axes.flatten()
    all_vals = []

    for ax, continent in zip(axes, CONTINENTS):
        cont_rows = [row for row in subset if row["continent"] == continent]
        ax.axvspan(-0.25, 0.25, color="#efdfdd", alpha=0.95, zorder=0.1)
        ax.axvspan(0.75, 1.25, color="#dfeedd", alpha=0.95, zorder=0.1)

        label_items = []
        for infer_meta in INFER_ORDER:
            points = []
            for train_meta in TRAIN_ORDER:
                combo = (train_meta, infer_meta)
                if combo not in visible_combos:
                    continue
                matches = [
                    row for row in cont_rows
                    if row["train_meta"] == train_meta and row["infer_meta"] == infer_meta
                ]
                if not matches:
                    continue
                row = matches[0]
                x = TRAIN_ORDER.index(train_meta) + X_OFFSET[infer_meta]
                mean = float(row["mean_ppl"])
                all_vals.append(mean)
                points.append((x, mean, float(row["ci_low"]), float(row["ci_high"]), train_meta))

            if len(points) == 2:
                xs = [points[0][0], points[1][0]]
                ys = [points[0][1], points[1][1]]
                line, = ax.plot(
                    xs,
                    ys,
                    color=COMBO_STYLES[(points[1][4], infer_meta)]["color"],
                    linestyle=LINESTYLE[infer_meta],
                    linewidth=3.2,
                    zorder=2.6,
                )
                line.set_path_effects([pe.Stroke(linewidth=4.1, foreground="white", alpha=0.68), pe.Normal()])

            for x, mean, ci_low, ci_high, train_meta in points:
                style = COMBO_STYLES[(train_meta, infer_meta)]
                ax.scatter(
                    [x],
                    [mean],
                    s=98,
                    color=style["color"],
                    marker=style["marker"],
                    edgecolors="black",
                    linewidths=0.85,
                    zorder=4,
                )
                ax.errorbar(
                    [x],
                    [mean],
                    yerr=[[mean - ci_low], [ci_high - mean]],
                    linestyle="none",
                    color=style["color"],
                    linewidth=1.55,
                    capsize=2.8,
                    zorder=3.2,
                )
                label_items.append((x, mean, train_meta, infer_meta, style["label"]))

        ax.text(
            0.5,
            0.96,
            continent,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12.7,
            fontweight="bold",
            bbox=dict(facecolor="#e2e2e2", edgecolor="#aaaaaa", alpha=0.95, boxstyle="round,pad=0.28"),
        )

        for x, mean, train_meta, infer_meta, label in label_items:
            if train_meta == "T-":
                ha = "right"
                dx = -0.08
                dy = 0.18 if infer_meta == "I+" else -0.15
            else:
                ha = "left"
                dx = 0.03
                dy = 0.03 if infer_meta == "I+" else -0.03
            ax.text(
                x + dx,
                mean + dy,
                label,
                fontsize=10.1,
                fontweight="black",
                color=LABEL_COLORS[(train_meta, infer_meta)],
                ha=ha,
                va="bottom" if dy >= 0 else "top",
                zorder=5,
                path_effects=[pe.withStroke(linewidth=1.9, foreground="white", alpha=0.96)],
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["T-", "T+"], fontsize=16, fontweight="bold")
        ax.tick_params(axis="y", labelsize=16)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    if not all_vals:
        raise RuntimeError("No visible values selected for the plot.")

    y_min = min(all_vals)
    y_max = max(all_vals)
    for ax in axes:
        ax.set_ylim(y_min - 0.35, y_max + 0.68)

    fig.text(0.5, 0.04, "Training metadata", ha="center", fontsize=18)
    fig.text(0.02, 0.5, "Perplexity (↓ better)", va="center", rotation="vertical", fontsize=18)
    fig.subplots_adjust(top=0.965, bottom=0.11, left=0.10, right=0.99, hspace=0.12, wspace=0.10)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{out_stem}.png"
    pdf_path = OUT_DIR / f"{out_stem}.pdf"
    fig.savefig(png_path, dpi=220, pad_inches=0.0)
    fig.savefig(pdf_path, dpi=600, pad_inches=0.0)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


def main() -> None:
    build_plot({("T-", "I-")}, "local_factorial_overlay_step1_tminus_iminus")
    build_plot({("T-", "I-"), ("T+", "I-")}, "local_factorial_overlay_step2_add_tplus_iminus")
    build_plot({("T-", "I-"), ("T+", "I-"), ("T-", "I+")}, "local_factorial_overlay_step3_add_tminus_iplus")
    build_plot({("T-", "I-"), ("T+", "I-"), ("T-", "I+"), ("T+", "I+")}, "local_factorial_overlay_step4_add_tplus_iplus")


if __name__ == "__main__":
    main()
