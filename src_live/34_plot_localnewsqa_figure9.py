#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
from matplotlib.lines import Line2D


CSV_PATH = Path("/scratch/amukher6/metacul/results/plots/plot8/plot_8_pretrained_target_split_multiseed.csv")
RESULTS_DIR = Path("/scratch/amukher6/metacul/results/plots/plot8")
LATEX_MAIN_FIG = Path("/scratch/amukher6/metacul/latex/figs/main/8_accuracy_apples_to_apples_answered_by_all.pdf")


def _series_row(df, family, series, split):
    row = df[(df["family"] == family) & (df["series"] == series) & (df["split"] == split)]
    if row.empty:
        return None
    return row.iloc[0]


def _series_value(df, family, series, split):
    row = _series_row(df, family, series, split)
    if row is None:
        return None
    return float(row["accuracy"])


def _t_critical_95(dfree):
    lookup = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        25: 2.060,
        30: 2.042,
    }
    if dfree <= 0:
        return None
    if dfree in lookup:
        return lookup[dfree]
    if dfree < 30:
        nearest = max(key for key in lookup if key < dfree)
        return lookup[nearest]
    return 1.96


def _series_ci_halfwidth(row):
    if row is None or "accuracy_std" not in row.index or "seed_count" not in row.index:
        return None
    if pd.isna(row["accuracy_std"]) or pd.isna(row["seed_count"]):
        return None
    seed_count = int(row["seed_count"])
    if seed_count <= 1:
        return None
    critical = _t_critical_95(seed_count - 1)
    if critical is None:
        return None
    return critical * float(row["accuracy_std"]) / (seed_count ** 0.5)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot LocalNewsQA Figure 9 style accuracy panels."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=CSV_PATH,
        help="Input CSV with family/series/split/accuracy columns.",
    )
    parser.add_argument(
        "--families",
        default="1B,3B",
        help="Comma-separated family panels to include, e.g. '1B' or '1B,3B'.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for auxiliary PNG/PDF outputs.",
    )
    parser.add_argument(
        "--latex-main-fig",
        type=Path,
        default=LATEX_MAIN_FIG,
        help="Primary paper figure PDF path.",
    )
    parser.add_argument(
        "--basename",
        default="accuracy_explicit_vs_ambiguous_pretrained_target",
        help="Base filename for results-dir PNG/PDF outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv_path)

    split_order = ["Explicit", "Ambiguous", "Overall"]
    y_positions = {"Explicit": 2, "Ambiguous": 1, "Overall": 0}
    series_order = ["T+/I+", "T+/I-", "T-/I+", "T-/I-"]
    series_offsets = {
        "T+/I+": 0.18,
        "T+/I-": 0.06,
        "T-/I+": -0.06,
        "T-/I-": -0.18,
    }

    family_titles = [family.strip() for family in args.families.split(",") if family.strip()]
    if not family_titles:
        raise ValueError("No families specified.")
    nrows = len(family_titles)
    fig_height = 4.6 if nrows == 1 else 8.2
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(8.6, fig_height),
        sharex=True,
        gridspec_kw={"hspace": 0.40},
    )
    if nrows == 1:
        axes = [axes]

    colors = {
        "T+/I+": "#2f9e44",
        "T+/I-": "#78c679",
        "T-/I+": "#2b8cbe",
        "T-/I-": "#8a919c",
        "band": "#ebf6ec",
    }

    marker_style = {
        "T+/I+": dict(marker="o", s=210, facecolor=colors["T+/I+"], edgecolor="black", linewidth=1.8),
        "T+/I-": dict(marker="^", s=210, facecolor=colors["T+/I-"], edgecolor="black", linewidth=1.8),
        "T-/I+": dict(marker="D", s=165, facecolor=colors["T-/I+"], edgecolor="black", linewidth=1.65),
        "T-/I-": dict(marker="s", s=165, facecolor=colors["T-/I-"], edgecolor="black", linewidth=1.65),
    }

    x_vals = (df["accuracy"] * 100.0).tolist()
    if "accuracy_std" in df.columns and "seed_count" in df.columns:
        ci_bounds = []
        for _, row in df.iterrows():
            ci_halfwidth = _series_ci_halfwidth(row)
            if ci_halfwidth is None:
                continue
            acc = float(row["accuracy"]) * 100.0
            ci = ci_halfwidth * 100.0
            ci_bounds.extend([acc - ci, acc + ci])
        if ci_bounds:
            x_vals.extend(ci_bounds)
    x_min = max(0.0, min(x_vals) - 1.5) if x_vals else 0.0
    x_max = min(100.0, max(x_vals) + 1.5) if x_vals else 100.0
    tick_fs = 16
    label_fs = 19
    title_fs = 21
    legend_fs = 13
    row_label_fs = 18
    bbox_props = dict(
        facecolor="lightgrey",
        edgecolor="black",
        linewidth=1.2,
        alpha=0.84,
        boxstyle="round,pad=0.3",
    )

    for ax, family in zip(axes, family_titles):
        ax.axhspan(0.5, 1.5, color=colors["band"], zorder=0)

        for split in split_order:
            base_y = y_positions[split]
            row_points = []
            for short_series in series_order:
                full_series = f"{family} {short_series}"
                row = _series_row(df, family, full_series, split)
                if row is None:
                    continue
                x_val = float(row["accuracy"]) * 100.0
                y_val = base_y + series_offsets[short_series]
                ci_halfwidth = _series_ci_halfwidth(row)
                ci_val = None if ci_halfwidth is None else ci_halfwidth * 100.0
                row_points.append((x_val, y_val, short_series, ci_val))

            row_points.sort(key=lambda item: item[0])
            if len(row_points) >= 2:
                connector = ax.plot(
                    [point[0] for point in row_points],
                    [point[1] for point in row_points],
                    color="#d6d9df",
                    linewidth=2.2,
                    alpha=0.85,
                    solid_capstyle="round",
                    zorder=1,
                )[0]
                connector.set_path_effects([pe.Stroke(linewidth=3.2, foreground="white", alpha=0.72), pe.Normal()])

            for x_val, y_val, short_series, ci_val in row_points:
                if ci_val is not None and ci_val > 0:
                    ax.errorbar(
                        [x_val],
                        [y_val],
                        xerr=[ci_val],
                        fmt="none",
                        ecolor=colors[short_series],
                        elinewidth=1.9,
                        capsize=4.5,
                        capthick=1.7,
                        alpha=0.95,
                        zorder=3,
                    )
                ax.scatter([x_val], [y_val], zorder=4, **marker_style[short_series])

        ax.set_yticks([y_positions[s] for s in split_order])
        ax.set_yticklabels(split_order, fontsize=row_label_fs)
        ax.set_title(
            family,
            fontsize=title_fs,
            fontweight="bold",
            pad=8,
            y=0.88,
            bbox=bbox_props,
        )
        ax.grid(axis="x", linestyle="-", linewidth=1.05, alpha=0.38, color="#c7ced8")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.35, 2.65)
        ax.tick_params(axis="x", labelsize=tick_fs)
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color("black")

    axes[-1].set_xlabel("Target accuracy (%)", fontsize=label_fs)

    present_series = []
    for short_series in series_order:
        for family in family_titles:
            if _series_value(df, family, f"{family} {short_series}", "Overall") is not None:
                present_series.append(short_series)
                break

    legend_label_map = {
        "T+/I+": ("o", colors["T+/I+"], "(T+,I+)"),
        "T+/I-": ("^", colors["T+/I-"], "(T+,I-)"),
        "T-/I+": ("D", colors["T-/I+"], "(T-,I+)"),
        "T-/I-": ("s", colors["T-/I-"], "(T-,I-)"),
    }
    legend_handles = [
        Line2D([0], [0], marker=legend_label_map[short_series][0], color="none",
               markerfacecolor=legend_label_map[short_series][1], markeredgecolor="black",
               markeredgewidth=1.5 if short_series.startswith("T+") else 1.4,
               markersize=10.5 if short_series.startswith("T+") else 9.5,
               label=legend_label_map[short_series][2])
        for short_series in present_series
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=max(1, len(legend_handles)),
        frameon=True,
        fancybox=True,
        framealpha=0.94,
        edgecolor="black",
        fontsize=legend_fs,
        bbox_to_anchor=(0.5, 0.99),
        handletextpad=0.5,
        columnspacing=1.0,
    )
    if "accuracy_std" in df.columns and "seed_count" in df.columns:
        fig.text(
            0.99,
            0.012,
            "Bars show 95% CI across shuffle seeds",
            ha="right",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.95))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.results_dir / f"{args.basename}.png"
    pdf_path = args.results_dir / f"{args.basename}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.latex_main_fig, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {args.latex_main_fig}")


if __name__ == "__main__":
    main()
