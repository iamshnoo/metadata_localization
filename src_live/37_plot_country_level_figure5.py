#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
from matplotlib.lines import Line2D


CSV_PATH = Path("/scratch/amukher6/metacul/results/plots/plot12/plot_14.csv")
RESULTS_DIR = Path("/scratch/amukher6/metacul/results/plots/plot12")
LATEX_FIG = Path("/scratch/amukher6/metacul/latex/figs/appendix/13_local_country_ppl_500m_1b.pdf")

CONTINENTS = ["Africa", "America", "Asia", "Europe"]
SIZE_COLORS = {"500M": "#f4a3a3", "1B": "#9ad1a6"}
CONTINENT_BG = {
    "Africa": "#edc3ca",
    "America": "#c7d9f4",
    "Asia": "#d5e7c1",
    "Europe": "#e7d0ae",
}


def main():
    df = pd.read_csv(CSV_PATH)
    fig, ax = plt.subplots(figsize=(19, 7.2))

    bbox_props = dict(
        facecolor="lightgrey",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.82,
        boxstyle="round",
        pad=0.35,
    )

    country_positions = []
    country_labels = []
    group_specs = []
    continent_means_by_size = {"500M": [], "1B": []}
    cursor = 0.0
    continent_gap = 1.25

    for continent in CONTINENTS:
        sub = df[df["continent"] == continent].copy()
        order = (
            sub[sub["size"] == "1B"]
            .sort_values("mean_ppl")["country_name"]
            .tolist()
        )
        if not order:
            order = sorted(sub["country_name"].unique().tolist())
        sub["country_name"] = pd.Categorical(sub["country_name"], categories=order, ordered=True)
        sub = sub.sort_values(["country_name", "size"]).reset_index(drop=True)

        base_width = 0.78 if continent != "Asia" else 0.60
        marker_offsets = {"500M": -0.16, "1B": 0.16}

        group_start = cursor
        x = pd.Series(range(len(order)), dtype=float).to_numpy() + cursor
        country_positions.extend(x.tolist())
        country_labels.extend(order)

        by_size = {}
        for size_label in ["500M", "1B"]:
            size_sub = (
                sub[sub["size"] == size_label]
                .set_index("country_name")
                .reindex(order)
                .reset_index()
            )
            by_size[size_label] = size_sub
            continent_mean = float(size_sub["continent_mean_ppl"].dropna().iloc[0])
            continent_means_by_size[size_label].append(
                {
                    "left": x[0] - base_width * 0.7,
                    "right": x[-1] + base_width * 0.7,
                    "mean": continent_mean,
                }
            )

        for idx, _country_name in enumerate(order):
            y_500 = float(by_size["500M"].iloc[idx]["mean_ppl"])
            y_1b = float(by_size["1B"].iloc[idx]["mean_ppl"])
            x_500 = x[idx] + marker_offsets["500M"]
            x_1b = x[idx] + marker_offsets["1B"]
            line = ax.plot(
                [x_500, x_1b],
                [y_500, y_1b],
                color="#7a7a7a",
                linewidth=2.8,
                alpha=0.86,
                zorder=2.8,
            )[0]
            line.set_path_effects([pe.Stroke(linewidth=3.7, foreground="white", alpha=0.65), pe.Normal()])

        for size_label in ["500M", "1B"]:
            face = SIZE_COLORS[size_label]
            size_sub = by_size[size_label]
            ax.scatter(
                x + marker_offsets[size_label],
                size_sub["mean_ppl"],
                s=122,
                color=face,
                edgecolors="black",
                linewidths=1.1,
                zorder=3.4,
            )

        group_end = x[-1]
        group_specs.append(
            {
                "continent": continent,
                "start": group_start,
                "end": group_end,
                "center": (group_start + group_end) / 2,
                "base_width": base_width,
            }
        )
        cursor = group_end + continent_gap

    for spec in group_specs:
        ax.axvspan(
            spec["start"] - spec["base_width"] * 0.75,
            spec["end"] + spec["base_width"] * 0.75,
            facecolor=CONTINENT_BG.get(spec["continent"], "#f5f5f5"),
            alpha=0.78,
            zorder=0.1,
            linewidth=0,
        )

    for size_label, segments in continent_means_by_size.items():
        face = SIZE_COLORS[size_label]
        line_x = []
        line_y = []
        for idx, seg in enumerate(segments):
            if idx > 0:
                prev = segments[idx - 1]
                join_x = (prev["right"] + seg["left"]) / 2
                line_x.extend([join_x, join_x])
                line_y.extend([prev["mean"], seg["mean"]])
            line_x.extend([seg["left"], seg["right"]])
            line_y.extend([seg["mean"], seg["mean"]])
        line = ax.plot(
            line_x,
            line_y,
            color=face,
            linestyle=(0, (4, 2)),
            linewidth=2.1,
            alpha=0.98,
            zorder=2.2,
        )[0]
        line.set_path_effects([pe.Stroke(linewidth=3.0, foreground="white", alpha=0.58), pe.Normal()])

    ax.set_xticks(country_positions)
    ax.set_xticklabels(country_labels, rotation=40, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.26)
    ax.set_ylim(6, 13)
    ax.set_xlim(min(country_positions) - 1.0, max(country_positions) + 1.0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")
    ax.set_ylabel("Perplexity (↓ better)", fontsize=18)

    for spec in group_specs:
        ax.text(
            spec["center"],
            12.25,
            spec["continent"],
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            bbox=bbox_props,
            zorder=5,
        )

    legend_handles = [
        Line2D(
            [],
            [],
            color=SIZE_COLORS["500M"],
            marker="o",
            linestyle="None",
            markersize=11.5,
            markeredgecolor="black",
            markeredgewidth=1.0,
            label="500M",
        ),
        Line2D(
            [],
            [],
            color=SIZE_COLORS["1B"],
            marker="o",
            linestyle="None",
            markersize=11.5,
            markeredgecolor="black",
            markeredgewidth=1.0,
            label="1B",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.93,
        edgecolor="black",
        fontsize=14,
        title="Model:",
        title_fontsize=14,
        bbox_to_anchor=(0.5, 0.81),
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.70, bottom=0.26, left=0.07, right=0.99)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIG.parent.mkdir(parents=True, exist_ok=True)
    png_path = RESULTS_DIR / "local_country_ppl_500m_1b.png"
    pdf_path = RESULTS_DIR / "local_country_ppl_500m_1b.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(LATEX_FIG, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {LATEX_FIG}")


if __name__ == "__main__":
    main()
