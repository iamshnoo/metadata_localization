#!/usr/bin/env python3
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


REPO_ROOT = Path("/path/to/metacul")
WVS_ROOT = Path(os.environ.get("WVS_ROOT", str(REPO_ROOT / "results" / "culture_map_wvs_country_eval")))
OUTPUT_PDF = Path(os.environ.get("OUTPUT_PDF", str(REPO_ROOT / "results" / "plots" / "plot8" / "wvs_variant_centroids.pdf")))
OUTPUT_PNG = Path(os.environ.get("OUTPUT_PNG", str(REPO_ROOT / "results" / "plots" / "plot8" / "wvs_variant_centroids.png")))

CULTURE_MAP_SRC = Path("/path/to/culture-map/src")
if str(CULTURE_MAP_SRC) not in sys.path:
    sys.path.insert(0, str(CULTURE_MAP_SRC))

from culture_map.paper_assets import load_country_map  # noqa: E402
from culture_map.plotting import (  # noqa: E402
    CATEGORY_COLORS,
    CATEGORY_LABEL_POSITIONS,
    CULTURE_ZONE_DRAW_ORDER,
    smooth_region_polygon,
)


VARIANT_STYLE = {
    "maple_1b_tminus_eminus": {"label": "MAPLE 1B (T-, I-)", "color": "#4c78a8", "marker": "o"},
    "maple_1b_tplus_eplus": {"label": "MAPLE 1B (T+, I+)", "color": "#f58518", "marker": "o"},
    "maple_3b_tminus_eminus": {"label": "MAPLE 3B (T-, I-)", "color": "#54a24b", "marker": "^"},
    "maple_3b_tplus_eplus": {"label": "MAPLE 3B (T+, I+)", "color": "#e45756", "marker": "^"},
}

XMIN = -2.5
XMAX = 3.5
YMIN = -2.5
YMAX = 2.0

LABEL_NUDGES = {
    "maple_1b_tminus_eminus": (0.00, -0.05),
    "maple_1b_tplus_eplus": (0.00, -0.32),
    "maple_3b_tminus_eminus": (0.00, -0.18),
    "maple_3b_tplus_eplus": (0.00, -0.46),
}


def _draw_background(ax, polygon_df, label_df):
    for category in CULTURE_ZONE_DRAW_ORDER:
        category_points = polygon_df.loc[polygon_df["Category"] == category, ["RC1_final", "RC2_final"]].to_numpy()
        if len(category_points) < 3:
            continue
        polygon = smooth_region_polygon(category_points)
        ax.fill(
            polygon[:, 0],
            polygon[:, 1],
            facecolor=CATEGORY_COLORS[category],
            edgecolor="black",
            linewidth=1.4,
            alpha=0.88,
            zorder=0,
        )

    ax.scatter(label_df["RC1_final"], label_df["RC2_final"], s=24, color="black", zorder=3)
    for _, row in label_df.iterrows():
        ax.text(row["RC1_final"] + 0.03, row["RC2_final"] + 0.03, row["country"], fontsize=6.5, color="black", zorder=4)

    for category, (x_pos, y_pos) in CATEGORY_LABEL_POSITIONS.items():
        label = category
        if category == "African-Islamic":
            label = "African-\nIslamic"
        elif category == "Latin America":
            label = "Latin\nAmerica"
        elif category == "Orthodox Europe":
            label = "Orthodox\nEurope"
        elif category == "West & South Asia":
            label = "West & South\nAsia"
        elif category == "Catholic Europe":
            label = "Catholic\nEurope"
        elif category == "Protestant Europe":
            label = "Protestant\nEurope"
        elif category == "English-Speaking":
            label = "English-\nSpeaking"
        ax.text(
            x_pos,
            y_pos,
            label,
            fontsize=10.5,
            fontstyle="italic",
            fontweight="semibold",
            ha="center",
            va="center",
            color="black",
            alpha=0.78,
            zorder=2,
        )

    ax.axvline(0.0, color="#4a4a4a", linewidth=1.4, alpha=0.8, linestyle=(0, (1, 3)), zorder=1)
    ax.axhline(0.0, color="#4a4a4a", linewidth=1.4, alpha=0.8, linestyle=(0, (1, 3)), zorder=1)
    ax.text(
        -2.25,
        1.35,
        "Strong\nSecular\nValues",
        ha="left",
        va="top",
        fontsize=19,
        fontweight="bold",
        color="black",
        zorder=8,
    )
    ax.text(
        3.25,
        -2.4,
        "Strong\nSelf-Expression\nValues",
        ha="right",
        va="bottom",
        fontsize=19,
        fontweight="bold",
        color="black",
        zorder=8,
    )

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks([-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    ax.set_yticks([-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("Survival vs. Self-Expression Values", fontsize=18, fontstyle="italic", fontweight="semibold")
    ax.set_ylabel("Traditional vs. Secular Values", fontsize=18, fontstyle="italic", fontweight="semibold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9c9c9")
    ax.spines["bottom"].set_color("#c9c9c9")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", labelsize=10.5, color="#c9c9c9", width=1.0, length=3)
    ax.grid(False)


def main():
    means = pd.read_csv(WVS_ROOT / "all_variant_country_mean_projection.csv")
    data_dir = Path("/path/to/culture-map/data/paper_osf")
    human_df_full = load_country_map(data_dir)
    keep = sorted(means["country"].unique())
    human_df = human_df_full[human_df_full["country"].isin(keep)].copy()

    fig, ax = plt.subplots(figsize=(14, 9.4))
    _draw_background(ax, human_df_full, human_df)

    for variant, style in VARIANT_STYLE.items():
        sub = means[means["variant"] == variant].copy()
        if sub.empty:
            continue
        avg_rc1 = float(sub["RC1"].mean())
        avg_rc2 = float(sub["RC2"].mean())
        display_rc1 = min(max(avg_rc1, XMIN + 0.08), XMAX - 0.08)
        display_rc2 = min(max(avg_rc2, YMIN + 0.08), YMAX - 0.12)
        off_map = (display_rc1 != avg_rc1) or (display_rc2 != avg_rc2)

        ax.scatter(
            [display_rc1],
            [display_rc2],
            s=220,
            color=style["color"],
            marker=style["marker"],
            edgecolors="black",
            linewidths=1.0,
            zorder=7,
        )
        if off_map:
            ax.annotate(
                "",
                xy=(display_rc1, YMAX - 0.03),
                xytext=(display_rc1, YMAX - 0.35),
                arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.0),
                zorder=7,
            )
        dx, dy = LABEL_NUDGES[variant]
        label_text = "{}{}\n({:.2f}, {:.2f})".format(
            style["label"],
            " off-map" if off_map else "",
            avg_rc1,
            avg_rc2,
        )
        ax.text(
            display_rc1 + dx,
            display_rc2 + dy,
            label_text,
            fontsize=9.2,
            color="black",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="black", linewidth=0.7, alpha=0.95),
            zorder=8,
        )

    legend_handles = [
        Line2D([0], [0], marker=style["marker"], color="w", label=style["label"], markerfacecolor=style["color"], markeredgecolor="black", markersize=9)
        for style in VARIANT_STYLE.values()
    ]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.98, 0.985), ncol=2, frameon=False, fontsize=10.5)
    fig.tight_layout()

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
