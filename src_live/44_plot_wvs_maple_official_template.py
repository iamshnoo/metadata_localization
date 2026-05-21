#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


REPO_ROOT = Path("/path/to/metacul")
CULTURE_MAP_SRC = Path("/path/to/culture-map/src")
if str(CULTURE_MAP_SRC) not in sys.path:
    sys.path.insert(0, str(CULTURE_MAP_SRC))

from culture_map.paper_assets import load_country_map  # noqa: E402
from culture_map.plotting import (  # noqa: E402
    CATEGORY_COLORS,
    CATEGORY_LABEL_FONT_SIZES,
    CATEGORY_LABEL_POSITIONS,
    CULTURE_ZONE_DRAW_ORDER,
    MUSLIM_COUNTRIES,
    _format_model_label,
    _model_label_bbox,
    extract_template_polygons,
)


WVS_ROOT = Path(
    os.environ.get(
        "WVS_ROOT",
        str(REPO_ROOT / "results" / "culture_map_wvs_best_maple_mixed"),
    )
)
TEMPLATE_IMAGE = Path(
    os.environ.get(
        "TEMPLATE_IMAGE",
        "/path/to/culture-map/data/paper_osf/Map2023NEWsmall.png",
    )
)
OUTPUT_PDF = Path(
    os.environ.get(
        "OUTPUT_PDF",
        str(REPO_ROOT / "results" / "plots" / "plot8" / "wvs_maple_official_template.pdf"),
    )
)
OUTPUT_PNG = Path(
    os.environ.get(
        "OUTPUT_PNG",
        str(REPO_ROOT / "results" / "plots" / "plot8" / "wvs_maple_official_template.png"),
    )
)

VARIANT_ORDER = [
    "maple_1b_tminus_eminus",
    "maple_1b_tplus_eplus",
    "maple_3b_tminus_eminus",
    "maple_3b_tplus_eplus",
]

LABEL_NUDGES = {
    "MAPLE 1B (T-, I-)": (0.15, -0.18),
    "MAPLE 1B (T+, I+)": (0.22, 0.18),
    "MAPLE 3B (T-, I-)": (0.10, -0.20),
    "MAPLE 3B (T+, I+)": (-0.15, 0.12),
}

XMIN = -2.5
XMAX = 3.5
YMIN = -2.5
YMAX = 2.0


def _category_label_text(category: str) -> str:
    if category == "African-Islamic":
        return "African-\nIslamic"
    if category == "Latin America":
        return "Latin\nAmerica"
    if category == "Orthodox Europe":
        return "Orthodox\nEurope"
    if category == "West & South Asia":
        return "West&South\nAsia"
    if category == "Catholic Europe":
        return "Catholic\nEurope"
    if category == "Protestant Europe":
        return "Protestant\nEurope"
    return category


def _load_centroids():
    means = pd.read_csv(WVS_ROOT / "all_variant_country_mean_projection.csv")
    overall = pd.read_csv(WVS_ROOT / "all_variant_overall_summary.csv")
    centroids = (
        means.groupby("variant", as_index=False)
        .agg(RC1=("RC1", "mean"), RC2=("RC2", "mean"))
        .merge(overall[["variant", "label"]], on="variant", how="left")
    )
    centroids["label"] = centroids["label"].str.replace('"', "", regex=False).str.replace("  ", " ", regex=False)
    return means, centroids


def _draw_background(ax, human_df, polygons):
    ax.set_facecolor("white")

    for category in CULTURE_ZONE_DRAW_ORDER:
        polygon = polygons[category]
        ax.fill(
            polygon[:, 0],
            polygon[:, 1],
            facecolor=CATEGORY_COLORS[category],
            edgecolor="none",
            alpha=0.92,
            zorder=0,
        )
    for category in CULTURE_ZONE_DRAW_ORDER:
        polygon = polygons[category]
        ax.plot(
            np.r_[polygon[:, 0], polygon[0, 0]],
            np.r_[polygon[:, 1], polygon[0, 1]],
            color="black",
            linewidth=2.35,
            zorder=1,
            solid_joinstyle="round",
            solid_capstyle="round",
            antialiased=True,
        )

    ax.scatter(human_df["RC1_final"], human_df["RC2_final"], s=36, color="black", zorder=3)
    for _, row in human_df.iterrows():
        ax.text(
            row["RC1_final"] + 0.03,
            row["RC2_final"] + 0.03,
            row["country"],
            fontsize=8.4,
            fontstyle="italic" if row["country"] in MUSLIM_COUNTRIES else "normal",
            color="black",
            zorder=4,
        )

    for category, (x_pos, y_pos) in CATEGORY_LABEL_POSITIONS.items():
        text = ax.text(
            x_pos,
            y_pos,
            _category_label_text(category),
            fontsize=CATEGORY_LABEL_FONT_SIZES[category],
            fontstyle="italic",
            fontweight="semibold",
            ha="center",
            va="center",
            color="black",
            zorder=2,
        )
        text.set_path_effects([patheffects.withStroke(linewidth=2.0, foreground=(1, 1, 1, 0.20))])

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(np.arange(-2.5, 3.51, 0.5))
    ax.set_yticks(np.arange(-2.5, 2.01, 0.5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("Survival vs. Self-Expression Values", fontsize=18, fontstyle="italic", fontweight="semibold")
    ax.set_ylabel("Traditional vs. Secular Values", fontsize=18, fontstyle="italic", fontweight="semibold")
    ax.text(
        0.01,
        0.03,
        "Muslim countries in italic",
        transform=ax.transAxes,
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "#999999", "boxstyle": "square,pad=0.25"},
        zorder=10,
    )
    ax.text(
        0.985,
        0.06,
        "Source: World Values Survey &\nEuropean Values Study\n(2005-2022)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="black",
        zorder=10,
    )
    ax.text(
        0.985,
        0.007,
        "www.worldvaluessurvey.org\nhttps://europeanvaluesstudy.eu/",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="#2663d8",
        zorder=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9c9c9")
    ax.spines["bottom"].set_color("#c9c9c9")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", labelsize=10.5, color="#c9c9c9", width=1.0, length=3)
    ax.grid(False)


def _draw_maple_points(ax, centroids):
    for variant in VARIANT_ORDER:
        row = centroids.loc[centroids["variant"] == variant]
        if row.empty:
            continue
        row = row.iloc[0]
        label = _format_model_label(str(row["label"]))
        rc1 = float(row["RC1"])
        rc2 = float(row["RC2"])
        display_rc1 = min(max(rc1, XMIN + 0.08), XMAX - 0.08)
        display_rc2 = min(max(rc2, YMIN + 0.08), YMAX - 0.08)
        off_map = (display_rc1 != rc1) or (display_rc2 != rc2)

        ax.scatter(
            [display_rc1],
            [display_rc2],
            s=360,
            marker="^",
            color="#d5a3ea",
            edgecolors="#4f425b",
            linewidths=1.4,
            zorder=6,
        )
        if off_map:
            if rc2 > YMAX:
                ax.annotate(
                    "",
                    xy=(display_rc1, YMAX - 0.02),
                    xytext=(display_rc1, YMAX - 0.28),
                    arrowprops=dict(arrowstyle="-|>", color="#4f425b", linewidth=1.1),
                    zorder=6,
                )
            elif rc1 < XMIN:
                ax.annotate(
                    "",
                    xy=(XMIN + 0.02, display_rc2),
                    xytext=(XMIN + 0.28, display_rc2),
                    arrowprops=dict(arrowstyle="-|>", color="#4f425b", linewidth=1.1),
                    zorder=6,
                )
        dx, dy = LABEL_NUDGES.get(label, (0.08, 0.08))
        ax.text(
            display_rc1 + dx,
            display_rc2 + dy,
            label if not off_map else f"{label} off-map",
            fontsize=12.5,
            color="black",
            bbox=_model_label_bbox(),
            zorder=7,
        )


def main():
    means, centroids = _load_centroids()
    keep = sorted(means["country"].unique())

    data_dir = Path("/path/to/culture-map/data/paper_osf")
    human_df = load_country_map(data_dir)
    human_df = human_df.loc[human_df["country"].isin(keep)].copy()

    polygons = extract_template_polygons(TEMPLATE_IMAGE)

    fig, ax = plt.subplots(figsize=(14, 10))
    _draw_background(ax, human_df, polygons)
    _draw_maple_points(ax, centroids)
    fig.tight_layout()

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
