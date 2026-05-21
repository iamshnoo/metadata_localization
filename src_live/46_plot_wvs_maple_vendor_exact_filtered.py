#!/usr/bin/env python3
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


REPO_ROOT = Path("/path/to/metacul")
WVS_ROOT = REPO_ROOT / "results" / "culture_map_wvs_best_maple_mixed"
OUTPUT_DIR = REPO_ROOT / "results" / "plots" / "plot8"

CULTURE_MAP_SRC = Path("/path/to/culture-map/src")
if str(CULTURE_MAP_SRC) not in sys.path:
    sys.path.insert(0, str(CULTURE_MAP_SRC))

from culture_map.constants import CATEGORY_LABEL_FONT_SIZES, CATEGORY_LABEL_POSITIONS, CULTURE_ZONE_DRAW_ORDER, MUSLIM_COUNTRIES  # noqa: E402
from culture_map.paper_assets import get_template_image_path, load_country_map  # noqa: E402
from culture_map.plotting import (  # noqa: E402
    CATEGORY_COLORS,
    MODEL_LABEL_FONT_SIZES,
    MODEL_LABEL_NUDGES,
    _format_model_label,
    _model_label_bbox,
    _category_expansion_params,
    _envelope_polygon_from_shape_and_points,
    _radial_expand_polygon,
    _resample_closed_curve,
    _smooth_closed_curve,
    extract_template_polygons,
)


XMIN = -2.5
XMAX = 3.5
YMIN = -2.5
YMAX = 2.0

LABEL_MAP = {
    "maple_1b_tminus_eminus": "MAPLE 1B (T-, I-)",
    "maple_1b_tplus_eplus": "MAPLE 1B (T+, I+)",
    "maple_3b_tminus_eminus": "MAPLE 3B (T-, I-)",
    "maple_3b_tplus_eplus": "MAPLE 3B (T+, I+)",
}

MAPLE_LABEL_NUDGES = {
    "MAPLE 1B (T-, I-)": (-0.35, 0.05),
    "MAPLE 1B (T+, I+)": (0.10, 0.10),
    "MAPLE 3B (T-, I-)": (0.08, 0.06),
    "MAPLE 3B (T+, I+)": (-0.35, 0.10),
}


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
    if category == "English-Speaking":
        return "English-Speaking"
    if category == "Protestant Europe":
        return "Protestant\nEurope"
    return category


def _build_polygons(human_df_full, template_image_path):
    polygons = extract_template_polygons(template_image_path)
    category_points = {
        category: human_df_full.loc[human_df_full["Category"] == category, ["RC1_final", "RC2_final"]].to_numpy()
        for category in CULTURE_ZONE_DRAW_ORDER
    }
    for category in CULTURE_ZONE_DRAW_ORDER:
        polygon = polygons[category]
        params = _category_expansion_params(category)
        polygon = _radial_expand_polygon(polygon, factor=params["factor"], pad=params["pad"])
        polygon = _envelope_polygon_from_shape_and_points(
            polygon,
            category_points[category],
            margin=params["margin"],
            samples=720,
            bandwidth=params["sigma"],
        )
        polygon = _resample_closed_curve(polygon, samples=500)
        polygon = _smooth_closed_curve(polygon, window=11, passes=3)
        polygons[category] = polygon
    return polygons


def main():
    means = pd.read_csv(WVS_ROOT / "all_variant_country_mean_projection.csv")
    keep_countries = sorted(means["country"].unique())

    data_dir = Path("/path/to/culture-map/data/paper_osf")
    human_df_full = load_country_map(data_dir)
    human_df_visible = human_df_full.loc[human_df_full["country"].isin(keep_countries)].copy()
    template_image_path = get_template_image_path(data_dir)
    polygons = _build_polygons(human_df_full, template_image_path)

    centroids = (
        means.groupby("variant", as_index=False)
        .agg(RC1=("RC1", "mean"), RC2=("RC2", "mean"))
        .assign(label=lambda df: df["variant"].map(LABEL_MAP))
    )

    fig, ax = plt.subplots(figsize=(14, 10))
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

    ax.scatter(human_df_visible["RC1_final"], human_df_visible["RC2_final"], s=36, color="black", zorder=3)
    for _, row in human_df_visible.iterrows():
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

    ax.scatter(
        centroids["RC1"],
        centroids["RC2"],
        s=360,
        marker="^",
        color="#d5a3ea",
        edgecolors="#4f425b",
        linewidths=1.4,
        zorder=6,
        clip_on=False,
    )
    for _, row in centroids.iterrows():
        display_label = _format_model_label(row["label"])
        nudge_x, nudge_y = MAPLE_LABEL_NUDGES.get(
            display_label,
            MODEL_LABEL_NUDGES.get(display_label, MODEL_LABEL_NUDGES.get(row["label"], (0.08, 0.08))),
        )
        ax.text(
            float(row["RC1"]) + nudge_x,
            float(row["RC2"]) + nudge_y,
            display_label,
            fontsize=MODEL_LABEL_FONT_SIZES.get(display_label, MODEL_LABEL_FONT_SIZES.get(row["label"], 18)),
            color="black",
            bbox=_model_label_bbox(),
            zorder=7,
            clip_on=False,
        )

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
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = OUTPUT_DIR / f"wvs_maple_vendor_exact_filtered.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(out)


if __name__ == "__main__":
    main()
