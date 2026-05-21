#!/usr/bin/env python3
import os
from pathlib import Path
import sys
from typing import Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path("/path/to/metacul")
WVS_ROOT = Path(os.environ.get("WVS_ROOT", str(REPO_ROOT / "results" / "culture_map_wvs_country_eval")))
FIGURE_PDF = Path(os.environ.get("FIGURE_PDF", str(REPO_ROOT / "latex" / "figs" / "appendix" / "20_wvs_country_projection_map.pdf")))
FIGURE_PNG = Path(os.environ.get("FIGURE_PNG", str(REPO_ROOT / "results" / "plots" / "plot8" / "20_wvs_country_projection_map.png")))
SUMMARY_CSV = WVS_ROOT / "all_variant_overall_summary.csv"
DISTANCE_CSV = WVS_ROOT / "all_variant_country_distance_summary.csv"
MEAN_CSV = WVS_ROOT / "all_variant_country_mean_projection.csv"
PAPER_TABLE_CSV = Path(os.environ.get("PAPER_TABLE_CSV", str(WVS_ROOT / "wvs_country_projection_summary_for_paper.csv")))

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


SELECTED_VARIANTS = [
    "maple_1b_tminus_eminus",
    "maple_1b_tplus_eplus",
    "maple_3b_tminus_eminus",
    "maple_3b_tplus_eplus",
]

CONTINENT_COLORS = {
    "Africa": "#d94841",
    "America": "#2b8cbe",
    "Asia": "#31a354",
    "Europe": "#756bb1",
}

LABEL_MAP = {
    "maple_1b_tminus_eminus": "MAPLE 1B (T-, I-)",
    "maple_1b_tplus_eplus": "MAPLE 1B (T+, I+)",
    "maple_3b_tminus_eminus": "MAPLE 3B (T-, I-)",
    "maple_3b_tplus_eplus": "MAPLE 3B (T+, I+)",
}


def _load_variant_frames(pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(WVS_ROOT.glob(pattern)):
        if path.name.startswith("all_variant_"):
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No files matched {pattern} in {WVS_ROOT}")
    return pd.concat(frames, ignore_index=True)


def rebuild_combined_summaries() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = _load_variant_frames("maple_*_overall_summary.csv").sort_values(["size", "train_metadata", "eval_metadata", "variant"]).reset_index(drop=True)
    distance = _load_variant_frames("maple_*_country_distance_summary.csv").sort_values(["variant", "country"]).reset_index(drop=True)
    means = _load_variant_frames("maple_*_country_mean_projection.csv").sort_values(["variant", "country"]).reset_index(drop=True)
    overall.to_csv(SUMMARY_CSV, index=False)
    distance.to_csv(DISTANCE_CSV, index=False)
    means.to_csv(MEAN_CSV, index=False)
    return overall, distance, means


def build_paper_summary(overall: pd.DataFrame, means: pd.DataFrame) -> pd.DataFrame:
    unique_counts = (
        means.groupby("variant")[["RC1", "RC2"]]
        .apply(lambda frame: frame.drop_duplicates().shape[0])
        .rename("unique_projection_points")
        .reset_index()
    )
    summary = overall.merge(unique_counts, on="variant", how="left")
    summary["top1_rate_pct"] = 100.0 * summary["top1_rate"]
    summary["top3_rate_pct"] = 100.0 * summary["top3_rate"]
    summary = summary[
        [
            "variant",
            "label",
            "size",
            "train_metadata",
            "eval_metadata",
            "mean_target_distance",
            "median_target_distance",
            "mean_target_rank",
            "top1_rate_pct",
            "top3_rate_pct",
            "unique_projection_points",
        ]
    ].copy()
    summary.to_csv(PAPER_TABLE_CSV, index=False)
    return summary


def _draw_background(ax, polygon_df: pd.DataFrame, label_df: pd.DataFrame) -> None:
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
            linewidth=1.2,
            alpha=0.88,
            zorder=0,
        )

    ax.scatter(label_df["RC1_final"], label_df["RC2_final"], s=18, color="black", zorder=3)
    for _, row in label_df.iterrows():
        ax.text(
            row["RC1_final"] + 0.03,
            row["RC2_final"] + 0.03,
            row["country"],
            fontsize=6.2,
            color="black",
            zorder=4,
        )

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
            fontsize=10,
            fontstyle="italic",
            fontweight="semibold",
            ha="center",
            va="center",
            color="black",
            alpha=0.8,
            zorder=2,
        )

    ax.set_xlim(-2.5, 3.5)
    ax.set_ylim(-2.7, 2.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_map(summary: pd.DataFrame, distance: pd.DataFrame, means: pd.DataFrame) -> None:
    data_dir = Path("/path/to/culture-map/data/paper_osf")
    human_df_full = load_country_map(data_dir)
    human_df = human_df_full[human_df_full["country"].isin(sorted(distance["country"].unique()))].copy()
    target_lookup = human_df.set_index("country")[["RC1_final", "RC2_final"]]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))

    for ax, variant in zip(axes.flatten(), SELECTED_VARIANTS):
        variant_summary = summary.loc[summary["variant"] == variant].iloc[0]
        variant_means = means.loc[means["variant"] == variant].copy()
        variant_distance = distance.loc[distance["variant"] == variant].copy()
        variant_means = variant_means.merge(target_lookup, left_on="country", right_index=True, how="left")
        variant_means = variant_means.merge(
            variant_distance[["country", "target_distance"]],
            on="country",
            how="left",
        )

        _draw_background(ax, human_df_full, human_df)
        for _, row in variant_means.iterrows():
            color = CONTINENT_COLORS[row["continent"]]
            ax.plot(
                [row["RC1_final"], row["RC1"]],
                [row["RC2_final"], row["RC2"]],
                color=color,
                alpha=0.45,
                linewidth=1.0,
                zorder=5,
            )
        ax.scatter(
            variant_means["RC1"],
            variant_means["RC2"],
            s=42,
            c=variant_means["continent"].map(CONTINENT_COLORS),
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
        )
        ax.set_title(
            "{}\nmean dist={:.2f}, top-1={:.1f}%, unique points={}".format(
                LABEL_MAP[variant],
                float(variant_summary["mean_target_distance"]),
                float(variant_summary["top1_rate_pct"]),
                int(variant_summary["unique_projection_points"]),
            ),
            fontsize=10.5,
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label=continent, markerfacecolor=color, markeredgecolor="black", markersize=7)
        for continent, color in CONTINENT_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        fontsize=9.5,
        title="Target-country continent",
        title_fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    overall, distance, means = rebuild_combined_summaries()
    summary = build_paper_summary(overall, means)
    render_map(summary, distance, means)


if __name__ == "__main__":
    main()
