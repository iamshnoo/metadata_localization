#!/usr/bin/env python3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path("/path/to/metacul")
CULTURE_MAP_ROOT = Path("/path/to/culture-map")
sys.path.insert(0, str(CULTURE_MAP_ROOT / "src"))

from culture_map.paper_assets import load_country_map  # noqa: E402


PLOT_DIR = REPO_ROOT / "results" / "plots" / "plot8"
LATEX_DIR = REPO_ROOT / "latex" / "figs" / "appendix"
POINTS_CSV = PLOT_DIR / "_wvs_vendor_mixed_metadata_points.csv"
RANKING_CSV = PLOT_DIR / "_wvs_vendor_mixed_metadata_distance_from_centroid.csv"
PNG_PATH = PLOT_DIR / "wvs_vendor_plus_maple_official_axes_distance_numberline.png"
PDF_PATH = PLOT_DIR / "wvs_vendor_plus_maple_official_axes_distance_numberline.pdf"
LATEX_PDF_PATH = LATEX_DIR / "21_wvs_vendor_plus_maple_official_axes_distance_numberline.pdf"

COLOR_BY_FAMILY = {
    "OpenAI": "#2b6cb0",
    "Anthropic": "#dd6b20",
    "Google": "#2f855a",
    "Meta": "#805ad5",
    "Qwen": "#718096",
    "DeepSeek": "#0f766e",
    "Moonshot": "#b7791f",
    "Zhipu": "#9f1239",
    "MAPLE": "#c05621",
    "Other": "#4a5568",
}


def _family_for_row(label: str, model: str) -> str:
    text = "{} {}".format(label, model).lower()
    if "maple" in text:
        return "MAPLE"
    if str(label).startswith("GPT") or str(model).startswith("gpt"):
        return "OpenAI"
    if "claude" in text:
        return "Anthropic"
    if "gemini" in text or "gemma" in text:
        return "Google"
    if "llama" in text:
        return "Meta"
    if "qwen" in text:
        return "Qwen"
    if "deepseek" in text:
        return "DeepSeek"
    if "kimi" in text:
        return "Moonshot"
    if "glm" in text:
        return "Zhipu"
    return "Other"


def _load_ranked_points() -> tuple[pd.DataFrame, float, float]:
    points = pd.read_csv(POINTS_CSV)
    points = points.loc[points["label"].notna()].copy()

    human_df = load_country_map(CULTURE_MAP_ROOT / "data")
    centroid_rc1 = float(human_df["RC1_final"].mean())
    centroid_rc2 = float(human_df["RC2_final"].mean())

    points["distance_from_country_centroid"] = (
        (points["RC1"] - centroid_rc1) ** 2 + (points["RC2"] - centroid_rc2) ** 2
    ) ** 0.5
    points["family"] = [
        _family_for_row(str(label), str(model))
        for label, model in zip(points["label"], points["model"].fillna(""), strict=False)
    ]
    points = points.sort_values("distance_from_country_centroid").reset_index(drop=True)
    points["rank"] = points.index + 1
    return points, centroid_rc1, centroid_rc2


def _assign_label_levels(points: pd.DataFrame) -> list[float]:
    levels = [0.18, -0.18, 0.34, -0.34, 0.50, -0.50]
    min_gap = 0.18
    last_x_by_level = {level: -1e9 for level in levels}
    assigned = []

    for distance in points["distance_from_country_centroid"]:
        valid_levels = [level for level in levels if distance - last_x_by_level[level] >= min_gap]
        if valid_levels:
            level = min(valid_levels, key=lambda value: (abs(value), last_x_by_level[value]))
        else:
            level = min(levels, key=lambda value: last_x_by_level[value])
        assigned.append(level)
        last_x_by_level[level] = float(distance)
    return assigned


def _draw_numberline(points: pd.DataFrame, centroid_rc1: float, centroid_rc2: float) -> None:
    del centroid_rc1, centroid_rc2

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    points.to_csv(RANKING_CSV, index=False)

    label_levels = _assign_label_levels(points)
    max_distance = float(points["distance_from_country_centroid"].max())

    fig = plt.figure(figsize=(13.6, 7.4))
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.5, 1.75], hspace=0.03)
    ax = fig.add_subplot(grid[0])
    list_ax = fig.add_subplot(grid[1])

    ax.axhline(0.0, color="#1a202c", linewidth=1.8, zorder=1)
    ax.scatter([0.0], [0.0], marker="D", s=76, color="black", zorder=4)

    seen_families = set()
    for (_, row), level in zip(points.iterrows(), label_levels, strict=False):
        family = row["family"]
        color = COLOR_BY_FAMILY.get(family, COLOR_BY_FAMILY["Other"])
        distance = float(row["distance_from_country_centroid"])
        rank = int(row["rank"])

        ax.plot([distance, distance], [0.0, level * 0.84], color=color, linewidth=1.1, alpha=0.9, zorder=2)
        scatter_kwargs = {}
        if family not in seen_families:
            scatter_kwargs["label"] = family
            seen_families.add(family)
        ax.scatter([distance], [0.0], s=66, color=color, edgecolors="white", linewidths=0.8, zorder=3, **scatter_kwargs)
        ax.text(
            distance,
            level,
            str(rank),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "circle,pad=0.16", "facecolor": "white", "edgecolor": color, "linewidth": 1.0},
            zorder=5,
        )

    ax.set_xlim(-0.12, max_distance + 0.18)
    ax.set_ylim(-0.68, 0.68)
    ax.set_yticks([])
    ax.text(
        0.5,
        0.055,
        "Distance From Country Centroid",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="#1a202c",
    )
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#4a5568")
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        frameon=False,
        ncol=2,
        fontsize=9,
        handletextpad=0.4,
        columnspacing=0.8,
        borderaxespad=0.2,
    )
    list_ax.axis("off")

    lines = [
        "{rank:>2}. {label:<24} {distance:>5.3f}".format(
            rank=int(row["rank"]),
            label=str(row["label"])[:24],
            distance=float(row["distance_from_country_centroid"]),
        )
        for _, row in points.iterrows()
    ]
    split_at = (len(lines) + 1) // 2
    columns = [lines[:split_at], lines[split_at:]]
    x_positions = [0.03, 0.52]
    for x_position, column_lines in zip(x_positions, columns, strict=False):
        for index, line in enumerate(column_lines):
            y_position = 0.93 - index * 0.088
            list_ax.text(
                x_position,
                y_position,
                line,
                transform=list_ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                family="monospace",
                color="#1a202c",
            )

    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(PDF_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(LATEX_PDF_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    points, centroid_rc1, centroid_rc2 = _load_ranked_points()
    _draw_numberline(points, centroid_rc1, centroid_rc2)
    print(RANKING_CSV)
    print(PNG_PATH)
    print(PDF_PATH)
    print(LATEX_PDF_PATH)


if __name__ == "__main__":
    main()
