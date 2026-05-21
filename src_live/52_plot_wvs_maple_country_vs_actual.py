#!/usr/bin/env python3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path("/path/to/metacul")
CULTURE_MAP_ROOT = Path("/path/to/culture-map")
sys.path.insert(0, str(CULTURE_MAP_ROOT / "src"))

from culture_map.paper_assets import load_country_map  # noqa: E402


WVS_ROOT = REPO_ROOT / "results" / "culture_map_wvs_all_human_typical_mixed"
PLOT_DIR = REPO_ROOT / "results" / "plots" / "plot8"
LATEX_DIR = REPO_ROOT / "latex" / "figs" / "appendix"
MEAN_CSV = WVS_ROOT / "all_variant_country_mean_projection.csv"
PAIRS_CSV = PLOT_DIR / "_wvs_maple_country_vs_actual_pairs.csv"
PNG_PATH = PLOT_DIR / "wvs_maple_country_vs_actual_official_axes.png"
PDF_PATH = PLOT_DIR / "wvs_maple_country_vs_actual_official_axes.pdf"
LATEX_PDF_PATH = LATEX_DIR / "22_wvs_maple_country_vs_actual_official_axes.pdf"

VARIANT = "maple_3b_tplus_eplus"
PANEL_LABEL = "MAPLE 3B (T+, I+)"
TOP_K = 11
EMPHASIZE_COUNTRY = "South Korea"

MAPLE_COLOR = "#7c3aed"
MAPLE_EDGE = "#4c1d95"
ACTUAL_COLOR = "#111827"
LINK_COLOR = "#94a3b8"

XMIN = -2.5
XMAX = 3.5
YMIN = -2.5
YMAX = 2.0


def _load_pairs() -> pd.DataFrame:
    human_df = load_country_map(CULTURE_MAP_ROOT / "data")
    means = pd.read_csv(MEAN_CSV)
    means = means.loc[means["variant"] == VARIANT].copy()

    pairs = means.merge(
        human_df[["country", "RC1_final", "RC2_final"]],
        on="country",
        how="inner",
        suffixes=("", "_human"),
    )
    pairs["distance_to_actual"] = (
        (pairs["RC1"] - pairs["RC1_final"]) ** 2 + (pairs["RC2"] - pairs["RC2_final"]) ** 2
    ) ** 0.5
    pairs = pairs.sort_values(["distance_to_actual", "country"]).reset_index(drop=True)
    pairs.to_csv(PAIRS_CSV, index=False)
    return pairs.head(TOP_K).copy()


def _format_axes(ax) -> None:
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks([-2, -1, 0, 1, 2, 3])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.axvline(0.0, color="#94a3b8", linewidth=0.9, alpha=0.8, linestyle=(0, (3, 3)), zorder=0.2)
    ax.axhline(0.0, color="#94a3b8", linewidth=0.9, alpha=0.8, linestyle=(0, (3, 3)), zorder=0.2)
    ax.grid(color="#e2e8f0", linewidth=0.8, alpha=0.9, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(labelsize=10)
    ax.set_xlabel("Survival vs. Self-Expression Values", fontsize=12, fontweight="bold")
    ax.set_ylabel("Traditional vs. Secular Values", fontsize=12, fontweight="bold")


def _label_position(row: pd.Series, index: int) -> tuple[float, float, str]:
    del index
    custom = {
        "Argentina": (0.10, 0.12, "left"),
        "Bulgaria": (-0.08, 0.26, "right"),
        "Hungary": (0.08, 0.14, "left"),
        "Lithuania": (-0.08, 0.06, "right"),
        "Slovakia": (0.14, 0.18, "left"),
        "Greece": (-0.10, -0.12, "right"),
        "Poland": (0.14, -0.08, "left"),
        "Singapore": (0.12, -0.06, "left"),
        "Estonia": (0.14, 0.02, "left"),
        "Mongolia": (0.16, -0.02, "left"),
        "South Korea": (-0.08, 0.24, "right"),
    }
    if row["country"] in custom:
        dx, dy, ha = custom[row["country"]]
        return float(row["RC1_final"]) + dx, float(row["RC2_final"]) + dy, ha
    dx = 0.10
    dy = 0.12
    ha = "left" if dx >= 0 else "right"
    return float(row["RC1_final"]) + dx, float(row["RC2_final"]) + dy, ha


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    frame = _load_pairs()
    base_frame = frame.loc[frame["country"] != EMPHASIZE_COUNTRY].copy()
    emphasize_frame = frame.loc[frame["country"] == EMPHASIZE_COUNTRY].copy()

    fig, ax = plt.subplots(figsize=(8.4, 6.5))
    _format_axes(ax)

    for index, (_, row) in enumerate(base_frame.iterrows()):
        ax.plot(
            [row["RC1_final"], row["RC1"]],
            [row["RC2_final"], row["RC2"]],
            color=LINK_COLOR,
            linewidth=1.6,
            alpha=0.85,
            zorder=2,
        )
    for _, row in emphasize_frame.iterrows():
        ax.plot(
            [row["RC1_final"], row["RC1"]],
            [row["RC2_final"], row["RC2"]],
            color="#64748b",
            linewidth=2.4,
            alpha=0.95,
            zorder=2.8,
        )

    ax.scatter(
        base_frame["RC1_final"],
        base_frame["RC2_final"],
        s=44,
        color=ACTUAL_COLOR,
        alpha=0.95,
        zorder=3,
    )
    if not emphasize_frame.empty:
        ax.scatter(
            emphasize_frame["RC1_final"],
            emphasize_frame["RC2_final"],
            s=72,
            color=ACTUAL_COLOR,
            edgecolors="white",
            linewidths=0.9,
            alpha=0.98,
            zorder=4.6,
        )
    ax.scatter(
        frame["RC1"],
        frame["RC2"],
        s=90,
        marker="^",
        color=MAPLE_COLOR,
        edgecolors=MAPLE_EDGE,
        linewidths=0.8,
        alpha=0.92,
        zorder=4,
    )

    for index, (_, row) in enumerate(frame.iterrows()):
        label_x, label_y, ha = _label_position(row, index)
        if row["country"] == EMPHASIZE_COUNTRY:
            ax.annotate(
                f"{row['country']} ({row['distance_to_actual']:.2f})",
                xy=(float(row["RC1_final"]), float(row["RC2_final"])),
                xytext=(label_x, label_y),
                textcoords="data",
                ha=ha,
                va="center",
                fontsize=10,
                color=ACTUAL_COLOR,
                bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.18"},
                arrowprops={"arrowstyle": "-", "color": "#64748b", "linewidth": 1.0, "shrinkA": 4.0, "shrinkB": 4.0},
                zorder=5.2,
            )
        else:
            ax.text(
                label_x,
                label_y,
                f"{row['country']} ({row['distance_to_actual']:.2f})",
                ha=ha,
                va="center",
                fontsize=10,
                color=ACTUAL_COLOR,
                bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.18"},
                zorder=5,
            )

    ax.set_title(
        f"{PANEL_LABEL}: {TOP_K} Closest Countries\nBlack = actual country, purple = MAPLE point",
        fontsize=13,
        pad=10,
    )

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=ACTUAL_COLOR, markeredgecolor=ACTUAL_COLOR, markersize=6.5, label="Actual country"),
        Line2D([0], [0], marker="^", linestyle="", markerfacecolor=MAPLE_COLOR, markeredgecolor=MAPLE_EDGE, markersize=8.0, label="MAPLE country-conditioned point"),
        Line2D([0], [0], color=LINK_COLOR, linewidth=1.6, label="Distance between them"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        fontsize=10,
        handletextpad=0.5,
    )

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(PDF_PATH, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(LATEX_PDF_PATH, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(PAIRS_CSV)
    print(PNG_PATH)
    print(PDF_PATH)
    print(LATEX_PDF_PATH)


if __name__ == "__main__":
    main()
