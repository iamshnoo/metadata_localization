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

from culture_map.local_country_runner import METACUL_COUNTRY_SPECS  # noqa: E402
from culture_map.paper_assets import load_country_map  # noqa: E402


WVS_ROOT = REPO_ROOT / "results" / "culture_map_wvs_all_human_typical_mixed"
PLOT_DIR = REPO_ROOT / "results" / "plots" / "plot8"
LATEX_DIR = REPO_ROOT / "latex" / "figs" / "appendix"
MEAN_CSV = WVS_ROOT / "all_variant_country_mean_projection.csv"
PAIRS_CSV = PLOT_DIR / "_wvs_maple_train15_country_vs_actual_pairs.csv"
PNG_PATH = PLOT_DIR / "wvs_maple_train15_country_vs_actual_official_axes.png"
PDF_PATH = PLOT_DIR / "wvs_maple_train15_country_vs_actual_official_axes.pdf"
LATEX_PDF_PATH = LATEX_DIR / "23_wvs_maple_train15_country_vs_actual_official_axes.pdf"

VARIANT = "maple_3b_tplus_eplus"
PANEL_LABEL = "MAPLE 3B (T+, I+)"

MAPLE_COLOR = "#7c3aed"
MAPLE_EDGE = "#4c1d95"
ACTUAL_COLOR = "#111827"
LINK_COLOR = "#94a3b8"

XMIN = -2.5
XMAX = 3.5
YMIN = -2.5
YMAX = 2.0

TRAIN_ORDER = list(METACUL_COUNTRY_SPECS.keys())
TRAIN_COUNTRY_TO_CODE = {spec["country"]: code.upper() for code, spec in METACUL_COUNTRY_SPECS.items()}
LABEL_OFFSETS = {
    "BD": (-0.18, -0.05, "right"),
    "CA": (0.14, -0.02, "left"),
    "GB": (0.16, 0.00, "left"),
    "GH": (-0.14, -0.10, "right"),
    "HK": (0.14, 0.07, "left"),
    "IE": (0.12, -0.08, "left"),
    "IN": (-0.10, 0.05, "right"),
    "KE": (-0.12, -0.03, "right"),
    "MY": (-0.12, 0.02, "right"),
    "NG": (-0.16, 0.02, "right"),
    "PH": (0.06, -0.08, "left"),
    "PK": (-0.12, 0.06, "right"),
    "SG": (0.10, 0.03, "left"),
    "US": (0.14, -0.02, "left"),
    "ZA": (0.12, -0.02, "left"),
}
COUNT_LABEL_OFFSETS = {
    (0.1209231911098346, 0.835050963851985): (0.16, 0.12),
    (0.3017213180182547, 0.5023623825273424): (0.14, 0.10),
    (0.0596060518369353, 1.1592220385819418): (0.10, 0.12),
}


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
    train_countries = [spec["country"] for spec in METACUL_COUNTRY_SPECS.values()]
    pairs = pairs.loc[pairs["country"].isin(train_countries)].copy()
    pairs["country_code_short"] = pairs["country"].map(TRAIN_COUNTRY_TO_CODE)
    pairs["train_order"] = pairs["country_code_short"].map({code.upper(): idx for idx, code in enumerate(TRAIN_ORDER)})
    pairs["distance_to_actual"] = (
        (pairs["RC1"] - pairs["RC1_final"]) ** 2 + (pairs["RC2"] - pairs["RC2_final"]) ** 2
    ) ** 0.5
    pairs = pairs.sort_values(["train_order", "country"]).reset_index(drop=True)
    pairs.to_csv(PAIRS_CSV, index=False)
    return pairs


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


def _label_position(row: pd.Series) -> tuple[float, float, str]:
    code = str(row["country_code_short"])
    dx, dy, ha = LABEL_OFFSETS.get(code, (0.10, 0.10, "left"))
    return float(row["RC1_final"]) + dx, float(row["RC2_final"]) + dy, ha


def _count_label_position(rc1: float, rc2: float) -> tuple[float, float]:
    dx, dy = COUNT_LABEL_OFFSETS.get((rc1, rc2), (0.14, 0.10))
    return rc1 + dx, rc2 + dy


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    frame = _load_pairs()
    triangle_counts = (
        frame.groupby(["RC1", "RC2"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "RC1", "RC2"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.22)
    _format_axes(ax)

    for _, row in frame.iterrows():
        ax.plot(
            [row["RC1_final"], row["RC1"]],
            [row["RC2_final"], row["RC2"]],
            color=LINK_COLOR,
            linewidth=1.55,
            alpha=0.82,
            zorder=2,
        )

    ax.scatter(
        frame["RC1_final"],
        frame["RC2_final"],
        s=46,
        color=ACTUAL_COLOR,
        alpha=0.96,
        zorder=3,
    )
    ax.scatter(
        triangle_counts["RC1"],
        triangle_counts["RC2"],
        s=[150 + 22 * (count - 1) for count in triangle_counts["count"]],
        marker="^",
        color=MAPLE_COLOR,
        edgecolors=MAPLE_EDGE,
        linewidths=1.0,
        alpha=0.92,
        zorder=4,
    )

    for _, row in frame.iterrows():
        label_x, label_y, ha = _label_position(row)
        ax.text(
            label_x,
            label_y,
            str(row["country_code_short"]),
            ha=ha,
            va="center",
            fontsize=9.2,
            color=ACTUAL_COLOR,
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.16"},
            zorder=5,
        )

    for _, row in triangle_counts.iterrows():
        label_x, label_y = _count_label_position(float(row["RC1"]), float(row["RC2"]))
        ax.text(
            label_x,
            label_y,
            f"x{int(row['count'])}",
            ha="left",
            va="center",
            fontsize=9.6,
            fontweight="bold",
            color=MAPLE_EDGE,
            bbox={"facecolor": "white", "edgecolor": MAPLE_EDGE, "boxstyle": "round,pad=0.18"},
            zorder=5.1,
        )

    ax.set_title(
        f"{PANEL_LABEL}: 15 Training Countries Only",
        fontsize=13,
        pad=10,
    )

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=ACTUAL_COLOR, markeredgecolor=ACTUAL_COLOR, markersize=6.5, label="Actual country"),
        Line2D([0], [0], marker="^", linestyle="", markerfacecolor=MAPLE_COLOR, markeredgecolor=MAPLE_EDGE, markersize=8.0, label="Unique MAPLE point"),
        Line2D([0], [0], color=LINK_COLOR, linewidth=1.6, label="Distance"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=10,
        handletextpad=0.5,
    )

    code_rows = [
        [("BD", "Bangladesh"), ("CA", "Canada"), ("GB", "Great Britain"), ("GH", "Ghana"), ("HK", "Hong Kong SAR")],
        [("IE", "Ireland"), ("IN", "India"), ("KE", "Kenya"), ("MY", "Malaysia"), ("NG", "Nigeria")],
        [("PH", "Philippines"), ("PK", "Pakistan"), ("SG", "Singapore"), ("US", "United States"), ("ZA", "South Africa")],
    ]
    code_lines = ["   ".join(f"{code} {name}" for code, name in row) for row in code_rows]
    fig.text(
        0.5,
        0.035,
        "Country codes\n" + "\n".join(code_lines),
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#334155",
        family="DejaVu Sans Mono",
    )

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
