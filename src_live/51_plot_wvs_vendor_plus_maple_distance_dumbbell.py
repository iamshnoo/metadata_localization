#!/usr/bin/env python3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path("/path/to/metacul")
PLOT_DIR = REPO_ROOT / "results" / "plots" / "plot8"
LATEX_DIR = REPO_ROOT / "latex" / "figs" / "appendix"
DELTA_CSV = PLOT_DIR / "_wvs_vendor_distance_delta_nometa_minus_meta.csv"
PNG_PATH = PLOT_DIR / "wvs_vendor_plus_maple_distance_dumbbell_nometa_vs_meta.png"
PDF_PATH = PLOT_DIR / "wvs_vendor_plus_maple_distance_dumbbell_nometa_vs_meta.pdf"
LATEX_PDF_PATH = LATEX_DIR / "21_wvs_vendor_plus_maple_distance_dumbbell_nometa_vs_meta.pdf"

NO_META_COLOR = "#94a3b8"
WITH_META_COLOR = "#0f766e"
LINK_COLOR = "#cbd5e1"
DELTA_TEXT_COLOR = "#166534"


def _load_delta_table() -> pd.DataFrame:
    frame = pd.read_csv(DELTA_CSV).copy()
    frame = frame.sort_values(["distance_with_meta", "label"], ascending=[True, True]).reset_index(drop=True)
    frame["rank"] = frame.index + 1
    return frame


def _draw_dumbbell(frame: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.8, 8.8))
    y_positions = list(range(len(frame)))
    y_labels = frame["label"].tolist()

    for y_position, (_, row) in zip(y_positions, frame.iterrows(), strict=False):
        no_meta = float(row["distance_no_meta"])
        with_meta = float(row["distance_with_meta"])
        delta = float(row["delta_distance"])
        label = str(row["label"])

        ax.plot([with_meta, no_meta], [y_position, y_position], color=LINK_COLOR, linewidth=2.3, zorder=1)
        ax.scatter([no_meta], [y_position], s=64, facecolors="white", edgecolors=NO_META_COLOR, linewidths=1.7, zorder=3)
        ax.scatter([with_meta], [y_position], s=66, color=WITH_META_COLOR, edgecolors="white", linewidths=0.7, zorder=4)
        ax.text(
            no_meta + 0.08,
            y_position,
            f"+{delta:.3f}",
            ha="left",
            va="center",
            fontsize=9,
            family="monospace",
            color=DELTA_TEXT_COLOR,
        )

        if label.startswith("MAPLE "):
            ax.text(
                with_meta + 0.10,
                y_position - 0.18,
                "(T+, I+)",
                ha="left",
                va="center",
                fontsize=8.5,
                family="monospace",
                color=WITH_META_COLOR,
            )
            ax.text(
                no_meta - 0.10,
                y_position - 0.18,
                "(T-, I-)",
                ha="right",
                va="center",
                fontsize=8.5,
                family="monospace",
                color="#64748b",
            )

    max_distance = float(frame[["distance_no_meta", "distance_with_meta"]].to_numpy().max())
    ax.set_xlim(-0.08, max_distance + 0.95)
    ax.set_ylim(-0.7, len(frame) - 0.3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Distance From Country Centroid", fontsize=12, fontweight="bold")

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=NO_META_COLOR, markeredgewidth=1.7, markersize=7.5, label="No meta"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=WITH_META_COLOR, markeredgecolor="white", markeredgewidth=0.7, markersize=7.5, label="With meta"),
        Line2D([0], [0], color=LINK_COLOR, linewidth=2.3, label="Improvement span"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        fontsize=9,
        handletextpad=0.6,
    )

    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#4a5568")
    ax.spines["bottom"].set_color("#4a5568")

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(PDF_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(LATEX_PDF_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    frame = _load_delta_table()
    _draw_dumbbell(frame)
    print(PNG_PATH)
    print(PDF_PATH)
    print(LATEX_PDF_PATH)


if __name__ == "__main__":
    main()
