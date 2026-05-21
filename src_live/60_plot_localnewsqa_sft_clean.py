#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = Path(
    "/path/to/metacul/results/plots/plot8/plot_8_sft_target_split_multiseed_cleaned.csv"
)
OUT_RESULTS_PDF = Path(
    "/path/to/metacul/results/plots/plot8/18_sft_localnewsqa_accuracy_apples_to_apples.pdf"
)
OUT_RESULTS_PNG = Path(
    "/path/to/metacul/results/plots/plot8/18_sft_localnewsqa_accuracy_apples_to_apples.png"
)
OUT_LATEX_PDF = Path(
    "/path/to/metacul/latex/figs/appendix/18_sft_localnewsqa_accuracy_apples_to_apples.pdf"
)

SPLITS = ["Overall", "Explicit", "Ambiguous"]
Y_POS = {"Overall": 2, "Explicit": 1, "Ambiguous": 0}
SERIES_ORDER = ["T+/I+", "T-/I-"]
SERIES_STYLE = {
    "T+/I+": {"marker": "o", "color": "#2f9e44", "label": r"$(T{+}, I{+})$"},
    "T-/I-": {"marker": "s", "color": "#8a919c", "label": r"$(T{-}, I{-})$"},
}
TEXT_Y_OFFSETS = {"T+/I+": 0.10, "T-/I-": -0.10}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MAPLE SFT LocalNewsQA target accuracy for the appendix."
    )
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    parser.add_argument("--out-results-pdf", type=Path, default=OUT_RESULTS_PDF)
    parser.add_argument("--out-results-png", type=Path, default=OUT_RESULTS_PNG)
    parser.add_argument("--out-latex-pdf", type=Path, default=OUT_LATEX_PDF)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["series_short"] = df["series"].str.extract(r"(T[+-]/I[+-])")
    df = df[df["series_short"].isin(SERIES_ORDER)].copy()
    return df


def plot_family(ax, df: pd.DataFrame, family: str, xlim: tuple[float, float]) -> None:
    fam = df[df["family"] == family]
    for split in SPLITS:
        y = Y_POS[split]
        row_vals = []
        for series in SERIES_ORDER:
            row = fam[(fam["split"] == split) & (fam["series_short"] == series)].iloc[0]
            x = row["accuracy"] * 100.0
            row_vals.append((series, x))
            style = SERIES_STYLE[series]
            ax.scatter(
                x,
                y,
                s=150,
                marker=style["marker"],
                color=style["color"],
                edgecolors="black",
                linewidths=0.9,
                zorder=3,
            )
            ax.text(
                x + 0.26,
                y + TEXT_Y_OFFSETS[series],
                f"{x:.2f}",
                va="center",
                ha="left",
                fontsize=10,
                color=style["color"],
            )
        xs = [x for _, x in row_vals]
        ax.hlines(y, min(xs), max(xs), color="#adb5bd", linewidth=1.4, zorder=1)

    ax.set_yticks([Y_POS[s] for s in SPLITS])
    ax.set_yticklabels(SPLITS, fontsize=11)
    ax.set_xlim(*xlim)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_title(f"{family} Chat", fontsize=13, weight="bold")
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.tick_params(axis="x", labelsize=10)


def main() -> int:
    args = parse_args()
    df = load_data(args.csv_path)
    x_values = df["accuracy"].astype(float) * 100.0
    x_min = max(0.0, float(x_values.min()) - 1.5)
    x_max = min(100.0, float(x_values.max()) + 2.5)
    xlim = (x_min, x_max)
    fig, axes = plt.subplots(2, 1, figsize=(9.3, 7.0), sharex=True)
    for ax, family in zip(axes, ["1B", "3B"]):
        plot_family(ax, df, family, xlim)

    handles = []
    labels = []
    for series in SERIES_ORDER:
        style = SERIES_STYLE[series]
        h = axes[0].scatter([], [], s=150, marker=style["marker"], color=style["color"], edgecolors="black", linewidths=0.9)
        handles.append(h)
        labels.append(style["label"])
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    for out in [args.out_results_pdf, args.out_results_png, args.out_latex_pdf]:
        ensure_parent(out)
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {args.out_results_pdf}")
    print(f"[ok] wrote {args.out_results_png}")
    print(f"[ok] wrote {args.out_latex_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
