#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, Rectangle


CSV_PATH = Path("/scratch/amukher6/metacul/results/plots/plot11/plot_13.csv")
OUT_DIR = Path("/scratch/amukher6/metacul/slides")


def load_rows():
    with CSV_PATH.open(newline="") as f:
        return list(csv.DictReader(f))

def interpolate_crossing(tokens, values, target):
    for x1, y1, x2, y2 in zip(tokens[:-1], values[:-1], tokens[1:], values[1:]):
        if y1 >= target and y2 <= target and y1 != y2:
            frac = (y1 - target) / (y1 - y2)
            return x1 + frac * (x2 - x1)
    return None


def add_hatched_marker(ax, x, y, shape, facecolor, size_px=24, hatch="///"):
    da = DrawingArea(size_px, size_px, 0, 0)
    if shape == "circle":
        artist = Circle(
            (size_px / 2, size_px / 2),
            size_px * 0.36,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=2.4,
            hatch=hatch,
        )
    else:
        side = size_px * 0.72
        artist = Rectangle(
            ((size_px - side) / 2, (size_px - side) / 2),
            side,
            side,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=2.4,
            hatch=hatch,
        )
    da.add_artist(artist)
    ab = AnnotationBbox(da, (x, y), frameon=False, box_alignment=(0.5, 0.5), pad=0)
    ab.set_zorder(7)
    ax.add_artist(ab)


def build_plot(include_metadata, add_guides, out_stem):
    rows = load_rows()
    size_order = ["500M", "1B", "3B"]
    bbox_props = dict(
        facecolor="#ececec",
        edgecolor="#888888",
        alpha=0.95,
        boxstyle="round,pad=0.35",
    )
    colors_map = {"T+": "#69c3a5", "T-": "#ee9a53"}
    markers_map = {"T+": "o", "T-": "s"}
    token_ticks = [8.388608, 16.777216, 33.554432, 41.94304]
    tick_labels = ["8.4", "16.8", "33.6", "41.9"]

    fig, axes = plt.subplots(1, len(size_order), figsize=(13.65, 4.35), sharey=True)

    for ax, size_label in zip(axes, size_order):
        curves = {"T+": [], "T-": []}
        for row in rows:
            if row["size"] != size_label:
                continue
            tag = row["train_tag"]
            if tag not in curves:
                continue
            curves[tag].append(
                {
                    "step": int(row["step"]),
                    "tokens_b": float(row["tokens_b"]),
                    "mean_ppl": float(row["mean_ppl"]),
                    "final_tminus_target_ppl": float(row["final_tminus_target_ppl"]),
                    "tplus_cross_tokens_b": float(row["tplus_cross_tokens_b"]),
                    "tminus_final_tokens_b": float(row["tminus_final_tokens_b"]),
                    "token_savings_frac": float(row["token_savings_frac"]),
                }
            )

        curves["T-"].sort(key=lambda r: r["step"])
        curves["T+"].sort(key=lambda r: r["step"])

        tminus_x = [r["tokens_b"] for r in curves["T-"]]
        tminus_y = [r["mean_ppl"] for r in curves["T-"]]
        ax.plot(
            tminus_x,
            tminus_y,
            color=colors_map["T-"],
            marker=markers_map["T-"],
            linestyle=(0, (5, 4)),
            linewidth=2.1,
            markersize=12.6,
            markeredgecolor="black",
            markeredgewidth=1.0,
            markerfacecolor=colors_map["T-"],
            zorder=3,
        )

        target_ppl = curves["T-"][-1]["final_tminus_target_ppl"]
        final_tokens_b = curves["T-"][-1]["tminus_final_tokens_b"]

        if include_metadata:
            tplus_x = [r["tokens_b"] for r in curves["T+"]]
            tplus_y = [r["mean_ppl"] for r in curves["T+"]]
            ax.plot(
                tplus_x,
                tplus_y,
                color=colors_map["T+"],
                marker=markers_map["T+"],
                linestyle=(0, (5, 4)),
                linewidth=2.1,
                markersize=12.6,
                markeredgecolor="black",
                markeredgewidth=1.0,
                markerfacecolor=colors_map["T+"],
                zorder=4,
            )

            cross_tokens_b = curves["T+"][0]["tplus_cross_tokens_b"]
            if add_guides and cross_tokens_b is not None:
                ax.axvspan(cross_tokens_b, final_tokens_b, color="#d9d9d9", alpha=0.14, zorder=0)
                ax.axhline(target_ppl, color="#6f6f6f", linestyle=":", linewidth=1.3, zorder=1)
                ax.axvline(cross_tokens_b, color="#7dbf9b", linestyle=":", linewidth=1.4, zorder=1)
                ax.axvline(final_tokens_b, color="#b5b5b5", linestyle=":", linewidth=1.4, zorder=1)
                add_hatched_marker(ax, cross_tokens_b, target_ppl, "circle", "#4fbf98", size_px=26, hatch="///")
                add_hatched_marker(ax, final_tokens_b, target_ppl, "square", "#f5bf8a", size_px=26, hatch="///")

                y_min = min(min(tplus_y), min(tminus_y))
                y_max = max(max(tplus_y), max(tminus_y))
                tminus_mid = [y for x, y in zip(tminus_x, tminus_y) if cross_tokens_b <= x <= final_tokens_b]
                if tminus_mid:
                    arrow_y = max(tminus_mid) + 0.28 * (y_max - y_min)
                else:
                    arrow_y = target_ppl + 0.32 * (y_max - y_min)
                savings_frac = curves["T+"][0]["token_savings_frac"]
                ax.annotate(
                    "",
                    xy=(cross_tokens_b, arrow_y),
                    xytext=(final_tokens_b, arrow_y),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#4f4f4f",
                        lw=2.8,
                        mutation_scale=24,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=6,
                )
                ax.text(
                    (cross_tokens_b + final_tokens_b) / 2,
                    arrow_y + 0.105 * (y_max - y_min),
                    f"{savings_frac * 100:.0f}% fewer tokens",
                    ha="center",
                    va="bottom",
                    fontsize=12.8,
                    fontweight="bold",
                    color="#3e3e3e",
                    bbox=dict(
                        facecolor="#fff1b8",
                        edgecolor="#c59d00",
                        boxstyle="round,pad=0.22",
                        alpha=0.98,
                    ),
                    zorder=7,
                )

        ax.text(
            0.94,
            0.94,
            size_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=13.5,
            fontweight="bold",
            bbox=bbox_props,
        )
        ax.set_xlabel("Training tokens (B)", fontsize=13.5)
        ax.set_xticks(token_ticks)
        ax.set_xticklabels(tick_labels, fontsize=11.8)
        ax.tick_params(axis="y", labelsize=11.8)
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.28)
        ax.set_xlim(7.5, 43.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.3)
        ax.spines["bottom"].set_linewidth(1.3)

    axes[0].set_ylabel("Perplexity (↓ better)", fontsize=15.5)

    legend_handles = [
        Line2D(
            [],
            [],
            color=colors_map["T-"],
            marker=markers_map["T-"],
            linestyle=(0, (5, 4)),
            linewidth=2.1,
            markersize=12.6,
            markeredgecolor="black",
            markerfacecolor=colors_map["T-"],
            label="Trained without metadata",
        ),
        Line2D(
            [],
            [],
            color=colors_map["T+"],
            marker=markers_map["T+"],
            linestyle=(0, (5, 4)),
            linewidth=2.1,
            markersize=12.6,
            markeredgecolor="black",
            markerfacecolor=colors_map["T+"],
            label="Trained with metadata",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="black",
        fontsize=11.6,
        bbox_to_anchor=(0.5, 0.992),
    )

    fig.subplots_adjust(top=0.865, bottom=0.13, left=0.055, right=0.995, wspace=0.08)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{out_stem}.png"
    pdf_path = OUT_DIR / f"{out_stem}.pdf"
    fig.savefig(png_path, dpi=220, pad_inches=0.0)
    fig.savefig(pdf_path, dpi=600, pad_inches=0.0)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


def main():
    build_plot(include_metadata=False, add_guides=False, out_stem="token_efficiency_overlay_step1_tminus_only")
    build_plot(include_metadata=True, add_guides=True, out_stem="token_efficiency_overlay_step2_add_tplus_and_guides")


if __name__ == "__main__":
    main()
