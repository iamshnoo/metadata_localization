#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


SIG_CSV = Path("/path/to/metacul/results/significance/plot4.csv")
OUT_DIR = Path("/path/to/metacul/slides")

TRAIN_CONTINENT = "africa"
TRAIN_LABEL = "Africa"
TEST_CONTINENTS = ["africa", "america", "asia", "europe"]
TEST_LABELS = ["Africa", "America", "Asia", "Europe"]
CARD_COLORS = {
    "Africa": "#8FD3B6",
    "America": "#F4B183",
    "Asia": "#9CC9F5",
    "Europe": "#CDB7F6",
}
STAGES = [
    ("cross_continent_africa_tplus_iplus_overlay_step1_africa_only", [0]),
    ("cross_continent_africa_tplus_iplus_overlay_step2_add_america", [0, 1]),
    ("cross_continent_africa_tplus_iplus_overlay_step3_add_asia", [0, 1, 2]),
    ("cross_continent_africa_tplus_iplus_overlay_step4_add_europe", [0, 1, 2, 3]),
]


def _hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _mix(color_a, color_b, alpha):
    a = _hex_to_rgb(color_a) if isinstance(color_a, str) else color_a
    b = _hex_to_rgb(color_b) if isinstance(color_b, str) else color_b
    return tuple((1 - alpha) * x + alpha * y for x, y in zip(a, b))


def _text_color(rgb):
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "#1f1f1f" if luminance > 0.62 else "white"


def load_africa_row():
    values = {}
    with SIG_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["plot"] != "plot4" or row["size"].lower() != "1b":
                continue
            if row["train_continent"] != TRAIN_CONTINENT or row["test_meta"] != "with_metadata":
                continue
            values[row["test_continent"]] = float(row["mean_ppl_b"])
    return [values[test_continent] for test_continent in TEST_CONTINENTS]


def _draw_round_box(ax, x, y, w, h, facecolor, edgecolor, linewidth=1.8, alpha=1.0, dashed=False, zorder=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=(0, (3, 3)) if dashed else "solid",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _draw_shadow(ax, x, y, w, h, alpha=0.10, zorder=1):
    shadow = FancyBboxPatch(
        (x + 0.03, y - 0.03),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="#000000",
        edgecolor="none",
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(shadow)


def render_stage(values, out_stem, visible_cols):
    fig = plt.figure(figsize=(11.8, 2.65), facecolor="#ffffff")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0.40, 11.86)
    ax.set_ylim(0.52, 2.44)
    ax.axis("off")

    train_x = 0.62
    card_y = 0.70
    card_w = 1.80
    card_h = 1.58
    card_gap = 0.24
    start_x = 3.58

    _draw_shadow(ax, train_x, card_y, 1.72, card_h, alpha=0.08)
    _draw_round_box(ax, train_x, card_y, 1.72, card_h, "#eef8f1", "#7eb594", linewidth=2.2, zorder=2)
    _draw_round_box(ax, train_x + 0.18, card_y + 1.15, 0.76, 0.26, "#d7eee0", "#78ab8f", linewidth=1.0, zorder=3)
    ax.text(train_x + 0.56, card_y + 1.28, "Train", ha="center", va="center", fontsize=10.2, color="#486156", zorder=4)
    ax.text(train_x + 0.86, card_y + 0.70, TRAIN_LABEL, ha="center", va="center", fontsize=23.0, fontweight="bold", color="#234536", zorder=4)

    arrow = FancyArrowPatch(
        (2.50, 1.42),
        (3.34, 1.42),
        arrowstyle="simple",
        mutation_scale=18,
        linewidth=0,
        color="#9da6b0",
        alpha=0.95,
        zorder=2,
    )
    ax.add_patch(arrow)

    active_idx = visible_cols[-1]

    for idx, label in enumerate(TEST_LABELS):
        x = start_x + idx * (card_w + card_gap)
        base = CARD_COLORS[label]
        visible = idx in visible_cols
        active = idx == active_idx

        if visible:
            fill = _mix(base, "#ffffff", 0.08 if active else 0.20)
            edge = _mix(base, "#000000", 0.36 if active else 0.25)
            _draw_shadow(ax, x, card_y, card_w, card_h, alpha=0.12 if active else 0.06)
            _draw_round_box(ax, x, card_y, card_w, card_h, fill, edge, linewidth=3.0 if active else 1.9, zorder=2)
            chip_fill = _mix(base, "#ffffff", 0.22)
            chip_edge = _mix(base, "#000000", 0.20)
            _draw_round_box(ax, x + 0.18, card_y + 1.14, 0.72, 0.26, chip_fill, chip_edge, linewidth=1.0, zorder=3)
            ax.text(x + 0.54, card_y + 1.27, "Test", ha="center", va="center", fontsize=10, color="#42515a", zorder=4)
            ax.text(x + card_w / 2, card_y + 0.88, label, ha="center", va="center", fontsize=16.5, fontweight="bold", color="#24313a", zorder=4)
            value_rgb = _mix(base, "#000000", 0.10 if active else 0.16)
            value_fill = _mix(base, "#ffffff", 0.03 if active else 0.10)
            value_edge = _mix(base, "#000000", 0.30)
            _draw_round_box(ax, x + 0.18, card_y + 0.18, card_w - 0.36, 0.54, value_fill, value_edge, linewidth=1.0, zorder=3)
            ax.text(x + card_w / 2, card_y + 0.45, f"{values[idx]:.1f}", ha="center", va="center", fontsize=23.5, fontweight="bold", color=_text_color(value_rgb), zorder=4)
        else:
            _draw_round_box(ax, x, card_y, card_w, card_h, "#ffffff", "#d3d9df", linewidth=1.5, dashed=True, zorder=2)
            ax.text(x + card_w / 2, card_y + 0.88, label, ha="center", va="center", fontsize=16.5, fontweight="bold", color="#b2bac3", zorder=3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{out_stem}.png"
    pdf_path = OUT_DIR / f"{out_stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.0, facecolor="#ffffff", edgecolor="#ffffff")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.0, facecolor="#ffffff", edgecolor="#ffffff")
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


def main():
    values = load_africa_row()
    for out_stem, visible_cols in STAGES:
        render_stage(values, out_stem, visible_cols)


if __name__ == "__main__":
    main()
