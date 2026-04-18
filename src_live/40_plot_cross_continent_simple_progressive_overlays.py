#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


SIG_CSV = Path("/scratch/amukher6/metacul/results/significance/plot4.csv")
OUT_DIR = Path("/scratch/amukher6/metacul/slides")

CONTINENTS = ["africa", "america", "asia", "europe"]
LABELS = ["Africa", "America", "Asia", "Europe"]
PANEL_STYLES = {
    "Local(T+,I+)": {"base": "#a3cea8", "title_fc": "#a3cea8", "title_ec": "black"},
    "Local(T+,I-)": {"base": "#eca7a4", "title_fc": "#eca7a4", "title_ec": "black"},
    "Local(T-,I+)": {"base": "#fad9b7", "title_fc": "#fad9b7", "title_ec": "black"},
    "Local(T-,I-)": {"base": "#d9d9d9", "title_fc": "#d9d9d9", "title_ec": "black"},
}
STAGES = [
    ("cross_continent_simple_overlay_step1_diagonal_only", {0, 1, 2, 3}, set()),
    ("cross_continent_simple_overlay_step2_add_africa_row", {0, 1, 2, 3}, {0}),
    ("cross_continent_simple_overlay_step3_add_america_row", {0, 1, 2, 3}, {0, 1}),
    ("cross_continent_simple_overlay_step4_add_asia_row", {0, 1, 2, 3}, {0, 1, 2}),
    ("cross_continent_simple_overlay_step5_add_europe_row", {0, 1, 2, 3}, {0, 1, 2, 3}),
]


def _mix(color, target, alpha):
    base_rgb = mcolors.to_rgb(color)
    target_rgb = mcolors.to_rgb(target)
    return tuple((1 - alpha) * b + alpha * t for b, t in zip(base_rgb, target_rgb))


def _panel_cmap(base):
    dark = _mix(base, "black", 0.28)
    mid = mcolors.to_rgb(base)
    light = _mix(base, "white", 0.62)
    return LinearSegmentedColormap.from_list("", [dark, mid, light])


def _text_color(rgb):
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "white" if luminance < 0.52 else "black"


def load_rows():
    rows = []
    with SIG_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["plot"] != "plot4" or row["size"].lower() != "1b":
                continue
            rows.append(row)
    return rows


def build_matrix(rows, test_meta, use_with_model):
    value_col = "mean_ppl_b" if use_with_model else "mean_ppl_a"
    matrix = []
    for train in CONTINENTS:
        matrix_row = []
        for test in CONTINENTS:
            match = next(
                (
                    row for row in rows
                    if row["train_continent"] == train
                    and row["test_continent"] == test
                    and row["test_meta"] == test_meta
                ),
                None,
            )
            matrix_row.append(float(match[value_col]) if match else float("nan"))
        matrix.append(matrix_row)
    return np.array(matrix, dtype=float)


def build_mask(diagonal_indices, visible_rows):
    mask = np.zeros((len(LABELS), len(LABELS)), dtype=bool)
    for idx in diagonal_indices:
        mask[idx, idx] = True
    for row_idx in visible_rows:
        mask[row_idx, :] = True
    return mask


def render_stage(panels, vmin, vmax, out_stem, diagonal_indices, visible_rows):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mask = build_mask(diagonal_indices, visible_rows)

    fig, axes = plt.subplots(2, 2, figsize=(7.08, 6.16))
    title_artists = []

    for idx, (title, panel) in enumerate(panels):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        style = PANEL_STYLES[title]
        cmap = _panel_cmap(style["base"])
        cmap.set_bad(color="white")

        masked_panel = np.ma.array(panel, mask=~mask)
        ax.imshow(masked_panel, cmap=cmap, norm=norm, aspect="equal")

        title_artist = ax.text(
            0.5,
            1.01,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12.3,
            fontweight="bold",
            clip_on=False,
            zorder=5,
            bbox=dict(
                facecolor=style["title_fc"],
                edgecolor=style["title_ec"],
                linewidth=0.8,
                alpha=0.95,
                boxstyle="round,pad=0.24",
            ),
        )
        title_artists.append(title_artist)

        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=28, ha="right", fontsize=10.0)
        ax.set_yticklabels(LABELS, fontsize=10.3)

        ax.set_xlabel("Test Region" if r == 1 else "", fontsize=10.8)
        ax.set_ylabel("Train Region" if c == 0 else "", fontsize=10.8)

        ax.set_xticks([x - 0.5 for x in range(1, len(LABELS))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(LABELS))], minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(length=0, axis="both", pad=4)

        for i in range(panel.shape[0]):
            for j in range(panel.shape[1]):
                if not mask[i, j]:
                    continue
                val = panel[i, j]
                rgb = cmap(norm(val))[:3]
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=10.3,
                    fontweight="bold" if i == j else "normal",
                    color=_text_color(rgb),
                )

    fig.subplots_adjust(left=0.085, right=0.992, bottom=0.115, top=0.94, wspace=0.12, hspace=0.30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{out_stem}.png"
    pdf_path = OUT_DIR / f"{out_stem}.pdf"
    fig.savefig(png_path, dpi=220, pad_inches=0.0)
    fig.savefig(pdf_path, dpi=600, pad_inches=0.0)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


def main():
    rows = load_rows()
    panels = [
        ("Local(T+,I+)", build_matrix(rows, "with_metadata", True)),
        ("Local(T-,I+)", build_matrix(rows, "with_metadata", False)),
        ("Local(T+,I-)", build_matrix(rows, "without_metadata", True)),
        ("Local(T-,I-)", build_matrix(rows, "without_metadata", False)),
    ]
    vmin = min(float(np.nanmin(panel)) for _, panel in panels)
    vmax = max(float(np.nanmax(panel)) for _, panel in panels)

    for out_stem, diagonal_indices, visible_rows in STAGES:
        render_stage(panels, vmin, vmax, out_stem, diagonal_indices, visible_rows)


if __name__ == "__main__":
    main()
