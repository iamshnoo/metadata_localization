#!/usr/bin/env python3
import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image


REPO_ROOT = Path("/path/to/metacul")
DEFAULT_WVS_ROOT = REPO_ROOT / "results" / "culture_map_wvs_all_human_typical_mixed"
DEFAULT_TEMPLATE = Path("/path/to/culture-map/data/paper_osf/Map2023NEWsmall.png")
DEFAULT_OUTPUT_PNG = REPO_ROOT / "results" / "plots" / "plot8" / "wvs_maple_template_inmap_only_nosource.png"
DEFAULT_OUTPUT_PDF = REPO_ROOT / "results" / "plots" / "plot8" / "wvs_maple_template_inmap_only_nosource.pdf"

# Pixel bounds of the actual data axes inside Map2023NEWsmall.png.
AXIS_LEFT = 89
AXIS_RIGHT = 1557
AXIS_TOP = 121
AXIS_BOTTOM = 1077

# The raster includes a title we do not want in the final figure.
TOP_CROP = 75

# Full displayed axes on the official template.
XMIN = -2.5
XMAX = 3.5
YMIN = -2.5
YMAX = 2.0

# Match the previously used output width so this swaps into existing slide/paper flows cleanly.
TARGET_WIDTH = 4200

VARIANT_LABELS = {
    "maple_1b_tplus_eplus": "MAPLE 1B (T+, I+)",
    "maple_3b_tplus_eplus": "MAPLE 3B (T+, I+)",
    "maple_1b_tminus_eminus": "MAPLE 1B (T-, I-)",
    "maple_3b_tminus_eminus": "MAPLE 3B (T-, I-)",
}

# Pixel nudges after resizing/cropping. These are tuned for the all-country centroids.
LABEL_NUDGES = {
    "MAPLE 1B (T+, I+)": (-650, 90),
    "MAPLE 3B (T+, I+)": (-380, -10),
}


def _load_centroids(path: Path):
    sums = defaultdict(lambda: [0.0, 0.0, 0])
    with path.open() as f:
        for row in csv.DictReader(f):
            variant = row["variant"]
            if variant not in VARIANT_LABELS:
                continue
            sums[variant][0] += float(row["RC1"])
            sums[variant][1] += float(row["RC2"])
            sums[variant][2] += 1
    centroids = {}
    for variant, (sx, sy, n) in sums.items():
        if n:
            centroids[variant] = (sx / n, sy / n)
    return centroids


def _to_px(x, y, scale):
    x_px = AXIS_LEFT + (x - XMIN) / (XMAX - XMIN) * (AXIS_RIGHT - AXIS_LEFT)
    y_px = AXIS_TOP + (YMAX - y) / (YMAX - YMIN) * (AXIS_BOTTOM - AXIS_TOP)
    return x_px * scale, (y_px - TOP_CROP) * scale


def _in_display_bounds(x, y):
    return XMIN <= x <= XMAX and YMIN <= y <= YMAX


def main():
    wvs_root = Path(os.environ.get("WVS_ROOT", str(DEFAULT_WVS_ROOT)))
    template_path = Path(os.environ.get("TEMPLATE_IMAGE", str(DEFAULT_TEMPLATE)))
    output_png = Path(os.environ.get("OUTPUT_PNG", str(DEFAULT_OUTPUT_PNG)))
    output_pdf = Path(os.environ.get("OUTPUT_PDF", str(DEFAULT_OUTPUT_PDF)))

    centroids = _load_centroids(wvs_root / "all_variant_country_mean_projection.csv")

    base = Image.open(template_path).convert("RGB")
    cropped = base.crop((0, TOP_CROP, base.width, base.height))
    scale = TARGET_WIDTH / cropped.width
    lanczos = getattr(Image, "Resampling", Image).LANCZOS
    resized = cropped.resize((TARGET_WIDTH, int(round(cropped.height * scale))), lanczos)

    fig_w = resized.width / 300
    fig_h = resized.height / 300
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.imshow(resized)
    ax.set_axis_off()

    # Remove the bottom-right source block while keeping the rest of the official map untouched.
    ax.add_patch(
        plt.Rectangle(
            (resized.width * 0.735, resized.height * 0.665),
            resized.width * 0.255,
            resized.height * 0.23,
            facecolor="white",
            edgecolor="none",
            zorder=2,
        )
    )

    # Add central zero guides in the actual plotted coordinate system.
    x0_px, y0_px = _to_px(0.0, 0.0, scale)
    left_px, _ = _to_px(XMIN, 0.0, scale)
    right_px, _ = _to_px(XMAX, 0.0, scale)
    _, top_px = _to_px(0.0, YMAX, scale)
    _, bottom_px = _to_px(0.0, YMIN, scale)
    ax.plot(
        [left_px, right_px],
        [y0_px, y0_px],
        linestyle=(0, (6, 6)),
        color="#666666",
        linewidth=1.7,
        alpha=0.9,
        zorder=3,
    )
    ax.plot(
        [x0_px, x0_px],
        [top_px, bottom_px],
        linestyle=(0, (6, 6)),
        color="#666666",
        linewidth=1.7,
        alpha=0.9,
        zorder=3,
    )

    for variant in ("maple_1b_tplus_eplus", "maple_3b_tplus_eplus", "maple_1b_tminus_eminus", "maple_3b_tminus_eminus"):
        if variant not in centroids:
            continue
        x, y = centroids[variant]
        if not _in_display_bounds(x, y):
            continue
        label = VARIANT_LABELS[variant]
        px, py = _to_px(x, y, scale)
        ax.scatter(
            [px],
            [py],
            s=680,
            marker="^",
            c="#d9a7ea",
            edgecolors="#4f425b",
            linewidths=2.2,
            zorder=5,
        )
        dx, dy = LABEL_NUDGES.get(label, (60, 20))
        ax.text(
            px + dx,
            py + dy,
            label,
            fontsize=22,
            color="black",
            bbox={
                "facecolor": "white",
                "edgecolor": "#777777",
                "boxstyle": "square,pad=0.20",
            },
            zorder=6,
        )

    fig.subplots_adjust(0, 0, 1, 1)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight", pad_inches=0)
    print(output_png)
    print(output_pdf)


if __name__ == "__main__":
    main()
