from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox


SIG_CSV = Path("/path/to/metacul/results/significance/plot4.csv")
OUT_DIR = Path("/path/to/metacul/results/plots/plot4")
OUT_PDF = OUT_DIR / "perplexity_cross_continent_1b_simple.pdf"
OUT_PNG = OUT_DIR / "perplexity_cross_continent_1b_simple.png"
LATEX_MAIN_FIG = Path(
    "/path/to/metacul/latex/figs/main/3_perplexity_cross_continent_simple_1b.pdf"
)
LATEX_OVERLEAF_FIG = Path(
    "/path/to/metacul/latex/overleaf_bundle/figs/main/3_perplexity_cross_continent_simple_1b.pdf"
)
FIGURE3_FOOTPRINT_IN = (221.788 / 72.0, 224.000 / 72.0)

CONTINENTS = ["africa", "america", "asia", "europe"]
LABELS = ["Africa", "America", "Asia", "Europe"]
PANEL_STYLES = {
    "Local (T+, I+)": {"base": "#a3cea8", "title_fc": "#a3cea8", "title_ec": "black"},
    "Local (T+, I-)": {"base": "#eca7a4", "title_fc": "#eca7a4", "title_ec": "black"},
    "Local (T-, I+)": {"base": "#fad9b7", "title_fc": "#fad9b7", "title_ec": "black"},
    "Local (T-, I-)": {"base": "#d9d9d9", "title_fc": "#d9d9d9", "title_ec": "black"},
}


def build_matrix(df: pd.DataFrame, test_meta: str, use_with_model: bool) -> pd.DataFrame:
    value_col = "mean_ppl_b" if use_with_model else "mean_ppl_a"
    values = []
    for train in CONTINENTS:
        row = []
        for test in CONTINENTS:
            match = df[
                (df["train_continent"] == train)
                & (df["test_continent"] == test)
                & (df["test_meta"] == test_meta)
            ]
            if match.empty:
                row.append(float("nan"))
            else:
                row.append(float(match.iloc[0][value_col]))
        values.append(row)
    return pd.DataFrame(values, index=LABELS, columns=LABELS)


def _mix(color, target, alpha):
    base_rgb = mcolors.to_rgb(color)
    target_rgb = mcolors.to_rgb(target)
    return tuple((1 - alpha) * b + alpha * t for b, t in zip(base_rgb, target_rgb))


def _panel_cmap(base: str) -> LinearSegmentedColormap:
    dark = _mix(base, "black", 0.28)
    mid = mcolors.to_rgb(base)
    light = _mix(base, "white", 0.62)
    return LinearSegmentedColormap.from_list("", [dark, mid, light])


def _text_color(rgb):
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "white" if luminance < 0.52 else "black"


def _fixed_footprint_bbox(fig) -> Bbox:
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    target_width = max(tight.width, FIGURE3_FOOTPRINT_IN[0])
    target_height = max(tight.height, FIGURE3_FOOTPRINT_IN[1])
    return Bbox.from_bounds(
        tight.x0 - (target_width - tight.width) / 2,
        tight.y0 - (target_height - tight.height) / 2 + 0.03,
        target_width,
        target_height,
    )


def main() -> None:
    df = pd.read_csv(SIG_CSV)
    df = df[(df["plot"] == "plot4") & (df["size"].str.lower() == "1b")].copy()

    panels = [
        ("Local (T+, I+)", build_matrix(df, "with_metadata", True)),
        ("Local (T-, I+)", build_matrix(df, "with_metadata", False)),
        ("Local (T+, I-)", build_matrix(df, "without_metadata", True)),
        ("Local (T-, I-)", build_matrix(df, "without_metadata", False)),
    ]

    all_values = np.concatenate([panel.values.astype(float).ravel() for _, panel in panels])
    vmin = float(np.nanmin(all_values))
    vmax = float(np.nanmax(all_values))

    fig, axes = plt.subplots(2, 2, figsize=(3.68, 3.05), sharex=True, sharey=True)
    axes = axes.reshape(2, 2)
    title_artists = []

    for idx, (ax, (title, panel)) in enumerate(zip(axes.ravel(), panels)):
        _, col_idx = divmod(idx, 2)
        ax.set_anchor("E" if col_idx == 0 else "W")
        values = panel.values.astype(float)
        style = PANEL_STYLES[title]
        norm = (values - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0.0, 1.0)
        img = _panel_cmap(style["base"])(norm)

        ax.imshow(img, vmin=0.0, vmax=1.0)

        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                is_diag = row_idx == col_idx
                cell_norm = np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)
                cell_rgb = _panel_cmap(style["base"])(cell_norm)
                text_alpha = 1.0 if is_diag else 0.72
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8.6,
                    fontweight="bold" if is_diag else "normal",
                    color=_text_color(cell_rgb[:3]),
                    alpha=text_alpha,
                )
                if is_diag:
                    ax.add_patch(
                        Rectangle(
                            (col_idx - 0.5, row_idx - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            edgecolor="black",
                            linewidth=1.45,
                            zorder=5,
                            joinstyle="miter",
                        )
                    )

        title_artist = ax.set_title(
            title,
            fontsize=9.7,
            fontweight="bold",
            pad=3.2,
            bbox=dict(
                facecolor=style["title_fc"],
                edgecolor=style["title_ec"],
                            linewidth=0.9,
                            boxstyle="round,pad=0.18",
            ),
        )
        title_artists.append(title_artist)

        ax.set_xticks(np.arange(len(LABELS)))
        ax.set_yticks(np.arange(len(LABELS)))
        if idx // 2 == 1:
            ax.set_xticklabels(LABELS, fontsize=8.5, rotation=35, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticklabels([])
        if idx % 2 == 0:
            ax.set_yticklabels(LABELS, fontsize=8.5)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="both", length=0, pad=1.5)

        ax.set_xticks(np.arange(-0.5, len(LABELS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(LABELS), 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.45)
        ax.tick_params(which="minor", bottom=False, left=False)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
            spine.set_color("black")

    axes[0, 0].set_ylabel("Train Region", fontsize=9.5, labelpad=5.0)
    axes[1, 0].set_ylabel("Train Region", fontsize=9.5, labelpad=5.0)
    axes[1, 0].set_xlabel("Test Region", fontsize=9.5, labelpad=-1.0)
    axes[1, 1].set_xlabel("Test Region", fontsize=9.5, labelpad=-1.0)

    fig.subplots_adjust(left=0.125, right=0.998, bottom=0.112, top=0.918, wspace=0.125, hspace=0.245)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_MAIN_FIG.parent.mkdir(parents=True, exist_ok=True)
    LATEX_OVERLEAF_FIG.parent.mkdir(parents=True, exist_ok=True)
    save_bbox = _fixed_footprint_bbox(fig)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches=save_bbox, pad_inches=0.0)
    fig.savefig(OUT_PDF, dpi=600, bbox_inches=save_bbox, pad_inches=0.0)
    fig.savefig(
        LATEX_MAIN_FIG,
        dpi=600,
        bbox_inches=save_bbox,
        pad_inches=0.0,
    )
    fig.savefig(
        LATEX_OVERLEAF_FIG,
        dpi=600,
        bbox_inches=save_bbox,
        pad_inches=0.0,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
