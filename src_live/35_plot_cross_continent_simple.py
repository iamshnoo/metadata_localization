from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


SIG_CSV = Path("/scratch/amukher6/metacul/results/significance/plot4.csv")
OUT_DIR = Path("/scratch/amukher6/metacul/results/plots/plot4")
OUT_PDF = OUT_DIR / "perplexity_cross_continent_1b_simple.pdf"
OUT_PNG = OUT_DIR / "perplexity_cross_continent_1b_simple.png"
LATEX_MAIN_FIG = Path(
    "/scratch/amukher6/metacul/latex/figs/main/3_perplexity_cross_continent_simple_1b.pdf"
)

CONTINENTS = ["africa", "america", "asia", "europe"]
LABELS = ["Africa", "America", "Asia", "Europe"]
PANEL_STYLES = {
    "Local(T+,I+)": {"base": "#a3cea8", "title_fc": "#a3cea8", "title_ec": "black"},
    "Local(T+,I-)": {"base": "#eca7a4", "title_fc": "#eca7a4", "title_ec": "black"},
    "Local(T-,I+)": {"base": "#fad9b7", "title_fc": "#fad9b7", "title_ec": "black"},
    "Local(T-,I-)": {"base": "#d9d9d9", "title_fc": "#d9d9d9", "title_ec": "black"},
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


def main() -> None:
    df = pd.read_csv(SIG_CSV)
    df = df[(df["plot"] == "plot4") & (df["size"].str.lower() == "1b")].copy()

    panels = [
        ("Local(T+,I+)", build_matrix(df, "with_metadata", True)),
        ("Local(T-,I+)", build_matrix(df, "with_metadata", False)),
        ("Local(T+,I-)", build_matrix(df, "without_metadata", True)),
        ("Local(T-,I-)", build_matrix(df, "without_metadata", False)),
    ]

    vmin = min(panel.values.min() for _, panel in panels)
    vmax = max(panel.values.max() for _, panel in panels)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.55))

    title_artists = []

    for idx, (title, panel) in enumerate(panels):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        style = PANEL_STYLES[title]
        cmap = _panel_cmap(style["base"])
        ax.imshow(panel.values, cmap=cmap, norm=norm, aspect="equal")

        title_artist = ax.text(
            0.5,
            1.035,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12.8,
            fontweight="bold",
            clip_on=False,
            zorder=5,
            bbox=dict(
                facecolor=style["title_fc"],
                edgecolor=style["title_ec"],
                linewidth=0.8,
                alpha=0.95,
                boxstyle="round,pad=0.25",
            ),
        )
        title_artists.append(title_artist)
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=28, ha="right", fontsize=9.5)
        ax.set_yticklabels(LABELS, fontsize=10)
        if r == 0:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Test Region", fontsize=10.5)
        if c == 0:
            ax.set_ylabel("Train Region", fontsize=10.5)
        else:
            ax.set_ylabel("")

        ax.set_xticks([x - 0.5 for x in range(1, len(LABELS))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(LABELS))], minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(length=0)
        for i in range(panel.shape[0]):
            for j in range(panel.shape[1]):
                val = panel.iloc[i, j]
                rgb = cmap(norm(val))[:3]
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=10.2,
                    fontweight="bold" if i == j else "normal",
                    color=_text_color(rgb),
                )

    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.88, wspace=0.16, hspace=0.30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_MAIN_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", pad_inches=0.06, bbox_extra_artists=title_artists)
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight", pad_inches=0.06, bbox_extra_artists=title_artists)
    fig.savefig(
        LATEX_MAIN_FIG,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.06,
        bbox_extra_artists=title_artists,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
