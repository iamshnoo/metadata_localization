from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


CSV_PATH = Path("/path/to/metacul/results/plots/plot1/plot_1.csv")
SIG_CSV = Path("/path/to/metacul/results/significance/plot4.csv")
OUT_PDF = Path("/path/to/metacul/results/plots/plot1/perplexity_continent_metadata_effect_1b_factorial.pdf")
OUT_PNG = Path("/path/to/metacul/results/plots/plot1/perplexity_continent_metadata_effect_1b_factorial.png")
LATEX_MAIN_FIG = Path("/path/to/metacul/latex/figs/main/1_perplexity_continent_metadata_effect_1b_factorial.pdf")

CONTINENTS = ["Africa", "America", "Asia", "Europe"]
PANEL_CONFIGS = [
    ("Local (T+, I+)", "T+", "I+", "#a3cea8"),
    ("Local (T-, I+)", "T-", "I+", "#fad9b7"),
    ("Local (T+, I-)", "T+", "I-", "#eca7a4"),
    ("Local (T-, I-)", "T-", "I-", "#d9d9d9"),
]


def parse_model_path(path):
    name = Path(path).name
    if "_with_metadata_" in name:
        continent, size = name.split("_with_metadata_")
        train = "T+"
    elif "_without_metadata_" in name:
        continent, size = name.split("_without_metadata_")
        train = "T-"
    else:
        raise ValueError(f"Unexpected model path: {path}")
    return continent.capitalize(), train, size


def parse_test_path(path):
    parts = [p for p in Path(path).parts if p]
    continent = parts[-2].capitalize()
    infer = "I+" if parts[-1] == "with_metadata" else "I-"
    return continent, infer


def load_plot_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    rows = []
    for _, row in df.iterrows():
        continent, train_meta, size = parse_model_path(row["model_path"])
        test_continent, infer_meta = parse_test_path(row["test_set_path"])
        if continent != test_continent:
            continue
        rows.append(
            {
                "continent": continent,
                "train_meta": train_meta,
                "infer_meta": infer_meta,
                "size": size,
                "mean_ppl": float(row["mean_ppl"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
            }
        )
    return pd.DataFrame(rows)


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


def load_figure3_range() -> tuple[float, float]:
    df = pd.read_csv(SIG_CSV)
    df = df[(df["plot"] == "plot4") & (df["size"].str.lower() == "1b")].copy()
    values = []
    for test_meta, value_cols in {
        "with_metadata": ["mean_ppl_a", "mean_ppl_b"],
        "without_metadata": ["mean_ppl_a", "mean_ppl_b"],
    }.items():
        matching = df[df["test_meta"] == test_meta]
        for col in value_cols:
            values.extend(float(v) for v in matching[col].dropna())
    return min(values), max(values)


def main() -> None:
    plot_df = load_plot_df()
    subset = plot_df[plot_df["size"] == "1b"].copy()

    vmin, vmax = load_figure3_range()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(3.46, 2.18))
    ax.set_xlim(-1.65, 3.55)
    ax.set_ylim(-0.65, 4.45)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(
        Rectangle(
            (-0.5, 3.96),
            2.0,
            0.36,
            facecolor="#f6f6f6",
            edgecolor="black",
            linewidth=0.55,
            zorder=1,
        )
    )
    ax.add_patch(
        Rectangle(
            (1.5, 3.96),
            2.0,
            0.36,
            facecolor="#f6f6f6",
            edgecolor="black",
            linewidth=0.55,
            zorder=1,
        )
    )
    ax.text(0.5, 4.14, "I+", ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text(2.5, 4.14, "I-", ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text(-0.68, 3.72, "Region", ha="right", va="center", fontsize=8.4, fontweight="bold")

    for x, (_, train_meta, _infer_meta, base_color) in enumerate(PANEL_CONFIGS):
        ax.add_patch(
            Rectangle(
                (x - 0.5, 3.55),
                1.0,
                0.35,
                facecolor=base_color,
                edgecolor="black",
                linewidth=0.55,
                alpha=0.95,
                zorder=1.5,
            )
        )
        ax.text(x, 3.725, train_meta, ha="center", va="center", fontsize=8.4, fontweight="bold")

    for row_idx, continent in enumerate(CONTINENTS):
        y = len(CONTINENTS) - 1 - row_idx
        ax.text(-0.68, y, continent, ha="right", va="center", fontsize=8.5)

        row_values = []
        for x, (_title, train_meta, infer_meta, _base_color) in enumerate(PANEL_CONFIGS):
            row = subset[
                (subset["continent"] == continent)
                & (subset["train_meta"] == train_meta)
                & (subset["infer_meta"] == infer_meta)
            ]
            if row.empty:
                row_values.append(float("inf"))
            else:
                row_values.append(float(row.iloc[0]["mean_ppl"]))
        best_x = min(range(len(row_values)), key=row_values.__getitem__)

        for x, (_title, train_meta, infer_meta, base_color) in enumerate(PANEL_CONFIGS):
            row = subset[
                (subset["continent"] == continent)
                & (subset["train_meta"] == train_meta)
                & (subset["infer_meta"] == infer_meta)
            ]
            if row.empty:
                continue
            mean = float(row.iloc[0]["mean_ppl"])
            cmap = _panel_cmap(base_color)
            rgb = cmap(norm(mean))[:3]
            ax.add_patch(
                Rectangle(
                    (x - 0.5, y - 0.5),
                    1.0,
                    1.0,
                    facecolor=rgb,
                    edgecolor="black" if x == best_x else "#cfcfcf",
                    linewidth=1.05 if x == best_x else 0.45,
                    zorder=1,
                )
            )
            ax.text(
                x,
                y,
                f"{mean:.1f}",
                ha="center",
                va="center",
                fontsize=8.1,
                fontweight="bold" if x == best_x else "normal",
                color=_text_color(rgb),
                zorder=3,
                path_effects=[pe.withStroke(linewidth=0.7, foreground="white", alpha=0.32)],
            )

    ax.plot([1.5, 1.5], [-0.5, 4.32], color="black", linewidth=0.65, linestyle=(0, (2.0, 2.0)))
    ax.text(1.5, -0.56, "train region = test region", ha="center", va="top", fontsize=8.0)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    LATEX_MAIN_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(LATEX_MAIN_FIG, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
