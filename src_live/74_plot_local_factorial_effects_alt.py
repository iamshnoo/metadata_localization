from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


CSV_PATH = Path("/path/to/metacul/results/plots/plot1/plot_1.csv")
OUT_DIR = Path("/path/to/metacul/results/plots/plot1")
OUT_PDF = OUT_DIR / "perplexity_continent_metadata_effect_1b_factorial_alt_effects.pdf"
OUT_PNG = OUT_DIR / "perplexity_continent_metadata_effect_1b_factorial_alt_effects.png"
OUT_CSV = OUT_DIR / "perplexity_continent_metadata_effect_1b_factorial_alt_effects.csv"
LATEX_ALT_FIG = Path(
    "/path/to/metacul/latex/figs/main/"
    "1_perplexity_continent_metadata_effect_1b_factorial_alt_effects.pdf"
)

CONTINENTS = ["Africa", "America", "Asia", "Europe"]
COLORS = {
    "weak": "#b76b6b",
    "infer_only": "#b76b6b",
    "full": "#2f9e44",
    "line": "#8b95a5",
}


def parse_model_path(path: str) -> tuple[str, str, str]:
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


def parse_test_path(path: str) -> tuple[str, str]:
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
        if continent != test_continent or size != "1b":
            continue
        rows.append(
            {
                "continent": continent,
                "train_meta": train_meta,
                "infer_meta": infer_meta,
                "mean_ppl": float(row["mean_ppl"]),
            }
        )
    return pd.DataFrame(rows)


def value(df: pd.DataFrame, continent: str, train: str, infer: str) -> float:
    row = df[
        (df["continent"] == continent)
        & (df["train_meta"] == train)
        & (df["infer_meta"] == infer)
    ]
    if row.empty:
        raise ValueError(f"Missing {continent} {train}/{infer}")
    return float(row.iloc[0]["mean_ppl"])


def add_value_label(ax: plt.Axes, x: float, y: float, color: str, above: bool) -> None:
    dx = 0.04 if x >= 0 else -0.04
    ha = "left" if x >= 0 else "right"
    txt = ax.text(
        x + dx,
        y + (0.09 if above else -0.09),
        f"{x:+.2f}",
        ha=ha,
        va="bottom" if above else "top",
        fontsize=7.0,
        fontweight="bold",
        color=color,
        zorder=8,
    )
    txt.set_path_effects([pe.Stroke(linewidth=2.0, foreground="white", alpha=0.94), pe.Normal()])


def draw_effect_pair(
    ax: plt.Axes,
    y: float,
    weak_effect: float,
    strong_effect: float,
    weak_color: str,
    strong_color: str,
) -> None:
    ax.plot(
        [weak_effect, strong_effect],
        [y, y],
        color=COLORS["line"],
        linewidth=1.3,
        alpha=0.58,
        zorder=2,
    )
    ax.scatter(
        [weak_effect],
        [y],
        s=48,
        marker="s",
        facecolor=weak_color,
        edgecolor="black",
        linewidth=0.85,
        zorder=5,
    )
    ax.scatter(
        [strong_effect],
        [y],
        s=64,
        marker="o",
        facecolor=strong_color,
        edgecolor="black",
        linewidth=0.95,
        zorder=6,
    )
    add_value_label(ax, weak_effect, y, weak_color, above=False)
    add_value_label(ax, strong_effect, y, strong_color, above=True)


def main() -> None:
    df = load_plot_df()
    rows = []
    for continent in CONTINENTS:
        tminus_iminus = value(df, continent, "T-", "I-")
        tplus_iminus = value(df, continent, "T+", "I-")
        tminus_iplus = value(df, continent, "T-", "I+")
        tplus_iplus = value(df, continent, "T+", "I+")
        rows.append(
            {
                "continent": continent,
                "train_effect_at_i_minus": tplus_iminus - tminus_iminus,
                "train_effect_at_i_plus": tplus_iplus - tminus_iplus,
                "inference_effect_at_t_minus": tminus_iplus - tminus_iminus,
                "inference_effect_at_t_plus": tplus_iplus - tplus_iminus,
                "interaction_train_effect": (tplus_iplus - tminus_iplus)
                - (tplus_iminus - tminus_iminus),
            }
        )
    effect_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(3.46, 2.12), sharey=True)
    y_positions = list(range(len(CONTINENTS)))[::-1]

    for y, row in zip(y_positions, effect_df.to_dict("records")):
        draw_effect_pair(
            axes[0],
            y,
            float(row["train_effect_at_i_minus"]),
            float(row["train_effect_at_i_plus"]),
            COLORS["weak"],
            COLORS["full"],
        )
        draw_effect_pair(
            axes[1],
            y,
            float(row["inference_effect_at_t_minus"]),
            float(row["inference_effect_at_t_plus"]),
            COLORS["infer_only"],
            COLORS["full"],
        )

    titles = [
        "Add train metadata\nT+ minus T-",
        "Add inference metadata\nI+ minus I-",
    ]
    for ax, title in zip(axes, titles):
        ax.axvline(0, color="#222222", linewidth=1.0, linestyle=(0, (3, 2)), alpha=0.78, zorder=1)
        ax.set_title(title, fontsize=8.1, fontweight="bold", pad=4)
        ax.grid(axis="x", linestyle="--", linewidth=0.45, alpha=0.28)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.tick_params(axis="y", length=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.95)

    all_effects = []
    for col in [
        "train_effect_at_i_minus",
        "train_effect_at_i_plus",
        "inference_effect_at_t_minus",
        "inference_effect_at_t_plus",
    ]:
        all_effects.extend(effect_df[col].astype(float).tolist())
    x_min = min(all_effects) - 0.28
    x_max = max(all_effects) + 0.32
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_xticks([-2, -1, 0, 1])

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(CONTINENTS, fontsize=8.3, fontweight="bold")
    axes[1].tick_params(labelleft=False)
    fig.text(0.5, 0.03, "Δ perplexity (negative is better)", ha="center", fontsize=8.5)

    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=COLORS["weak"],
            markeredgecolor="black",
            markersize=5.5,
            label="Effect without paired metadata",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["full"],
            markeredgecolor="black",
            markersize=6.1,
            label="Effect with paired metadata",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor="black",
        fontsize=6.9,
        borderpad=0.35,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.21, right=0.99, top=0.72, bottom=0.20, wspace=0.12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_ALT_FIG.parent.mkdir(parents=True, exist_ok=True)
    effect_df.to_csv(OUT_CSV, index=False)
    fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(LATEX_ALT_FIG, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
