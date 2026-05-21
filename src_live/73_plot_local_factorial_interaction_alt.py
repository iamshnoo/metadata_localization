from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


CSV_PATH = Path("/path/to/metacul/results/plots/plot1/plot_1.csv")
OUT_DIR = Path("/path/to/metacul/results/plots/plot1")
OUT_PDF = OUT_DIR / "perplexity_continent_metadata_effect_1b_factorial_alt_interaction.pdf"
OUT_PNG = OUT_DIR / "perplexity_continent_metadata_effect_1b_factorial_alt_interaction.png"
OUT_CSV = OUT_DIR / "perplexity_continent_metadata_effect_1b_factorial_alt_interaction.csv"
LATEX_ALT_FIG = Path(
    "/path/to/metacul/latex/figs/main/"
    "1_perplexity_continent_metadata_effect_1b_factorial_alt_interaction.pdf"
)

CONTINENTS = ["Africa", "America", "Asia", "Europe"]
TRAIN_ORDER = ["T-", "T+"]
INFER_ORDER = ["I-", "I+"]

COND_STYLES = {
    ("T-", "I-"): {"color": "#d9d9d9", "marker": "s", "label": "(T-, I-)"},
    ("T+", "I-"): {"color": "#eca7a4", "marker": "s", "label": "(T+, I-)"},
    ("T-", "I+"): {"color": "#fad9b7", "marker": "o", "label": "(T-, I+)"},
    ("T+", "I+"): {"color": "#a3cea8", "marker": "o", "label": "(T+, I+)"},
}
SERIES_STYLES = {
    "I-": {"color": "#b76b6b", "linestyle": (0, (4, 2)), "label": "Evaluate without metadata (I-)"},
    "I+": {"color": "#2f9e44", "linestyle": "-", "label": "Evaluate with metadata (I+)"},
}
X_OFFSET = {"I-": -0.035, "I+": 0.035}


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
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
            }
        )
    return pd.DataFrame(rows)


def get_value(df: pd.DataFrame, continent: str, train_meta: str, infer_meta: str) -> pd.Series:
    row = df[
        (df["continent"] == continent)
        & (df["train_meta"] == train_meta)
        & (df["infer_meta"] == infer_meta)
    ]
    if row.empty:
        raise ValueError(f"Missing {continent} {train_meta}/{infer_meta}")
    return row.iloc[0]


def delta_label(delta: float) -> str:
    return f"{delta:+.2f}"


def label_effect(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    text: str,
    color: str,
    y_pad: float,
) -> None:
    x_mid = (x0 + x1) / 2
    y_mid = (y0 + y1) / 2 + y_pad
    txt = ax.text(
        x_mid,
        y_mid,
        text,
        ha="center",
        va="bottom" if y_pad >= 0 else "top",
        fontsize=7.0,
        fontweight="bold",
        color=color,
        zorder=8,
    )
    txt.set_path_effects([pe.Stroke(linewidth=2.1, foreground="white", alpha=0.94), pe.Normal()])


def main() -> None:
    df = load_plot_df()
    fig, axes = plt.subplots(2, 2, figsize=(3.46, 2.92), sharex=True, sharey=True)
    axes = axes.flatten()

    all_values = []
    summary_rows = []
    for ax, continent in zip(axes, CONTINENTS):
        values = {
            (train, infer): get_value(df, continent, train, infer)
            for train in TRAIN_ORDER
            for infer in INFER_ORDER
        }
        ax.axvspan(-0.25, 0.25, color="#f4eceb", alpha=0.96, zorder=0)
        ax.axvspan(0.75, 1.25, color="#edf6ed", alpha=0.96, zorder=0)

        effects = {}
        for infer_meta in INFER_ORDER:
            xs, ys = [], []
            for train_meta in TRAIN_ORDER:
                row = values[(train_meta, infer_meta)]
                x = TRAIN_ORDER.index(train_meta) + X_OFFSET[infer_meta]
                y = float(row["mean_ppl"])
                xs.append(x)
                ys.append(y)
                all_values.append(y)
                style = COND_STYLES[(train_meta, infer_meta)]
                ax.errorbar(
                    [x],
                    [y],
                    yerr=[[y - float(row["ci_low"])], [float(row["ci_high"]) - y]],
                    fmt="none",
                    ecolor=style["color"],
                    elinewidth=1.05,
                    capsize=2.0,
                    capthick=1.0,
                    alpha=0.85,
                    zorder=4,
                )
                ax.scatter(
                    [x],
                    [y],
                    s=55 if (train_meta, infer_meta) != ("T+", "I+") else 70,
                    marker=style["marker"],
                    facecolor=style["color"],
                    edgecolor="black",
                    linewidth=0.9,
                    zorder=5,
                )
                summary_rows.append(
                    {
                        "continent": continent,
                        "condition": style["label"],
                        "train_meta": train_meta,
                        "infer_meta": infer_meta,
                        "mean_ppl": y,
                    }
                )

            series_style = SERIES_STYLES[infer_meta]
            line = ax.plot(
                xs,
                ys,
                color=series_style["color"],
                linestyle=series_style["linestyle"],
                linewidth=1.9,
                zorder=3,
            )[0]
            line.set_path_effects([pe.Stroke(linewidth=3.3, foreground="white", alpha=0.76), pe.Normal()])
            effects[infer_meta] = ys[1] - ys[0]

        y_span = max(all_values) - min(all_values) if all_values else 1.0
        label_effect(
            ax,
            0 + X_OFFSET["I+"],
            1 + X_OFFSET["I+"],
            float(values[("T-", "I+")]["mean_ppl"]),
            float(values[("T+", "I+")]["mean_ppl"]),
            f"ΔT|I+ {delta_label(effects['I+'])}",
            SERIES_STYLES["I+"]["color"],
            0.05 * y_span,
        )
        label_effect(
            ax,
            0 + X_OFFSET["I-"],
            1 + X_OFFSET["I-"],
            float(values[("T-", "I-")]["mean_ppl"]),
            float(values[("T+", "I-")]["mean_ppl"]),
            f"ΔT|I- {delta_label(effects['I-'])}",
            "#8d4f4f",
            -0.06 * y_span,
        )

        interaction = effects["I+"] - effects["I-"]
        ax.text(
            0.5,
            0.95,
            f"{continent}  ΔΔ {delta_label(interaction)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.4,
            fontweight="bold",
            bbox=dict(facecolor="#e2e2e2", edgecolor="#9b9b9b", linewidth=0.7, boxstyle="round,pad=0.20"),
            zorder=10,
        )
        for key, value in effects.items():
            summary_rows.append(
                {
                    "continent": continent,
                    "condition": f"train_effect_at_{key}",
                    "train_meta": "T+ minus T-",
                    "infer_meta": key,
                    "mean_ppl": value,
                }
            )
        summary_rows.append(
            {
                "continent": continent,
                "condition": "difference_in_differences",
                "train_meta": "interaction",
                "infer_meta": "I+ minus I-",
                "mean_ppl": interaction,
            }
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["T-", "T+"], fontsize=8.6, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8.0)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.28)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)

    y_min = min(all_values)
    y_max = max(all_values)
    for ax in axes:
        ax.set_ylim(y_min - 0.45, y_max + 0.72)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=SERIES_STYLES["I+"]["color"],
            linestyle=SERIES_STYLES["I+"]["linestyle"],
            linewidth=2.0,
            marker="o",
            markerfacecolor=COND_STYLES[("T+", "I+")]["color"],
            markeredgecolor="black",
            label=SERIES_STYLES["I+"]["label"],
        ),
        Line2D(
            [0],
            [0],
            color=SERIES_STYLES["I-"]["color"],
            linestyle=SERIES_STYLES["I-"]["linestyle"],
            linewidth=2.0,
            marker="s",
            markerfacecolor=COND_STYLES[("T+", "I-")]["color"],
            markeredgecolor="black",
            label=SERIES_STYLES["I-"]["label"],
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=1,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor="black",
        fontsize=7.3,
        borderpad=0.35,
        handlelength=2.4,
    )
    fig.text(0.5, 0.045, "Training metadata", ha="center", fontsize=9.0)
    fig.text(0.02, 0.50, "Perplexity (lower is better)", va="center", rotation="vertical", fontsize=9.0)
    fig.subplots_adjust(top=0.78, bottom=0.15, left=0.16, right=0.99, hspace=0.20, wspace=0.13)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_ALT_FIG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(OUT_CSV, index=False)
    fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(LATEX_ALT_FIG, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
