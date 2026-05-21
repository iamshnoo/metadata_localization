from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


CSV_PATH = Path("/path/to/metacul/results/plots/plot1/plot_1.csv")
OUT_DIR = Path("/path/to/metacul/results/plots/plot1")
OUT_PDF = OUT_DIR / "perplexity_continent_metadata_effect_1b_factorial_alt_ladder.pdf"
OUT_PNG = OUT_DIR / "perplexity_continent_metadata_effect_1b_factorial_alt_ladder.png"
OUT_CSV = OUT_DIR / "perplexity_continent_metadata_effect_1b_factorial_alt_ladder.csv"
LATEX_ALT_FIG = Path(
    "/path/to/metacul/latex/figs/main/"
    "1_perplexity_continent_metadata_effect_1b_factorial_alt_ladder.pdf"
)

CONTINENTS = ["Africa", "America", "Asia", "Europe"]
CONDITIONS = [
    ("T-", "I-", "No metadata", "#d9d9d9", "s", 62, -0.12),
    ("T+", "I-", "Train only", "#eca7a4", "s", 46, -0.12),
    ("T-", "I+", "Infer only", "#fad9b7", "o", 46, 0.12),
    ("T+", "I+", "Train + infer", "#a3cea8", "o", 72, 0.12),
]
COMPARISON_ENDPOINTS = {("T-", "I-"), ("T+", "I+")}
TRAIN_ARROW = "#2f9e44"
TRAIN_ARROW_MUTED = "#d77f7c"
INFER_LINK = "#5b677a"


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


def draw_arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: str, lw: float) -> None:
    ann = ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            linewidth=lw,
            mutation_scale=11,
            shrinkA=6,
            shrinkB=7,
        ),
        zorder=3.2,
    )
    if ann.arrow_patch is not None:
        ann.arrow_patch.set_path_effects(
            [pe.Stroke(linewidth=lw + 1.35, foreground="white", alpha=0.82), pe.Normal()]
        )


def main() -> None:
    df = load_plot_df()
    fig, ax = plt.subplots(figsize=(3.46, 2.36))

    summary_rows = []
    all_x = []
    for y, continent in zip(range(len(CONTINENTS))[::-1], CONTINENTS):
        values = {
            (train, infer): get_value(df, continent, train, infer)
            for train, infer, *_ in CONDITIONS
        }
        xs = [float(row["mean_ppl"]) for row in values.values()]
        all_x.extend(xs)
        baseline = float(values[("T-", "I-")]["mean_ppl"])
        full = float(values[("T+", "I+")]["mean_ppl"])
        gain = baseline - full

        y_low = y - 0.12
        y_high = y + 0.12
        row_min = min(xs) - 0.12
        row_max = max(xs) + 0.12
        ax.hlines([y_low, y_high], row_min, row_max, color="#e3e7ec", linewidth=1.1, zorder=0)
        ax.plot(
            [baseline, float(values[("T-", "I+")]["mean_ppl"])],
            [y_low, y_high],
            color=INFER_LINK,
            linestyle=(0, (2.2, 1.6)),
            linewidth=1.15,
            alpha=0.36,
            zorder=1.6,
        )
        ax.plot(
            [float(values[("T+", "I-")]["mean_ppl"]), full],
            [y_low, y_high],
            color=INFER_LINK,
            linestyle=(0, (2.2, 1.6)),
            linewidth=1.15,
            alpha=0.36,
            zorder=1.6,
        )
        draw_arrow(
            ax,
            baseline,
            y_low,
            float(values[("T+", "I-")]["mean_ppl"]),
            y_low,
            TRAIN_ARROW_MUTED,
            1.5,
        )
        draw_arrow(
            ax,
            float(values[("T-", "I+")]["mean_ppl"]),
            y_high,
            full,
            y_high,
            TRAIN_ARROW,
            2.1,
        )

        for train_meta, infer_meta, label, color, marker, size, y_offset in CONDITIONS:
            row = values[(train_meta, infer_meta)]
            x = float(row["mean_ppl"])
            y_point = y + y_offset
            xerr = [[x - float(row["ci_low"])], [float(row["ci_high"]) - x]]
            is_endpoint = (train_meta, infer_meta) in COMPARISON_ENDPOINTS
            ax.errorbar(
                [x],
                [y_point],
                xerr=xerr,
                fmt="none",
                ecolor=color,
                elinewidth=1.1 if is_endpoint else 0.8,
                capsize=2.4 if is_endpoint else 2.0,
                capthick=1.05 if is_endpoint else 0.8,
                alpha=0.88 if is_endpoint else 0.58,
                zorder=2.0 if is_endpoint else 1.4,
            )
            summary_rows.append(
                {
                    "continent": continent,
                    "condition": label,
                    "notation": f"({train_meta}, {infer_meta})",
                    "mean_ppl": x,
                    "delta_ppl_vs_no_metadata": x - baseline,
                    "train_effect_at_i_minus": (
                        float(values[("T+", "I-")]["mean_ppl"]) - baseline
                    ),
                    "train_effect_at_i_plus": (
                        full - float(values[("T-", "I+")]["mean_ppl"])
                    ),
                    "inference_effect_at_t_minus": (
                        float(values[("T-", "I+")]["mean_ppl"]) - baseline
                    ),
                    "inference_effect_at_t_plus": (
                        full - float(values[("T+", "I-")]["mean_ppl"])
                    ),
                    "gain_vs_no_metadata": gain if label == "Train + infer" else baseline - x,
                }
            )

        for draw_endpoints in (False, True):
            for train_meta, infer_meta, label, color, marker, size, y_offset in CONDITIONS:
                is_endpoint = (train_meta, infer_meta) in COMPARISON_ENDPOINTS
                if is_endpoint != draw_endpoints:
                    continue
                row = values[(train_meta, infer_meta)]
                x = float(row["mean_ppl"])
                y_point = y + y_offset
                scatter_kwargs = dict(
                    s=size,
                    marker=marker,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.35 if is_endpoint else 0.75,
                    zorder=5.0 if is_endpoint else 4.5,
                )
                ax.scatter([x], [y_point], **scatter_kwargs)

        label_x = (float(values[("T-", "I+")]["mean_ppl"]) + full) / 2
        train_effect_i_plus = full - float(values[("T-", "I+")]["mean_ppl"])
        txt = ax.text(
            label_x,
            y_high + 0.08,
            f"{train_effect_i_plus:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.4,
            fontweight="bold",
            color="#1b7f36",
            zorder=5,
        )
        txt.set_path_effects([pe.Stroke(linewidth=2.0, foreground="white", alpha=0.92), pe.Normal()])

    ax.set_yticks(range(len(CONTINENTS)))
    ax.set_yticklabels(CONTINENTS[::-1], fontsize=8.3, fontweight="bold")
    x_min = min(all_x) - 0.34
    x_max = max(all_x) + 0.34
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.58, len(CONTINENTS) - 0.16)
    ax.grid(axis="x", linestyle="--", linewidth=0.45, alpha=0.33)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Perplexity (lower is better)", fontsize=8.8)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.0)

    effect_handles = [
        Line2D(
            [0],
            [0],
            color=TRAIN_ARROW,
            linewidth=2.1,
            marker=">",
            markerfacecolor=TRAIN_ARROW,
            markeredgecolor=TRAIN_ARROW,
            label="Train effect at fixed I",
        ),
        Line2D(
            [0],
            [0],
            color=INFER_LINK,
            linestyle=(0, (2.2, 1.6)),
            linewidth=1.3,
            label="Inference link at fixed T",
        ),
    ]
    condition_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=6.0 if (train, infer) in COMPARISON_ENDPOINTS else 5.0,
            markeredgewidth=1.15 if (train, infer) in COMPARISON_ENDPOINTS else 0.75,
            label=f"{label} ({train}, {infer})",
        )
        for train, infer, label, color, marker, _, _ in CONDITIONS
    ]
    handles = effect_handles + condition_handles
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor="black",
        fontsize=6.05,
        columnspacing=0.5,
        handletextpad=0.35,
        borderpad=0.35,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_ALT_FIG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(OUT_CSV, index=False)
    fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(LATEX_ALT_FIG, dpi=600, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


if __name__ == "__main__":
    main()
