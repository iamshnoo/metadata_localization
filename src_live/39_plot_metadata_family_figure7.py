#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


CSV_PATH = Path("/path/to/metacul/results/plots/plot10/plot_10.csv")
RESULTS_DIR = Path("/path/to/metacul/results/plots/plot10")
LATEX_FIG = Path("/path/to/metacul/latex/figs/main/6_perplexity_metadata_ablations_1b_main.pdf")


STEPS = [2000, 4000, 8000, 10000]
STEP_LABELS = ["2k", "4k", "8k", "10k"]

TESTS = [
    ("[URL] (I+)", "/path/to/metacul/training_data/meco_datasets/combined_only_url/with_metadata/"),
    ("[URL][Country] (I+)", "/path/to/metacul/training_data/meco_datasets/combined_only_url_country/with_metadata/"),
    ("[URL][Continent] (I+)", "/path/to/metacul/training_data/meco_datasets/combined_only_url_continent/with_metadata/"),
    ("[Country] (I+)", "/path/to/metacul/training_data/meco_datasets/combined_only_country/with_metadata/"),
    ("[Continent] (I+)", "/path/to/metacul/training_data/meco_datasets/combined_only_continent/with_metadata/"),
    ("[URL][Country][Continent] (I+)", "/path/to/metacul/training_data/meco_datasets/combined/with_metadata/"),
    ("No metadata (I-)", "/path/to/metacul/training_data/meco_datasets/combined/without_metadata/"),
]

MODEL_GROUPS = {
    "combined_with": {
        "label": "[URL][Country][Continent] (T+)",
        "final": "/path/to/metacul/models/combined_with_metadata_1b",
        "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_with_metadata_1b_step{step}k",
        "color": "#2b8c66",
        "marker": "o",
        "linestyle": "-",
        "markerfacecolor": "#2b8c66",
    },
    "combined_without": {
        "label": "[No metadata] (T-)",
        "final": "/path/to/metacul/models/combined_without_metadata_1b",
        "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_without_metadata_1b_step{step}k",
        "color": "#7f7f7f",
        "marker": "s",
        "linestyle": (0, (5, 2)),
        "markerfacecolor": "#7f7f7f",
    },
    "url": {
        "label": "[URL] (T+)",
        "final": "/path/to/metacul/models/ablations/metadata/combined_only_url_with_metadata_1b",
        "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_url_with_metadata_1b_step{step}k",
        "color": "#f4a3a3",
        "marker": "D",
        "linestyle": (0, (3, 1, 1, 1)),
        "markerfacecolor": "#f4a3a3",
    },
    "country_only": {
        "label": "[Country] (T+)",
        "final": "/path/to/metacul/models/combined_only_country_with_metadata_1b",
        "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_country_with_metadata_1b_step{step}k",
        "color": "#f0c36d",
        "marker": "P",
        "linestyle": (0, (1, 1)),
        "markerfacecolor": "white",
    },
    "continent_only": {
        "label": "[Continent] (T+)",
        "final": "/path/to/metacul/models/combined_only_continent_with_metadata_1b",
        "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_continent_with_metadata_1b_step{step}k",
        "color": "#74c476",
        "marker": "X",
        "linestyle": (0, (7, 2, 1.2, 2)),
        "markerfacecolor": "white",
    },
}


def _lookup(df, model_path, test_path):
    row = df[(df["model_path"] == model_path) & (df["test_set_path"] == test_path)]
    if row.empty:
        return np.nan, np.nan, np.nan
    r = row.iloc[0]
    return float(r["mean_ppl"]), float(r["ci_low"]), float(r["ci_high"])


def _series_for_test_path(df, test_path, keys):
    series = {}
    for key in keys:
        cfg = MODEL_GROUPS[key]
        y_vals, lo_vals, hi_vals = [], [], []
        for step in STEPS:
            model_path = cfg["final"] if step == 10000 else cfg["steps"].format(step=step // 1000)
            m, lo, hi = _lookup(df, model_path, test_path)
            y_vals.append(m)
            lo_vals.append(lo)
            hi_vals.append(hi)
        series[key] = (np.array(y_vals), np.array(lo_vals), np.array(hi_vals))
    return series


def _series_for_own_average(df, keys):
    own_test_paths = [path for _, path in TESTS[:5]]
    series = {}
    for key in keys:
        cfg = MODEL_GROUPS[key]
        y_vals, lo_vals, hi_vals = [], [], []
        for step in STEPS:
            model_path = cfg["final"] if step == 10000 else cfg["steps"].format(step=step // 1000)
            rows = df[(df["model_path"] == model_path) & (df["test_set_path"].isin(own_test_paths))]
            if rows.empty:
                y_vals.append(np.nan); lo_vals.append(np.nan); hi_vals.append(np.nan)
            else:
                y_vals.append(float(rows["mean_ppl"].mean()))
                lo_vals.append(float(rows["ci_low"].mean()))
                hi_vals.append(float(rows["ci_high"].mean()))
        series[key] = (np.array(y_vals), np.array(lo_vals), np.array(hi_vals))
    return series


def _style_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.28)
    ax.set_axisbelow(True)


def _draw_panel(ax, title, series_map, keys):
    draw_order = [k for k in keys if k not in {"combined_with", "combined_without"}] + ["combined_with", "combined_without"]
    x_vals = np.arange(len(STEP_LABELS), dtype=float)
    values = []
    for key in draw_order:
        cfg = MODEL_GROUPS[key]
        y_vals, lo_vals, hi_vals = series_map[key]
        mask = ~np.isnan(y_vals)
        if not np.any(mask):
            continue
        is_global = key in {"combined_with", "combined_without"}
        ax.plot(
            x_vals,
            y_vals,
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle=cfg["linestyle"],
            linewidth=2.15 if is_global else 1.75,
            markersize=5.7,
            markerfacecolor=cfg.get("markerfacecolor", cfg["color"]),
            markeredgecolor="black",
            markeredgewidth=0.85,
            label=cfg["label"],
            zorder=4 if is_global else 3,
        )
        band_mask = mask & ~np.isnan(lo_vals) & ~np.isnan(hi_vals)
        if np.any(band_mask):
            ax.fill_between(
                x_vals,
                lo_vals,
                hi_vals,
                color=cfg["color"],
                alpha=0.10 if is_global else 0.12,
                linewidth=0.3,
                zorder=1,
            )
        values.extend([v for v in y_vals if not np.isnan(v)])

    _style_axes(ax)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(STEP_LABELS, fontsize=9.0)
    ax.tick_params(axis="y", labelsize=9.0)
    bbox_props = dict(facecolor="#e2e2e2", edgecolor="black", linewidth=0.65, alpha=0.95, boxstyle="round,pad=0.24")
    ax.text(
        0.5,
        0.90,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        bbox=bbox_props,
        zorder=10,
        clip_on=False,
    )
    return values


def main():
    df = pd.read_csv(CSV_PATH)
    fig, axes = plt.subplots(1, 3, figsize=(6.95, 2.82), sharey=True)
    panels = [
        ("Avg. metadata-formatted\ntest sets", "own_average"),
        ("Test set\n[URL][Country][Continent] (I+)", TESTS[5][1]),
        ("Test set\nNo metadata (I-)", TESTS[6][1]),
    ]
    keys = ["url", "country_only", "continent_only", "combined_with", "combined_without"]

    all_values = []
    for ax, (title, payload) in zip(axes, panels):
        if payload == "own_average":
            series_map = _series_for_own_average(df, keys)
        else:
            series_map = _series_for_test_path(df, payload, keys)
        all_values.extend(_draw_panel(ax, title, series_map, keys))

    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        for ax in axes:
            ax.set_ylim(y_min - 0.18, y_max + 0.30)

    axes[0].set_ylabel("Perplexity (↓ better)", fontsize=10.0)
    fig.text(0.5, 0.045, "Training steps", ha="center", fontsize=10.0)

    legend_handles = [
        Line2D([], [], color=MODEL_GROUPS[key]["color"], marker=MODEL_GROUPS[key]["marker"],
               linestyle=MODEL_GROUPS[key]["linestyle"], linewidth=1.75, markersize=5.5,
               markerfacecolor=MODEL_GROUPS[key].get("markerfacecolor", MODEL_GROUPS[key]["color"]),
               markeredgecolor="black", markeredgewidth=0.8, label=MODEL_GROUPS[key]["label"])
        for key in keys
    ]
    fig.legend(
        handles=legend_handles,
        title="Models",
        frameon=True,
        fancybox=True,
        framealpha=0.93,
        edgecolor="black",
        fontsize=8.1,
        title_fontsize=8.3,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.985),
        borderpad=0.45,
        columnspacing=0.85,
        handletextpad=0.45,
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.70, bottom=0.18, left=0.075, right=0.995, wspace=0.11)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIG.parent.mkdir(parents=True, exist_ok=True)
    png_path = RESULTS_DIR / "perplexity_metadata_family_main_1b.png"
    pdf_path = RESULTS_DIR / "perplexity_metadata_family_main_1b.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(LATEX_FIG, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {LATEX_FIG}")


if __name__ == "__main__":
    main()
