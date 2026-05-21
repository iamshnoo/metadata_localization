#!/usr/bin/env python3
"""
Perplexity plots from metacul/results/perplexity_eval.csv.
"""

import os
import re
import json
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator

RESULTS_CSV = "/path/to/metacul/results/perplexity_eval.csv"
PLOTS_DIR = "/path/to/metacul/results/plots"


sns.set(font_scale=1.4)
sns.set_theme(style="whitegrid")
sns.set_context("paper", rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20})
plt.rcParams["hatch.linewidth"] = 0.8
plt.rcParams["hatch.color"] = "black"


def _parse_model_info(model_path):
    match = re.search(
        r"/models/(?P<continent>africa|america|asia|europe)_(?P<meta>with_metadata|without_metadata)_(?P<size>500m|1b|3b)$",
        model_path,
    )
    if not match:
        return None
    return match.groupdict()


def _parse_test_info(test_path):
    match = re.search(
        r"/training_data/meco_datasets/continents/(?P<continent>africa|america|asia|europe)/(?P<meta>with_metadata|without_metadata)/$",
        test_path,
    )
    if not match:
        return None
    return match.groupdict()


def _load_perplexity_df():
    df = pd.read_csv(RESULTS_CSV)
    df["mean_ppl"] = pd.to_numeric(df["mean_ppl"], errors="coerce")
    df["ci_low"] = pd.to_numeric(df["ci_low"], errors="coerce")
    df["ci_high"] = pd.to_numeric(df["ci_high"], errors="coerce")
    return df


def _subset_by_pairs(df, pairs):
    if not pairs:
        return df.head(0)
    pairs_df = pd.DataFrame(list(pairs), columns=["model_path", "test_set_path"])
    return df.merge(pairs_df, on=["model_path", "test_set_path"], how="inner")


def _write_plot_csv(output_dir, plot_index, df):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"plot_{plot_index}.csv")
    df.to_csv(output_path, index=False)


def _has_finite_values(*arrays):
    for arr in arrays:
        values = np.asarray(arr, dtype=float)
        if np.isfinite(values).any():
            return True
    return False


def _load_significance_map(path, key_fields):
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "p_value" not in df.columns:
        return {}
    sig = {}
    for _, row in df.iterrows():
        key = tuple(str(row.get(field, "")).lower() for field in key_fields)
        try:
            pval = float(row.get("p_value", "nan"))
        except Exception:
            pval = float("nan")
        sig[key] = bool(pval < 0.05)
    return sig


def plot_continent_models_metadata_effect():
    # Plot: continent models (500m, 1b) on their own continent, showing metadata effects.
    # Output: /path/to/metacul/results/plots/plot1/perplexity_continent_metadata_effect_{size}.pdf
    df = _load_perplexity_df()
    axis_label_fs = 22
    tick_fs = 19
    legend_fs = 18
    title_fs = 16
    continents = ["africa", "america", "asia", "europe"]
    size_order = ["500m", "1b"]
    meta_order = ["with_metadata", "without_metadata"]
    pairs = set()
    for cont in continents:
        for size in size_order:
            for model_meta in meta_order:
                for test_meta in meta_order:
                    model_path = (
                        f"/path/to/metacul/models/{cont}_{model_meta}_{size}"
                    )
                    test_path = (
                        "/path/to/metacul/training_data/meco_datasets/"
                        f"continents/{cont}/{test_meta}/"
                    )
                    pairs.add((model_path, test_path))

    records = []
    for _, row in df.iterrows():
        model_info = _parse_model_info(row["model_path"])
        test_info = _parse_test_info(row["test_set_path"])
        if not model_info or not test_info:
            continue
        if model_info["continent"] != test_info["continent"]:
            continue
        if pd.isna(row["mean_ppl"]):
            continue
        records.append(
            {
                "continent": model_info["continent"].capitalize(),
                "model_meta": model_info["meta"],
                "test_meta": test_info["meta"],
                "size": model_info["size"],
                "mean_ppl": row["mean_ppl"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
            }
        )

    if not records:
        print("No continent model records found for metadata effect plot.")
        return

    plot_df = pd.DataFrame(records)
    def _combo_label(model_meta, test_meta):
        train_tag = "T+" if model_meta == "with_metadata" else "T-"
        eval_tag = "I+" if test_meta == "with_metadata" else "I-"
        return f"Local({train_tag},{eval_tag})"

    plot_df["combo"] = [
        _combo_label(m, t) for m, t in zip(plot_df["model_meta"], plot_df["test_meta"])
    ]

    combo_order = [
        "Local(T+,I+)",
        "Local(T+,I-)",
        "Local(T-,I+)",
        "Local(T-,I-)",
    ]
    continents = ["Africa", "America", "Asia", "Europe"]

    width = 0.18
    gap = 0.06
    x = np.arange(len(continents))
    combo_colors = ["#9ad1a6", "#f4a3a3", "#fdd9b5", "#d9d9d9"]
    combo_hatches = ["\\", "*", "..", ""]

    for size in size_order:
        fig, ax = plt.subplots(figsize=(9, 6))
        subset = plot_df[plot_df["size"] == size]
        for i, combo in enumerate(combo_order):
            combo_subset = subset[subset["combo"] == combo]
            means = []
            yerr = []
            for cont in continents:
                row = combo_subset[combo_subset["continent"] == cont]
                if row.empty:
                    means.append(np.nan)
                    yerr.append((0.0, 0.0))
                    continue
                mean = row["mean_ppl"].values[0]
                ci_low = row["ci_low"].values[0]
                ci_high = row["ci_high"].values[0]
                means.append(mean)
                yerr.append((mean - ci_low, ci_high - mean))
            yerr = np.array(yerr).T
            offset = i * width + (gap if i >= 2 else 0.0)
            ax.bar(
                x + offset,
                means,
                width,
                yerr=yerr,
                capsize=3,
                label=combo,
                color=combo_colors[i],
                hatch=combo_hatches[i],
                edgecolor="none" if i in (1, 3) else "black",
                linewidth=0.6,
            )

        ax.set_title("")
        ax.set_xticks(x + (3 * width + gap) / 2)
        ax.set_xticklabels(continents, rotation=0, fontsize=tick_fs)
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.set_ylim(bottom=6)

        ax.tick_params(axis="y", labelsize=tick_fs)
        ax.set_ylabel("Perplexity (↓ better)", fontsize=axis_label_fs)
        bbox_props = dict(
            facecolor="lightgrey",
            edgecolor="grey",
            alpha=0.7,
            boxstyle="round",
            pad=0.3,
        )
        ax.set_title(
            "Local test sets",
            fontsize=title_fs,
            weight="bold",
            pad=6,
            y=0.70,
            bbox=bbox_props,
        )
        ax.legend(
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            edgecolor="black",
            fontsize=legend_fs,
            loc="upper right",
            ncol=2,
            bbox_to_anchor=(0.98, 0.99),
        )

        output_dir = os.path.join(PLOTS_DIR, "plot1")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"perplexity_continent_metadata_effect_{size}.pdf"
        )
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.06)
        plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    output_dir = os.path.join(PLOTS_DIR, "plot1")
    subset = _subset_by_pairs(df, pairs)
    _write_plot_csv(output_dir, 1, subset)


def _row_to_values(row):
    mean = float(row["mean_ppl"])
    ci_low = float(row["ci_low"])
    ci_high = float(row["ci_high"])
    return mean, ci_low, ci_high


def _lookup_with_ci(df, model_path, test_path):
    row = df[
        (df["model_path"] == model_path)
        & (df["test_set_path"] == test_path)
    ]
    if row.empty or pd.isna(row["mean_ppl"].values[0]):
        return np.nan, np.nan, np.nan
    return (
        float(row["mean_ppl"].values[0]),
        float(row["ci_low"].values[0]),
        float(row["ci_high"].values[0]),
    )


def _aggregate_rows(rows):
    means = []
    lows = []
    highs = []
    for row in rows:
        mean, ci_low, ci_high = _row_to_values(row)
        means.append(mean)
        lows.append(ci_low)
        highs.append(ci_high)
    if not means:
        return None
    mean_val = float(np.mean(means))
    ci_low = float(np.mean(lows))
    ci_high = float(np.mean(highs))
    return mean_val, ci_low, ci_high


def plot_local_vs_global_on_local_and_global():
    # Plot: local vs global models on local test sets + global test set.
    # Output: /path/to/metacul/results/plots/plot2/perplexity_local_vs_global_{size}.pdf
    df = _load_perplexity_df()
    pairs = set()
    axis_label_fs = 22
    tick_fs = 19
    legend_fs = 18
    title_fs = 16

    continents = ["africa", "america", "asia", "europe"]
    region_labels = ["Africa", "America", "Asia", "Europe", "All"]
    size_order = ["500m", "1b"]

    combos = [
        {"label": "Global(T+,I+)", "scope": "global", "meta": "with_metadata"},
        {"label": "Local(T+,I+)", "scope": "local", "meta": "with_metadata"},
        {"label": "Global(T-,I-)", "scope": "global", "meta": "without_metadata"},
        {"label": "Local(T-,I-)", "scope": "local", "meta": "without_metadata"},
    ]

    records = []
    for size in size_order:
        for cont in continents:
            for combo in combos:
                meta = combo["meta"]
                if combo["scope"] == "local":
                    model_path = f"/path/to/metacul/models/{cont}_{meta}_{size}"
                else:
                    model_path = f"/path/to/metacul/models/combined_{meta}_{size}"
                test_path = (
                    f"/path/to/metacul/training_data/meco_datasets/continents/{cont}/{meta}/"
                )
                pairs.add((model_path, test_path))
                row = df[
                    (df["model_path"] == model_path)
                    & (df["test_set_path"] == test_path)
                ]
                if row.empty or pd.isna(row["mean_ppl"].values[0]):
                    continue
                mean, ci_low, ci_high = _row_to_values(row.iloc[0])
                records.append(
                    {
                        "region": cont.capitalize(),
                        "combo": combo["label"],
                        "size": size,
                        "mean_ppl": mean,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )

        for combo in combos:
            meta = combo["meta"]
            test_path = (
                f"/path/to/metacul/training_data/meco_datasets/combined/{meta}/"
            )
            if combo["scope"] == "global":
                model_path = f"/path/to/metacul/models/combined_{meta}_{size}"
                pairs.add((model_path, test_path))
                row = df[
                    (df["model_path"] == model_path)
                    & (df["test_set_path"] == test_path)
                ]
                if row.empty or pd.isna(row["mean_ppl"].values[0]):
                    continue
                mean, ci_low, ci_high = _row_to_values(row.iloc[0])
            else:
                local_rows = []
                for cont in continents:
                    model_path = (
                        f"/path/to/metacul/models/{cont}_{meta}_{size}"
                    )
                    pairs.add((model_path, test_path))
                    row = df[
                        (df["model_path"] == model_path)
                        & (df["test_set_path"] == test_path)
                    ]
                    if row.empty or pd.isna(row["mean_ppl"].values[0]):
                        continue
                    local_rows.append(row.iloc[0])
                aggregated = _aggregate_rows(local_rows)
                if aggregated is None:
                    continue
                mean, ci_low, ci_high = aggregated

            records.append(
                {
                    "region": "All",
                    "combo": combo["label"],
                    "size": size,
                    "mean_ppl": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    if not records:
        print("No records found for local vs global plot.")
        return

    plot_df = pd.DataFrame(records)

    width = 0.18
    gap = 0.06
    x = np.arange(len(region_labels))
    combo_styles = {
        "Global(T+,I+)": {"color": "#a6cee3", "hatch": "o"},
        "Global(T-,I-)": {"color": "#7f7f7f", "hatch": ""},
        "Local(T+,I+)": {"color": "#9ad1a6", "hatch": "\\"},
        "Local(T-,I-)": {"color": "#d9d9d9", "hatch": ""},
    }
    plot_order = [
        "Global(T+,I+)",
        "Global(T-,I-)",
        "Local(T+,I+)",
        "Local(T-,I-)",
    ]

    def _plot_panel(ax, subset, regions, labels):
        x_pos = np.arange(len(labels))
        for i, combo in enumerate(plot_order):
            combo_subset = subset[subset["combo"] == combo]
            means = []
            yerr = []
            for region in regions:
                row = combo_subset[combo_subset["region"] == region]
                if row.empty:
                    means.append(np.nan)
                    yerr.append((0.0, 0.0))
                    continue
                mean = row["mean_ppl"].values[0]
                ci_low = row["ci_low"].values[0]
                ci_high = row["ci_high"].values[0]
                means.append(mean)
                yerr.append((mean - ci_low, ci_high - mean))
            yerr = np.array(yerr).T
            style = combo_styles[combo]
            offset = i * width + (gap if i >= 2 else 0.0)
            ax.bar(
                x_pos + offset,
                means,
                width,
                yerr=yerr,
                capsize=3,
                label=combo,
                color=style["color"],
                hatch=style["hatch"],
                edgecolor="none" if combo in ("Global(T-,I-)", "Local(T-,I-)") else "black",
                linewidth=0.6,
            )

        ax.set_xticks(x_pos + (3 * width + gap) / 2)
        ax.set_xticklabels(labels, rotation=0, fontsize=tick_fs)
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.set_ylim(bottom=6)
        ax.tick_params(axis="y", labelsize=tick_fs)
        ax.tick_params(axis="x", labelsize=tick_fs)

    for size in size_order:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 6),
            sharey=True,
            gridspec_kw={"width_ratios": [4, 1], "wspace": 0.15},
        )
        subset = plot_df[plot_df["size"] == size]
        global_max = subset["mean_ppl"].max()
        y_top = global_max + 1.0 if not np.isnan(global_max) else None
        _plot_panel(axes[0], subset, region_labels[:-1], region_labels[:-1])
        _plot_panel(axes[1], subset, ["All"], ["All"])

        bbox_props = dict(
            facecolor="lightgrey",
            edgecolor="grey",
            alpha=0.7,
            boxstyle="round",
            pad=0.3,
        )
        axes[0].set_title(
            "Local test sets",
            fontsize=title_fs,
            weight="bold",
            pad=6,
            y=0.62,
            bbox=bbox_props,
        )
        axes[1].set_title(
            "Global test set",
            fontsize=title_fs,
            weight="bold",
            pad=6,
            y=0.62,
            bbox=bbox_props,
        )
        axes[0].set_ylabel("Perplexity (↓ better)", fontsize=axis_label_fs)
        axes[1].set_ylabel("")
        if y_top is not None:
            axes[0].set_ylim(top=y_top)
            axes[1].set_ylim(top=y_top)
        axes[0].legend(
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            edgecolor="black",
            fontsize=legend_fs,
            loc="upper right",
            ncol=2,
            bbox_to_anchor=(0.98, 0.99),
        )

        output_dir = os.path.join(PLOTS_DIR, "plot2")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"perplexity_local_vs_global_{size}.pdf"
        )
        plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)

    output_dir = os.path.join(PLOTS_DIR, "plot2")
    subset = _subset_by_pairs(df, pairs)
    _write_plot_csv(output_dir, 2, subset)


def plot_scaling_global_models():
    # Plot: scaling impact for global models as a dumbbell with delta annotations.
    # Output: /path/to/metacul/results/plots/plot3/perplexity_scaling_global_delta.pdf
    df = _load_perplexity_df()
    pairs = set()
    axis_label_fs = 22
    tick_fs = 19
    legend_fs = 18
    title_fs = 16
    delta_fs = 19

    regions = ["africa", "america", "asia", "europe", "combined"]
    region_labels = ["Africa", "America", "Asia", "Europe", "All"]
    meta_order = ["with_metadata", "without_metadata"]

    size_order = ["500m", "1b", "3b"]
    records = []
    for meta in meta_order:
        for size in size_order:
            model_path = f"/path/to/metacul/models/combined_{meta}_{size}"
            for region, label in zip(regions, region_labels):
                if region == "combined":
                    test_path = (
                        f"/path/to/metacul/training_data/meco_datasets/combined/{meta}/"
                    )
                else:
                    test_path = (
                        f"/path/to/metacul/training_data/meco_datasets/continents/{region}/{meta}/"
                    )
                pairs.add((model_path, test_path))
                row = df[
                    (df["model_path"] == model_path)
                    & (df["test_set_path"] == test_path)
                ]
                if row.empty or pd.isna(row["mean_ppl"].values[0]):
                    continue
                mean, ci_low, ci_high = _row_to_values(row.iloc[0])
                records.append(
                    {
                        "meta": meta,
                        "size": size,
                        "region": label,
                        "mean_ppl": mean,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )

    if not records:
        print("No records found for scaling plot.")
        return

    plot_df = pd.DataFrame(records)
    line_color = "#7f7f7f"
    colors = {"500m": "#f4a3a3", "1b": "#9ad1a6", "3b": "#8fb6ff"}
    markers = {"500m": "o", "1b": "s", "3b": "^"}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        sharey=True,
        gridspec_kw={"width_ratios": [4, 1], "wspace": 0.15},
    )
    meta_offsets = {"with_metadata": -0.15, "without_metadata": 0.15}
    meta_styles = {"with_metadata": "solid", "without_metadata": "dashed"}
    local_labels = region_labels[:-1]
    global_labels = ["All"]

    def _plot_panel(ax, labels):
        x_positions = np.arange(len(labels))
        if len(labels) == 1:
            offsets = {"with_metadata": -0.05, "without_metadata": 0.05}
        else:
            offsets = meta_offsets
        for meta in meta_order:
            subset = plot_df[plot_df["meta"] == meta]
            for idx, region in enumerate(labels):
                rows = []
                for size in size_order:
                    row = subset[
                        (subset["size"] == size) & (subset["region"] == region)
                    ]
                    if row.empty:
                        continue
                    rows.append((size, row.iloc[0]))
                if len(rows) < 2:
                    continue
                ys = [row["mean_ppl"] for _, row in rows]
                delta = ys[-1] - ys[0]

                x_pos = x_positions[idx] + offsets[meta]
                ax.plot(
                    [x_pos] * len(ys),
                    ys,
                    color=line_color,
                    linestyle=meta_styles[meta],
                    linewidth=1.5,
                )
                for size, row in rows:
                    ax.scatter(
                        [x_pos],
                        [row["mean_ppl"]],
                        color=colors[size],
                        edgecolor="black",
                        marker=markers[size],
                        s=70,
                        zorder=3,
                        label=f"{size} ({size})" if idx == 0 and meta == "with_metadata" else None,
                    )
                ax.text(
                    x_pos,
                    min(ys) - 0.2,
                    f"Δ {delta:+.2f}",
                    va="top",
                    ha="center",
                    fontsize=delta_fs,
                    color="#444444",
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, fontsize=tick_fs)
        ax.set_xlabel("")
        ax.set_ylim(bottom=6)
        ax.tick_params(axis="y", labelsize=tick_fs)
        ax.tick_params(axis="x", labelsize=tick_fs)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    _plot_panel(axes[0], local_labels)
    _plot_panel(axes[1], global_labels)
    axes[0].set_ylabel("Perplexity (↓ better)", fontsize=axis_label_fs)
    axes[1].set_ylabel("")
    y_top = plot_df["mean_ppl"].max()
    if not np.isnan(y_top):
        axes[0].set_ylim(top=y_top + 1.0)
        axes[1].set_ylim(top=y_top + 1.0)
    bbox_props = dict(
        facecolor="lightgrey",
        edgecolor="grey",
        alpha=0.7,
        boxstyle="round",
        pad=0.3,
    )
    axes[0].text(
        0.5,
        0.92,
        "Local test sets",
        transform=axes[0].transAxes,
        ha="center",
        va="bottom",
        fontsize=title_fs,
        weight="bold",
        bbox=bbox_props,
    )
    axes[1].text(
        0.5,
        0.92,
        "Global test set",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=title_fs,
        weight="bold",
        bbox=bbox_props,
    )
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=markers["500m"],
            color="none",
            markerfacecolor=colors["500m"],
            markeredgecolor="black",
            label="500m",
            markersize=8,
        ),
        plt.Line2D(
            [0],
            [0],
            marker=markers["1b"],
            color="none",
            markerfacecolor=colors["1b"],
            markeredgecolor="black",
            label="1b",
            markersize=8,
        ),
        plt.Line2D(
            [0],
            [0],
            marker=markers["3b"],
            color="none",
            markerfacecolor=colors["3b"],
            markeredgecolor="black",
            label="3b",
            markersize=8,
        ),
        plt.Line2D(
            [0],
            [0],
            color=line_color,
            linestyle="solid",
            label="Global(T+,I+)",
        ),
        plt.Line2D(
            [0],
            [0],
            color=line_color,
            linestyle="dashed",
            label="Global(T-,I-)",
        ),
    ]
    axes[0].legend(
        handles=legend_handles,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="black",
        fontsize=legend_fs,
        loc="lower right",
        ncol=2,
        bbox_to_anchor=(0.98, 0.01),
    )

    output_dir = os.path.join(PLOTS_DIR, "plot3")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "perplexity_scaling_global_delta.pdf")
    plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    subset = _subset_by_pairs(df, pairs)
    _write_plot_csv(output_dir, 3, subset)


def plot_cross_continent_generalization():
    # Plot: cross-continent generalization (local models/tests), with deltas.
    # Output: /path/to/metacul/results/plots/plot4/perplexity_cross_continent_{size}.pdf
    df = _load_perplexity_df()
    pairs = set()
    sig_plot4 = _load_significance_map(
        "/path/to/metacul/results/significance/plot4.csv",
        ["size", "train_continent", "test_continent", "test_meta"],
    )
    sig_plot4_avg = _load_significance_map(
        "/path/to/metacul/results/significance/plot4_avg_by_test.csv",
        ["size", "test_continent", "test_meta"],
    )

    continents = ["africa", "america", "asia", "europe"]
    labels = ["Africa", "America", "Asia", "Europe"]
    size_order = ["500m", "1b"]
    annot_fs = 17
    title_fs = 16
    axis_label_fs = 18
    tick_fs = 15
    cbar_fs = 12
    bar_label_fs = 18
    bar_tick_fs = 14
    bar_value_fs = 15

    def _matrix(model_meta, test_meta, size):
        values = []
        for train_cont in continents:
            row_vals = []
            model_path = f"/path/to/metacul/models/{train_cont}_{model_meta}_{size}"
            for test_cont in continents:
                test_path = (
                    f"/path/to/metacul/training_data/meco_datasets/continents/{test_cont}/{test_meta}/"
                )
                pairs.add((model_path, test_path))
                row = df[
                    (df["model_path"] == model_path)
                    & (df["test_set_path"] == test_path)
                ]
                if row.empty or pd.isna(row["mean_ppl"].values[0]):
                    row_vals.append(np.nan)
                else:
                    row_vals.append(float(row["mean_ppl"].values[0]))
            values.append(row_vals)
        return pd.DataFrame(values, index=labels, columns=labels)

    for size in size_order:
        with_with = _matrix("with_metadata", "with_metadata", size)
        without_with = _matrix("without_metadata", "with_metadata", size)
        with_without = _matrix("with_metadata", "without_metadata", size)
        without_without = _matrix("without_metadata", "without_metadata", size)

        if not _has_finite_values(
            with_with.values,
            without_with.values,
            with_without.values,
            without_without.values,
        ):
            print(f"No cross-continent records found for size={size}; skipping plot4.")
            continue

        delta_with = without_with - with_with
        delta_without = without_without - with_without

        avg_delta_with = delta_with.mean(axis=0)
        avg_delta_without = delta_without.mean(axis=0)

        value_min = np.nanmin([with_with.values, without_with.values, with_without.values, without_without.values])
        value_max = np.nanmax([with_with.values, without_with.values, with_without.values, without_without.values])
        if not np.isfinite(value_min) or not np.isfinite(value_max):
            print(f"Non-finite cross-continent value range for size={size}; skipping plot4.")
            continue
        if value_min == value_max:
            value_min -= 1e-6
            value_max += 1e-6

        if _has_finite_values(delta_with.values, delta_without.values):
            delta_max = np.nanmax([np.abs(delta_with.values), np.abs(delta_without.values)])
            if not np.isfinite(delta_max) or delta_max == 0:
                delta_max = 1e-6
        else:
            delta_max = 1e-6

        if _has_finite_values(avg_delta_with.values, avg_delta_without.values):
            bar_min = float(np.nanmin([avg_delta_with.values, avg_delta_without.values]))
            bar_max = float(np.nanmax([avg_delta_with.values, avg_delta_without.values]))
            if not np.isfinite(bar_min) or not np.isfinite(bar_max):
                bar_min, bar_max = -1.0, 1.0
            elif bar_min == bar_max:
                bar_min -= 0.5
                bar_max += 0.5
        else:
            bar_min, bar_max = -1.0, 1.0

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        bbox_props = dict(
            facecolor="lightgrey",
            edgecolor="grey",
            alpha=0.7,
            boxstyle="round",
            pad=0.3,
        )

        heatmap_kws = dict(
            annot=False,
            fmt=".2f",
            linewidths=0.8,
            linecolor="black",
            square=True,
            cbar=True,
            cbar_kws={"shrink": 0.8},
        )

        def _annotate_heatmap(ax, data, cmap, norm, sig_mask=None):
            values = data.values
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    val = values[i, j]
                    if np.isnan(val):
                        continue
                    r, g, b, _ = cmap(norm(val))
                    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    text_color = "white" if luminance < 0.5 else "black"
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=annot_fs,
                        color=text_color,
                        fontweight=(
                            "bold"
                            if sig_mask is not None and sig_mask[i, j]
                            else "normal"
                        ),
                    )
                    if sig_mask is not None and sig_mask[i, j]:
                        ax.text(
                            j + 0.82,
                            i + 0.18,
                            "★",
                            ha="center",
                            va="center",
                            fontsize=14,
                            color="black",
                        )

        value_norm = colors.Normalize(vmin=value_min, vmax=value_max)
        delta_norm = colors.TwoSlopeNorm(vmin=-delta_max, vcenter=0, vmax=delta_max)

        sns.heatmap(
            with_with,
            ax=axes[0, 0],
            cmap="Greens",
            norm=value_norm,
            **heatmap_kws,
        )
        axes[0, 0].collections[0].colorbar.ax.locator = MaxNLocator(4)
        axes[0, 0].collections[0].colorbar.update_ticks()
        axes[0, 0].collections[0].colorbar.ax.tick_params(labelsize=cbar_fs)
        _annotate_heatmap(axes[0, 0], with_with, plt.get_cmap("Greens"), value_norm)
        axes[0, 0].set_title(
            "Local(T+,I+)",
            fontsize=title_fs,
            weight="bold",
            pad=12,
            bbox=bbox_props,
        )
        axes[0, 0].set_xlabel("")
        axes[0, 0].set_ylabel("Train Region", fontsize=axis_label_fs)

        sns.heatmap(
            without_with,
            ax=axes[0, 1],
            cmap="YlOrBr",
            norm=value_norm,
            **heatmap_kws,
        )
        axes[0, 1].collections[0].colorbar.ax.locator = MaxNLocator(4)
        axes[0, 1].collections[0].colorbar.update_ticks()
        axes[0, 1].collections[0].colorbar.ax.tick_params(labelsize=cbar_fs)
        _annotate_heatmap(axes[0, 1], without_with, plt.get_cmap("YlOrBr"), value_norm)
        axes[0, 1].set_title(
            "Local(T-,I+)",
            fontsize=title_fs,
            weight="bold",
            pad=12,
            bbox=bbox_props,
        )
        axes[0, 1].set_xlabel("")
        axes[0, 1].set_ylabel("")
        axes[0, 1].tick_params(axis="x", labelsize=15)

        sns.heatmap(
            delta_with,
            ax=axes[0, 2],
            cmap="vlag",
            norm=delta_norm,
            **heatmap_kws,
        )
        axes[0, 2].collections[0].colorbar.ax.locator = MaxNLocator(4)
        axes[0, 2].collections[0].colorbar.update_ticks()
        axes[0, 2].collections[0].colorbar.ax.tick_params(labelsize=cbar_fs)
        sig_mask_with = np.zeros_like(delta_with.values, dtype=bool)
        for i, train in enumerate(continents):
            for j, test in enumerate(continents):
                key = (size.lower(), train, test, "with_metadata")
                sig_mask_with[i, j] = sig_plot4.get(key, False)
        _annotate_heatmap(
            axes[0, 2], delta_with, plt.get_cmap("vlag"), delta_norm, sig_mask_with
        )
        axes[0, 2].set_title(
            "Δ: Local(T-,I+) − Local(T+,I+)",
            fontsize=title_fs,
            weight="bold",
            pad=12,
            bbox=bbox_props,
        )
        axes[0, 2].set_xlabel("")
        axes[0, 2].set_ylabel("")

        axes[0, 3].set_title(
            "Avg Δ by test region (I+)",
            fontsize=title_fs,
            weight="bold",
            pad=8,
            y=0.90,
            bbox=bbox_props,
        )
        axes[0, 3].set_xlabel("")
        axes[0, 3].set_ylabel("")
        axes[0, 3].set_xticks([])
        axes[0, 3].set_yticks([])
        axes[0, 3].set_facecolor("none")
        for spine in axes[0, 3].spines.values():
            spine.set_visible(False)
        bar_ax_top = axes[0, 3].inset_axes([0.12, 0.08, 0.82, 0.7])
        bar_x = np.arange(len(labels))
        top_vals = avg_delta_with.values
        top_colors = ["#9ad1a6" if v >= 0 else "#f4a3a3" for v in top_vals]
        bar_ax_top.bar(bar_x, top_vals, color=top_colors, edgecolor="black")
        bar_ax_top.axhline(0, color="black", linewidth=0.8)
        bar_ax_top.set_ylim(bar_min - 0.5, bar_max + 0.5)
        bar_ax_top.set_xlabel("")
        bar_ax_top.set_ylabel("Avg Δ (↑ better)", fontsize=bar_label_fs)
        bar_ax_top.set_xticks(bar_x)
        bar_ax_top.set_xticklabels(labels, fontsize=bar_tick_fs)
        bar_ax_top.tick_params(axis="x", labelsize=bar_tick_fs)
        bar_ax_top.tick_params(axis="y", labelsize=bar_value_fs)
        for i, v in enumerate(top_vals):
            key = (size.lower(), labels[i].lower(), "with_metadata")
            is_sig = sig_plot4_avg.get(key, False)
            bar_ax_top.text(
                i,
                v + (0.05 if v >= 0 else -0.05),
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=bar_value_fs,
                fontweight="bold" if is_sig else "normal",
            )
            if is_sig:
                bar_ax_top.text(
                    i,
                    v + (0.28 if v >= 0 else -0.28),
                    "★",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=14,
                    color="black",
                )

        sns.heatmap(
            with_without,
            ax=axes[1, 0],
            cmap="OrRd",
            norm=value_norm,
            **heatmap_kws,
        )
        axes[1, 0].collections[0].colorbar.ax.locator = MaxNLocator(4)
        axes[1, 0].collections[0].colorbar.update_ticks()
        axes[1, 0].collections[0].colorbar.ax.tick_params(labelsize=cbar_fs)
        _annotate_heatmap(axes[1, 0], with_without, plt.get_cmap("OrRd"), value_norm)
        axes[1, 0].set_title(
            "Local(T+,I-)",
            fontsize=title_fs,
            weight="bold",
            pad=12,
            bbox=bbox_props,
        )
        axes[1, 0].set_xlabel("Test Region", fontsize=axis_label_fs)
        axes[1, 0].set_ylabel("Train Region", fontsize=axis_label_fs)

        sns.heatmap(
            without_without,
            ax=axes[1, 1],
            cmap="Greys",
            norm=value_norm,
            **heatmap_kws,
        )
        axes[1, 1].collections[0].colorbar.ax.locator = MaxNLocator(4)
        axes[1, 1].collections[0].colorbar.update_ticks()
        axes[1, 1].collections[0].colorbar.ax.tick_params(labelsize=cbar_fs)
        _annotate_heatmap(axes[1, 1], without_without, plt.get_cmap("Greys"), value_norm)
        axes[1, 1].set_title(
            "Local(T-,I-)",
            fontsize=title_fs,
            weight="bold",
            pad=12,
            bbox=bbox_props,
        )
        axes[1, 1].set_xlabel("Test Region", fontsize=axis_label_fs)
        axes[1, 1].set_ylabel("")
        axes[1, 1].tick_params(axis="x", labelsize=15)

        sns.heatmap(
            delta_without,
            ax=axes[1, 2],
            cmap="vlag",
            norm=delta_norm,
            **heatmap_kws,
        )
        axes[1, 2].collections[0].colorbar.ax.locator = MaxNLocator(4)
        axes[1, 2].collections[0].colorbar.update_ticks()
        axes[1, 2].collections[0].colorbar.ax.tick_params(labelsize=cbar_fs)
        sig_mask_without = np.zeros_like(delta_without.values, dtype=bool)
        for i, train in enumerate(continents):
            for j, test in enumerate(continents):
                key = (size.lower(), train, test, "without_metadata")
                sig_mask_without[i, j] = sig_plot4.get(key, False)
        _annotate_heatmap(
            axes[1, 2], delta_without, plt.get_cmap("vlag"), delta_norm, sig_mask_without
        )
        axes[1, 2].set_title(
            "Δ: Local(T-,I-) − Local(T+,I-)",
            fontsize=title_fs,
            weight="bold",
            pad=12,
            bbox=bbox_props,
        )
        axes[1, 2].set_xlabel("Test Region", fontsize=axis_label_fs)
        axes[1, 2].set_ylabel("")

        axes[1, 3].set_title(
            "Avg Δ by test region (I-)",
            fontsize=title_fs,
            weight="bold",
            pad=8,
            y=0.90,
            bbox=bbox_props,
        )
        axes[1, 3].set_xlabel("")
        axes[1, 3].set_ylabel("")
        axes[1, 3].set_xticks([])
        axes[1, 3].set_yticks([])
        axes[1, 3].set_facecolor("none")
        for spine in axes[1, 3].spines.values():
            spine.set_visible(False)
        bar_ax_bottom = axes[1, 3].inset_axes([0.12, 0.08, 0.82, 0.7])
        bottom_vals = avg_delta_without.values
        bottom_colors = ["#9ad1a6" if v >= 0 else "#f4a3a3" for v in bottom_vals]
        bar_ax_bottom.bar(bar_x, bottom_vals, color=bottom_colors, edgecolor="black")
        bar_ax_bottom.axhline(0, color="black", linewidth=0.8)
        bar_ax_bottom.set_ylim(bar_min - 0.5, bar_max + 0.5)
        bar_ax_bottom.set_xlabel("Test Region", fontsize=bar_label_fs)
        bar_ax_bottom.set_ylabel("Avg Δ (↑ better)", fontsize=bar_label_fs)
        bar_ax_bottom.set_xticks(bar_x)
        bar_ax_bottom.set_xticklabels(labels, fontsize=bar_tick_fs)
        bar_ax_bottom.tick_params(axis="x", labelsize=bar_tick_fs)
        bar_ax_bottom.tick_params(axis="y", labelsize=bar_value_fs)
        for i, v in enumerate(bottom_vals):
            key = (size.lower(), labels[i].lower(), "without_metadata")
            is_sig = sig_plot4_avg.get(key, False)
            bar_ax_bottom.text(
                i,
                v + (0.05 if v >= 0 else -0.05),
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=bar_value_fs,
                fontweight="bold" if is_sig else "normal",
            )
            if is_sig:
                bar_ax_bottom.text(
                    i,
                    v + (0.28 if v >= 0 else -0.28),
                    "★",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=14,
                    color="black",
                )

        for ax_row in axes:
            for ax in ax_row:
                ax.tick_params(axis="y", labelsize=tick_fs)
                ax.tick_params(axis="x", labelsize=tick_fs)

        output_dir = os.path.join(PLOTS_DIR, "plot4")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"perplexity_cross_continent_{size}.pdf"
        )
        plt.tight_layout()
        # Ensure text extents (e.g., axis labels) are available for tight bounding boxes.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        # Add row-level grouping labels to clarify what each row represents.
        row_labels = {
            0: "Evaluation with metadata (I+)",
            1: "Evaluation without metadata (I-)",
        }
        row_bounds = {}
        for row_idx in row_labels:
            row_axes = [axes[row_idx, c] for c in range(4)]
            # Include axis decorations (not just the raw axes rectangle) so the
            # "Train Region" y-label remains inside the row grouping box.
            tight_x0 = min(
                ax.get_tightbbox(renderer)
                .transformed(fig.transFigure.inverted())
                .x0
                for ax in row_axes
            )
            row_bounds[row_idx] = {
                "x0": min(min(ax.get_position().x0 for ax in row_axes), tight_x0),
                "x1": max(ax.get_position().x1 for ax in row_axes),
                "y0": min(ax.get_position().y0 for ax in row_axes),
                "y1": max(ax.get_position().y1 for ax in row_axes),
            }
        inter_row_gap = row_bounds[0]["y0"] - row_bounds[1]["y1"]
        # Keep padding small enough to prevent row-box overlap while
        # giving extra space below the second row for x-axis labels.
        base_pad_y = max(0.003, min(0.009, inter_row_gap * 0.32))
        for row_idx, row_label in row_labels.items():
            x0 = row_bounds[row_idx]["x0"]
            x1 = row_bounds[row_idx]["x1"]
            y0 = row_bounds[row_idx]["y0"]
            y1 = row_bounds[row_idx]["y1"]

            pad_x = 0.008
            pad_top = base_pad_y if row_idx == 0 else max(0.002, base_pad_y * 0.7)
            pad_bottom = (
                max(0.002, base_pad_y * 0.7)
                if row_idx == 0
                else base_pad_y + 0.008
            )
            row_box = FancyBboxPatch(
                (x0 - pad_x, y0 - pad_bottom),
                (x1 - x0) + 2 * pad_x,
                (y1 - y0) + pad_top + pad_bottom,
                boxstyle="round,pad=0.008,rounding_size=0.015",
                transform=fig.transFigure,
                fill=False,
                linewidth=1.1,
                edgecolor="#5a5a5a",
                zorder=0,
                clip_on=False,
            )
            fig.add_artist(row_box)
            row_box_left = x0 - pad_x
            fig.text(
                row_box_left - 0.022,
                (y0 + y1) / 2,
                row_label,
                rotation=90,
                va="center",
                ha="center",
                rotation_mode="anchor",
                fontsize=title_fs,
                fontweight="semibold",
                color="#333333",
                transform=fig.transFigure,
            )
        # Add lighter column-level grouping marks for training-time metadata condition.
        col_labels = {
            0: "Training with metadata",
            1: "Training without metadata",
        }
        all_row_y0 = min(row_bounds[r]["y0"] for r in row_bounds)
        all_row_y1 = max(row_bounds[r]["y1"] for r in row_bounds)
        for col_idx, col_label in col_labels.items():
            col_axes = [axes[r, col_idx] for r in range(2)]
            # Include the colorbar axes for heatmaps so scale ticks are inside the group box.
            colorbar_axes = []
            for ax in col_axes:
                if ax.collections:
                    cbar = getattr(ax.collections[0], "colorbar", None)
                    if cbar is not None and cbar.ax is not None:
                        colorbar_axes.append(cbar.ax)
            col_axes_all = col_axes + colorbar_axes
            cx0 = min(ax.get_position().x0 for ax in col_axes_all)
            cx1 = max(ax.get_position().x1 for ax in col_axes_all)
            cy1 = all_row_y1
            cpad_x = 0.003
            bracket_y = cy1 + 0.007
            tick_h = 0.006
            # Bracket line
            fig.add_artist(
                plt.Line2D(
                    [cx0 - cpad_x, cx1 + cpad_x],
                    [bracket_y, bracket_y],
                    transform=fig.transFigure,
                    color="#666666",
                    linewidth=1.0,
                    zorder=2,
                )
            )
            # Bracket end ticks
            fig.add_artist(
                plt.Line2D(
                    [cx0 - cpad_x, cx0 - cpad_x],
                    [bracket_y, bracket_y - tick_h],
                    transform=fig.transFigure,
                    color="#666666",
                    linewidth=1.0,
                    zorder=2,
                )
            )
            fig.add_artist(
                plt.Line2D(
                    [cx1 + cpad_x, cx1 + cpad_x],
                    [bracket_y, bracket_y - tick_h],
                    transform=fig.transFigure,
                    color="#666666",
                    linewidth=1.0,
                    zorder=2,
                )
            )
            fig.text(
                (cx0 + cx1) / 2,
                bracket_y + 0.009,
                col_label,
                va="bottom",
                ha="center",
                fontsize=title_fs,
                fontweight="semibold",
                color="#333333",
                transform=fig.transFigure,
            )
        plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    output_dir = os.path.join(PLOTS_DIR, "plot4")
    subset = _subset_by_pairs(df, pairs)
    _write_plot_csv(output_dir, 4, subset)


def plot_cross_continent_asymmetry():
    # Plot: cross-continent asymmetry for local models/tests (Local(T+,I+)).
    # Output: /path/to/metacul/results/plots/plot5/perplexity_asymmetry_{size}.pdf
    df = _load_perplexity_df()
    pairs = set()
    sig_plot5 = _load_significance_map(
        "/path/to/metacul/results/significance/plot5.csv",
        ["size", "train_continent", "test_continent"],
    )
    axis_label_fs = 22
    tick_fs = 19
    title_fs = 16
    cbar_fs = 12
    annot_fs = 17

    continents = ["africa", "america", "asia", "europe"]
    labels = ["Africa", "America", "Asia", "Europe"]
    size_order = ["500m", "1b"]

    def _matrix(size):
        values = []
        for train_cont in continents:
            row_vals = []
            model_path = f"/path/to/metacul/models/{train_cont}_with_metadata_{size}"
            for test_cont in continents:
                test_path = (
                    f"/path/to/metacul/training_data/meco_datasets/continents/{test_cont}/with_metadata/"
                )
                pairs.add((model_path, test_path))
                row = df[
                    (df["model_path"] == model_path)
                    & (df["test_set_path"] == test_path)
                ]
                if row.empty or pd.isna(row["mean_ppl"].values[0]):
                    row_vals.append(np.nan)
                else:
                    row_vals.append(float(row["mean_ppl"].values[0]))
            values.append(row_vals)
        return pd.DataFrame(values, index=labels, columns=labels)

    for size in size_order:
        base = _matrix(size)
        if not _has_finite_values(base.values):
            print(f"No asymmetry records found for size={size}; skipping plot5.")
            continue
        asym = base - base.T
        asym_values = asym.values
        asym_max = np.nanmax(np.abs(asym_values))
        if not np.isfinite(asym_max) or asym_max == 0:
            asym_max = 1e-6

        fig, ax = plt.subplots(figsize=(8, 6))
        bbox_props = dict(
            facecolor="lightgrey",
            edgecolor="grey",
            alpha=0.7,
            boxstyle="round",
            pad=0.3,
        )

        heatmap_kws = dict(
            annot=False,
            fmt=".2f",
            linewidths=0.8,
            linecolor="black",
            square=True,
            cbar=True,
            cbar_kws={"shrink": 0.85},
        )

        asym_norm = colors.TwoSlopeNorm(vmin=-asym_max, vcenter=0, vmax=asym_max)
        sns.heatmap(
            asym,
            ax=ax,
            cmap="vlag",
            norm=asym_norm,
            **heatmap_kws,
        )
        ax.collections[0].colorbar.ax.locator = MaxNLocator(4)
        ax.collections[0].colorbar.update_ticks()
        ax.collections[0].colorbar.ax.tick_params(labelsize=cbar_fs)

        for i in range(asym_values.shape[0]):
            for j in range(asym_values.shape[1]):
                val = asym_values[i, j]
                if np.isnan(val):
                    continue
                r, g, b, _ = plt.get_cmap("vlag")(asym_norm(val))
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=annot_fs,
                    color=text_color,
                )
                key = (size.lower(), continents[i], continents[j])
                if sig_plot5.get(key, False):
                    ax.text(
                        j + 0.82,
                        i + 0.18,
                        "★",
                        ha="center",
                        va="center",
                        fontsize=14,
                        color="black",
                    )

        ax.set_title(
            "Asymmetry (ppl(i→j) − ppl(j→i))",
            fontsize=title_fs,
            weight="bold",
            pad=12,
            bbox=bbox_props,
        )
        ax.set_xlabel("Test Region", fontsize=axis_label_fs)
        ax.set_ylabel("Train Region", fontsize=axis_label_fs)
        ax.tick_params(axis="x", labelsize=tick_fs)
        ax.tick_params(axis="y", labelsize=tick_fs)

        output_dir = os.path.join(PLOTS_DIR, "plot5")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"perplexity_asymmetry_{size}.pdf")
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.09)
        plt.close(fig)

    output_dir = os.path.join(PLOTS_DIR, "plot5")
    subset = _subset_by_pairs(df, pairs)
    _write_plot_csv(output_dir, 5, subset)


def plot_sft_accuracy_apples_to_apples():
    # Plot: apples-to-apples QA accuracy (answered_by_all=1) across continents.
    # Output: /path/to/metacul/results/plots/plot8/accuracy_apples_to_apples.pdf
    axis_label_fs = 22
    tick_fs = 19
    legend_fs = 18
    title_fs = 16
    value_fs = 15
    df = pd.read_csv("/path/to/metacul/results/qa_metacul_eval.csv")
    # If corruption rates are present, use only the clean (c0) rows for plot8.
    if "url_corruption_rate" in df.columns:
        df = df[df["url_corruption_rate"] == 0.0].copy()

    # If metadata-only runs were used, backfill missing variants from the original eval CSV.
    original_path = "/path/to/metacul/results/qa_metacul_eval_original.csv"
    if os.path.exists(original_path):
        df_orig = pd.read_csv(original_path)
        missing_cols = [
            col
            for col in [
                "custom_without_metadata_correct",
                "custom_without_metadata_incorrect",
                "custom_without_metadata_skipped",
                "llama3_chat_without_metadata_correct",
                "llama3_chat_without_metadata_incorrect",
                "llama3_chat_without_metadata_skipped",
            ]
            if col not in df.columns and col in df_orig.columns
        ]
        if missing_cols:
            merge_keys = [
                "question_id",
                "country",
                "continent",
                "generated_by",
                "base_url",
            ]
            df = df.merge(
                df_orig[merge_keys + missing_cols],
                on=merge_keys,
                how="left",
            )

    df = df[df["answered_by_all"] == 1]
    df["continent"] = df["continent"].str.capitalize()

    models = {
        "Global(T-,I-)": {
            "correct": "custom_without_metadata_correct",
            "incorrect": "custom_without_metadata_incorrect",
            "color": "#f7a1b5",
            "hatch": "",
        },
        "Global(T+,I+)": {
            "correct": "custom_with_metadata_correct",
            "incorrect": "custom_with_metadata_incorrect",
            "color": "#a6cee3",
            "hatch": "o",
        },
        "LLaMA-3.2(I-)": {
            "correct": "llama3_chat_without_metadata_correct",
            "incorrect": "llama3_chat_without_metadata_incorrect",
            "color": "#d9d9d9",
            "hatch": "",
        },
        "LLaMA-3.2(I+)": {
            "correct": "llama3_chat_with_metadata_correct",
            "incorrect": "llama3_chat_with_metadata_incorrect",
            "color": "#7f7f7f",
            "hatch": "..",
        },
    }
    # Drop model variants that are not present in the CSV (e.g., metadata-only runs).
    available_cols = set(df.columns)
    models = {
        name: cfg
        for name, cfg in models.items()
        if cfg["correct"] in available_cols and cfg["incorrect"] in available_cols
    }
    if not models:
        print("[!] No SFT variants found in qa_metacul_eval.csv for plot8.")
        return

    continents = ["Africa", "Europe", "Asia", "America"]

    records = []
    for model, cfg in models.items():
        for (url, cont), sub in df.groupby(["base_url", "continent"]):
            correct = sub[cfg["correct"]].sum()
            incorrect = (sub[cfg["incorrect"]] == -1).sum()
            denom = correct + incorrect
            acc = correct / denom if denom > 0 else np.nan

            records.append(
                {
                    "model": model,
                    "continent": cont,
                    "base_url": url,
                    "accuracy": acc,
                }
            )

    long = pd.DataFrame(records)
    summary = (
        long.groupby(["model", "continent"])["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["count"])

    order = [
        name
        for name in [
            "Global(T+,I+)",
            "Global(T-,I-)",
            "LLaMA-3.2(I+)",
            "LLaMA-3.2(I-)",
        ]
        if name in models
    ]

    cont_question_counts = df.groupby("continent")["question_id"].nunique().to_dict()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        sharey=True,
        gridspec_kw={"width_ratios": [4, 1], "wspace": 0.15},
    )
    ax = axes[0]
    ax_summary = axes[1]
    width = 0.18
    gap = 0.06
    x = np.arange(len(continents))

    for i, model in enumerate(order):
        subset = summary[summary["model"] == model]
        means = [subset[subset["continent"] == c]["mean"].values[0] for c in continents]
        cis = [subset[subset["continent"] == c]["ci95"].values[0] for c in continents]

        offset = i * width + (gap if i >= 2 else 0.0)
        ax.bar(
            x + offset,
            means,
            width,
            yerr=cis,
            capsize=3,
            label=model,
            color=models[model]["color"],
            hatch=models[model]["hatch"],
            edgecolor="none" if model in ("Global(T-,I-)", "LLaMA-3.2(I-)") else "black",
            linewidth=0.6,
        )

    ax.set_xticks(x + (3 * width + gap) / 2)
    weight_labels = [
        f"{c}\n({cont_question_counts.get(c, 0)} Q/s)" for c in continents
    ]
    ax.set_xticklabels(weight_labels, fontsize=tick_fs)
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (↑ better)", fontsize=axis_label_fs)
    ax.set_ylim(0.55, 0.80)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.legend(
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="black",
        fontsize=legend_fs,
        loc="upper left",
        ncol=2,
        bbox_to_anchor=(0.02, 1.14),
    )

    micro_models = [m for m in ["Global(T+,I+)", "LLaMA-3.2(I-)"] if m in models]
    micro_records = []
    for model in micro_models:
        cfg = models[model]
        per_url_weighted = []
        for url, sub in df.groupby("base_url"):
            cont_acc = {}
            for cont, cont_sub in sub.groupby("continent"):
                correct = cont_sub[cfg["correct"]].sum()
                incorrect = (cont_sub[cfg["incorrect"]] == -1).sum()
                denom = correct + incorrect
                acc = correct / denom if denom > 0 else np.nan
                cont_acc[cont] = acc
            weights = []
            values = []
            for cont, acc in cont_acc.items():
                if np.isnan(acc):
                    continue
                weight = cont_question_counts.get(cont, 0)
                if weight <= 0:
                    continue
                weights.append(weight)
                values.append(acc)
            if weights:
                weighted = float(np.average(values, weights=weights))
                per_url_weighted.append(weighted)
        if per_url_weighted:
            mean = float(np.mean(per_url_weighted))
            std = (
                float(np.std(per_url_weighted, ddof=1))
                if len(per_url_weighted) > 1
                else 0.0
            )
            ci = (
                1.96 * std / np.sqrt(len(per_url_weighted))
                if len(per_url_weighted) > 1
                else np.nan
            )
        else:
            mean = np.nan
            ci = np.nan
        micro_records.append((model, mean, ci))

    label_bbox = dict(
        facecolor="white",
        edgecolor="grey",
        alpha=0.9,
        boxstyle="round",
        pad=0.2,
    )
    x_summary = np.array([0.0, 0.5])
    for idx, (model, acc, ci) in enumerate(micro_records):
        ax_summary.errorbar(
            x_summary[idx],
            acc,
            yerr=ci,
            fmt="o",
            color=models[model]["color"],
            markeredgecolor="black",
            markersize=6,
            capsize=3,
            linewidth=1.2,
            zorder=3,
        )
        if not np.isnan(acc):
            ax_summary.text(
                x_summary[idx] + 0.06,
                acc,
                f"{acc:.3f}",
                ha="left",
                va="center",
                fontsize=value_fs,
                color="#333333",
            )
            is_llama = model == "LLaMA-3.2(I-)"
            label_offset = -0.02 if is_llama else 0.02
            label_va = "top" if is_llama else "bottom"
            label_dx = 0.01 if is_llama else 0.06
            label_face = models[model]["color"]
            label_text_color = "black"
            label_edge = "none" if is_llama else "grey"
            ax_summary.text(
                x_summary[idx] + label_dx,
                acc + label_offset,
                model,
                ha="center",
                va=label_va,
                fontsize=value_fs,
                color=label_text_color,
                bbox={**label_bbox, "facecolor": label_face, "edgecolor": label_edge},
            )
    ax_summary.set_xticks([])
    ax_summary.set_xlabel("")
    ax_summary.tick_params(axis="y", labelsize=tick_fs)
    ax_summary.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax_summary.set_ylabel("")
    bbox_props = dict(
        facecolor="lightgrey",
        edgecolor="grey",
        alpha=0.7,
        boxstyle="round",
        pad=0.3,
    )
    ax_summary.set_title(
        "Overall\n(micro average)",
        fontsize=title_fs,
        weight="bold",
        pad=6,
        y=0.90,
        bbox=bbox_props,
    )

    output_dir = os.path.join(PLOTS_DIR, "plot8")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, "accuracy_apples_to_apples_answered_by_all.pdf"
    )
    plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    _write_plot_csv(output_dir, 8, df)


def _filter_explicit_qa(df, qa_path):
    if not os.path.exists(qa_path):
        print(f"[!] QA dataset not found for explicit filter: {qa_path}")
        return df
    id_to_explicit = {}
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            if not qid:
                parts = [
                    row.get("question", ""),
                    json.dumps(row.get("options", []), ensure_ascii=False),
                    row.get("correct_answer", ""),
                    row.get("country", ""),
                    row.get("continent", ""),
                    row.get("generated_by", ""),
                ]
                blob = "||".join(parts).encode("utf-8")
                qid = hashlib.sha1(blob).hexdigest()
            question = row.get("question", "")
            country = row.get("country", "")
            continent = row.get("continent", "")
            text = question.lower()
            has_country = bool(country) and country.lower() in text
            has_continent = bool(continent) and continent.lower() in text
            id_to_explicit[qid] = (has_country or has_continent)
    if "question_id" not in df.columns:
        print("[!] question_id missing from qa_metacul_eval.csv; skipping explicit filter.")
        return df
    mask = df["question_id"].map(id_to_explicit).fillna(False)
    filtered = df[~mask].copy()
    print(f"[✔] Plot9 explicit filter kept {len(filtered)}/{len(df)} rows.")
    return filtered


def plot_adversarial_url_accuracy(exclude_explicit=False, output_name="qa_adversarial_accuracy.pdf"):
    # Plot: QA accuracy vs URL corruption rate.
    # Output: /path/to/metacul/results/plots/plot9/{output_name}
    axis_label_fs = 22
    tick_fs = 19
    legend_fs = 18
    df_path = "/path/to/metacul/results/qa_metacul_eval.csv"
    if not os.path.exists(df_path):
        print(f"[!] Missing adversarial CSV: {df_path}")
        return

    df = pd.read_csv(df_path)
    if df.empty:
        print("[!] Adversarial CSV is empty.")
        return
    if exclude_explicit:
        df = _filter_explicit_qa(df, "/path/to/metacul/qa_data/hf_dataset.jsonl")

    variants = []
    for col in df.columns:
        if col.endswith("_correct"):
            variants.append(col[: -len("_correct")])
    variants = sorted(set(variants))
    if not variants:
        print("[!] No metadata variants found for plot9.")
        return

    records = []
    for (rate, url), sub in df.groupby(["url_corruption_rate", "base_url"]):
        for variant in variants:
            correct = sub[f"{variant}_correct"].sum()
            incorrect = (sub[f"{variant}_incorrect"] == -1).sum()
            denom = correct + incorrect
            if denom > 0:
                acc = correct / denom
                records.append(
                    {
                        "url_corruption_rate": rate,
                        "base_url": url,
                        "variant": variant,
                        "accuracy": acc,
                    }
                )

    per_url = pd.DataFrame(records)
    summary = (
        per_url.groupby(["url_corruption_rate", "variant"])["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary = summary[summary["count"] > 0].copy()
    summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["count"])
    rates_all = sorted(summary["url_corruption_rate"].unique())

    # For without-metadata variants (only c0 exists), extend a flat line across all rates.
    expanded = [summary]
    for variant in variants:
        if variant.endswith("without_metadata"):
            sub = summary[(summary["variant"] == variant)]
            if sub.empty:
                continue
            if len(sub["url_corruption_rate"].unique()) == 1:
                base_row = sub.iloc[0]
                missing_rates = [r for r in rates_all if r != base_row["url_corruption_rate"]]
                if missing_rates:
                    extra = pd.DataFrame(
                        {
                            "url_corruption_rate": missing_rates,
                            "variant": variant,
                            "mean": base_row["mean"],
                            "std": base_row["std"],
                            "count": base_row["count"],
                            "ci95": base_row["ci95"],
                        }
                    )
                    expanded.append(extra)
    if len(expanded) > 1:
        summary = pd.concat(expanded, ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = summary.sort_values("url_corruption_rate")
    label_map = {
        "custom_with_metadata": "Global(T+,I+)",
        "custom_without_metadata": "Global(T-,I-)",
        "llama3_chat_with_metadata": "LLaMA-3.2(I+)",
        "llama3_chat_without_metadata": "LLaMA-3.2(I-)",
    }
    plot_df["variant_label"] = plot_df["variant"].map(label_map).fillna(plot_df["variant"])
    palette = {
        "Global(T+,I+)": "#a6cee3",
        "Global(T-,I-)": "#f7a1b5",
        "LLaMA-3.2(I+)": "#7f7f7f",
        "LLaMA-3.2(I-)": "#d9d9d9",
    }
    linestyle_map = {
        "Global(T+,I+)": "--",
        "LLaMA-3.2(I+)": "--",
    }
    marker_map = {
        "Global(T+,I+)": "o",
        "LLaMA-3.2(I+)": "o",
        "Global(T-,I-)": "s",
        "LLaMA-3.2(I-)": "s",
    }
    marker_size_map = {
        "Global(T-,I-)": 7,
        "LLaMA-3.2(I-)": 7,
    }

    for label, sub in plot_df.groupby("variant_label"):
        sub = sub.sort_values("url_corruption_rate")
        line_width = 2.0 if label == "LLaMA-3.2(I-)" else 1.6
        marker_edge = "black" if label == "LLaMA-3.2(I-)" else None
        ax.plot(
            sub["url_corruption_rate"],
            sub["mean"],
            marker=marker_map.get(label, "o"),
            color=palette.get(label),
            linestyle=linestyle_map.get(label, "-"),
            linewidth=line_width,
            markeredgecolor=marker_edge,
            markersize=marker_size_map.get(label, 5.5),
            label=label,
        )
        ax.fill_between(
            sub["url_corruption_rate"],
            sub["mean"] - sub["ci95"],
            sub["mean"] + sub["ci95"],
            color=palette.get(label),
            alpha=0.2,
        )
    ax.set_xlabel("Fraction of samples corrupted", fontsize=axis_label_fs)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"], fontsize=tick_fs)
    ax.set_ylabel("Accuracy (↑ better)", fontsize=axis_label_fs)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.legend(
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="black",
        fontsize=legend_fs,
        loc="lower right",
    )

    output_dir = os.path.join(PLOTS_DIR, "plot9")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def plot_metadata_ablations():
    # Plot: metadata ablations across checkpoints on metadata/combined test sets.
    # Output: /path/to/metacul/results/plots/plot6/perplexity_metadata_ablations_1b.pdf
    df = _load_perplexity_df()
    pairs = set()
    axis_label_fs = 18
    tick_fs = 15
    title_fs = 16
    legend_fs = 15

    steps = [2000, 4000, 8000, 10000]
    step_labels = ["2k", "4k", "8k", "10k"]

    metadata_tests = {
        "url": "/path/to/metacul/training_data/meco_datasets/combined_only_url/with_metadata/",
        "url_continent": "/path/to/metacul/training_data/meco_datasets/combined_only_url_continent/with_metadata/",
        "url_country": "/path/to/metacul/training_data/meco_datasets/combined_only_url_country/with_metadata/",
    }

    combined_tests = {
        "with": "/path/to/metacul/training_data/meco_datasets/combined/with_metadata/",
        "without": "/path/to/metacul/training_data/meco_datasets/combined/without_metadata/",
    }

    model_groups = {
        "combined_with": {
            "final": "/path/to/metacul/models/combined_with_metadata_1b",
            "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_with_metadata_1b_step{step}k",
        },
        "combined_without": {
            "final": "/path/to/metacul/models/combined_without_metadata_1b",
            "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_without_metadata_1b_step{step}k",
        },
        "url": {
            "final": "/path/to/metacul/models/ablations/metadata/combined_only_url_with_metadata_1b",
            "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_url_with_metadata_1b_step{step}k",
        },
        "url_continent": {
            "final": "/path/to/metacul/models/ablations/metadata/combined_only_url_continent_with_metadata_1b",
            "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_url_continent_with_metadata_1b_step{step}k",
        },
        "url_country": {
            "final": "/path/to/metacul/models/ablations/metadata/combined_only_url_country_with_metadata_1b",
            "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_url_country_with_metadata_1b_step{step}k",
        },
    }

    colors = {
        "combined_with": "#5fae78",
        "combined_without": "#7f7f7f",
        "url": "#f4a3a3",
        "url_continent": "#6baed6",
        "url_country": "#b8a1d9",
    }
    markers = {
        "combined_with": "o",
        "combined_without": "s",
        "url": "D",
        "url_continent": "^",
        "url_country": "v",
    }
    linestyles = {
        "combined_with": "--",
        "combined_without": "--",
        "url": "-",
        "url_continent": "-",
        "url_country": "-",
    }
    labels = {
        "combined_with": "Global [URL][Country][Continent] (T+)",
        "combined_without": "Global (T-)",
        "url": "URL-only [URL] (T+)",
        "url_continent": "URL+Continent [URL][Continent] (T+)",
        "url_country": "URL+Country [URL][Country] (T+)",
    }

    def _lookup(model_path, test_path):
        row = df[
            (df["model_path"] == model_path)
            & (df["test_set_path"] == test_path)
        ]
        if row.empty or pd.isna(row["mean_ppl"].values[0]):
            return np.nan
        return float(row["mean_ppl"].values[0])

    def _series_for_panel(panel, model_keys, own_mode):
        series = {}
        series_ci = {}
        for key in model_keys:
            cfg = model_groups[key]
            values = []
            ci_lows = []
            ci_highs = []
            for step in steps:
                if step == 10000:
                    model_path = cfg["final"]
                else:
                    model_path = cfg["steps"].format(step=step // 1000)
                if panel == "own":
                    if own_mode == "single_url":
                        test_path = metadata_tests["url"]
                        pairs.add((model_path, test_path))
                        mean, ci_low, ci_high = _lookup_with_ci(
                            df, model_path, test_path
                        )
                        val = mean
                        ci_lows.append(ci_low)
                        ci_highs.append(ci_high)
                    else:
                        vals = []
                        lows = []
                        highs = []
                        for test_path in (
                            metadata_tests["url"],
                            metadata_tests["url_continent"],
                            metadata_tests["url_country"],
                        ):
                            pairs.add((model_path, test_path))
                            mean, ci_low, ci_high = _lookup_with_ci(
                                df, model_path, test_path
                            )
                            if not np.isnan(mean):
                                vals.append(mean)
                                lows.append(ci_low)
                                highs.append(ci_high)
                        val = float(np.mean(vals)) if vals else np.nan
                        ci_lows.append(float(np.mean(lows)) if lows else np.nan)
                        ci_highs.append(float(np.mean(highs)) if highs else np.nan)
                elif panel == "combined_with":
                    pairs.add((model_path, combined_tests["with"]))
                    mean, ci_low, ci_high = _lookup_with_ci(
                        df, model_path, combined_tests["with"]
                    )
                    val = mean
                    ci_lows.append(ci_low)
                    ci_highs.append(ci_high)
                else:
                    pairs.add((model_path, combined_tests["without"]))
                    mean, ci_low, ci_high = _lookup_with_ci(
                        df, model_path, combined_tests["without"]
                    )
                    val = mean
                    ci_lows.append(ci_low)
                    ci_highs.append(ci_high)
                values.append(val)
            series[key] = values
            series_ci[key] = (ci_lows, ci_highs)
        return series, series_ci

    def _plot_metadata_figure(panels, model_keys, own_mode, output_suffix, legend_order):
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 5), sharey=True)
        is_appendix_layout = len(legend_order) > 4
        title_y = 0.84 if is_appendix_layout else 0.90
        bbox_props = dict(
            facecolor="lightgrey",
            edgecolor="grey",
            alpha=0.7,
            boxstyle="round",
            pad=0.3,
        )
        all_values = []
        for ax, (panel_key, title) in zip(axes, panels):
            series, series_ci = _series_for_panel(panel_key, model_keys, own_mode)
            draw_order = [
                key
                for key in model_keys
                if key not in {"combined_with", "combined_without"}
            ] + [key for key in model_keys if key in {"combined_with", "combined_without"}]
            for key in draw_order:
                values = series[key]
                ci_lows, ci_highs = series_ci[key]
                x_vals = np.arange(len(step_labels))
                y_vals = np.array(values, dtype=float)
                lo_vals = np.array(ci_lows, dtype=float)
                hi_vals = np.array(ci_highs, dtype=float)
                is_global = key in {"combined_with", "combined_without"}
                ax.plot(
                    x_vals,
                    y_vals,
                    marker=markers[key],
                    color=colors[key],
                    linestyle=linestyles[key],
                    label=labels[key],
                    linewidth=2.6 if is_global else 2,
                    markersize=6.5 if is_global else 6,
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    zorder=5 if is_global else 3,
                )
                mask = ~np.isnan(y_vals) & ~np.isnan(lo_vals) & ~np.isnan(hi_vals)
                if np.any(mask):
                    ax.fill_between(
                        x_vals,
                        lo_vals,
                        hi_vals,
                        color=colors[key],
                        alpha=0.22 if is_global else 0.35,
                        linewidth=0.4,
                        edgecolor=colors[key],
                        where=mask,
                        interpolate=True,
                        zorder=2 if is_global else 1,
                    )
                all_values.extend([v for v in values if not np.isnan(v)])
            title_y_this = title_y
            if is_appendix_layout and panel_key in ("own", "combined_with"):
                title_y_this = 0.79
            elif not is_appendix_layout and panel_key == "combined_with":
                title_y_this = 0.80
            elif not is_appendix_layout and panel_key == "combined_without":
                title_y_this = 0.85
            title_obj = ax.set_title(
                title,
                fontsize=title_fs,
                weight="bold",
                pad=6,
                y=title_y_this,
                bbox=bbox_props,
            )
            title_obj.set_clip_on(False)
            ax.set_xlabel("")
            ax.set_xticks(x_vals)
            ax.set_xticklabels(step_labels, fontsize=tick_fs)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
            ax.tick_params(axis="x", labelsize=tick_fs)
            ax.tick_params(axis="y", labelsize=tick_fs)
            ax.set_ylim(bottom=9)

        if all_values:
            y_max = max(all_values)
            for ax in axes:
                ax.set_ylim(top=y_max + 1.0)

        axes[0].set_ylabel("Perplexity (↓ better)", fontsize=axis_label_fs)
        label_to_key = {labels[key]: key for key in labels}
        legend_handles = []
        for label in legend_order:
            key = label_to_key.get(label)
            if key is None:
                continue
            handle = Line2D(
                [],
                [],
                color=colors[key],
                marker=markers[key],
                linestyle=linestyles[key],
                linewidth=2,
                label=label,
            )
            legend_handles.append(handle)
        fig.legend(
            handles=legend_handles,
            labels=[handle.get_label() for handle in legend_handles],
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            edgecolor="black",
            fontsize=legend_fs,
            loc="upper center",
            ncol=3 if is_appendix_layout else len(legend_order),
            bbox_to_anchor=(0.5, 0.98 if is_appendix_layout else 0.93),
        )

        output_path = os.path.join(
            output_dir, f"perplexity_metadata_ablations_1b{output_suffix}.pdf"
        )
        fig.text(0.5, 0.02, "Training steps", ha="center", fontsize=axis_label_fs)
        plt.tight_layout()
        plt.subplots_adjust(top=0.79 if is_appendix_layout else 0.82)
        plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)

    output_dir = os.path.join(PLOTS_DIR, "plot6")
    os.makedirs(output_dir, exist_ok=True)
    main_panels = [
        ("own", "[URL]-only test (I+)"),
        ("combined_with", "Global\n[URL][Country][Continent]\ntest (I+)"),
        ("combined_without", "Global No-metadata\ntest (I-)"),
    ]
    appendix_panels = [
        ("own", "Avg over [URL],\n[URL][Continent],\n[URL][Country] test (I+)"),
        ("combined_with", "Global\n[URL][Country][Continent]\ntest (I+)"),
        ("combined_without", "Global No-metadata\ntest (I-)"),
    ]
    main_model_keys = ["combined_with", "url", "combined_without"]
    appendix_model_keys = [
        "combined_with",
        "url",
        "url_continent",
        "url_country",
        "combined_without",
    ]
    main_legend_order = [
        labels["combined_with"],
        labels["url"],
        labels["combined_without"],
    ]
    appendix_legend_order = [
        labels["combined_with"],
        labels["url"],
        labels["url_continent"],
        labels["url_country"],
        labels["combined_without"],
    ]

    _plot_metadata_figure(
        main_panels, main_model_keys, "single_url", "_main", main_legend_order
    )
    _plot_metadata_figure(
        appendix_panels,
        appendix_model_keys,
        "average_all",
        "_appendix",
        appendix_legend_order,
    )

    subset = _subset_by_pairs(df, pairs)
    _write_plot_csv(output_dir, 6, subset)


def plot_metadata_family_full_grid():
    # Plot: metadata-family ablations with a 3-panel main figure and a 5x7 appendix grid.
    # Outputs:
    #   /path/to/metacul/results/plots/plot10/perplexity_metadata_family_main_1b.pdf
    #   /path/to/metacul/results/plots/plot11/perplexity_metadata_family_full_grid_1b.pdf
    df = _load_perplexity_df()
    pairs_main = set()
    pairs_appendix = set()

    steps = [2000, 4000, 8000, 10000]
    step_labels = ["2k", "4k", "8k", "10k"]
    axis_label_fs = 18
    tick_fs = 13
    title_fs = 14
    legend_fs = 13

    tests = [
        (
            "URL-only (I+)",
            "/path/to/metacul/training_data/meco_datasets/combined_only_url/with_metadata/",
        ),
        (
            "URL+Country (I+)",
            "/path/to/metacul/training_data/meco_datasets/combined_only_url_country/with_metadata/",
        ),
        (
            "URL+Continent (I+)",
            "/path/to/metacul/training_data/meco_datasets/combined_only_url_continent/with_metadata/",
        ),
        (
            "Country-only (I+)",
            "/path/to/metacul/training_data/meco_datasets/combined_only_country/with_metadata/",
        ),
        (
            "Continent-only (I+)",
            "/path/to/metacul/training_data/meco_datasets/combined_only_continent/with_metadata/",
        ),
        (
            "Global metadata (I+)",
            "/path/to/metacul/training_data/meco_datasets/combined/with_metadata/",
        ),
        (
            "Global no-metadata (I-)",
            "/path/to/metacul/training_data/meco_datasets/combined/without_metadata/",
        ),
    ]

    model_groups = {
        "combined_with": {
            "label": "Global [URL][Country][Continent] (T+)",
            "final": "/path/to/metacul/models/combined_with_metadata_1b",
            "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_with_metadata_1b_step{step}k",
            "color": "#2b8c66",
            "marker": "o",
            "linestyle": "--",
        },
        "combined_without": {
            "label": "Global (T-)",
            "final": "/path/to/metacul/models/combined_without_metadata_1b",
            "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_without_metadata_1b_step{step}k",
            "color": "#7f7f7f",
            "marker": "s",
            "linestyle": "--",
        },
        "url": {
            "label": "URL-only [URL] (T+)",
            "final": "/path/to/metacul/models/ablations/metadata/combined_only_url_with_metadata_1b",
            "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_url_with_metadata_1b_step{step}k",
            "color": "#f4a3a3",
            "marker": "D",
            "linestyle": "-",
        },
        "url_country": {
            "label": "URL+Country [URL][Country] (T+)",
            "final": "/path/to/metacul/models/ablations/metadata/combined_only_url_country_with_metadata_1b",
            "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_url_country_with_metadata_1b_step{step}k",
            "color": "#b8a1d9",
            "marker": "v",
            "linestyle": "-",
        },
        "url_continent": {
            "label": "URL+Continent [URL][Continent] (T+)",
            "final": "/path/to/metacul/models/ablations/metadata/combined_only_url_continent_with_metadata_1b",
            "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_url_continent_with_metadata_1b_step{step}k",
            "color": "#6baed6",
            "marker": "^",
            "linestyle": "-",
        },
        "country_only": {
            "label": "Country-only [Country] (T+)",
            "final": "/path/to/metacul/models/combined_only_country_with_metadata_1b",
            "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_country_with_metadata_1b_step{step}k",
            "color": "#f0c36d",
            "marker": "P",
            "linestyle": "-",
        },
        "continent_only": {
            "label": "Continent-only [Continent] (T+)",
            "final": "/path/to/metacul/models/combined_only_continent_with_metadata_1b",
            "steps": "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_continent_with_metadata_1b_step{step}k",
            "color": "#74c476",
            "marker": "o",
            "linestyle": "-",
        },
    }

    all_values = []

    def _series_for_test_path(test_path, keys, pair_sink):
        series = {}
        for key in keys:
            cfg = model_groups[key]
            y_vals = []
            lo_vals = []
            hi_vals = []
            for step in steps:
                model_path = (
                    cfg["final"]
                    if step == 10000
                    else cfg["steps"].format(step=step // 1000)
                )
                pair_sink.add((model_path, test_path))
                mean, ci_low, ci_high = _lookup_with_ci(df, model_path, test_path)
                y_vals.append(mean)
                lo_vals.append(ci_low)
                hi_vals.append(ci_high)
            series[key] = (
                np.array(y_vals, dtype=float),
                np.array(lo_vals, dtype=float),
                np.array(hi_vals, dtype=float),
            )
        return series

    own_test_paths = [path for _, path in tests[:5]]

    def _series_for_own_average(keys, pair_sink):
        series = {}
        for key in keys:
            cfg = model_groups[key]
            y_vals = []
            lo_vals = []
            hi_vals = []
            for step in steps:
                model_path = (
                    cfg["final"]
                    if step == 10000
                    else cfg["steps"].format(step=step // 1000)
                )
                rows = []
                for test_path in own_test_paths:
                    pair_sink.add((model_path, test_path))
                    row = df[
                        (df["model_path"] == model_path)
                        & (df["test_set_path"] == test_path)
                    ]
                    if row.empty or pd.isna(row["mean_ppl"].values[0]):
                        continue
                    rows.append(row.iloc[0])
                agg = _aggregate_rows(rows)
                if agg is None:
                    y_vals.append(np.nan)
                    lo_vals.append(np.nan)
                    hi_vals.append(np.nan)
                else:
                    mean, ci_low, ci_high = agg
                    y_vals.append(mean)
                    lo_vals.append(ci_low)
                    hi_vals.append(ci_high)
            series[key] = (
                np.array(y_vals, dtype=float),
                np.array(lo_vals, dtype=float),
                np.array(hi_vals, dtype=float),
            )
        return series

    def _draw_series(ax, title, series_map, keys):
        draw_order = [key for key in keys if key not in {"combined_with", "combined_without"}]
        draw_order += [key for key in keys if key in {"combined_with", "combined_without"}]
        for key in draw_order:
            cfg = model_groups[key]
            y_vals, lo_vals, hi_vals = series_map[key]
            valid = ~np.isnan(y_vals)
            if not np.any(valid):
                continue
            x_vals = np.arange(len(step_labels))
            is_global = key in {"combined_with", "combined_without"}
            ax.plot(
                x_vals,
                y_vals,
                color=cfg["color"],
                marker=cfg["marker"],
                linestyle=cfg["linestyle"],
                linewidth=2.6 if is_global else 2,
                markersize=6.5 if is_global else 6,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label=cfg["label"],
                zorder=5 if is_global else 3,
            )
            band_mask = ~np.isnan(y_vals) & ~np.isnan(lo_vals) & ~np.isnan(hi_vals)
            if np.any(band_mask):
                ax.fill_between(
                    x_vals,
                    lo_vals,
                    hi_vals,
                    color=cfg["color"],
                    alpha=0.14 if is_global else 0.2,
                    linewidth=0.4,
                    edgecolor=cfg["color"],
                    where=band_mask,
                    interpolate=True,
                    zorder=2 if is_global else 1,
                )
            all_values.extend([v for v in y_vals if not np.isnan(v)])

        bbox_props = dict(
            facecolor="lightgrey",
            edgecolor="grey",
            alpha=0.7,
            boxstyle="round,pad=0.45",
        )
        ax.set_title(title, fontsize=title_fs, weight="bold", pad=6, bbox=bbox_props)
        ax.set_xticks(np.arange(len(step_labels)))
        ax.set_xticklabels(step_labels, fontsize=tick_fs)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.tick_params(axis="y", labelsize=tick_fs)
        ax.tick_params(axis="x", labelsize=tick_fs)
        ax.set_ylim(bottom=8.5)

    main_panels = [
        ("Avg over family tests (I+)", own_test_paths[0], "own_average"),
        ("Global metadata (I+)", tests[5][1], "test_path"),
        ("Global no-metadata (I-)", tests[6][1], "test_path"),
    ]
    main_model_keys = [
        "url",
        "country_only",
        "continent_only",
        "combined_with",
        "combined_without",
    ]
    appendix_model_keys = list(model_groups.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5), sharey=True)
    for ax, (title, payload, mode) in zip(axes, main_panels):
        if mode == "own_average":
            series_map = _series_for_own_average(main_model_keys, pairs_main)
        else:
            series_map = _series_for_test_path(payload, main_model_keys, pairs_main)
        _draw_series(ax, title, series_map, main_model_keys)

    if all_values:
        y_max = max(all_values)
        for ax in axes:
            ax.set_ylim(top=y_max + 1.0)

    axes[0].set_ylabel("Perplexity (↓ better)", fontsize=axis_label_fs)
    fig.text(0.5, 0.03, "Training steps", ha="center", fontsize=axis_label_fs)

    legend_handles = [
        Line2D(
            [],
            [],
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle=cfg["linestyle"],
            linewidth=2,
            markeredgecolor="black",
            label=cfg["label"],
        )
        for key in main_model_keys
        for cfg in [model_groups[key]]
    ]
    fig.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="black",
        fontsize=legend_fs,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.04),
    )

    output_dir = os.path.join(PLOTS_DIR, "plot10")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "perplexity_metadata_family_main_1b.pdf")
    plt.tight_layout()
    plt.subplots_adjust(top=0.74, bottom=0.12)
    plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    axes = axes.flatten()
    appendix_values = []

    for ax, (title, test_path) in zip(axes, tests):
        series_map = _series_for_test_path(test_path, appendix_model_keys, pairs_appendix)
        _draw_series(ax, title, series_map, appendix_model_keys)
        for y_vals, _, _ in series_map.values():
            appendix_values.extend([v for v in y_vals if not np.isnan(v)])

    axes[-1].axis("off")
    if appendix_values:
        y_max = max(appendix_values)
        for ax in axes[:-1]:
            ax.set_ylim(top=y_max + 1.0)

    axes[0].set_ylabel("Perplexity (↓ better)", fontsize=axis_label_fs)
    axes[4].set_ylabel("Perplexity (↓ better)", fontsize=axis_label_fs)
    fig.text(0.5, 0.03, "Training steps", ha="center", fontsize=axis_label_fs)

    legend_handles = [
        Line2D(
            [],
            [],
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle=cfg["linestyle"],
            linewidth=2,
            markeredgecolor="black",
            label=cfg["label"],
        )
        for key in appendix_model_keys
        for cfg in [model_groups[key]]
    ]
    fig.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="black",
        fontsize=legend_fs,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.995),
    )

    output_dir = os.path.join(PLOTS_DIR, "plot11")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "perplexity_metadata_family_full_grid_1b.pdf")
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.10)
    plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    subset_main = _subset_by_pairs(df, pairs_main)
    subset_appendix = _subset_by_pairs(df, pairs_appendix)
    _write_plot_csv(os.path.join(PLOTS_DIR, "plot10"), 10, subset_main)
    _write_plot_csv(output_dir, 11, subset_appendix)


def plot_leave_one_out_ablations():
    # Plot: leave-one-out ablations (with/without metadata).
    # Output: /path/to/metacul/results/plots/plot7/leave_one_out_{meta}.pdf
    df = _load_perplexity_df()
    pairs = set()

    steps = [2000, 4000, 8000, 10000]
    step_labels = ["2k", "4k", "8k", "10k"]
    continents = ["africa", "america", "asia", "europe"]

    combined_tests = {
        "with_metadata": "/path/to/metacul/training_data/meco_datasets/combined/with_metadata/",
        "without_metadata": "/path/to/metacul/training_data/meco_datasets/combined/without_metadata/",
    }

    def _lookup(model_path, test_path):
        row = df[
            (df["model_path"] == model_path)
            & (df["test_set_path"] == test_path)
        ]
        if row.empty or pd.isna(row["mean_ppl"].values[0]):
            return np.nan
        return float(row["mean_ppl"].values[0])

    def _model_path(meta, step):
        if step == 10000:
            return f"/path/to/metacul/models/combined_{meta}_1b"
        return (
            "/path/to/metacul/models/ablation_intermediates/metadata/"
            f"combined_{meta}_1b_step{step // 1000}k"
        )

    def _loo_model_path(continent, meta, step):
        if step == 10000:
            return (
                "/path/to/metacul/models/ablations/leave_one_out/"
                f"combined_no_{continent}_{meta}_1b"
            )
        return (
            "/path/to/metacul/models/ablation_intermediates/leave_one_out/"
            f"combined_no_{continent}_{meta}_1b_step{step // 1000}k"
        )

    colors = {"combined": "#7f7f7f", "loo": "#5fae78"}
    markers = {"combined": "o", "loo": "s"}
    linestyles = {"combined": "--", "loo": "-"}
    axis_label_fs = 22
    tick_fs = 19
    title_fs = 16
    legend_fs = 18

    bbox_props = dict(
        facecolor="lightgrey",
        edgecolor="grey",
        alpha=0.7,
        boxstyle="round",
        pad=0.3,
    )

    for meta in ["with_metadata", "without_metadata"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
        train_notation = "T+" if meta == "with_metadata" else "T-"
        eval_notation = "I+" if meta == "with_metadata" else "I-"

        combined_series = []
        combined_ci = []
        loo_series_by_cont = {cont: [] for cont in continents}
        loo_ci_by_cont = {cont: [] for cont in continents}
        combined_by_cont = {cont: [] for cont in continents}
        for step in steps:
            combined_vals = []
            combined_lows = []
            combined_highs = []
            combined_model = _model_path(meta, step)
            for cont in continents:
                left_out_test = (
                    f"/path/to/metacul/training_data/meco_datasets/continents/{cont}/{meta}/"
                )
                pairs.add((combined_model, left_out_test))
                mean, ci_low, ci_high = _lookup_with_ci(
                    df, combined_model, left_out_test
                )
                combined_by_cont[cont].append(mean)
                if not np.isnan(mean):
                    combined_vals.append(mean)
                    combined_lows.append(ci_low)
                    combined_highs.append(ci_high)
                loo_model = _loo_model_path(cont, meta, step)
                pairs.add((loo_model, left_out_test))
                loo_mean, loo_low, loo_high = _lookup_with_ci(
                    df, loo_model, left_out_test
                )
                loo_series_by_cont[cont].append(loo_mean)
                loo_ci_by_cont[cont].append((loo_low, loo_high))
            combined_vals = [v for v in combined_vals if not np.isnan(v)]
            combined_series.append(float(np.mean(combined_vals)) if combined_vals else np.nan)
            combined_ci.append(
                (
                    float(np.mean(combined_lows)) if combined_lows else np.nan,
                    float(np.mean(combined_highs)) if combined_highs else np.nan,
                )
            )

        combined_all_series = []
        combined_all_ci = []
        loo_all_series_by_cont = {cont: [] for cont in continents}
        loo_all_ci_by_cont = {cont: [] for cont in continents}
        for step in steps:
            combined_model = _model_path(meta, step)
            pairs.add((combined_model, combined_tests[meta]))
            mean, ci_low, ci_high = _lookup_with_ci(
                df, combined_model, combined_tests[meta]
            )
            combined_all_series.append(mean)
            combined_all_ci.append((ci_low, ci_high))
            for cont in continents:
                loo_model = _loo_model_path(cont, meta, step)
                pairs.add((loo_model, combined_tests[meta]))
                loo_mean, loo_low, loo_high = _lookup_with_ci(
                    df, loo_model, combined_tests[meta]
                )
                loo_all_series_by_cont[cont].append(loo_mean)
                loo_all_ci_by_cont[cont].append((loo_low, loo_high))

        loo_continent_colors = {
            "africa": "#fbc4a9",
            "america": "#f7b6b2",
            "asia": "#b2df8a",
            "europe": "#cbb7e5",
        }
        loo_continent_markers = {
            "africa": "o",
            "america": "s",
            "asia": "^",
            "europe": "D",
        }
        delta_values = []
        for cont in continents:
            loo_label = f"No{cont.capitalize()} ({train_notation}) vs Global ({train_notation})"
            marker_size = 7 if cont in ("america", "asia") else 5
            delta_series = []
            for loo_val, all_val in zip(
                loo_series_by_cont[cont], combined_by_cont[cont]
            ):
                if np.isnan(loo_val) or np.isnan(all_val):
                    delta_series.append(np.nan)
                else:
                    delta_val = loo_val - all_val
                    delta_series.append(delta_val)
                    delta_values.append(delta_val)
            axes[0].plot(
                step_labels,
                delta_series,
                marker=loo_continent_markers[cont],
                color=loo_continent_colors[cont],
                linestyle=linestyles["loo"],
                linewidth=2,
                markersize=marker_size,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label=loo_label,
            )
        axes[0].set_title(
            f"Δ: held-out Local test ({eval_notation})",
            fontsize=title_fs,
            weight="bold",
            pad=6,
            y=0.90,
            bbox=bbox_props,
        )

        for cont in continents:
            loo_label = f"No{cont.capitalize()} ({train_notation}) vs Global ({train_notation})"
            marker_size = 7 if cont in ("america", "asia") else 5
            delta_all_series = []
            for loo_val, all_val in zip(
                loo_all_series_by_cont[cont], combined_all_series
            ):
                if np.isnan(loo_val) or np.isnan(all_val):
                    delta_all_series.append(np.nan)
                else:
                    delta_all_series.append(loo_val - all_val)
            axes[1].plot(
                step_labels,
                delta_all_series,
                marker=loo_continent_markers[cont],
                color=loo_continent_colors[cont],
                linestyle=linestyles["loo"],
                linewidth=2,
                markersize=marker_size,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label=loo_label,
            )
        axes[1].set_title(
            f"Δ: Global test ({eval_notation})",
            fontsize=title_fs,
            weight="bold",
            pad=6,
            y=0.90,
            bbox=bbox_props,
        )

        for ax in axes:
            ax.set_xlabel("")
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
            ax.tick_params(axis="x", labelsize=tick_fs)
            ax.tick_params(axis="y", labelsize=tick_fs)

        axes[0].set_ylabel("Δ Perplexity", fontsize=axis_label_fs)
        all_delta_values = delta_values[:]
        for cont in continents:
            for loo_val, all_val in zip(
                loo_all_series_by_cont[cont], combined_all_series
            ):
                if np.isnan(loo_val) or np.isnan(all_val):
                    continue
                all_delta_values.append(loo_val - all_val)
        if all_delta_values:
            delta_min = min(all_delta_values)
            delta_max = max(all_delta_values)
            pad = 0.5
            axes[0].set_ylim(delta_min - pad, delta_max + pad)
            axes[1].set_ylim(delta_min - pad, delta_max + pad)
        legend_handles = []
        for cont in continents:
            loo_label = f"No{cont.capitalize()} ({train_notation}) vs Global ({train_notation})"
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    color=loo_continent_colors[cont],
                    marker=loo_continent_markers[cont],
                    linestyle=linestyles["loo"],
                    linewidth=2,
                    markeredgecolor="black",
                    label=loo_label,
                )
            )
        fig.legend(
            handles=legend_handles,
            labels=[handle.get_label() for handle in legend_handles],
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            edgecolor="black",
            fontsize=legend_fs,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.01),
        )

        output_dir = os.path.join(PLOTS_DIR, "plot7")
        os.makedirs(output_dir, exist_ok=True)
        suffix = "with_metadata" if meta == "with_metadata" else "without_metadata"
        output_path = os.path.join(output_dir, f"leave_one_out_{suffix}.pdf")
        fig.text(0.5, 0.02, "Training steps", ha="center", fontsize=axis_label_fs)
        plt.tight_layout()
        plt.subplots_adjust(top=0.78)
        plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)

    output_dir = os.path.join(PLOTS_DIR, "plot7")
    subset = _subset_by_pairs(df, pairs)
    _write_plot_csv(output_dir, 7, subset)


def _estimate_llama_params_from_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    hidden = int(cfg["hidden_size"])
    intermediate = int(cfg["intermediate_size"])
    layers = int(cfg["num_hidden_layers"])
    vocab = int(cfg["vocab_size"])

    # Llama-style dense blocks with untied input/output embeddings.
    per_layer = 4 * hidden * hidden + 3 * hidden * intermediate + 2 * hidden
    total_params = layers * per_layer + 2 * vocab * hidden + hidden
    return total_params


def _load_qa_accuracy_from_jsonl(results_dir, metadata_label, slugs):
    correct = 0
    total = 0
    used_slugs = []

    for slug in slugs:
        path = os.path.join(
            results_dir, f"qa_metacul_eval_{metadata_label}_custom_{slug}_c0.jsonl"
        )
        if not os.path.exists(path):
            continue
        file_total = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                processed_answer = row.get("processed_answer")
                if processed_answer is None:
                    continue
                is_correct = row.get("is_correct")
                if is_correct is None:
                    is_correct = processed_answer == row.get("correct_answer")
                total += 1
                file_total += 1
                correct += int(bool(is_correct))
        if file_total > 0:
            used_slugs.append(slug)

    accuracy = correct / total if total else np.nan
    return accuracy, correct, total, used_slugs


def _list_usable_qa_slugs(results_dir, metadata_label):
    usable = set()
    pattern = re.compile(rf"qa_metacul_eval_{metadata_label}_custom_(.+)_c0\.jsonl$")
    for filename in os.listdir(results_dir):
        match = pattern.match(filename)
        if not match:
            continue
        slug = match.group(1)
        path = os.path.join(results_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("processed_answer") is not None:
                    usable.add(slug)
                    break
    return usable


def plot_compute_tradeoff_global_models():
    # Plot 12: 1B/3B global models, matched PPL and QA accuracy vs estimated FLOPs.
    # QA currently uses the strict intersection of URL subsets already available across
    # all four final custom chat models.
    df = _load_perplexity_df()

    model_specs = [
        {
            "size": "1B",
            "train_tag": "T+",
            "label": "1B T+",
            "color": "#1b9e77",
            "marker": "o",
            "model_path": "/path/to/metacul/models/combined_with_metadata_1b",
            "test_set_path": "/path/to/metacul/training_data/meco_datasets/combined/with_metadata/",
            "config_path": "/path/to/metacul/models/combined_with_metadata_1b/config.json",
            "results_dir": "/path/to/metacul/results/downstream",
            "results_metadata_label": "with_metadata",
        },
        {
            "size": "1B",
            "train_tag": "T-",
            "label": "1B T-",
            "color": "#d95f02",
            "marker": "o",
            "model_path": "/path/to/metacul/models/combined_without_metadata_1b",
            "test_set_path": "/path/to/metacul/training_data/meco_datasets/combined/without_metadata/",
            "config_path": "/path/to/metacul/models/combined_without_metadata_1b/config.json",
            "results_dir": "/path/to/metacul/results/downstream",
            "results_metadata_label": "without_metadata",
        },
        {
            "size": "3B",
            "train_tag": "T+",
            "label": "3B T+",
            "color": "#1b9e77",
            "marker": "s",
            "model_path": "/path/to/metacul/models/combined_with_metadata_3b",
            "test_set_path": "/path/to/metacul/training_data/meco_datasets/combined/with_metadata/",
            "config_path": "/path/to/metacul/models/combined_with_metadata_3b/config.json",
            "results_dir": "/path/to/metacul/results/downstream_3b",
            "results_metadata_label": "with_metadata",
        },
        {
            "size": "3B",
            "train_tag": "T-",
            "label": "3B T-",
            "color": "#d95f02",
            "marker": "s",
            "model_path": "/path/to/metacul/models/combined_without_metadata_3b",
            "test_set_path": "/path/to/metacul/training_data/meco_datasets/combined/without_metadata/",
            "config_path": "/path/to/metacul/models/combined_without_metadata_3b/config.json",
            "results_dir": "/path/to/metacul/results/downstream_3b",
            "results_metadata_label": "without_metadata",
        },
    ]

    # Final pretraining configs use:
    # dp=4, micro_batch_size=8, batch_accumulation_per_replica=64,
    # sequence_length=2048, train_steps=10000.
    train_tokens = 4 * 8 * 64 * 2048 * 10000

    available_slugs_by_variant = []
    for spec in model_specs:
        available_slugs_by_variant.append(
            _list_usable_qa_slugs(
                spec["results_dir"], spec["results_metadata_label"]
            )
        )

    common_slugs = sorted(set.intersection(*available_slugs_by_variant))
    if not common_slugs:
        print("No common QA URL subset found across 1B/3B final chat models.")
        return

    plot_rows = []
    for spec in model_specs:
        param_count = _estimate_llama_params_from_config(spec["config_path"])
        est_flops = float(6 * param_count * train_tokens)

        ppl_row = df[
            (df["model_path"] == spec["model_path"])
            & (df["test_set_path"] == spec["test_set_path"])
        ]
        if ppl_row.empty or pd.isna(ppl_row["mean_ppl"].iloc[0]):
            ppl_value = np.nan
        else:
            ppl_value = float(ppl_row["mean_ppl"].iloc[0])

        qa_acc, qa_correct, qa_total, used_slugs = _load_qa_accuracy_from_jsonl(
            spec["results_dir"], spec["results_metadata_label"], common_slugs
        )

        plot_rows.append(
            {
                "panel": "PPL vs FLOPs",
                "model_label": spec["label"],
                "size": spec["size"],
                "train_tag": spec["train_tag"],
                "model_path": spec["model_path"],
                "metric": "mean_ppl",
                "metric_value": ppl_value,
                "estimated_params": param_count,
                "train_tokens": train_tokens,
                "estimated_flops": est_flops,
                "estimated_flops_e20": est_flops / 1e20,
                "eval_target": spec["test_set_path"],
                "note": "Matched global test set",
            }
        )
        plot_rows.append(
            {
                "panel": "QA Accuracy vs FLOPs",
                "model_label": spec["label"],
                "size": spec["size"],
                "train_tag": spec["train_tag"],
                "model_path": spec["model_path"],
                "metric": "qa_accuracy",
                "metric_value": qa_acc,
                "estimated_params": param_count,
                "train_tokens": train_tokens,
                "estimated_flops": est_flops,
                "estimated_flops_e20": est_flops / 1e20,
                "eval_target": ",".join(used_slugs),
                "note": f"Common {len(common_slugs)}-URL subset; {qa_correct}/{qa_total}",
            }
        )

    plot_df = pd.DataFrame(plot_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
    bbox_props = dict(
        facecolor="lightgrey",
        edgecolor="grey",
        alpha=0.7,
        boxstyle="round",
        pad=0.35,
    )
    panel_specs = [
        ("PPL vs FLOPs", "mean_ppl", "Perplexity (↓ better)", 0),
        ("QA Accuracy vs FLOPs", "qa_accuracy", "QA accuracy (↑ better)", 1),
    ]

    for panel_name, metric_name, ylabel, ax_idx in panel_specs:
        ax = axes[ax_idx]
        subset = plot_df[plot_df["metric"] == metric_name].copy()
        subset = subset.sort_values(["estimated_flops", "train_tag", "size"])
        for _, row in subset.iterrows():
            spec = next(s for s in model_specs if s["label"] == row["model_label"])
            ax.scatter(
                row["estimated_flops"],
                row["metric_value"],
                s=95,
                color=spec["color"],
                marker=spec["marker"],
                edgecolors="black",
                linewidths=0.7,
                zorder=3,
            )
            y_offset = 0.10 if metric_name == "mean_ppl" else 0.006
            ax.annotate(
                row["model_label"],
                (row["estimated_flops"], row["metric_value"]),
                xytext=(6, 6 if row["train_tag"] == "T+" else -12),
                textcoords="offset points",
                fontsize=12,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Estimated training FLOPs", fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.set_title(panel_name, fontsize=15, weight="bold", y=1.01, bbox=bbox_props)

        if metric_name == "qa_accuracy":
            finite = subset["metric_value"][np.isfinite(subset["metric_value"])]
            if not finite.empty:
                ax.set_ylim(max(0.0, finite.min() - 0.05), min(1.0, finite.max() + 0.05))

    legend_handles = [
        Line2D([], [], color="#1b9e77", marker="o", linestyle="None", markersize=9, markeredgecolor="black", label="1B T+"),
        Line2D([], [], color="#d95f02", marker="o", linestyle="None", markersize=9, markeredgecolor="black", label="1B T-"),
        Line2D([], [], color="#1b9e77", marker="s", linestyle="None", markersize=9, markeredgecolor="black", label="3B T+"),
        Line2D([], [], color="#d95f02", marker="s", linestyle="None", markersize=9, markeredgecolor="black", label="3B T-"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="black",
        fontsize=12,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.text(
        0.5,
        0.01,
        "Estimated FLOPs use 6ND with N from the final model config and D = 41.94B training tokens.",
        ha="center",
        fontsize=11,
    )

    output_dir = os.path.join(PLOTS_DIR, "plot12")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "compute_tradeoff_1b_3b.pdf")
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, bottom=0.18, wspace=0.28)
    plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    plot_df["common_qa_url_count"] = len(common_slugs)
    plot_df["common_qa_urls"] = ",".join(common_slugs)
    _write_plot_csv(output_dir, 12, plot_df)


def plot_token_efficiency_global_ppl():
    # Plot 13: matched global PPL vs exact training tokens for T+ and T-,
    # with annotation for how much earlier T+ reaches final T- quality.
    df = _load_perplexity_df()
    per_step_tokens = 4 * 8 * 64 * 2048

    size_specs = {
        "1B": {
            "tplus": [
                (
                    2000,
                    "/path/to/metacul/models/ablation_intermediates/metadata/combined_with_metadata_1b_step2k",
                    "/path/to/metacul/training_data/meco_datasets/combined/with_metadata/",
                ),
                (
                    4000,
                    "/path/to/metacul/models/ablation_intermediates/metadata/combined_with_metadata_1b_step4k",
                    "/path/to/metacul/training_data/meco_datasets/combined/with_metadata/",
                ),
                (
                    8000,
                    "/path/to/metacul/models/ablation_intermediates/metadata/combined_with_metadata_1b_step8k",
                    "/path/to/metacul/training_data/meco_datasets/combined/with_metadata/",
                ),
                (
                    10000,
                    "/path/to/metacul/models/combined_with_metadata_1b",
                    "/path/to/metacul/training_data/meco_datasets/combined/with_metadata/",
                ),
            ],
            "tminus": [
                (
                    2000,
                    "/path/to/metacul/models/ablation_intermediates/metadata/combined_without_metadata_1b_step2k",
                    "/path/to/metacul/training_data/meco_datasets/combined/without_metadata/",
                ),
                (
                    4000,
                    "/path/to/metacul/models/ablation_intermediates/metadata/combined_without_metadata_1b_step4k",
                    "/path/to/metacul/training_data/meco_datasets/combined/without_metadata/",
                ),
                (
                    8000,
                    "/path/to/metacul/models/ablation_intermediates/metadata/combined_without_metadata_1b_step8k",
                    "/path/to/metacul/training_data/meco_datasets/combined/without_metadata/",
                ),
                (
                    10000,
                    "/path/to/metacul/models/combined_without_metadata_1b",
                    "/path/to/metacul/training_data/meco_datasets/combined/without_metadata/",
                ),
            ],
        },
        "3B": {
            "tplus": [
                (
                    2000,
                    "/path/to/metacul/models/ablation_intermediates/metadata/combined_with_metadata_3b_step2k",
                    "/path/to/metacul/training_data/meco_datasets/combined/with_metadata/",
                ),
                (
                    4000,
                    "/path/to/metacul/models/ablation_intermediates/metadata/combined_with_metadata_3b_step4k",
                    "/path/to/metacul/training_data/meco_datasets/combined/with_metadata/",
                ),
                (
                    8000,
                    "/path/to/metacul/models/ablation_intermediates/metadata/combined_with_metadata_3b_step8k",
                    "/path/to/metacul/training_data/meco_datasets/combined/with_metadata/",
                ),
                (
                    10000,
                    "/path/to/metacul/models/combined_with_metadata_3b",
                    "/path/to/metacul/training_data/meco_datasets/combined/with_metadata/",
                ),
            ],
            "tminus": [
                (
                    2000,
                    "/path/to/metacul/models/ablation_intermediates/metadata/combined_without_metadata_3b_step2k",
                    "/path/to/metacul/training_data/meco_datasets/combined/without_metadata/",
                ),
                (
                    4000,
                    "/path/to/metacul/models/ablation_intermediates/metadata/combined_without_metadata_3b_step4k",
                    "/path/to/metacul/training_data/meco_datasets/combined/without_metadata/",
                ),
                (
                    8000,
                    "/path/to/metacul/models/ablation_intermediates/metadata/combined_without_metadata_3b_step8k",
                    "/path/to/metacul/training_data/meco_datasets/combined/without_metadata/",
                ),
                (
                    10000,
                    "/path/to/metacul/models/combined_without_metadata_3b",
                    "/path/to/metacul/training_data/meco_datasets/combined/without_metadata/",
                ),
            ],
        },
    }

    pairs = set()
    records = []

    for size_label, size_spec in size_specs.items():
        for train_tag, checkpoints in size_spec.items():
            for step, model_path, test_path in checkpoints:
                pairs.add((model_path, test_path))
                row = df[
                    (df["model_path"] == model_path)
                    & (df["test_set_path"] == test_path)
                ]
                if row.empty or pd.isna(row["mean_ppl"].iloc[0]):
                    continue
                records.append(
                    {
                        "size": size_label,
                        "train_tag": "T+" if train_tag == "tplus" else "T-",
                        "step": step,
                        "tokens_b": step * per_step_tokens / 1e9,
                        "mean_ppl": float(row["mean_ppl"].iloc[0]),
                        "ci_low": float(row["ci_low"].iloc[0]),
                        "ci_high": float(row["ci_high"].iloc[0]),
                        "model_path": model_path,
                        "test_set_path": test_path,
                    }
                )

    if not records:
        print("No records found for token-efficiency global PPL plot.")
        return

    plot_df = pd.DataFrame(records)
    plot_df = plot_df.sort_values(["size", "train_tag", "step"]).reset_index(drop=True)

    def _interpolate_crossing(tokens, values, target):
        for x1, y1, x2, y2 in zip(tokens[:-1], values[:-1], tokens[1:], values[1:]):
            if y1 >= target and y2 <= target and y1 != y2:
                frac = (y1 - target) / (y1 - y2)
                return x1 + frac * (x2 - x1)
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    bbox_props = dict(
        facecolor="lightgrey",
        edgecolor="grey",
        alpha=0.7,
        boxstyle="round",
        pad=0.35,
    )
    colors_map = {"T+": "#1b9e77", "T-": "#d95f02"}
    markers_map = {"T+": "o", "T-": "s"}
    token_ticks = [8.388608, 16.777216, 23.393796, 33.554432, 41.94304]
    tick_labels = ["8.4", "16.8", "23.4", "33.6", "41.9"]
    summary_rows = []

    for ax, size_label in zip(axes, ["1B", "3B"]):
        subset = plot_df[plot_df["size"] == size_label].copy()
        tplus = subset[subset["train_tag"] == "T+"].sort_values("step")
        tminus = subset[subset["train_tag"] == "T-"].sort_values("step")

        for train_tag, curve_df in [("T+", tplus), ("T-", tminus)]:
            ax.plot(
                curve_df["tokens_b"],
                curve_df["mean_ppl"],
                color=colors_map[train_tag],
                marker=markers_map[train_tag],
                linewidth=2.4,
                markersize=7.5,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label=train_tag,
                zorder=3,
            )

        target_ppl = float(tminus[tminus["step"] == 10000]["mean_ppl"].iloc[0])
        final_tokens_b = float(tminus[tminus["step"] == 10000]["tokens_b"].iloc[0])
        cross_tokens_b = _interpolate_crossing(
            tplus["tokens_b"].tolist(),
            tplus["mean_ppl"].tolist(),
            target_ppl,
        )

        if cross_tokens_b is not None:
            savings_frac = (final_tokens_b - cross_tokens_b) / final_tokens_b
            ax.axhline(
                target_ppl,
                color="#8f8f8f",
                linestyle=":",
                linewidth=1.6,
                zorder=1,
            )
            ax.axvline(
                final_tokens_b,
                color="#b5b5b5",
                linestyle=":",
                linewidth=1.6,
                zorder=1,
            )
            arrow_y = target_ppl + (0.55 if size_label == "1B" else 0.48)
            ax.annotate(
                "",
                xy=(cross_tokens_b, arrow_y),
                xytext=(final_tokens_b, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="#8a8a8a", lw=1.6),
            )
            ax.text(
                (cross_tokens_b + final_tokens_b) / 2,
                arrow_y + 0.08,
                f"{savings_frac * 100:.1f}% fewer tokens",
                ha="center",
                va="bottom",
                fontsize=12,
                color="#6f6f6f",
            )
            ax.scatter(
                [cross_tokens_b],
                [target_ppl],
                color=colors_map["T+"],
                marker="D",
                s=38,
                edgecolors="black",
                linewidths=0.6,
                zorder=4,
            )
        else:
            savings_frac = np.nan

        ax.set_title(size_label, fontsize=15, weight="bold", y=0.90, pad=2, bbox=bbox_props)
        ax.set_xlabel("Training tokens (B)", fontsize=18)
        ax.set_xticks(token_ticks)
        ax.set_xticklabels(tick_labels, fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.set_xlim(7.5, 43.2)

        summary_rows.append(
            {
                "size": size_label,
                "final_tminus_target_ppl": target_ppl,
                "tplus_cross_tokens_b": cross_tokens_b,
                "tminus_final_tokens_b": final_tokens_b,
                "token_savings_frac": savings_frac,
            }
        )

    axes[0].set_ylabel("Perplexity (↓ better)", fontsize=18)

    legend_handles = [
        Line2D([], [], color=colors_map["T+"], marker=markers_map["T+"], linestyle="-", linewidth=2.4, markersize=7.5, markeredgecolor="black", label="T+"),
        Line2D([], [], color=colors_map["T-"], marker=markers_map["T-"], linestyle="-", linewidth=2.4, markersize=7.5, markeredgecolor="black", label="T-"),
        Line2D([], [], color=colors_map["T+"], marker="D", linestyle="None", markersize=6.5, markeredgecolor="black", label="T+ reaches final T- quality"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="black",
        fontsize=12,
        bbox_to_anchor=(0.5, 0.98),
    )

    output_dir = os.path.join(PLOTS_DIR, "plot13")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "global_token_efficiency_ppl.pdf")
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.10, wspace=0.18)
    plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    merged = plot_df.merge(summary_df, on="size", how="left")
    _write_plot_csv(output_dir, 13, merged)


def main():
    plot_continent_models_metadata_effect()
    plot_local_vs_global_on_local_and_global()
    plot_scaling_global_models()
    plot_cross_continent_generalization()
    plot_cross_continent_asymmetry()
    plot_sft_accuracy_apples_to_apples()
    plot_metadata_ablations()
    plot_metadata_family_full_grid()
    plot_leave_one_out_ablations()
    plot_adversarial_url_accuracy()
    plot_adversarial_url_accuracy(
        exclude_explicit=True,
        output_name="qa_adversarial_accuracy_noexplicit.pdf",
    )
    plot_compute_tradeoff_global_models()
    plot_token_efficiency_global_ppl()


if __name__ == "__main__":
    main()
