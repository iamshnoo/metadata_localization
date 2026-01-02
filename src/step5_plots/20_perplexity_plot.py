#!/usr/bin/env python3
"""
Perplexity plots from metacul/results/perplexity_eval.csv.
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

RESULTS_CSV = "/scratch/$USER$/metacul/results/perplexity_eval.csv"
PLOTS_DIR = "/scratch/$USER$/metacul/results/plots"


sns.set(font_scale=1.4)
sns.set_theme(style="whitegrid")
sns.set_context("paper", rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20})
plt.rcParams["hatch.linewidth"] = 0.1


def _parse_model_info(model_path):
    match = re.search(
        r"/models/(?P<continent>africa|america|asia|europe)_(?P<meta>with_metadata|without_metadata)_(?P<size>500m|1b)$",
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


def plot_continent_models_metadata_effect():
    # Plot: continent models (500m, 1b) on their own continent, showing metadata effects.
    # Output: /scratch/$USER$/metacul/results/plots/plot1/perplexity_continent_metadata_effect_{size}.pdf
    df = _load_perplexity_df()
    continents = ["africa", "america", "asia", "europe"]
    size_order = ["500m", "1b"]
    meta_order = ["with_metadata", "without_metadata"]
    pairs = set()
    for cont in continents:
        for size in size_order:
            for model_meta in meta_order:
                for test_meta in meta_order:
                    model_path = (
                        f"/scratch/$USER$/metacul/models/{cont}_{model_meta}_{size}"
                    )
                    test_path = (
                        "/scratch/$USER$/metacul/training_data/meco_datasets/"
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
        model_label = "[Local]" if model_meta == "with_metadata" else "Local"
        test_label = "[Test]" if test_meta == "with_metadata" else "Test"
        return f"{model_label} | {test_label}"

    plot_df["combo"] = [
        _combo_label(m, t) for m, t in zip(plot_df["model_meta"], plot_df["test_meta"])
    ]

    combo_order = [
        "[Local] | [Test]",
        "[Local] | Test",
        "Local | [Test]",
        "Local | Test",
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
                edgecolor="black",
                linewidth=0.6,
            )

        ax.set_title("")
        ax.set_xticks(x + (3 * width + gap) / 2)
        ax.set_xticklabels(continents, rotation=0, fontsize=12)
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.set_ylim(bottom=6)

        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylabel("Perplexity", fontsize=11)
        bbox_props = dict(
            facecolor="lightgrey",
            edgecolor="grey",
            alpha=0.7,
            boxstyle="round",
            pad=0.3,
        )
        ax.set_title(
            "Local test sets",
            fontsize=11,
            weight="bold",
            pad=6,
            y=0.85,
            bbox=bbox_props,
        )
        ax.legend(
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            edgecolor="black",
            fontsize=10,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.99),
        )

        output_dir = os.path.join(PLOTS_DIR, "plot1")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"perplexity_continent_metadata_effect_{size}.pdf"
        )
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.06)
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
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
    # Output: /scratch/$USER$/metacul/results/plots/plot2/perplexity_local_vs_global_{size}.pdf
    df = _load_perplexity_df()
    pairs = set()

    continents = ["africa", "america", "asia", "europe"]
    region_labels = ["Africa", "America", "Asia", "Europe", "All"]
    size_order = ["500m", "1b"]

    combos = [
        {"label": "[Global] | [Test]", "scope": "global", "meta": "with_metadata"},
        {"label": "[Local] | [Test]", "scope": "local", "meta": "with_metadata"},
        {"label": "Global | Test", "scope": "global", "meta": "without_metadata"},
        {"label": "Local | Test", "scope": "local", "meta": "without_metadata"},
    ]

    records = []
    for size in size_order:
        for cont in continents:
            for combo in combos:
                meta = combo["meta"]
                if combo["scope"] == "local":
                    model_path = f"/scratch/$USER$/metacul/models/{cont}_{meta}_{size}"
                else:
                    model_path = f"/scratch/$USER$/metacul/models/combined_{meta}_{size}"
                test_path = (
                    f"/scratch/$USER$/metacul/training_data/meco_datasets/continents/{cont}/{meta}/"
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
                f"/scratch/$USER$/metacul/training_data/meco_datasets/combined/{meta}/"
            )
            if combo["scope"] == "global":
                model_path = f"/scratch/$USER$/metacul/models/combined_{meta}_{size}"
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
                        f"/scratch/$USER$/metacul/models/{cont}_{meta}_{size}"
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
        "[Global] | [Test]": {"color": "#a6cee3", "hatch": "o"},
        "Global | Test": {"color": "#7f7f7f", "hatch": ""},
        "[Local] | [Test]": {"color": "#9ad1a6", "hatch": "\\"},
        "Local | Test": {"color": "#d9d9d9", "hatch": ""},
    }
    plot_order = [
        "[Global] | [Test]",
        "Global | Test",
        "[Local] | [Test]",
        "Local | Test",
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
                edgecolor="black",
                linewidth=0.6,
            )

        ax.set_xticks(x_pos + (3 * width + gap) / 2)
        ax.set_xticklabels(labels, rotation=0, fontsize=12)
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.set_ylim(bottom=6)
        ax.tick_params(axis="y", labelsize=12)

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
            fontsize=11,
            weight="bold",
            pad=6,
            y=0.83,
            bbox=bbox_props,
        )
        axes[1].set_title(
            "Global test set",
            fontsize=11,
            weight="bold",
            pad=6,
            y=0.83,
            bbox=bbox_props,
        )
        axes[0].set_ylabel("Perplexity", fontsize=11)
        axes[1].set_ylabel("")
        if y_top is not None:
            axes[0].set_ylim(top=y_top)
            axes[1].set_ylim(top=y_top)
        axes[0].legend(
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            edgecolor="black",
            fontsize=10,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.99),
        )

        output_dir = os.path.join(PLOTS_DIR, "plot2")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"perplexity_local_vs_global_{size}.pdf"
        )
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

    output_dir = os.path.join(PLOTS_DIR, "plot2")
    subset = _subset_by_pairs(df, pairs)
    _write_plot_csv(output_dir, 2, subset)


def plot_scaling_global_models():
    # Plot: scaling impact for global models as a dumbbell with delta annotations.
    # Output: /scratch/$USER$/metacul/results/plots/plot3/perplexity_scaling_global_delta.pdf
    df = _load_perplexity_df()
    pairs = set()

    regions = ["africa", "america", "asia", "europe", "combined"]
    region_labels = ["Africa", "America", "Asia", "Europe", "All"]
    meta_order = ["with_metadata", "without_metadata"]

    records = []
    for meta in meta_order:
        for size in ["500m", "1b"]:
            model_path = f"/scratch/$USER$/metacul/models/combined_{meta}_{size}"
            for region, label in zip(regions, region_labels):
                if region == "combined":
                    test_path = (
                        f"/scratch/$USER$/metacul/training_data/meco_datasets/combined/{meta}/"
                    )
                else:
                    test_path = (
                        f"/scratch/$USER$/metacul/training_data/meco_datasets/continents/{region}/{meta}/"
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
    colors = {"500m": "#f4a3a3", "1b": "#9ad1a6"}
    markers = {"500m": "o", "1b": "s"}

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
                row_500 = subset[
                    (subset["size"] == "500m") & (subset["region"] == region)
                ]
                row_1b = subset[
                    (subset["size"] == "1b") & (subset["region"] == region)
                ]
                if row_500.empty or row_1b.empty:
                    continue
                mean_500 = row_500["mean_ppl"].values[0]
                mean_1b = row_1b["mean_ppl"].values[0]
                delta = mean_1b - mean_500

                x_pos = x_positions[idx] + offsets[meta]
                ax.plot(
                    [x_pos, x_pos],
                    [mean_500, mean_1b],
                    color=line_color,
                    linestyle=meta_styles[meta],
                    linewidth=1.5,
                )
                ax.scatter(
                    [x_pos],
                    [mean_500],
                    color=colors["500m"],
                    edgecolor="black",
                    marker=markers["500m"],
                    s=70,
                    zorder=3,
                    label="500m (red)" if idx == 0 and meta == "with_metadata" else None,
                )
                ax.scatter(
                    [x_pos],
                    [mean_1b],
                    color=colors["1b"],
                    edgecolor="black",
                    marker=markers["1b"],
                    s=70,
                    zorder=3,
                    label="1b (green)" if idx == 0 and meta == "with_metadata" else None,
                )
                ax.text(
                    x_pos,
                    min(mean_500, mean_1b) - 0.2,
                    f"Δ {delta:+.2f}",
                    va="top",
                    ha="center",
                    fontsize=10,
                    color="#444444",
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylim(bottom=6)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    _plot_panel(axes[0], local_labels)
    _plot_panel(axes[1], global_labels)
    axes[0].set_ylabel("Perplexity", fontsize=11)
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
        0.05,
        "Local test sets",
        transform=axes[0].transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        weight="bold",
        bbox=bbox_props,
    )
    axes[1].text(
        0.5,
        0.05,
        "Global test set",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
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
            label="500m (red)",
            markersize=8,
        ),
        plt.Line2D(
            [0],
            [0],
            marker=markers["1b"],
            color="none",
            markerfacecolor=colors["1b"],
            markeredgecolor="black",
            label="1b (green)",
            markersize=8,
        ),
        plt.Line2D(
            [0],
            [0],
            color=line_color,
            linestyle="solid",
            label="[Global] | [Test]",
        ),
        plt.Line2D(
            [0],
            [0],
            color=line_color,
            linestyle="dashed",
            label="Global | Test",
        ),
    ]
    axes[0].legend(
        handles=legend_handles,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="black",
        fontsize=10,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.99),
    )

    output_dir = os.path.join(PLOTS_DIR, "plot3")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "perplexity_scaling_global_delta.pdf")
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    subset = _subset_by_pairs(df, pairs)
    _write_plot_csv(output_dir, 3, subset)


def plot_cross_continent_generalization():
    # Plot: cross-continent generalization (local models/tests), with deltas.
    # Output: /scratch/$USER$/metacul/results/plots/plot4/perplexity_cross_continent_{size}.pdf
    df = _load_perplexity_df()
    pairs = set()

    continents = ["africa", "america", "asia", "europe"]
    labels = ["Africa", "America", "Asia", "Europe"]
    size_order = ["500m", "1b"]

    def _matrix(model_meta, test_meta, size):
        values = []
        for train_cont in continents:
            row_vals = []
            model_path = f"/scratch/$USER$/metacul/models/{train_cont}_{model_meta}_{size}"
            for test_cont in continents:
                test_path = (
                    f"/scratch/$USER$/metacul/training_data/meco_datasets/continents/{test_cont}/{test_meta}/"
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

        delta_with = without_with - with_with
        delta_without = without_without - with_without

        avg_delta_with = delta_with.mean(axis=0)
        avg_delta_without = delta_without.mean(axis=0)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        value_min = np.nanmin([with_with.values, without_with.values, with_without.values, without_without.values])
        value_max = np.nanmax([with_with.values, without_with.values, with_without.values, without_without.values])
        delta_max = np.nanmax([np.abs(delta_with.values), np.abs(delta_without.values)])
        bar_min = float(np.nanmin([avg_delta_with.values, avg_delta_without.values]))
        bar_max = float(np.nanmax([avg_delta_with.values, avg_delta_without.values]))

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

        def _annotate_heatmap(ax, data, cmap, norm):
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
                        fontsize=12,
                        color=text_color,
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
        axes[0, 0].collections[0].colorbar.ax.tick_params(labelsize=11)
        _annotate_heatmap(axes[0, 0], with_with, plt.get_cmap("Greens"), value_norm)
        axes[0, 0].set_title(
            "[Local] | [Test]",
            fontsize=11,
            weight="bold",
            pad=12,
            bbox=bbox_props,
        )
        axes[0, 0].set_xlabel("")
        axes[0, 0].set_ylabel("Train Region", fontsize=12)

        sns.heatmap(
            without_with,
            ax=axes[0, 1],
            cmap="YlOrBr",
            norm=value_norm,
            **heatmap_kws,
        )
        axes[0, 1].collections[0].colorbar.ax.locator = MaxNLocator(4)
        axes[0, 1].collections[0].colorbar.update_ticks()
        axes[0, 1].collections[0].colorbar.ax.tick_params(labelsize=11)
        _annotate_heatmap(axes[0, 1], without_with, plt.get_cmap("YlOrBr"), value_norm)
        axes[0, 1].set_title(
            "Local | [Test]",
            fontsize=11,
            weight="bold",
            pad=12,
            bbox=bbox_props,
        )
        axes[0, 1].set_xlabel("")
        axes[0, 1].set_ylabel("")

        sns.heatmap(
            delta_with,
            ax=axes[0, 2],
            cmap="vlag",
            norm=delta_norm,
            **heatmap_kws,
        )
        axes[0, 2].collections[0].colorbar.ax.locator = MaxNLocator(4)
        axes[0, 2].collections[0].colorbar.update_ticks()
        axes[0, 2].collections[0].colorbar.ax.tick_params(labelsize=11)
        _annotate_heatmap(axes[0, 2], delta_with, plt.get_cmap("vlag"), delta_norm)
        axes[0, 2].set_title(
            "Δ (Local − [Local]) | [Test]",
            fontsize=11,
            weight="bold",
            pad=12,
            bbox=bbox_props,
        )
        axes[0, 2].set_xlabel("")
        axes[0, 2].set_ylabel("")

        axes[0, 3].set_title(
            "Avg Δ by test | [Test]",
            fontsize=11,
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
        bar_ax_top.set_ylabel("Avg Δ (↑ better)", fontsize=11)
        bar_ax_top.set_xticks(bar_x)
        bar_ax_top.set_xticklabels(labels, fontsize=12)
        bar_ax_top.tick_params(axis="x", labelsize=12)
        bar_ax_top.tick_params(axis="y", labelsize=13)
        for i, v in enumerate(top_vals):
            bar_ax_top.text(
                i,
                v + (0.05 if v >= 0 else -0.05),
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=10,
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
        axes[1, 0].collections[0].colorbar.ax.tick_params(labelsize=11)
        _annotate_heatmap(axes[1, 0], with_without, plt.get_cmap("OrRd"), value_norm)
        axes[1, 0].set_title(
            "[Local] | Test",
            fontsize=11,
            weight="bold",
            pad=12,
            bbox=bbox_props,
        )
        axes[1, 0].set_xlabel("Test Region", fontsize=12)
        axes[1, 0].set_ylabel("Train Region", fontsize=12)

        sns.heatmap(
            without_without,
            ax=axes[1, 1],
            cmap="Greys",
            norm=value_norm,
            **heatmap_kws,
        )
        axes[1, 1].collections[0].colorbar.ax.locator = MaxNLocator(4)
        axes[1, 1].collections[0].colorbar.update_ticks()
        axes[1, 1].collections[0].colorbar.ax.tick_params(labelsize=11)
        _annotate_heatmap(axes[1, 1], without_without, plt.get_cmap("Greys"), value_norm)
        axes[1, 1].set_title(
            "Local | Test",
            fontsize=11,
            weight="bold",
            pad=12,
            bbox=bbox_props,
        )
        axes[1, 1].set_xlabel("Test Region", fontsize=12)
        axes[1, 1].set_ylabel("")

        sns.heatmap(
            delta_without,
            ax=axes[1, 2],
            cmap="vlag",
            norm=delta_norm,
            **heatmap_kws,
        )
        axes[1, 2].collections[0].colorbar.ax.locator = MaxNLocator(4)
        axes[1, 2].collections[0].colorbar.update_ticks()
        axes[1, 2].collections[0].colorbar.ax.tick_params(labelsize=11)
        _annotate_heatmap(axes[1, 2], delta_without, plt.get_cmap("vlag"), delta_norm)
        axes[1, 2].set_title(
            "Δ (Local − [Local]) | Test",
            fontsize=11,
            weight="bold",
            pad=12,
            bbox=bbox_props,
        )
        axes[1, 2].set_xlabel("Test Region", fontsize=12)
        axes[1, 2].set_ylabel("")

        axes[1, 3].set_title(
            "Avg Δ by test | Test",
            fontsize=11,
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
        bar_ax_bottom.set_xlabel("Test Region", fontsize=12)
        bar_ax_bottom.set_ylabel("Avg Δ (↑ better)", fontsize=11)
        bar_ax_bottom.set_xticks(bar_x)
        bar_ax_bottom.set_xticklabels(labels, fontsize=12)
        bar_ax_bottom.tick_params(axis="x", labelsize=12)
        bar_ax_bottom.tick_params(axis="y", labelsize=13)
        for i, v in enumerate(bottom_vals):
            bar_ax_bottom.text(
                i,
                v + (0.05 if v >= 0 else -0.05),
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=10,
            )

        for ax_row in axes:
            for ax in ax_row:
                ax.tick_params(axis="y", labelsize=13)
                ax.tick_params(axis="x", labelsize=13)

        output_dir = os.path.join(PLOTS_DIR, "plot4")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"perplexity_cross_continent_{size}.pdf"
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

    output_dir = os.path.join(PLOTS_DIR, "plot4")
    subset = _subset_by_pairs(df, pairs)
    _write_plot_csv(output_dir, 4, subset)


def plot_cross_continent_asymmetry():
    # Plot: cross-continent asymmetry for local models/tests ([Local] | [Test]).
    # Output: /scratch/$USER$/metacul/results/plots/plot5/perplexity_asymmetry_{size}.pdf
    df = _load_perplexity_df()
    pairs = set()

    continents = ["africa", "america", "asia", "europe"]
    labels = ["Africa", "America", "Asia", "Europe"]
    size_order = ["500m", "1b"]

    def _matrix(size):
        values = []
        for train_cont in continents:
            row_vals = []
            model_path = f"/scratch/$USER$/metacul/models/{train_cont}_with_metadata_{size}"
            for test_cont in continents:
                test_path = (
                    f"/scratch/$USER$/metacul/training_data/meco_datasets/continents/{test_cont}/with_metadata/"
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
        asym = base - base.T
        asym_values = asym.values
        asym_max = np.nanmax(np.abs(asym_values))

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
        ax.collections[0].colorbar.ax.tick_params(labelsize=11)

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
                    fontsize=12,
                    color=text_color,
                )

        ax.set_title(
            "Asymmetry (ppl(i→j) − ppl(j→i))",
            fontsize=11,
            weight="bold",
            pad=12,
            bbox=bbox_props,
        )
        ax.set_xlabel("Test Region", fontsize=12)
        ax.set_ylabel("Train Region", fontsize=12)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=13)

        output_dir = os.path.join(PLOTS_DIR, "plot5")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"perplexity_asymmetry_{size}.pdf")
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

    output_dir = os.path.join(PLOTS_DIR, "plot5")
    subset = _subset_by_pairs(df, pairs)
    _write_plot_csv(output_dir, 5, subset)


def plot_sft_accuracy_apples_to_apples():
    # Plot: apples-to-apples QA accuracy (answered_by_all=1) across continents.
    # Output: /scratch/$USER$/metacul/results/plots/plot8/accuracy_apples_to_apples.pdf
    df = pd.read_csv("/scratch/$USER$/metacul/results/qa_metacul_eval.csv")

    df = df[df["answered_by_all"] == 1]
    df["continent"] = df["continent"].str.capitalize()

    models = {
        "Global | QA": {
            "correct": "custom_without_metadata_correct",
            "incorrect": "custom_without_metadata_incorrect",
            "color": "#f7a1b5",
            "hatch": "",
        },
        "[Global] | [QA]": {
            "correct": "custom_with_metadata_correct",
            "incorrect": "custom_with_metadata_incorrect",
            "color": "#a6cee3",
            "hatch": "o",
        },
        "LLaMA-3 | QA": {
            "correct": "llama3_chat_without_metadata_correct",
            "incorrect": "llama3_chat_without_metadata_incorrect",
            "color": "#d9d9d9",
            "hatch": "",
        },
        "LLaMA-3 | [QA]": {
            "correct": "llama3_chat_with_metadata_correct",
            "incorrect": "llama3_chat_with_metadata_incorrect",
            "color": "#7f7f7f",
            "hatch": "..",
        },
    }

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
        "[Global] | [QA]",
        "Global | QA",
        "LLaMA-3 | [QA]",
        "LLaMA-3 | QA",
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
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_xticks(x + (3 * width + gap) / 2)
    weight_labels = [
        f"{c}\n({cont_question_counts.get(c, 0)} Q/s)" for c in continents
    ]
    ax.set_xticklabels(weight_labels, fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0.55, 0.80)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.legend(
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="black",
        fontsize=10,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.99),
    )

    micro_models = ["[Global] | [QA]", "LLaMA-3 | QA"]
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
                fontsize=9,
                color="#333333",
            )
    ax_summary.set_xticks(x_summary)
    ax_summary.set_xticklabels(micro_models, fontsize=10, rotation=20)
    ax_summary.set_xlabel("")
    ax_summary.tick_params(axis="y", labelsize=12)
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
        "Overall (micro average)",
        fontsize=10,
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
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    _write_plot_csv(output_dir, 8, df)


def plot_metadata_ablations():
    # Plot: metadata ablations across checkpoints on metadata/combined test sets.
    # Output: /scratch/$USER$/metacul/results/plots/plot6/perplexity_metadata_ablations_1b.pdf
    df = _load_perplexity_df()
    pairs = set()

    steps = [2000, 4000, 8000, 10000]
    step_labels = ["2k", "4k", "8k", "10k"]

    metadata_tests = {
        "url": "/scratch/$USER$/metacul/training_data/meco_datasets/combined_only_url/with_metadata/",
        "url_continent": "/scratch/$USER$/metacul/training_data/meco_datasets/combined_only_url_continent/with_metadata/",
        "url_country": "/scratch/$USER$/metacul/training_data/meco_datasets/combined_only_url_country/with_metadata/",
    }

    combined_tests = {
        "with": "/scratch/$USER$/metacul/training_data/meco_datasets/combined/with_metadata/",
        "without": "/scratch/$USER$/metacul/training_data/meco_datasets/combined/without_metadata/",
    }

    model_groups = {
        "combined_with": {
            "final": "/scratch/$USER$/metacul/models/combined_with_metadata_1b",
            "steps": "/scratch/$USER$/metacul/models/ablation_intermediates/metadata/combined_with_metadata_1b_step{step}k",
        },
        "combined_without": {
            "final": "/scratch/$USER$/metacul/models/combined_without_metadata_1b",
            "steps": "/scratch/$USER$/metacul/models/ablation_intermediates/metadata/combined_without_metadata_1b_step{step}k",
        },
        "url": {
            "final": "/scratch/$USER$/metacul/models/ablations/metadata/combined_only_url_with_metadata_1b",
            "steps": "/scratch/$USER$/metacul/models/ablation_intermediates/metadata/combined_only_url_with_metadata_1b_step{step}k",
        },
        "url_continent": {
            "final": "/scratch/$USER$/metacul/models/ablations/metadata/combined_only_url_continent_with_metadata_1b",
            "steps": "/scratch/$USER$/metacul/models/ablation_intermediates/metadata/combined_only_url_continent_with_metadata_1b_step{step}k",
        },
        "url_country": {
            "final": "/scratch/$USER$/metacul/models/ablations/metadata/combined_only_url_country_with_metadata_1b",
            "steps": "/scratch/$USER$/metacul/models/ablation_intermediates/metadata/combined_only_url_country_with_metadata_1b_step{step}k",
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
        "combined_with": "[URL][Country][Continent]",
        "combined_without": "NoMetadata",
        "url": "[URL]",
        "url_continent": "[URL][Continent]",
        "url_country": "[URL][Country]",
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
            for key, values in series.items():
                ci_lows, ci_highs = series_ci[key]
                x_vals = np.arange(len(step_labels))
                y_vals = np.array(values, dtype=float)
                lo_vals = np.array(ci_lows, dtype=float)
                hi_vals = np.array(ci_highs, dtype=float)
                ax.plot(
                    x_vals,
                    y_vals,
                    marker=markers[key],
                    color=colors[key],
                    linestyle=linestyles[key],
                    label=labels[key],
                    linewidth=2,
                    markersize=6,
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    zorder=3,
                )
                mask = ~np.isnan(y_vals) & ~np.isnan(lo_vals) & ~np.isnan(hi_vals)
                if np.any(mask):
                    ax.fill_between(
                        x_vals,
                        lo_vals,
                        hi_vals,
                        color=colors[key],
                        alpha=0.35,
                        linewidth=0.4,
                        edgecolor=colors[key],
                        where=mask,
                        interpolate=True,
                        zorder=1,
                    )
                all_values.extend([v for v in values if not np.isnan(v)])
            ax.set_title(
                title,
                fontsize=10,
                weight="bold",
                pad=6,
                y=0.90,
                bbox=bbox_props,
            )
            ax.set_xlabel("")
            ax.set_xticks(x_vals)
            ax.set_xticklabels(step_labels)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
            ax.tick_params(axis="x", labelsize=12)
            ax.tick_params(axis="y", labelsize=12)
            ax.set_ylim(bottom=9)

        if all_values:
            y_max = max(all_values)
            for ax in axes:
                ax.set_ylim(top=y_max + 1.0)

        axes[0].set_ylabel("Perplexity", fontsize=11)
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
            fontsize=10,
            loc="upper center",
            ncol=len(legend_order),
            bbox_to_anchor=(0.5, 0.9),
        )

        output_path = os.path.join(
            output_dir, f"perplexity_metadata_ablations_1b{output_suffix}.pdf"
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.82)
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

    output_dir = os.path.join(PLOTS_DIR, "plot6")
    os.makedirs(output_dir, exist_ok=True)
    main_panels = [
        ("own", "[URL]"),
        ("combined_with", "[ALL]"),
        ("combined_without", "NoMetadata"),
    ]
    appendix_panels = [
        ("own", "[URL] / [URL][Continent] / [URL][Country]"),
        ("combined_with", "[ALL]"),
        ("combined_without", "NoMetadata"),
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


def plot_leave_one_out_ablations():
    # Plot: leave-one-out ablations (with/without metadata).
    # Output: /scratch/$USER$/metacul/results/plots/plot7/leave_one_out_{meta}.pdf
    df = _load_perplexity_df()
    pairs = set()

    steps = [2000, 4000, 8000, 10000]
    step_labels = ["2k", "4k", "8k", "10k"]
    continents = ["africa", "america", "asia", "europe"]

    combined_tests = {
        "with_metadata": "/scratch/$USER$/metacul/training_data/meco_datasets/combined/with_metadata/",
        "without_metadata": "/scratch/$USER$/metacul/training_data/meco_datasets/combined/without_metadata/",
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
            return f"/scratch/$USER$/metacul/models/combined_{meta}_1b"
        return (
            "/scratch/$USER$/metacul/models/ablation_intermediates/metadata/"
            f"combined_{meta}_1b_step{step // 1000}k"
        )

    def _loo_model_path(continent, meta, step):
        if step == 10000:
            return (
                "/scratch/$USER$/metacul/models/ablations/leave_one_out/"
                f"combined_no_{continent}_{meta}_1b"
            )
        return (
            "/scratch/$USER$/metacul/models/ablation_intermediates/leave_one_out/"
            f"combined_no_{continent}_{meta}_1b_step{step // 1000}k"
        )

    colors = {"combined": "#7f7f7f", "loo": "#5fae78"}
    markers = {"combined": "o", "loo": "s"}
    linestyles = {"combined": "--", "loo": "-"}

    bbox_props = dict(
        facecolor="lightgrey",
        edgecolor="grey",
        alpha=0.7,
        boxstyle="round",
        pad=0.3,
    )

    for meta in ["with_metadata", "without_metadata"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

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
                    f"/scratch/$USER$/metacul/training_data/meco_datasets/continents/{cont}/{meta}/"
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

        combined_label = "[ALL]" if meta == "with_metadata" else "ALL"
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
            loo_label = (
                f"[No{cont.capitalize()}] - [ALL]"
                if meta == "with_metadata"
                else f"No{cont.capitalize()} - All"
            )
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
            "Δ on [Local] test set" if meta == "with_metadata" else "Δ on Local test set",
            fontsize=10,
            weight="bold",
            pad=6,
            y=0.90,
            bbox=bbox_props,
        )

        for cont in continents:
            loo_label = (
                f"[No{cont.capitalize()}]"
                if meta == "with_metadata"
                else f"No{cont.capitalize()}"
            )
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
            "Δ on [ALL] test set" if meta == "with_metadata" else "Δ on ALL test set",
            fontsize=10,
            weight="bold",
            pad=6,
            y=0.90,
            bbox=bbox_props,
        )

        for ax in axes:
            ax.set_xlabel("")
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
            ax.tick_params(axis="x", labelsize=12)
            ax.tick_params(axis="y", labelsize=12)

        axes[0].set_ylabel("Δ Perplexity", fontsize=11)
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
            loo_label = (
                f"[No{cont.capitalize()}] - [ALL]"
                if meta == "with_metadata"
                else f"No{cont.capitalize()} - All"
            )
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    color=loo_continent_colors[cont],
                    marker=loo_continent_markers[cont],
                    linestyle=linestyles["loo"],
                    linewidth=2,
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
            fontsize=10,
            loc="upper center",
            ncol=5,
            bbox_to_anchor=(0.5, 0.92),
        )

        output_dir = os.path.join(PLOTS_DIR, "plot7")
        os.makedirs(output_dir, exist_ok=True)
        suffix = "with_metadata" if meta == "with_metadata" else "without_metadata"
        output_path = os.path.join(output_dir, f"leave_one_out_{suffix}.pdf")
        plt.tight_layout()
        plt.subplots_adjust(top=0.84)
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

    output_dir = os.path.join(PLOTS_DIR, "plot7")
    subset = _subset_by_pairs(df, pairs)
    _write_plot_csv(output_dir, 7, subset)


def main():
    plot_continent_models_metadata_effect()
    plot_local_vs_global_on_local_and_global()
    plot_scaling_global_models()
    plot_cross_continent_generalization()
    plot_cross_continent_asymmetry()
    plot_sft_accuracy_apples_to_apples()
    plot_metadata_ablations()
    plot_leave_one_out_ablations()


if __name__ == "__main__":
    main()
