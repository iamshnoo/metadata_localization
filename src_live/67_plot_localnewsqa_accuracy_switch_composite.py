#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPO_ROOT = Path("/path/to/metacul")
ACCURACY_CSV = REPO_ROOT / "results/plots/plot8/plot_8_pretrained_target_split_seed41_bootstrap.csv"
SWITCH_CSV = REPO_ROOT / "results/analysis/localnewsqa_locale_switch_seed41/summary.csv"
GAIN_CSV = REPO_ROOT / "results/appendix_model_gain_tables_20260505_check/localnewsqa_model_gains_long.csv"
OUT_DIR = REPO_ROOT / "results/plots/plot8"
LATEX_FIG = REPO_ROOT / "latex/figs/main/8_localnewsqa_accuracy_switch_composite.pdf"
LATEX_TARGET_PANEL = REPO_ROOT / "latex/figs/main/8_localnewsqa_accuracy_target_panel.pdf"
LATEX_SWITCH_PANEL = REPO_ROOT / "latex/figs/main/8_localnewsqa_accuracy_switch_panel.pdf"
LATEX_ONECOL = REPO_ROOT / "latex/figs/main/8_localnewsqa_accuracy_switch_onecol.pdf"


def accuracy_row(df, family, series, split):
    row = df[(df["family"] == family) & (df["series"] == series) & (df["split"] == split)]
    if row.empty:
        raise KeyError(f"missing accuracy row: {family} {series} {split}")
    return row.iloc[0]


def accuracy_xerr(row):
    x = float(row["accuracy"]) * 100.0
    if "ci_low" in row and "ci_high" in row:
        lo = float(row["ci_low"]) * 100.0
        hi = float(row["ci_high"]) * 100.0
        return np.array([[x - lo], [hi - x]])
    return np.array([[0.0], [0.0]])


def accuracy_interval_pct(row):
    y = float(row["accuracy"]) * 100.0
    if "ci_low" in row and "ci_high" in row:
        return y, float(row["ci_low"]) * 100.0, float(row["ci_high"]) * 100.0
    return y, y, y


def gain_row(df, family, split):
    metric = {
        "Explicit": "localnewsqa_explicit",
        "Ambiguous": "localnewsqa_ambiguous",
        "Overall": "localnewsqa_overall",
    }[split]
    row = df[
        (df["row_key"] == f"maple_{family.lower()}")
        & (df["track"] == "base")
        & (df["metric_key"] == metric)
    ]
    if row.empty:
        raise KeyError(f"missing gain row: {family} {split}")
    return row.iloc[0]


def draw_accuracy_gain_panel(ax, acc, gains, families, splits, scale_colors, compact=False):
    short_labels = {
        "Explicit": "expl.",
        "Ambiguous": "ambig.",
        "Overall": "all",
    }
    title_fs = 9.8 if compact else 12.5
    label_fs = 7.6 if compact else 9.2
    tick_fs = 7.5 if compact else 9.0
    value_fs = 6.5 if compact else 7.9
    bar_width = 0.22 if compact else 0.20
    label_pad = 0.18 if compact else 0.22
    edge_colors = {
        "1B": "#4E7890",
        "3B": "#AE7132",
    }

    ax.set_title("(A) Target accuracy delta", loc="left", fontsize=title_fs, fontweight="bold", pad=2.0)

    x_centers = np.arange(len(splits))
    x_lookup = dict(zip(splits, x_centers))
    offsets = {"1B": -0.12, "3B": 0.12}

    gain_values = {}
    for split in splits:
        gain_values[split] = {}
        for family in families:
            grow = gain_row(gains, family, split)
            gain_values[split][family] = (
                float(grow["delta"]),
                float(grow["ci_low"]),
                float(grow["ci_high"]),
            )
    max_gain_high = max(
        hi
        for split_values in gain_values.values()
        for _, _, hi in split_values.values()
    )
    y_top = max(10.0, np.ceil((max_gain_high + label_pad + 0.95) * 2.0) / 2.0)

    for split in splits:
        x = x_lookup[split]
        if split == "Overall":
            band_color = "#eef1f4"
            band_alpha = 1.0
        elif split == "Explicit":
            band_color = "#f8f8f8"
            band_alpha = 1.0
        else:
            band_color = "#eef1f4"
            band_alpha = 1.0
        ax.axvspan(
            x - 0.38,
            x + 0.38,
            color=band_color,
            alpha=band_alpha,
            zorder=0,
        )

    for boundary in x_centers[:-1] + 0.5:
        ax.axvline(
            boundary,
            color="#7f8893",
            linewidth=0.9 if compact else 1.05,
            linestyle="-",
            alpha=0.95,
            zorder=2,
        )

    for family in families:
        for split in splits:
            gain, lo, hi = gain_values[split][family]
            x = x_lookup[split] + offsets[family]
            ax.bar(
                [x],
                [gain],
                width=bar_width,
                color=scale_colors[family],
                edgecolor=edge_colors[family],
                linewidth=0.95 if compact else 1.05,
                zorder=3,
            )
            ax.errorbar(
                [x],
                [gain],
                yerr=np.array([[gain - lo], [hi - gain]]),
                fmt="none",
                ecolor=edge_colors[family],
                elinewidth=1.0 if compact else 1.15,
                capsize=2.5 if compact else 3.0,
                zorder=4,
            )
            ax.text(
                x,
                hi + label_pad,
                f"+{gain:.1f}",
                ha="center",
                va="bottom",
                fontsize=value_fs,
                fontweight="bold",
                color="black",
                zorder=5,
            )

    ax.axhline(0, color="#6f7780", linewidth=0.8, zorder=1)
    ax.set_xlim(-0.55, len(splits) - 0.45)
    ax.set_ylim(-0.35, y_top)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([short_labels[s] if compact else s for s in splits], fontsize=label_fs)
    ylabel = (
        "(T+, I+) - (T-, I-) (accuracy points)"
        if compact
        else r"$\Delta$ accuracy, (T+, I+) - (T-, I-) (accuracy points)"
    )
    ax.set_ylabel(ylabel, fontsize=8.0 if compact else 9.4, labelpad=1.0)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.tick_params(axis="x", labelsize=tick_fs, pad=1.5)
    ax.tick_params(axis="y", labelsize=tick_fs, pad=1.5)
    ax.grid(axis="y", color="#d3d8de", linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)
    # legend is added once at the figure level in caller.


def switch_row(df, family, variant):
    row = df[(df["family"] == family) & (df["variant"] == variant)]
    if row.empty:
        raise KeyError(f"missing switch row: {family} {variant}")
    return row.iloc[0]


def draw_unified_localnewsqa_axis(ax, acc, sw, family, show_rows):
    rows = [
        ("Accuracy", "All", "accuracy", "Overall"),
        ("Accuracy", "Explicit", "accuracy", "Explicit"),
        ("Accuracy", "Ambiguous", "accuracy", "Ambiguous"),
        ("Exact pair", "Ambiguous", "exact_pair", None),
        ("Margin switch", "Ambiguous", "margin_switch", None),
    ]
    y_positions = np.arange(len(rows))[::-1]
    plus_color = "#2f9e44"
    minus_color = "#8a919c"
    connector_color = "#5f6b76"

    def metric_values(metric, split):
        if metric == "accuracy":
            plus = accuracy_row(acc, family, f"{family} T+/I+", split)
            minus = accuracy_row(acc, family, f"{family} T-/I-", split)
            plus_val, plus_lo, plus_hi = accuracy_interval_pct(plus)
            minus_val, minus_lo, minus_hi = accuracy_interval_pct(minus)
            return (minus_val, minus_lo, minus_hi), (plus_val, plus_lo, plus_hi)

        plus = switch_row(sw, f"MAPLE {family}", "tplus_eplus")
        minus = switch_row(sw, f"MAPLE {family}", "tminus_eminus")
        if metric == "exact_pair":
            keys = (
                "exact_pair_accuracy_pct",
                "exact_pair_ci_low_pct",
                "exact_pair_ci_high_pct",
            )
        else:
            keys = (
                "margin_switch_rate_pct",
                "margin_switch_ci_low_pct",
                "margin_switch_ci_high_pct",
            )
        return (
            float(minus[keys[0]]),
            float(minus[keys[1]]),
            float(minus[keys[2]]),
        ), (
            float(plus[keys[0]]),
            float(plus[keys[1]]),
            float(plus[keys[2]]),
        )

    for idx, ((metric_label, split_label, metric, split), y) in enumerate(zip(rows, y_positions)):
        band = "#eef7ef" if split_label == "Ambiguous" else "#f6f8fa"
        ax.axhspan(y - 0.42, y + 0.42, color=band, alpha=0.95 if idx % 2 == 0 else 0.62, zorder=0)

        minus, plus = metric_values(metric, split)
        minus_val, minus_lo, minus_hi = minus
        plus_val, plus_lo, plus_hi = plus
        ax.plot(
            [minus_val, plus_val],
            [y, y],
            color=connector_color,
            linewidth=1.45,
            alpha=0.82,
            zorder=2,
        )
        ax.errorbar(
            [minus_val],
            [y],
            xerr=[[minus_val - minus_lo], [minus_hi - minus_val]],
            fmt="s",
            markersize=5.4,
            markerfacecolor=minus_color,
            markeredgecolor="black",
            markeredgewidth=0.75,
            ecolor=minus_color,
            elinewidth=0.95,
            capsize=2.0,
            zorder=4,
        )
        ax.errorbar(
            [plus_val],
            [y],
            xerr=[[plus_val - plus_lo], [plus_hi - plus_val]],
            fmt="o",
            markersize=5.8,
            markerfacecolor=plus_color,
            markeredgecolor="black",
            markeredgewidth=0.75,
            ecolor=plus_color,
            elinewidth=0.95,
            capsize=2.0,
            zorder=5,
        )
        delta = plus_val - minus_val
        ax.text(
            min(44.5, max(plus_val, minus_val) + 1.3),
            y,
            f"+{delta:.1f}",
            ha="left",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color="black",
            zorder=6,
        )

    ax.set_title(f"{family}", fontsize=9.2, fontweight="bold", pad=3.0)
    ax.set_xlim(-0.8, 45.0)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xticks([0, 15, 30, 45])
    ax.tick_params(axis="x", labelsize=7.4, pad=1.5)
    ax.grid(axis="x", color="#d3d8de", linewidth=0.55, alpha=0.72)
    ax.set_axisbelow(True)
    if show_rows:
        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [f"{metric}\n{split}" for metric, split, _, _ in rows],
            fontsize=7.0,
            linespacing=0.88,
        )
    else:
        ax.set_yticks(y_positions)
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, pad=2.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("black")


def add_joint_figure_legend(fig, scale_colors, scale_edges):
    legend_handles = [
        Patch(
            facecolor=scale_colors["1B"],
            edgecolor=scale_edges["1B"],
            linewidth=1.0,
            label="1B",
        ),
        Patch(
            facecolor=scale_colors["3B"],
            edgecolor=scale_edges["3B"],
            linewidth=1.0,
            label="3B",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            markersize=5.4,
            label="(T-, I-) baseline (B)",
        ),
    ]
    fig_width = fig.get_size_inches()[0]
    is_compact = fig_width < 4.5
    anchor_x = 0.58 if is_compact else 0.56
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(anchor_x, 0.952 if is_compact else 0.985),
        fontsize=7.2 if is_compact else 8.0,
        frameon=True,
        ncol=3,
        columnspacing=0.55 if is_compact else 0.58,
        handlelength=0.8 if is_compact else 1.0,
        handletextpad=0.26 if is_compact else 0.3,
        borderpad=0.34 if is_compact else 0.4,
    )


def main():
    acc = pd.read_csv(ACCURACY_CSV)
    sw = pd.read_csv(SWITCH_CSV)
    gains = pd.read_csv(GAIN_CSV)

    families = ["1B", "3B"]
    splits = ["Overall", "Explicit", "Ambiguous"]
    series_order = ["T-/I-", "T-/I+", "T+/I-", "T+/I+"]
    scale_colors = {
        "1B": "#B8D9EA",
        "3B": "#F7CEA0",
    }
    scale_edges = {
        "1B": "#4E7890",
        "3B": "#AE7132",
    }

    fig = plt.figure(figsize=(8.8, 5.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.68, 1.08], wspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    draw_accuracy_gain_panel(ax_a, acc, gains, families, splits, scale_colors, compact=False)

    metric_names = ["Exact pair", "Margin switch"]
    x_centers = np.arange(len(metric_names))
    width = 0.32
    for idx, family in enumerate(["1B", "3B"]):
        row_plus = switch_row(sw, f"MAPLE {family}", "tplus_eplus")
        values = [float(row_plus["exact_pair_accuracy_pct"]), float(row_plus["margin_switch_rate_pct"])]
        lows = [float(row_plus["exact_pair_ci_low_pct"]), float(row_plus["margin_switch_ci_low_pct"])]
        highs = [float(row_plus["exact_pair_ci_high_pct"]), float(row_plus["margin_switch_ci_high_pct"])]
        errs = np.array([[v - lo for v, lo in zip(values, lows)], [hi - v for v, hi in zip(values, highs)]])
        xpos = x_centers + (idx - 0.5) * width
        bars = ax_b.bar(
            xpos,
            values,
            width=width,
            color=scale_colors[family],
            edgecolor=scale_edges[family],
            linewidth=0.85,
            label=f"{family} (T+, I+)",
            zorder=3,
        )
        ax_b.scatter(
            xpos,
            [0, 0],
            marker="o",
            s=40,
            facecolors="white",
            edgecolors=scale_edges[family],
            linewidths=1.35,
            zorder=5,
            clip_on=False,
        )
        ax_b.errorbar(xpos, values, yerr=errs, fmt="none", ecolor="black", elinewidth=1.0, capsize=2.8, zorder=4)
        for bar, value, high in zip(bars, values, highs):
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                high + 0.55,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
            )

    ax_b.axhline(0, color="black", linewidth=1.0)
    ax_b.set_xticks(x_centers)
    ax_b.set_xticklabels(metric_names, fontsize=9.6)
    ax_b.set_ylabel("Successful ambiguous pairs (%)", fontsize=10.8)
    switch_high = max(
        float(sw["exact_pair_ci_high_pct"].max()),
        float(sw["margin_switch_ci_high_pct"].max()),
    )
    switch_top = max(30.0, np.ceil((switch_high + 2.0) / 5.0) * 5.0)
    ax_b.set_ylim(-0.8, switch_top)
    ax_b.set_yticks(np.arange(0, switch_top + 0.1, 10.0))
    ax_b.set_title("(B) Paired switching from (T-, I-) to (T+, I+)", loc="left", fontsize=12.5, fontweight="bold")
    ax_b.grid(axis="y", color="#d3d8de", linewidth=0.7, alpha=0.75)
    ax_b.tick_params(axis="y", labelsize=9.8)


    for ax in [ax_a, ax_b]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("black")

    add_joint_figure_legend(fig, scale_colors, scale_edges)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.12, top=0.88, wspace=0.28)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in [
        OUT_DIR / "localnewsqa_accuracy_switch_composite.pdf",
        OUT_DIR / "localnewsqa_accuracy_switch_composite.png",
        LATEX_FIG,
    ]:
        fig.savefig(path, dpi=600 if path.suffix == ".pdf" else 240, bbox_inches="tight", pad_inches=0.02)
        print(f"wrote {path}")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, path in [
        (ax_a, OUT_DIR / "localnewsqa_accuracy_target_panel.pdf"),
        (ax_b, OUT_DIR / "localnewsqa_accuracy_switch_panel.pdf"),
        (ax_a, LATEX_TARGET_PANEL),
        (ax_b, LATEX_SWITCH_PANEL),
    ]:
        bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.03, 1.05)
        fig.savefig(path, dpi=600, bbox_inches=bbox, pad_inches=0.01)
        print(f"wrote {path}")
    plt.close(fig)

    compact, (compact_ax_a, compact_ax_b) = plt.subplots(
        2,
        1,
        figsize=(3.34, 3.78),
        gridspec_kw={"height_ratios": [1.10, 0.88]},
    )
    draw_accuracy_gain_panel(compact_ax_a, acc, gains, families, splits, scale_colors, compact=True)

    metric_names = ["Exact\npair", "Margin\nswitch"]
    x_centers = np.arange(len(metric_names))
    width = 0.32
    compact_ax_b.axhspan(-0.42, 0.42, color="#edf1f5", alpha=0.9, zorder=0)
    compact_ax_b.axhline(0, color="#111827", linewidth=1.65, zorder=2)
    for idx, family in enumerate(["1B", "3B"]):
        row_plus = switch_row(sw, f"MAPLE {family}", "tplus_eplus")
        values = [float(row_plus["exact_pair_accuracy_pct"]), float(row_plus["margin_switch_rate_pct"])]
        lows = [float(row_plus["exact_pair_ci_low_pct"]), float(row_plus["margin_switch_ci_low_pct"])]
        highs = [float(row_plus["exact_pair_ci_high_pct"]), float(row_plus["margin_switch_ci_high_pct"])]
        errs = np.array([[v - lo for v, lo in zip(values, lows)], [hi - v for v, hi in zip(values, highs)]])
        xpos = x_centers + (idx - 0.5) * width
        bars = compact_ax_b.bar(
            xpos,
            values,
            width=width,
            color=scale_colors[family],
            edgecolor=scale_edges[family],
            linewidth=0.85,
            label=f"{family} (T+, I+)",
            zorder=3,
        )
        compact_ax_b.scatter(
            xpos,
            [0, 0],
            marker="o",
            s=58,
            facecolors="white",
            edgecolors="#111827",
            linewidths=1.55,
            zorder=6,
            clip_on=False,
        )
        compact_ax_b.errorbar(
            xpos,
            values,
            yerr=errs,
            fmt="none",
            ecolor="black",
            elinewidth=0.95,
            capsize=2.6,
            zorder=4,
        )
        for bar, value, high in zip(bars, values, highs):
            compact_ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                high + 0.55,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.4,
            )

    compact_ax_b.set_xticks(x_centers)
    compact_ax_b.set_xticklabels(metric_names, fontsize=8.1)
    compact_ax_b.set_ylabel("Successful pairs (%)", fontsize=8.4, labelpad=1.5)
    switch_high = max(
        float(sw["exact_pair_ci_high_pct"].max()),
        float(sw["margin_switch_ci_high_pct"].max()),
    )
    switch_top = max(30.0, np.ceil((switch_high + 2.0) / 5.0) * 5.0)
    compact_ax_b.set_ylim(-0.9, switch_top)
    compact_ax_b.set_yticks(np.arange(0, switch_top + 0.1, 10.0))
    compact_ax_b.set_title("(B) Paired switching", loc="left", fontsize=9.8, fontweight="bold", pad=2.0)
    compact_ax_b.grid(axis="y", color="#d3d8de", linewidth=0.65, alpha=0.75)
    compact_ax_b.tick_params(axis="y", labelsize=8.0, pad=1.5)
    add_joint_figure_legend(compact, scale_colors, scale_edges)
    for ax in [compact_ax_a, compact_ax_b]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("black")

    compact.subplots_adjust(left=0.19, right=0.99, bottom=0.11, top=0.83, hspace=0.27)
    for path in [
        OUT_DIR / "localnewsqa_accuracy_switch_onecol.pdf",
        OUT_DIR / "localnewsqa_accuracy_switch_onecol.png",
        LATEX_ONECOL,
    ]:
        compact.savefig(path, dpi=600 if path.suffix == ".pdf" else 260, bbox_inches="tight", pad_inches=0.03)
        print(f"wrote {path}")
    plt.close(compact)


if __name__ == "__main__":
    main()
