#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_APPENDIX_DIR = Path(
    "/path/to/metacul/results/appendix_model_gain_tables_20260505"
)
DEFAULT_RESULTS_DIR = Path("/path/to/metacul/results/plots/plot15")

EXTERNAL_METRIC_LABELS = {
    "blend": "BLEND",
    "geomlama": "GeoMLaMA",
    "globalmmlu_cs": "GlobalMMLU-CS",
    "globalopinionqa": "GlobalOpinionQA",
    "normad": "NormAD",
    "worldvaluebench": "WorldValuesBench",
}

FAMILY_COLORS = {
    "MAPLE family": "#2f9e44",
    "LLaMA family": "#aeb4bd",
    "Qwen2.5 family": "#aeb4bd",
    "Qwen3.5 family": "#aeb4bd",
    "Gemma-4 family": "#aeb4bd",
    "Ministral family": "#aeb4bd",
}
MAPLE_COLOR = "#2f9e44"
OTHER_MODEL_COLOR = "#aeb4bd"

FRONTIER_SIZE_STYLES = {
    "gt_1b": {
        "legend_label": "3B+ frontier",
        "color": "#111827",
        "linestyle": "-",
    },
    "le_1b": {
        "legend_label": r"$\leq$1B frontier",
        "color": "#4b5563",
        "linestyle": (0, (4.5, 2.2)),
    },
}

AMBIGUOUS_LABEL_POSITIONS = {
    "MAPLE 1B base": (46.7, 3.45, "left", "center"),
    "MAPLE 1B chat": (31.0, 4.85, "right", "center"),
    "Qwen3.5 2B base": (55.0, 9.38, "left", "bottom"),
    "Qwen3.5 2B chat": (65.4, 12.25, "right", "center"),
    "MAPLE 3B base": (61.9, 6.25, "right", "center"),
    "MAPLE 3B chat": (46.8, 12.1, "right", "center"),
    "Gemma 4 E4B base": (69.6, 0.90, "right", "bottom"),
}

AMBIGUOUS_LABEL_SKIP = set()
ESSENTIAL_AMBIGUOUS_LABELS = {
    "MAPLE 1B base",
    "MAPLE 3B base",
    "Qwen3.5 2B chat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a pair-level Pareto frontier between external downstream performance "
            "and LocalNewsQA metadata gain."
        )
    )
    parser.add_argument(
        "--appendix-dir",
        type=Path,
        default=DEFAULT_APPENDIX_DIR,
        help="Directory containing external_model_gains_long.csv and localnewsqa_model_gains_long.csv.",
    )
    parser.add_argument(
        "--external-gains",
        type=Path,
        default=None,
        help="Optional path to external_model_gains_long.csv. Defaults to --appendix-dir.",
    )
    parser.add_argument(
        "--local-gains",
        type=Path,
        default=None,
        help="Optional path to localnewsqa_model_gains_long.csv. Defaults to --appendix-dir.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where the figure and source CSV are written.",
    )
    parser.add_argument(
        "--basename",
        default="downstream_ambiguous_gain_pareto_frontier",
        help="Base filename for output files.",
    )
    parser.add_argument(
        "--local-metric",
        default="localnewsqa_ambiguous",
        choices=["localnewsqa_ambiguous", "localnewsqa_exact_pair", "localnewsqa_overall"],
        help="LocalNewsQA gain metric to place on the y-axis.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["frontier", "frontier_and_maple", "essential", "none"],
        default="frontier",
        help="Which points to annotate.",
    )
    return parser.parse_args()


def model_stem(label: str) -> str:
    replacements = [
        ("LLaMA-3.2-", "LLaMA "),
        ("Qwen2.5-", "Qwen2.5 "),
        ("Qwen3.5-", "Qwen3.5 "),
        ("Gemma-4-", "Gemma 4 "),
        ("Ministral-3-", "Ministral "),
        (" Base", ""),
        (" Inst.", ""),
        (" Chat", ""),
        ("-it", ""),
    ]
    out = label
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def plot_label(label: str, track: str) -> str:
    return f"{model_stem(label)} {'chat' if track == 'chat' else 'base'}"


def parse_model_size_b(label: str, row_key: str = "") -> float:
    for text in (label, row_key.replace("p", ".")):
        b_matches = re.findall(r"(\d+(?:\.\d+)?)\s*B\b", text, flags=re.IGNORECASE)
        if b_matches:
            return float(b_matches[-1])
        m_matches = re.findall(r"(\d+(?:\.\d+)?)\s*M\b", text, flags=re.IGNORECASE)
        if m_matches:
            return float(m_matches[-1]) / 1000.0
    return float("nan")


def model_size_bucket(size_b: float) -> str:
    if not math.isfinite(float(size_b)):
        return "unknown"
    if float(size_b) > 1.0:
        return ">1B"
    return "<=1B"


def pair_name(row_key: str) -> Tuple[str, str]:
    if row_key.startswith("maple_"):
        return "(T+, I+)", "(T-, I-)"
    return "metadata", "no metadata"


def finite_values(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def score_value(value: float, min_val: float, max_val: float, lower_is_better: bool) -> float:
    if math.isclose(max_val, min_val):
        return 50.0
    if lower_is_better:
        return (max_val - value) / (max_val - min_val) * 100.0
    return (value - min_val) / (max_val - min_val) * 100.0


def load_pair_points(
    appendix_dir: Path,
    local_metric: str,
    external_gains: Path | None = None,
    local_gains: Path | None = None,
) -> pd.DataFrame:
    external_path = external_gains or appendix_dir / "external_model_gains_long.csv"
    local_path = local_gains or appendix_dir / "localnewsqa_model_gains_long.csv"
    if not external_path.exists():
        raise FileNotFoundError(f"Missing {external_path}")
    if not local_path.exists():
        raise FileNotFoundError(f"Missing {local_path}")

    external = pd.read_csv(external_path)
    local = pd.read_csv(local_path)

    external = external[external["metric_key"].isin(EXTERNAL_METRIC_LABELS)].copy()
    local_metric_rows = local[local["metric_key"] == local_metric].copy()
    if local_metric_rows.empty:
        raise ValueError(f"No rows found for {local_metric} in {local_path}")

    rows: Dict[Tuple[str, str], Dict[str, object]] = {}
    metric_ranges: Dict[str, Tuple[float, float]] = {}
    for metric, metric_df in external.groupby("metric_key"):
        values = finite_values(metric_df["plus_value"].tolist() + metric_df["minus_value"].tolist())
        if values:
            metric_ranges[str(metric)] = (min(values), max(values))

    for _, row in external.iterrows():
        row_key = str(row["row_key"])
        track = str(row["track"])
        key = (row_key, track)
        plus_name, minus_name = pair_name(row_key)
        rows.setdefault(
            key,
            {
                "row_key": row_key,
                "track": track,
                "label": str(row["label"]),
                "short_label": plot_label(str(row["label"]), track),
                "model_size_b": parse_model_size_b(str(row["label"]), row_key),
                "group": str(row["group"]),
                "plus_condition": plus_name,
                "minus_condition": minus_name,
            },
        )
        metric = str(row["metric_key"])
        rows[key][f"external_{metric}_plus"] = float(row["plus_value"])
        rows[key][f"external_{metric}_minus"] = float(row["minus_value"])
        rows[key][f"external_{metric}_delta"] = float(row["delta"])

    for _, row in local_metric_rows.iterrows():
        key = (str(row["row_key"]), str(row["track"]))
        if key not in rows:
            continue
        rows[key]["local_gain"] = float(row["delta"])
        rows[key]["local_ci_low"] = float(row["ci_low"])
        rows[key]["local_ci_high"] = float(row["ci_high"])
        rows[key]["local_plus_value"] = float(row["plus_value"])
        rows[key]["local_minus_value"] = float(row["minus_value"])
        rows[key]["local_n"] = int(row["n"])

    df = pd.DataFrame(rows.values())
    if df.empty:
        raise ValueError("No aligned pair records were loaded.")

    plus_score_cols: List[str] = []
    minus_score_cols: List[str] = []
    for metric in sorted(EXTERNAL_METRIC_LABELS):
        if metric not in metric_ranges:
            continue
        min_val, max_val = metric_ranges[metric]
        lower_is_better = metric == "worldvaluebench"
        plus_raw = f"external_{metric}_plus"
        minus_raw = f"external_{metric}_minus"
        if plus_raw in df.columns:
            plus_score = f"{plus_raw}_score"
            df[plus_score] = df[plus_raw].apply(
                lambda value: score_value(float(value), min_val, max_val, lower_is_better)
            )
            plus_score_cols.append(plus_score)
        if minus_raw in df.columns:
            minus_score = f"{minus_raw}_score"
            df[minus_score] = df[minus_raw].apply(
                lambda value: score_value(float(value), min_val, max_val, lower_is_better)
            )
            minus_score_cols.append(minus_score)

    df["external_downstream_index_plus"] = df[plus_score_cols].mean(axis=1)
    df["external_downstream_index_minus"] = df[minus_score_cols].mean(axis=1)
    df["external_downstream_index_delta"] = (
        df["external_downstream_index_plus"] - df["external_downstream_index_minus"]
    )
    df["external_metric_count"] = df[plus_score_cols].notna().sum(axis=1)
    df["plot_label"] = df["short_label"]
    df["model_size_bucket"] = df["model_size_b"].apply(model_size_bucket)
    df = df.dropna(subset=["external_downstream_index_plus", "local_gain"]).copy()
    return df.sort_values(["group", "row_key", "track"]).reset_index(drop=True)


def pareto_mask(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    mask = []
    xs = df[x_col].astype(float).tolist()
    ys = df[y_col].astype(float).tolist()
    for i, (x_i, y_i) in enumerate(zip(xs, ys)):
        dominated = False
        for j, (x_j, y_j) in enumerate(zip(xs, ys)):
            if i == j:
                continue
            if x_j >= x_i and y_j >= y_i and (x_j > x_i or y_j > y_i):
                dominated = True
                break
        mask.append(not dominated)
    return pd.Series(mask, index=df.index)


def annotate_points(ax: plt.Axes, points: pd.DataFrame, x_col: str, y_col: str) -> None:
    points = points.sort_values([x_col, y_col]).reset_index(drop=True)
    offsets = [(8, 10), (9, -13), (-10, 11), (-10, -12), (9, 0), (-10, 0)]
    for idx, row in points.iterrows():
        label = str(row["plot_label"])
        if label in AMBIGUOUS_LABEL_SKIP:
            continue
        xy = (float(row[x_col]), float(row[y_col]))
        if label in AMBIGUOUS_LABEL_POSITIONS:
            x_text, y_text, ha, va = AMBIGUOUS_LABEL_POSITIONS[label]
            label_bbox = None
            fontsize = 8.5 if label.startswith("MAPLE") else 8.1
            if label.startswith("MAPLE"):
                label_bbox = dict(
                    boxstyle=(
                        "round,pad=0.18,rounding_size=0.12"
                        if label == "MAPLE 3B base"
                        else "round,pad=0.22,rounding_size=0.12"
                    ),
                    facecolor="white",
                    edgecolor=FAMILY_COLORS["MAPLE family"],
                    linewidth=1.15,
                    alpha=0.94,
                )
            text = ax.annotate(
                label,
                xy=xy,
                xytext=(x_text, y_text),
                xycoords="data",
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-",
                    color="#4b5563",
                    linewidth=0.85,
                    alpha=0.82,
                    shrinkA=2,
                    shrinkB=3,
                ),
                fontsize=fontsize,
                color="#1f2933",
                ha=ha,
                va=va,
                bbox=label_bbox,
                zorder=8,
            )
            text.set_path_effects([pe.Stroke(linewidth=3.0, foreground="white", alpha=0.88), pe.Normal()])
            continue

        dx, dy = offsets[idx % len(offsets)]
        text = ax.annotate(
            label,
            xy=xy,
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.6,
            color="#1f2933",
            ha="left" if dx >= 0 else "right",
            va="center",
            zorder=8,
        )
        text.set_path_effects([pe.Stroke(linewidth=3.0, foreground="white", alpha=0.88), pe.Normal()])


def draw_points(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    marker_by_track = {"base": "o", "chat": "^"}
    plot_df = df.copy()
    plot_df["_is_maple"] = plot_df["group"].astype(str).eq("MAPLE family")
    plot_df = plot_df.sort_values("_is_maple")
    for _, row in plot_df.iterrows():
        is_maple = bool(row["_is_maple"])
        color = MAPLE_COLOR if is_maple else OTHER_MODEL_COLOR
        marker = marker_by_track.get(str(row["track"]), "s")
        point_size = (104 if marker == "o" else 118) if is_maple else (82 if marker == "o" else 92)
        yerr_low = max(0.0, float(row[y_col]) - float(row["local_ci_low"]))
        yerr_high = max(0.0, float(row["local_ci_high"]) - float(row[y_col]))
        ax.errorbar(
            [float(row[x_col])],
            [float(row[y_col])],
            yerr=[[yerr_low], [yerr_high]],
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=3.2,
            capthick=1.2,
            alpha=0.45,
            zorder=3 if is_maple else 2,
        )
        ax.scatter(
            [float(row[x_col])],
            [float(row[y_col])],
            s=point_size,
            marker=marker,
            facecolors=color,
            edgecolors="black",
            linewidths=1.45 if is_maple else 1.25,
            alpha=0.96 if is_maple else 0.93,
            zorder=8 if is_maple else 6,
        )


def draw_frontier(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color: str,
    linestyle: object,
) -> pd.DataFrame:
    frontier = df[pareto_mask(df, x_col, y_col)].copy()
    frontier = frontier.sort_values([x_col, y_col])
    if len(frontier) >= 2:
        line = ax.plot(
            frontier[x_col],
            frontier[y_col],
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            zorder=4,
        )[0]
        line.set_path_effects([pe.Stroke(linewidth=4.4, foreground="white", alpha=0.76), pe.Normal()])
    elif len(frontier) == 1:
        row = frontier.iloc[0]
        marker = {"base": "o", "chat": "^"}.get(str(row["track"]), "o")
        ax.scatter(
            [float(row[x_col])],
            [float(row[y_col])],
            s=190,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=2.35,
            linestyle=linestyle,
            zorder=8,
        )
    return frontier

def frontier_mask_for_band(df: pd.DataFrame, band: str) -> pd.Series:
    sizes = df["model_size_b"].astype(float)
    if band == "gt_1b":
        return sizes > 1.0
    if band == "le_1b":
        return sizes <= 1.0
    raise ValueError(f"Unknown frontier band: {band}")


def draw_size_frontiers(
    ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str
) -> Dict[str, pd.DataFrame]:
    frontiers: Dict[str, pd.DataFrame] = {}
    for band, style in FRONTIER_SIZE_STYLES.items():
        band_df = df[frontier_mask_for_band(df, band)]
        if band_df.empty:
            frontiers[band] = band_df.copy()
            continue
        frontiers[band] = draw_frontier(
            ax,
            band_df,
            x_col,
            y_col,
            color=str(style["color"]),
            linestyle=style["linestyle"],
        )
    return frontiers


def annotate_size_frontiers(
    ax: plt.Axes,
    frontiers: Dict[str, pd.DataFrame],
    x_col: str,
    y_col: str,
) -> None:
    for band, frontier in frontiers.items():
        if frontier.empty:
            continue
        style = FRONTIER_SIZE_STYLES[band]
        label = str(style["legend_label"])
        color = str(style["color"])
        frontier = frontier.sort_values([x_col, y_col]).reset_index(drop=True)
        if band == "le_1b":
            row = frontier.iloc[-1]
            xy = (float(row[x_col]), float(row[y_col]))
            xytext = (xy[0] - 5.4, xy[1] + 1.45)
            ha = "right"
        elif len(frontier) >= 2:
            x_values = frontier[x_col].astype(float).to_numpy()
            y_values = frontier[y_col].astype(float).to_numpy()
            xy = (float(x_values.mean()), float(y_values.mean()))
            xytext = (xy[0] + 0.2, xy[1] + 2.0)
            ha = "center"
        else:
            row = frontier.iloc[len(frontier) // 2]
            xy = (float(row[x_col]), float(row[y_col]))
            xytext = (xy[0] - 1.8, xy[1] + 1.15)
            ha = "left"
        text = ax.annotate(
            label,
            xy=xy,
            xytext=xytext,
            xycoords="data",
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                linewidth=1.0,
                shrinkA=2,
                shrinkB=4,
            ),
            fontsize=8.3,
            fontweight="bold",
            color=color,
            ha=ha,
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.18,rounding_size=0.10",
                facecolor="white",
                edgecolor=color,
                linewidth=0.9,
                alpha=0.95,
            ),
            zorder=9,
        )
        text.set_path_effects([pe.Stroke(linewidth=2.4, foreground="white", alpha=0.88), pe.Normal()])


def build_legend(ax: plt.Axes) -> None:
    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=MAPLE_COLOR,
            markeredgecolor="black",
            markersize=7.0,
            label="MAPLE",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=OTHER_MODEL_COLOR,
            markeredgecolor="black",
            markersize=7.0,
            label="Other models",
        ),
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#111827",
            markerfacecolor="white",
            linestyle="none",
            markersize=7.0,
            label="Base",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="#111827",
            markerfacecolor="white",
            linestyle="none",
            markersize=7.4,
            label="Chat",
        ),
        *[
            Line2D(
                [0],
                [0],
                color=str(style["color"]),
                linestyle=style["linestyle"],
                linewidth=2.5,
                label=str(style["legend_label"]),
            )
            for style in FRONTIER_SIZE_STYLES.values()
        ],
    ]
    ax.legend(
        handles=family_handles + style_handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=0.93,
        edgecolor="black",
        fontsize=8.2,
        bbox_to_anchor=(0.5, 1.012),
        borderpad=0.55,
        columnspacing=0.55,
        handlelength=1.45,
        handletextpad=0.34,
    )


def metric_axis_label(local_metric: str) -> str:
    if local_metric == "localnewsqa_ambiguous":
        return "LocalNewsQA ambiguous accuracy gain\n(percentage points)"
    if local_metric == "localnewsqa_exact_pair":
        return "LocalNewsQA exact-pair switch gain\n(percentage points)"
    return "LocalNewsQA overall accuracy gain\n(percentage points)"


def metric_title(local_metric: str) -> str:
    if local_metric == "localnewsqa_ambiguous":
        return "Local robustness gain versus downstream performance"
    if local_metric == "localnewsqa_exact_pair":
        return "Locale-switching gain versus downstream performance"
    return "LocalNewsQA gain versus downstream performance"


def plot(df: pd.DataFrame, results_dir: Path, basename: str, local_metric: str, label_mode: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    x_col = "external_downstream_index_plus"
    y_col = "local_gain"

    fig, ax = plt.subplots(figsize=(4.7, 3.05))
    draw_points(ax, df, x_col, y_col)
    # The panel caption carries the takeaway; omitting a title keeps the legend clear.
    frontiers = draw_size_frontiers(ax, df, x_col, y_col)

    if label_mode != "none":
        frontier_frames = [frontier for frontier in frontiers.values() if not frontier.empty]
        label_df = (
            pd.concat(frontier_frames, ignore_index=False)
            if frontier_frames
            else df.iloc[0:0].copy()
        )
        if label_mode == "frontier_and_maple":
            label_df = pd.concat(
                [label_df, df[df["group"] == "MAPLE family"]],
                ignore_index=False,
            ).drop_duplicates(subset=["row_key", "track"])
        elif label_mode == "essential":
            label_df = df[df["plot_label"].isin(ESSENTIAL_AMBIGUOUS_LABELS)].copy()
        annotate_points(ax, label_df, x_col, y_col)

    ax.axhline(0, color="#687385", linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.8, zorder=1)
    ax.set_xlabel("Normalized external score", fontsize=13.6)
    ax.set_ylabel(metric_axis_label(local_metric), fontsize=11.0)
    ax.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.28)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")

    x_vals = df[x_col].astype(float)
    y_vals = pd.concat([df["local_ci_low"], df["local_ci_high"], df[y_col]]).astype(float)
    x_pad = max(2.0, (float(x_vals.max()) - float(x_vals.min())) * 0.06)
    y_pad = max(0.6, (float(y_vals.max()) - float(y_vals.min())) * 0.10)
    ax.set_xlim(float(x_vals.min()) - x_pad, float(x_vals.max()) + x_pad)
    ax.set_ylim(float(y_vals.min()) - y_pad, float(y_vals.max()) + y_pad)
    if label_mode != "essential":
        annotate_size_frontiers(ax, frontiers, x_col, y_col)

    build_legend(ax)
    fig.tight_layout()
    fig.subplots_adjust(left=0.24, right=0.985, bottom=0.18, top=0.80)

    png_path = results_dir / f"{basename}.png"
    pdf_path = results_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    out_df = df.copy()
    out_df[f"pareto_{local_metric}"] = False
    for band, frontier in frontiers.items():
        col = f"pareto_{local_metric}_{band}"
        out_df[col] = out_df.index.isin(frontier.index)
        out_df[f"pareto_{local_metric}"] = (
            out_df[f"pareto_{local_metric}"] | out_df[col]
        )
    csv_path = results_dir / f"{basename}_source.csv"
    out_df.to_csv(csv_path, index=False)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {csv_path}")


def main() -> None:
    args = parse_args()
    df = load_pair_points(
        args.appendix_dir,
        args.local_metric,
        external_gains=args.external_gains,
        local_gains=args.local_gains,
    )
    plot(df, args.results_dir, args.basename, args.local_metric, args.label_mode)


if __name__ == "__main__":
    main()
