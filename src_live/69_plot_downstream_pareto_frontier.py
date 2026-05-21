#!/usr/bin/env python3
import argparse
import math
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

LOCAL_METRICS = [
    ("localnewsqa_overall", "Overall"),
    ("localnewsqa_ambiguous", "Ambiguous"),
]

FAMILY_COLORS = {
    "MAPLE family": "#2f9e44",
    "LLaMA family": "#2b6cb0",
    "Qwen2.5 family": "#dd7f22",
    "Qwen3.5 family": "#805ad5",
    "Gemma-4 family": "#c53030",
    "Ministral family": "#4a5568",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a Pareto frontier between external downstream and LocalNewsQA performance."
    )
    parser.add_argument(
        "--appendix-dir",
        type=Path,
        default=DEFAULT_APPENDIX_DIR,
        help="Directory containing external_model_gains_long.csv and localnewsqa_model_gains_long.csv.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where the figure and source CSV are written.",
    )
    parser.add_argument(
        "--basename",
        default="downstream_pareto_frontier",
        help="Base filename for output files.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["frontier", "frontier_and_maple", "none"],
        default="frontier",
        help="Which points to annotate.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def short_label(label: str) -> str:
    replacements = [
        ("LLaMA-3.2-", "LLaMA "),
        ("Qwen2.5-", "Qwen2.5 "),
        ("Qwen3.5-", "Qwen3.5 "),
        ("Gemma-4-", "Gemma "),
        ("Ministral-3-", "Ministral "),
        ("MAPLE ", "MAPLE "),
        (" Base", "-B"),
        (" Inst.", "-I"),
        (" Chat", "-C"),
        ("-it", "-I"),
    ]
    out = label
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def condition_label(row_key: str, side: str) -> Tuple[str, str]:
    if row_key.startswith("maple_"):
        return ("(T+, I+)", "+") if side == "plus" else ("(T-, I-)", "-")
    return ("With metadata", "+") if side == "plus" else ("Without metadata", "-")


def normalize_external_metrics(df: pd.DataFrame) -> pd.DataFrame:
    score_cols: List[str] = []
    for metric in sorted(EXTERNAL_METRIC_LABELS):
        raw_col = f"external_{metric}"
        if raw_col not in df.columns:
            continue
        values = df[raw_col].dropna()
        if values.empty:
            continue
        min_val = float(values.min())
        max_val = float(values.max())
        score_col = f"{raw_col}_score"
        if math.isclose(max_val, min_val):
            df[score_col] = 50.0
        elif metric == "worldvaluebench":
            df[score_col] = (max_val - df[raw_col]) / (max_val - min_val) * 100.0
        else:
            df[score_col] = (df[raw_col] - min_val) / (max_val - min_val) * 100.0
        score_cols.append(score_col)

    df["external_downstream_index"] = df[score_cols].mean(axis=1)
    df["external_metric_count"] = df[score_cols].notna().sum(axis=1)
    return df


def load_points(appendix_dir: Path) -> pd.DataFrame:
    external_path = appendix_dir / "external_model_gains_long.csv"
    local_path = appendix_dir / "localnewsqa_model_gains_long.csv"
    if not external_path.exists():
        raise FileNotFoundError(f"Missing {external_path}")
    if not local_path.exists():
        raise FileNotFoundError(f"Missing {local_path}")

    external = pd.read_csv(external_path)
    local = pd.read_csv(local_path)
    records: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    def base_record(row: pd.Series, side: str) -> Dict[str, object]:
        row_key = str(row["row_key"])
        condition, condition_marker = condition_label(row_key, side)
        return {
            "row_key": row_key,
            "track": str(row["track"]),
            "label": str(row["label"]),
            "short_label": short_label(str(row["label"])),
            "group": str(row["group"]),
            "side": side,
            "condition": condition,
            "condition_marker": condition_marker,
        }

    for _, row in external.iterrows():
        metric = str(row["metric_key"])
        if metric not in EXTERNAL_METRIC_LABELS:
            continue
        for side, value_col in (("plus", "plus_value"), ("minus", "minus_value")):
            key = (str(row["row_key"]), str(row["track"]), side)
            records.setdefault(key, base_record(row, side))
            records[key][f"external_{metric}"] = float(row[value_col])
            records[key][f"external_{metric}_metric_type"] = str(row["metric_type"])

    local_keep = {
        "localnewsqa_overall",
        "localnewsqa_ambiguous",
        "localnewsqa_explicit",
        "localnewsqa_exact_pair",
        "localnewsqa_margin_switch",
    }
    for _, row in local.iterrows():
        metric = str(row["metric_key"])
        if metric not in local_keep:
            continue
        for side, value_col in (("plus", "plus_value"), ("minus", "minus_value")):
            key = (str(row["row_key"]), str(row["track"]), side)
            records.setdefault(key, base_record(row, side))
            records[key][metric] = float(row[value_col])

    df = pd.DataFrame(records.values())
    if df.empty:
        raise ValueError("No aligned downstream records were loaded.")
    df = normalize_external_metrics(df)
    df["plot_label"] = df["short_label"] + " " + df["condition_marker"]
    df = df.sort_values(["group", "row_key", "track", "side"]).reset_index(drop=True)
    return df


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
    points = points.sort_values(x_col).reset_index(drop=True)
    offsets = [(6, 8), (6, -12), (-8, 10), (-8, -12), (8, 2), (-8, 2)]
    for idx, row in points.iterrows():
        dx, dy = offsets[idx % len(offsets)]
        text = ax.annotate(
            str(row["plot_label"]),
            xy=(float(row[x_col]), float(row[y_col])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.2,
            color="#1f2933",
            ha="left" if dx >= 0 else "right",
            va="center",
            zorder=7,
        )
        text.set_path_effects([pe.Stroke(linewidth=3.0, foreground="white", alpha=0.86), pe.Normal()])


def scatter_points(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    marker_by_track = {"base": "o", "chat": "^"}
    for _, row in df.iterrows():
        color = FAMILY_COLORS.get(str(row["group"]), "#333333")
        marker = marker_by_track.get(str(row["track"]), "s")
        filled = row["side"] == "plus"
        ax.scatter(
            [float(row[x_col])],
            [float(row[y_col])],
            s=88 if marker == "o" else 96,
            marker=marker,
            facecolors=color if filled else "white",
            edgecolors=color,
            linewidths=1.7,
            alpha=0.92 if filled else 0.86,
            zorder=4,
        )


def draw_frontier(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    frontier = df[pareto_mask(df, x_col, y_col)].copy()
    frontier = frontier.sort_values([x_col, y_col])
    if len(frontier) >= 2:
        line = ax.plot(
            frontier[x_col],
            frontier[y_col],
            color="#111827",
            linewidth=2.2,
            marker=None,
            zorder=5,
        )[0]
        line.set_path_effects([pe.Stroke(linewidth=4.0, foreground="white", alpha=0.75), pe.Normal()])
    return frontier


def build_legend(fig: plt.Figure) -> None:
    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=7.5,
            label=group.replace(" family", ""),
        )
        for group, color in FAMILY_COLORS.items()
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#111827",
            markerfacecolor="white",
            linestyle="none",
            markersize=7.5,
            label="Base",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="#111827",
            markerfacecolor="white",
            linestyle="none",
            markersize=8.0,
            label="Chat/Inst.",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="#111827",
            markerfacecolor="#111827",
            linestyle="none",
            markersize=7.5,
            label="Meta+",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="#111827",
            markerfacecolor="white",
            linestyle="none",
            markersize=7.5,
            label="Meta-",
        ),
        Line2D([0], [0], color="#111827", linewidth=2.2, label="Pareto frontier"),
    ]
    fig.legend(
        handles=family_handles + style_handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.2,
        handletextpad=0.5,
        bbox_to_anchor=(0.5, -0.01),
    )


def plot(df: pd.DataFrame, results_dir: Path, basename: str, label_mode: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    x_col = "external_downstream_index"

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharex=True)
    frontier_flags: Dict[str, pd.Series] = {}
    all_x = df[x_col].astype(float)
    all_y: List[float] = []

    for ax, (y_col, title) in zip(axes, LOCAL_METRICS):
        plot_df = df.dropna(subset=[x_col, y_col]).copy()
        all_y.extend(plot_df[y_col].astype(float).tolist())
        scatter_points(ax, plot_df, x_col, y_col)
        frontier = draw_frontier(ax, plot_df, x_col, y_col)
        frontier_flags[y_col] = plot_df.index.isin(frontier.index)

        if label_mode != "none":
            label_df = frontier.copy()
            if label_mode == "frontier_and_maple":
                label_df = pd.concat(
                    [label_df, plot_df[plot_df["group"] == "MAPLE family"]],
                    ignore_index=False,
                ).drop_duplicates(subset=["row_key", "track", "side"])
            annotate_points(ax, label_df, x_col, y_col)

        ax.set_title(title, fontsize=15, fontweight="bold", pad=9)
        ax.set_xlabel("External downstream performance index", fontsize=12.5)
        ax.grid(axis="both", color="#d9dee7", linewidth=0.8, alpha=0.72)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=10.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("LocalNewsQA target accuracy (%)", fontsize=12.5)
    x_pad = max(2.0, (float(all_x.max()) - float(all_x.min())) * 0.05)
    y_pad = max(1.2, (max(all_y) - min(all_y)) * 0.06)
    for ax in axes:
        ax.set_xlim(float(all_x.min()) - x_pad, float(all_x.max()) + x_pad)
        ax.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)

    fig.suptitle(
        "Pareto frontier of downstream performance",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.922,
        "External index is a macro average over BLEND, GeoMLaMA, GlobalMMLU-CS, GlobalOpinionQA, NormAD, and inverted WorldValuesBench EMD.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#3f4652",
    )
    build_legend(fig)
    fig.tight_layout(rect=(0.02, 0.075, 0.995, 0.90), w_pad=2.4)

    png_path = results_dir / f"{basename}.png"
    pdf_path = results_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    out_df = df.copy()
    for y_col, flags in frontier_flags.items():
        out_df[f"pareto_{y_col}"] = False
        out_df.loc[df.dropna(subset=[x_col, y_col]).index, f"pareto_{y_col}"] = flags
    csv_path = results_dir / f"{basename}_source.csv"
    out_df.to_csv(csv_path, index=False)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {csv_path}")


def main() -> None:
    args = parse_args()
    df = load_points(args.appendix_dir)
    plot(df, args.results_dir, args.basename, args.label_mode)


if __name__ == "__main__":
    main()
