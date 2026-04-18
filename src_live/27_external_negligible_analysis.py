#!/usr/bin/env python3
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class PairSpec:
    benchmark: str
    scale: str
    plus_path: str
    minus_path: str
    metric_kind: str  # "accuracy" or "emd"


ROOT = "/scratch/amukher6/metacul/results"
OUT_DIR = "/scratch/amukher6/metacul/results/plots/plot14"
N_BOOT = 10000
SEED = 42
NEGLIGIBLE_BAND = 0.01  # 1 percentage point


SPECS: List[PairSpec] = [
    PairSpec(
        "GeoMLaMA",
        "1B",
        f"{ROOT}/external_benchmarks_pretrained_fullgrid_1b/geomlama/custom_tplus_eplus/geomlama_eval_matrix.csv",
        f"{ROOT}/external_benchmarks_pretrained_fullgrid_1b/geomlama/custom_tminus_eminus/geomlama_eval_matrix.csv",
        "accuracy",
    ),
    PairSpec(
        "GeoMLaMA",
        "3B",
        f"{ROOT}/external_benchmarks_pretrained_fullgrid_3b_fixregex/geomlama/custom_tplus_eplus/geomlama_eval_matrix.csv",
        f"{ROOT}/external_benchmarks_pretrained_fullgrid_3b_fixregex/geomlama/custom_tminus_eminus/geomlama_eval_matrix.csv",
        "accuracy",
    ),
    PairSpec(
        "GlobalOpinionQA",
        "1B",
        f"{ROOT}/external_benchmarks_pretrained_fullgrid_1b_fixregex_all/globalopinionqa/custom_tplus_eplus/globalopinionqa_eval_matrix.csv",
        f"{ROOT}/external_benchmarks_pretrained_fullgrid_1b_fixregex_all/globalopinionqa/custom_tminus_eminus/globalopinionqa_eval_matrix.csv",
        "accuracy",
    ),
    PairSpec(
        "GlobalOpinionQA",
        "3B",
        f"{ROOT}/external_benchmarks_pretrained_fullgrid_3b_fixregex_all/globalopinionqa/custom_tplus_eplus/globalopinionqa_eval_matrix.csv",
        f"{ROOT}/external_benchmarks_pretrained_fullgrid_3b_fixregex_all/globalopinionqa/custom_tminus_eminus/globalopinionqa_eval_matrix.csv",
        "accuracy",
    ),
    PairSpec(
        "WorldValuesBench",
        "1B",
        f"{ROOT}/external_benchmarks_pretrained_fullgrid_1b_fixregex_all/worldvaluebench/custom_tplus_eplus/worldvaluebench_custom_tplus_eplus_emd_by_group.csv",
        f"{ROOT}/external_benchmarks_pretrained_fullgrid_1b_fixregex_all/worldvaluebench/custom_tminus_eminus/worldvaluebench_custom_tminus_eminus_emd_by_group.csv",
        "emd",
    ),
    PairSpec(
        "WorldValuesBench",
        "3B",
        f"{ROOT}/external_benchmarks_pretrained_fullgrid_3b_fixregex_all/worldvaluebench/custom_tplus_eplus/worldvaluebench_custom_tplus_eplus_emd_by_group.csv",
        f"{ROOT}/external_benchmarks_pretrained_fullgrid_3b_fixregex_all/worldvaluebench/custom_tminus_eminus/worldvaluebench_custom_tminus_eminus_emd_by_group.csv",
        "emd",
    ),
    PairSpec(
        "MMLU",
        "1B",
        f"{ROOT}/external_benchmarks_pretrained_fullgrid_1b_fixregex_all/mmlu/custom_tplus_eplus/mmlu_eval_matrix.csv",
        f"{ROOT}/external_benchmarks_pretrained_fullgrid_1b_fixregex_all/mmlu/custom_tminus_eminus/mmlu_eval_matrix.csv",
        "accuracy",
    ),
    PairSpec(
        "MMLU",
        "3B",
        f"{ROOT}/external_benchmarks_pretrained_fullgrid_3b_fixregex_all/mmlu/custom_tplus_eplus/mmlu_eval_matrix.csv",
        f"{ROOT}/external_benchmarks_pretrained_fullgrid_3b_fixregex_all/mmlu/custom_tminus_eminus/mmlu_eval_matrix.csv",
        "accuracy",
    ),
]


def load_correct_column(path: str) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    cands = [c for c in df.columns if c.endswith("_correct")]
    if len(cands) != 1:
        raise ValueError(f"Expected exactly one '*_correct' column in {path}, found: {cands}")
    ccol = cands[0]
    # convention: 1=correct, 0 or -1 incorrect
    out = df[["question_id", ccol]].copy()
    out[ccol] = (out[ccol] == 1).astype(np.int8)
    return out, ccol


def paired_bootstrap_delta(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    point = float(x.mean() - y.mean())
    return point, float(lo), float(hi)


def load_emd_by_group(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"question_key", "continent", "country", "emd"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    out = df[["question_key", "continent", "country", "emd"]].copy()
    out["group_id"] = (
        out["question_key"].astype(str)
        + "||"
        + out["continent"].astype(str)
        + "||"
        + out["country"].astype(str)
    )
    return out[["group_id", "emd"]]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows: List[Dict] = []

    for i, spec in enumerate(SPECS):
        if spec.metric_kind == "accuracy":
            dfa, col_a = load_correct_column(spec.plus_path)
            dfb, col_b = load_correct_column(spec.minus_path)
            merged = dfa.merge(dfb, on="question_id", how="inner")
            x = merged[col_a].to_numpy(dtype=np.float64)
            y = merged[col_b].to_numpy(dtype=np.float64)
            point, lo, hi = paired_bootstrap_delta(x, y, N_BOOT, SEED + i)
            metric_label = "accuracy"
        else:
            dfa = load_emd_by_group(spec.plus_path)
            dfb = load_emd_by_group(spec.minus_path)
            merged = dfa.merge(dfb, on="group_id", how="inner", suffixes=("_plus", "_minus"))
            x = merged["emd_plus"].to_numpy(dtype=np.float64)
            y = merged["emd_minus"].to_numpy(dtype=np.float64)
            point, lo, hi = paired_bootstrap_delta(x, y, N_BOOT, SEED + i)
            metric_label = "emd"
        rows.append(
            {
                "benchmark": spec.benchmark,
                "scale": spec.scale,
                "metric_kind": metric_label,
                "n_overlap": len(merged),
                "metric_tplus_iplus": float(x.mean()),
                "metric_tminus_iminus": float(y.mean()),
                "delta_tplus_minus_tminus": point,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "abs_delta": abs(point),
                "ci_within_1pp_band": (lo >= -NEGLIGIBLE_BAND and hi <= NEGLIGIBLE_BAND),
            }
        )

    res = pd.DataFrame(rows)
    bench_order = ["GeoMLaMA", "GlobalOpinionQA", "WorldValuesBench", "MMLU"]
    scale_order = ["1B", "3B"]
    res["benchmark"] = pd.Categorical(res["benchmark"], bench_order, ordered=True)
    res["scale"] = pd.Categorical(res["scale"], scale_order, ordered=True)
    res = res.sort_values(["benchmark", "scale"]).reset_index(drop=True)

    csv_path = f"{OUT_DIR}/external_negligible_delta_bootstrap.csv"
    res.to_csv(csv_path, index=False)

    # Plot (forest style)
    labels = [f"{b} ({s})" for b, s in zip(res["benchmark"], res["scale"])]
    y = np.arange(len(res))[::-1]
    d = res["delta_tplus_minus_tminus"].to_numpy()
    lo = res["ci95_lo"].to_numpy()
    hi = res["ci95_hi"].to_numpy()
    xerr = np.vstack([d - lo, hi - d])

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.axvspan(-NEGLIGIBLE_BAND, NEGLIGIBLE_BAND, color="#d9f2d9", alpha=0.65, zorder=0)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)

    colors = ["#1971c2" if s == "1B" else "#495057" for s in res["scale"]]
    ax.errorbar(d, y, xerr=xerr, fmt="none", ecolor="#495057", elinewidth=1.6, capsize=3, zorder=2)
    ax.scatter(d, y, s=54, c=colors, edgecolor="black", linewidth=0.5, zorder=3)

    for yi, n in zip(y, res["n_overlap"]):
        ax.text(0.055, yi, f"n={n}", va="center", ha="left", fontsize=8, color="#495057")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Metric delta: $(T+,I+) - (T-,I-)$")
    ax.set_title("Paired bootstrap metric deltas with 95% CI (external benchmarks)")
    ax.set_xlim(-0.12, 0.07)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    pdf_path = f"{OUT_DIR}/external_negligible_delta_bootstrap.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    png_path = f"{OUT_DIR}/external_negligible_delta_bootstrap.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote: {pdf_path}")
    print(f"[OK] Wrote: {png_path}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
