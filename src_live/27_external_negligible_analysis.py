#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OURS_1B_ROOT = Path(
    "/path/to/metacul/results/external_benchmarks_pretrained_multiseed/ours_1b"
)
DEFAULT_OURS_3B_ROOT = Path(
    "/path/to/metacul/results/external_benchmarks_pretrained_multiseed/ours_3b"
)
DEFAULT_OUT_DIR = Path("/path/to/metacul/results/plots/plot14")
DEFAULT_SEEDS = "41,42,43,44,45"
N_BOOT = 10000
SEED = 42
NEGLIGIBLE_BAND = 0.01  # 1 percentage point


@dataclass
class PairSpec:
    benchmark: str
    scale: str
    root_key: str
    plus_variant: str
    minus_variant: str
    metric_kind: str  # "accuracy" or "emd"


SPECS: List[PairSpec] = [
    PairSpec("GeoMLaMA", "1B", "ours_1b", "custom_tplus_eplus", "custom_tminus_eminus", "accuracy"),
    PairSpec("GeoMLaMA", "3B", "ours_3b", "custom_tplus_eplus", "custom_tminus_iminus", "accuracy"),
    PairSpec("GlobalOpinionQA", "1B", "ours_1b", "custom_tplus_eplus", "custom_tminus_iminus", "accuracy"),
    PairSpec("GlobalOpinionQA", "3B", "ours_3b", "custom_tplus_eplus", "custom_tminus_iminus", "accuracy"),
    PairSpec("WorldValuesBench", "1B", "ours_1b", "custom_tplus_eplus", "custom_tminus_iminus", "emd"),
    PairSpec("WorldValuesBench", "3B", "ours_3b", "custom_tplus_eplus", "custom_tminus_iminus", "emd"),
    PairSpec("MMLU", "1B", "ours_1b", "custom_tplus_eplus", "custom_tminus_iminus", "accuracy"),
    PairSpec("MMLU", "3B", "ours_3b", "custom_tplus_eplus", "custom_tminus_iminus", "accuracy"),
]

# Typo-safe normalization: we only want one canonical spelling downstream.
for spec in SPECS:
    if spec.minus_variant == "custom_tminus_iminus":
        spec.minus_variant = "custom_tminus_eminus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired bootstrap deltas for MAPLE external benchmarks."
    )
    parser.add_argument("--ours-1b-root", type=Path, default=DEFAULT_OURS_1B_ROOT)
    parser.add_argument("--ours-3b-root", type=Path, default=DEFAULT_OURS_3B_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--negligible-band", type=float, default=NEGLIGIBLE_BAND)
    return parser.parse_args()


def parse_seed_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def load_correct_column(path: Path) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    cands = [c for c in df.columns if c.endswith("_correct")]
    if len(cands) != 1:
        raise ValueError(f"Expected exactly one '*_correct' column in {path}, found: {cands}")
    ccol = cands[0]
    out = df[["question_id", ccol]].copy()
    out[ccol] = (out[ccol] == 1).astype(np.int8)
    return out, ccol


def paired_bootstrap_delta(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    point = float(x.mean() - y.mean())
    return point, float(lo), float(hi)


def load_emd_by_group(path: Path) -> pd.DataFrame:
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


def benchmark_slug(label: str) -> str:
    return {
        "GeoMLaMA": "geomlama",
        "GlobalOpinionQA": "globalopinionqa",
        "WorldValuesBench": "worldvaluebench",
        "MMLU": "mmlu",
    }[label]


def collect_accuracy_pairs(root: Path, benchmark: str, plus_variant: str, minus_variant: str, seeds: List[int]) -> pd.DataFrame:
    frames = []
    slug = benchmark_slug(benchmark)
    for eval_seed in seeds:
        plus_path = root / f"seed_{eval_seed}" / slug / plus_variant / f"{slug}_eval_matrix.csv"
        minus_path = root / f"seed_{eval_seed}" / slug / minus_variant / f"{slug}_eval_matrix.csv"
        if not plus_path.exists() or not minus_path.exists():
            print(f"[missing] bootstrap accuracy seed={eval_seed} benchmark={slug} root={root}")
            continue
        dfa, col_a = load_correct_column(plus_path)
        dfb, col_b = load_correct_column(minus_path)
        dfa["sample_id"] = dfa["question_id"].astype(str).map(lambda q: f"{eval_seed}::{q}")
        dfb["sample_id"] = dfb["question_id"].astype(str).map(lambda q: f"{eval_seed}::{q}")
        merged = dfa[["sample_id", col_a]].merge(dfb[["sample_id", col_b]], on="sample_id", how="inner")
        if not merged.empty:
            frames.append(merged)
    if not frames:
        return pd.DataFrame(columns=["plus", "minus"])
    merged = pd.concat(frames, ignore_index=True)
    merged.rename(columns={merged.columns[1]: "plus", merged.columns[2]: "minus"}, inplace=True)
    return merged


def collect_emd_pairs(root: Path, benchmark: str, plus_variant: str, minus_variant: str, seeds: List[int]) -> pd.DataFrame:
    frames = []
    slug = benchmark_slug(benchmark)
    for eval_seed in seeds:
        plus_path = root / f"seed_{eval_seed}" / slug / plus_variant / f"{slug}_{plus_variant}_emd_by_group.csv"
        minus_path = root / f"seed_{eval_seed}" / slug / minus_variant / f"{slug}_{minus_variant}_emd_by_group.csv"
        if not plus_path.exists() or not minus_path.exists():
            print(f"[missing] bootstrap emd seed={eval_seed} benchmark={slug} root={root}")
            continue
        dfa = load_emd_by_group(plus_path)
        dfb = load_emd_by_group(minus_path)
        dfa["sample_id"] = dfa["group_id"].astype(str).map(lambda g: f"{eval_seed}::{g}")
        dfb["sample_id"] = dfb["group_id"].astype(str).map(lambda g: f"{eval_seed}::{g}")
        merged = dfa[["sample_id", "emd"]].merge(
            dfb[["sample_id", "emd"]], on="sample_id", how="inner", suffixes=("_plus", "_minus")
        )
        if not merged.empty:
            frames.append(merged)
    if not frames:
        return pd.DataFrame(columns=["plus", "minus"])
    merged = pd.concat(frames, ignore_index=True)
    merged.rename(columns={"emd_plus": "plus", "emd_minus": "minus"}, inplace=True)
    return merged


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_list(args.seeds)
    root_map: Dict[str, Path] = {
        "ours_1b": args.ours_1b_root,
        "ours_3b": args.ours_3b_root,
    }

    rows: List[Dict[str, object]] = []
    for i, spec in enumerate(SPECS):
        root = root_map[spec.root_key]
        if spec.metric_kind == "accuracy":
            merged = collect_accuracy_pairs(root, spec.benchmark, spec.plus_variant, spec.minus_variant, seeds)
        else:
            merged = collect_emd_pairs(root, spec.benchmark, spec.plus_variant, spec.minus_variant, seeds)

        if merged.empty:
            print(f"[warn] No paired rows for {spec.benchmark} ({spec.scale})")
            continue

        x = merged["plus"].to_numpy(dtype=np.float64)
        y = merged["minus"].to_numpy(dtype=np.float64)
        point, lo, hi = paired_bootstrap_delta(x, y, args.n_boot, args.seed + i)
        rows.append(
            {
                "benchmark": spec.benchmark,
                "scale": spec.scale,
                "metric_kind": spec.metric_kind,
                "n_overlap": len(merged),
                "metric_tplus_iplus": float(x.mean()),
                "metric_tminus_iminus": float(y.mean()),
                "delta_tplus_minus_tminus": point,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "abs_delta": abs(point),
                "ci_within_1pp_band": (
                    lo >= -args.negligible_band and hi <= args.negligible_band
                ),
            }
        )

    if not rows:
        print("[warn] No completed bootstrap rows found.")
        return

    res = pd.DataFrame(rows)
    bench_order = ["GeoMLaMA", "GlobalOpinionQA", "WorldValuesBench", "MMLU"]
    scale_order = ["1B", "3B"]
    res["benchmark"] = pd.Categorical(res["benchmark"], bench_order, ordered=True)
    res["scale"] = pd.Categorical(res["scale"], scale_order, ordered=True)
    res = res.sort_values(["benchmark", "scale"]).reset_index(drop=True)

    csv_path = args.out_dir / "external_negligible_delta_bootstrap.csv"
    res.to_csv(csv_path, index=False)

    labels = [f"{b} ({s})" for b, s in zip(res["benchmark"], res["scale"])]
    y = np.arange(len(res))[::-1]
    d = res["delta_tplus_minus_tminus"].to_numpy()
    lo = res["ci95_lo"].to_numpy()
    hi = res["ci95_hi"].to_numpy()
    xerr = np.vstack([d - lo, hi - d])

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.axvspan(-args.negligible_band, args.negligible_band, color="#d9f2d9", alpha=0.65, zorder=0)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)

    colors = ["#1971c2" if s == "1B" else "#495057" for s in res["scale"]]
    ax.errorbar(d, y, xerr=xerr, fmt="none", ecolor="#495057", elinewidth=1.6, capsize=3, zorder=2)
    ax.scatter(d, y, s=54, c=colors, edgecolor="black", linewidth=0.5, zorder=3)

    x_min = min(float(lo.min()), -args.negligible_band)
    x_max = max(float(hi.max()), args.negligible_band)
    x_pad = max(0.01, 0.08 * (x_max - x_min))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

    for yi, n in zip(y, res["n_overlap"]):
        ax.text(ax.get_xlim()[1] - 0.002, yi, f"n={n}", va="center", ha="right", fontsize=8, color="#495057")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Metric delta: $(T+, I+) - (T-, I-)$")
    ax.set_title("Paired bootstrap metric deltas with 95% CI (external benchmarks)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()

    pdf_path = args.out_dir / "external_negligible_delta_bootstrap.pdf"
    png_path = args.out_dir / "external_negligible_delta_bootstrap.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote: {pdf_path}")
    print(f"[OK] Wrote: {png_path}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
