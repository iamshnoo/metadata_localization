#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path("/path/to/metacul")
FINAL_ROOT = ROOT / "results/final_benchmark_matrix"
PAPER_EXT_ROOT = ROOT / "results/external_paper_table_tokenizerfix_20260504"
TINKER_LONG = (
    ROOT
    / "results/external_scale3b_tinker_20260505/_reports/"
    / "paper_table_external_full_long_scale3b_tinker_20260505_030440.csv"
)
PROTOCOL_BEST = (
    PAPER_EXT_ROOT / "_reports/protocol_probe_best_20260505_014625.csv"
)
OUT_DIR = ROOT / "results/external_appendix_gain_table_20260505"


EXTERNAL_BENCHES = [
    ("geomlama", "Geo"),
    ("globalopinionqa", "GOQA"),
    ("globalmmlu_cs", "MMLU-CS"),
    ("normad", "NormAD"),
    ("blend", "BLEnD"),
    ("worldvaluebench", "WVB"),
]

LNQA_SPLITS = [
    ("localnewsqa_overall", "LNQA-O"),
    ("localnewsqa_ambiguous", "LNQA-Amb"),
    ("localnewsqa_explicit", "LNQA-Exp"),
]

LNQA_CHAT_SLUGS = {
    "gemma4_e2b": "gemma4_e2b_it",
    "gemma4_e4b": "gemma4_e4b_it",
    "ministral3_3b": "ministral3_3b",
}


@dataclass(frozen=True)
class RowSpec:
    group: str
    provider: str
    key: str
    label: str
    track: str
    kind: str


ROWS: List[RowSpec] = [
    RowSpec("MAPLE family", r"\providerMAPLE{}", "maple_1b", "MAPLE 1B Base", "base", "maple"),
    RowSpec("MAPLE family", r"\providerMAPLE{}", "maple_1b", "MAPLE 1B Chat", "chat", "maple"),
    RowSpec("MAPLE family", r"\providerMAPLE{}", "maple_3b", "MAPLE 3B Base", "base", "maple"),
    RowSpec("MAPLE family", r"\providerMAPLE{}", "maple_3b", "MAPLE 3B Chat", "chat", "maple"),
    RowSpec("LLaMA family", r"\providerMeta{}", "llama32_1b", "LLaMA-3.2-1B Base", "base", "external"),
    RowSpec("LLaMA family", r"\providerMeta{}", "llama32_1b", "LLaMA-3.2-1B Inst.", "chat", "external"),
    RowSpec("LLaMA family", r"\providerMeta{}", "llama32_3b", "LLaMA-3.2-3B Base", "base", "external"),
    RowSpec("LLaMA family", r"\providerMeta{}", "llama32_3b", "LLaMA-3.2-3B Inst.", "chat", "external"),
    RowSpec("Qwen2.5 family", r"\providerQwen{}", "qwen25_1p5b", "Qwen2.5-1.5B Base", "base", "external"),
    RowSpec("Qwen2.5 family", r"\providerQwen{}", "qwen25_1p5b", "Qwen2.5-1.5B Inst.", "chat", "external"),
    RowSpec("Qwen2.5 family", r"\providerQwen{}", "qwen25_3b", "Qwen2.5-3B Base", "base", "external"),
    RowSpec("Qwen2.5 family", r"\providerQwen{}", "qwen25_3b", "Qwen2.5-3B Inst.", "chat", "external"),
    RowSpec("Qwen3.5 family", r"\providerQwen{}", "qwen35_0p8b", "Qwen3.5-0.8B Base", "base", "external"),
    RowSpec("Qwen3.5 family", r"\providerQwen{}", "qwen35_0p8b", "Qwen3.5-0.8B Chat", "chat", "external"),
    RowSpec("Qwen3.5 family", r"\providerQwen{}", "qwen35_2b", "Qwen3.5-2B Base", "base", "external"),
    RowSpec("Qwen3.5 family", r"\providerQwen{}", "qwen35_2b", "Qwen3.5-2B Chat", "chat", "external"),
    RowSpec("Qwen3.5 family", r"\providerQwen{}", "qwen35_4b", "Qwen3.5-4B Base", "base", "external"),
    RowSpec("Qwen3.5 family", r"\providerQwen{}", "qwen35_4b", "Qwen3.5-4B Chat", "chat", "external"),
    RowSpec("Gemma-4 family", r"\providerGoogle{}", "gemma4_e2b", "Gemma-4-E2B Base", "base", "external"),
    RowSpec("Gemma-4 family", r"\providerGoogle{}", "gemma4_e2b", "Gemma-4-E2B-it", "chat", "external"),
    RowSpec("Gemma-4 family", r"\providerGoogle{}", "gemma4_e4b", "Gemma-4-E4B Base", "base", "external"),
    RowSpec("Gemma-4 family", r"\providerGoogle{}", "gemma4_e4b", "Gemma-4-E4B-it", "chat", "external"),
    RowSpec("Ministral family", r"\providerMistral{}", "ministral3_3b", "Ministral-3-3B Base", "base", "external"),
    RowSpec("Ministral family", r"\providerMistral{}", "ministral3_3b", "Ministral-3-3B Inst.", "chat", "external"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the compact external+LocalNewsQA appendix gain table."
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260505)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def item_key(row: Dict) -> str:
    for key in ("generation_custom_id", "question_id"):
        val = row.get(key)
        if val:
            return str(val)
    return "|".join(
        str(row.get(k, ""))
        for k in ("benchmark", "question_key", "participant_id", "source_index", "question")
    )


def split_name(row: Dict) -> str:
    split = str(row.get("split_type", "")).strip().lower()
    return "ambiguous" if split == "ambiguous" else "explicit"


def candidate_lengths(row: Dict) -> List[float]:
    sums = row.get("option_loglikelihood_sums") or []
    avgs = row.get("option_loglikelihood_avgs") or []
    lengths = []
    for score_sum, score_avg in zip(sums, avgs):
        s = float(score_sum)
        a = float(score_avg)
        lengths.append(max(1.0, abs(s / a)) if a != 0 else 1.0)
    return lengths


def selected_scores(row: Dict, alpha: Optional[float], beta: float) -> List[float]:
    sums = [float(x) for x in (row.get("option_loglikelihood_sums") or [])]
    if not sums:
        return []
    if alpha is None:
        primary = [float(x) for x in (row.get("option_loglikelihood_selected_scores") or [])]
        return primary if primary else sums
    lengths = candidate_lengths(row)
    primary = [s / (length ** alpha) for s, length in zip(sums, lengths)]
    cal_sums = row.get("null_calibration_option_loglikelihood_sums")
    cal_avgs = row.get("null_calibration_option_loglikelihood_avgs")
    if not cal_sums or beta == 0:
        return primary
    cal_lengths = []
    for score_sum, score_avg in zip(cal_sums, cal_avgs or []):
        s = float(score_sum)
        a = float(score_avg)
        cal_lengths.append(max(1.0, abs(s / a)) if a != 0 else 1.0)
    calibration = [
        float(s) / (length ** alpha) for s, length in zip(cal_sums, cal_lengths)
    ]
    return [p - beta * c for p, c in zip(primary, calibration)]


def correctness(row: Dict, alpha: Optional[float] = None, beta: float = 0.0) -> float:
    if alpha is None and beta == 0.0:
        return 1.0 if bool(row.get("is_correct")) else 0.0
    scores = selected_scores(row, alpha=alpha, beta=beta)
    options = row.get("prompt_options") or row.get("options") or []
    gold = row.get("correct_answer") or row.get("eval_correct_answer")
    if not scores or gold not in options:
        return 1.0 if bool(row.get("is_correct")) else 0.0
    pred_idx = int(np.argmax(np.array(scores, dtype=float)))
    return 1.0 if options[pred_idx] == gold else 0.0


def parse_rescore(scoring: str) -> Tuple[Optional[float], float]:
    if not isinstance(scoring, str) or not scoring.startswith("rescore"):
        return None, 0.0
    match = re.search(r"alpha=([0-9.]+)_beta=([0-9.]+)", scoring)
    if not match:
        return None, 0.0
    return float(match.group(1)), float(match.group(2))


def paired_bootstrap(
    plus: Dict[str, float],
    minus: Dict[str, float],
    rng: np.random.Generator,
    n_boot: int,
    scale: float = 100.0,
) -> Tuple[float, float, float, int]:
    keys = sorted(set(plus) & set(minus))
    if not keys:
        raise RuntimeError("No paired keys for bootstrap.")
    diff = np.array([plus[k] - minus[k] for k in keys], dtype=float)
    point = float(diff.mean() * scale)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    samples = diff[idx].mean(axis=1) * scale
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return point, float(lo), float(hi), len(diff)


def localnewsqa_maps(
    plus_paths: Sequence[Path],
    minus_paths: Sequence[Path],
) -> Dict[str, Tuple[Dict[str, float], Dict[str, float]]]:
    split_values = {
        "overall": ({}, {}),
        "ambiguous": ({}, {}),
        "explicit": ({}, {}),
    }
    accum: Dict[str, Dict[str, List[float]]] = {}

    for side, paths in (("plus", plus_paths), ("minus", minus_paths)):
        for path in paths:
            for row in read_jsonl(path):
                key = item_key(row)
                split = split_name(row)
                val = 1.0 if bool(row.get("is_correct")) else 0.0
                for bucket in ("overall", split):
                    accum.setdefault(f"{side}:{bucket}", {}).setdefault(key, []).append(val)

    for bucket in ("overall", "ambiguous", "explicit"):
        plus = {
            key: float(np.mean(vals))
            for key, vals in accum.get(f"plus:{bucket}", {}).items()
        }
        minus = {
            key: float(np.mean(vals))
            for key, vals in accum.get(f"minus:{bucket}", {}).items()
        }
        split_values[bucket] = (plus, minus)
    return split_values


def glob_one(pattern: str) -> Path:
    matches = sorted(ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]


def localnewsqa_paths(row: RowSpec, seed: int) -> Tuple[List[Path], List[Path]]:
    if row.kind == "maple" and row.track == "base":
        size = "1b" if row.key == "maple_1b" else "3b"
        return (
            [
                ROOT
                / f"results/downstream_localnewsqa_pretrained_multiseed/seed_{s}/"
                / f"localnewsqa_eval_target_with_metadata_custom_{size}_tplus_eplus_c0.jsonl"
                for s in (41, 42, 43, 44, 45)
            ],
            [
                ROOT
                / f"results/downstream_localnewsqa_pretrained_multiseed/seed_{s}/"
                / f"localnewsqa_eval_target_without_metadata_custom_{size}_tminus_eminus_c0.jsonl"
                for s in (41, 42, 43, 44, 45)
            ],
        )
    if row.kind == "maple" and row.track == "chat" and row.key == "maple_1b":
        root = (
            ROOT
            / "results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_multiseed_contrib2"
            / "ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos"
        )
        return (
            [glob_one(str(root.relative_to(ROOT)) + f"/seed_{s}/*tplus_eplus_seed{s}_c0.jsonl") for s in (41, 42, 43, 44, 45)],
            [glob_one(str(root.relative_to(ROOT)) + f"/seed_{s}/*tminus_eminus_seed{s}_c0.jsonl") for s in (41, 42, 43, 44, 45)],
        )
    if row.kind == "maple" and row.track == "chat" and row.key == "maple_3b":
        root = (
            ROOT
            / "results/downstream_localnewsqa_sft_figure9_full_multiseed"
            / "3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos"
        )
        return (
            [glob_one(str(root.relative_to(ROOT)) + f"/seed_{s}/*tplus_eplus_seed{s}_c0.jsonl") for s in (41, 42, 43, 44, 45)],
            [glob_one(str(root.relative_to(ROOT)) + f"/seed_{s}/*tminus_eminus_seed{s}_c0.jsonl") for s in (41, 42, 43, 44, 45)],
        )
    if row.kind == "external" and row.track == "chat":
        slug = LNQA_CHAT_SLUGS.get(row.key, row.key)
        return (
            [
                ROOT
                / f"results/downstream_localnewsqa_external_baselines_multiseed/seed_{s}/"
                / f"localnewsqa_{slug}_with_metadata_target.jsonl"
                for s in (41, 42, 43, 44, 45)
            ],
            [
                ROOT
                / f"results/downstream_localnewsqa_external_baselines_multiseed/seed_{s}/"
                / f"localnewsqa_{slug}_without_metadata_target.jsonl"
                for s in (41, 42, 43, 44, 45)
            ],
        )
    # External base rows were run under the final-matrix target-only wrapper.
    root = FINAL_ROOT / "localnewsqa" / row.key / row.track / f"seed_{seed}" / "target"
    return (
        sorted(root.glob(f"localnewsqa_{row.key}_{row.track}_with_metadata_target*.jsonl")),
        sorted(root.glob(f"localnewsqa_{row.key}_{row.track}_without_metadata_target*.jsonl")),
    )


def final_jsonl(model_key: str, track: str, seed: int, bench: str, variant: str) -> Optional[Path]:
    base = FINAL_ROOT / "closed_form" / model_key / track / f"seed_{seed}" / bench
    candidates = [
        base / variant / f"{bench}_{variant}.jsonl",
        base / f"{bench}_{variant}.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def protocol_jsonls(family: str, bench: str) -> Tuple[Path, Path]:
    df = pd.read_csv(PROTOCOL_BEST)
    block = df[(df["family"] == family) & (df["benchmark"] == bench)]
    if block.empty:
        raise RuntimeError(f"No protocol row for {family}/{bench}")
    row = block.iloc[0]
    return Path(row["positive_jsonl"]), Path(row["negative_jsonl"])


def tinker_source_jsonls(row: pd.Series) -> Tuple[Path, Path, Optional[float], float]:
    family = str(row["family"])
    bench = str(row["benchmark"])
    scoring = str(row["scoring"])
    alpha, beta = parse_rescore(scoring)
    if str(row["source"]).endswith("results/external_paper_table_tokenizerfix_20260504"):
        plus, minus = protocol_jsonls(family, bench)
    else:
        root = ROOT / str(row["source"])
        plus = root / "custom_tplus_eplus" / f"{bench}_custom_tplus_eplus.jsonl"
        minus = root / "custom_tminus_eminus" / f"{bench}_custom_tminus_eminus.jsonl"
    return plus, minus, alpha, beta


def external_paths(row: RowSpec, bench: str, seed: int) -> Tuple[Path, Path, Optional[float], float]:
    if row.kind == "maple" and row.track == "base":
        df = pd.read_csv(TINKER_LONG)
        block = df[
            (df["family"] == row.key)
            & (df["benchmark"] == bench)
            & (df["comparison"] == "T+I+ vs T-I-")
        ]
        if block.empty:
            raise RuntimeError(f"No tinker source for {row.key}/{bench}")
        return tinker_source_jsonls(block.iloc[0])
    if row.kind == "maple":
        plus = final_jsonl(row.key, row.track, seed, bench, "custom_tplus_eplus")
        minus = final_jsonl(row.key, row.track, seed, bench, "custom_tminus_eminus")
    else:
        plus = final_jsonl(row.key, row.track, seed, bench, "llama3_chat_with_metadata")
        minus = final_jsonl(row.key, row.track, seed, bench, "llama3_chat_without_metadata")
    if plus is None or minus is None:
        raise FileNotFoundError(f"Missing external pair for {row.key}/{row.track}/{bench}")
    return plus, minus, None, 0.0


def accuracy_maps(
    plus_path: Path,
    minus_path: Path,
    alpha: Optional[float],
    beta: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    plus = {item_key(r): correctness(r, alpha=alpha, beta=beta) for r in read_jsonl(plus_path)}
    minus = {item_key(r): correctness(r, alpha=alpha, beta=beta) for r in read_jsonl(minus_path)}
    return plus, minus


def wasserstein_equal_weight(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if not values_a or not values_b:
        return float("nan")
    a = np.sort(np.array(values_a, dtype=float))
    b = np.sort(np.array(values_b, dtype=float))
    if len(a) == len(b):
        return float(np.mean(np.abs(a - b)))
    n = max(len(a), len(b))
    q = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def wvb_groups(path: Path) -> Dict[Tuple[str, str, str], Tuple[int, float]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {}
    for row in read_jsonl(path):
        qkey = row.get("question_key")
        pred = row.get("processed_answer")
        gold = row.get("correct_answer")
        if qkey is None or pred is None:
            continue
        try:
            pred_i = int(pred)
            gold_i = int(gold)
            opts = [int(x) for x in (row.get("options") or [])]
        except Exception:
            continue
        if not opts or max(opts) <= min(opts):
            continue
        pred_n = (pred_i - min(opts)) / (max(opts) - min(opts))
        gold_n = (gold_i - min(opts)) / (max(opts) - min(opts))
        key = (str(qkey), str(row.get("continent")), str(row.get("country")))
        slot = grouped.setdefault(key, {"gold": [], "pred": []})
        slot["gold"].append(gold_n)
        slot["pred"].append(pred_n)
    return {
        key: (len(vals["pred"]), wasserstein_equal_weight(vals["gold"], vals["pred"]))
        for key, vals in grouped.items()
    }


def wvb_bootstrap(
    plus_path: Path,
    minus_path: Path,
    rng: np.random.Generator,
    n_boot: int,
) -> Tuple[float, float, float, int]:
    plus = wvb_groups(plus_path)
    minus = wvb_groups(minus_path)
    keys = sorted(set(plus) & set(minus))
    if not keys:
        raise RuntimeError("No paired WVB groups.")
    plus_n = np.array([plus[k][0] for k in keys], dtype=float)
    minus_n = np.array([minus[k][0] for k in keys], dtype=float)
    weights = (plus_n + minus_n) / 2.0
    plus_emd = np.array([plus[k][1] for k in keys], dtype=float)
    minus_emd = np.array([minus[k][1] for k in keys], dtype=float)
    point = float((np.sum(weights * minus_emd) / np.sum(weights)) - (np.sum(weights * plus_emd) / np.sum(weights)))
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    sample_weights = weights[idx]
    sample_plus = plus_emd[idx]
    sample_minus = minus_emd[idx]
    samples = (
        np.sum(sample_weights * sample_minus, axis=1) / np.sum(sample_weights, axis=1)
        - np.sum(sample_weights * sample_plus, axis=1) / np.sum(sample_weights, axis=1)
    )
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return point, float(lo), float(hi), len(keys)


def fmt_signed(value: float, decimals: int) -> str:
    return f"{value:+.{decimals}f}"


def ci_excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


def latex_cell(metric: str, point: float, lo: float, hi: float) -> str:
    decimals = 4 if metric == "wvb_emd_gain" else 1
    p = fmt_signed(point, decimals)
    l = fmt_signed(lo, decimals)
    h = fmt_signed(hi, decimals)
    if point > 0:
        macro = "cigainposstrong" if ci_excludes_zero(lo, hi) else "cigainpos"
    elif point < 0:
        macro = "cigainnegstrong" if ci_excludes_zero(lo, hi) else "cigainneg"
    else:
        macro = "cigainzero"
    return rf"\{macro}{{{p}}}{{{l}}}{{{h}}}"


def build_rows(results: pd.DataFrame) -> str:
    columns = [key for key, _ in LNQA_SPLITS] + [key for key, _ in EXTERNAL_BENCHES]
    lines: List[str] = []
    last_group = None
    for spec in ROWS:
        if spec.group != last_group:
            if last_group is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{spec.provider}~{spec.group}}}}} \\")
            last_group = spec.group
        label = spec.label
        cells = [label]
        for col in columns:
            block = results[
                (results["row_key"] == spec.key)
                & (results["track"] == spec.track)
                & (results["metric_key"] == col)
            ]
            if block.empty:
                cells.append(r"\textemdash")
                continue
            r = block.iloc[0]
            cells.append(latex_cell(str(r["metric_type"]), float(r["delta"]), float(r["ci_low"]), float(r["ci_high"])))
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.bootstrap_seed)
    records: List[Dict[str, object]] = []

    for spec in ROWS:
        plus_paths, minus_paths = localnewsqa_paths(spec, args.seed)
        if not plus_paths or not minus_paths:
            raise FileNotFoundError(f"Missing LocalNewsQA paths for {spec}")
        for path in list(plus_paths) + list(minus_paths):
            if not path.exists():
                raise FileNotFoundError(path)
        split_maps = localnewsqa_maps(plus_paths, minus_paths)
        for split_key, _ in LNQA_SPLITS:
            split = split_key.replace("localnewsqa_", "")
            plus, minus = split_maps[split]
            point, lo, hi, n = paired_bootstrap(plus, minus, rng, args.bootstrap, scale=100.0)
            records.append(
                {
                    "row_key": spec.key,
                    "track": spec.track,
                    "label": spec.label,
                    "group": spec.group,
                    "metric_key": split_key,
                    "metric_type": "accuracy_pp_gain",
                    "delta": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": n,
                    "plus_value": float(np.mean(list(plus.values())) * 100.0),
                    "minus_value": float(np.mean(list(minus.values())) * 100.0),
                    "source_plus": ";".join(str(p) for p in plus_paths),
                    "source_minus": ";".join(str(p) for p in minus_paths),
                }
            )

        for bench, _ in EXTERNAL_BENCHES:
            plus_path, minus_path, alpha, beta = external_paths(spec, bench, args.seed)
            if bench == "worldvaluebench":
                point, lo, hi, n = wvb_bootstrap(plus_path, minus_path, rng, args.bootstrap)
                plus_value = math.nan
                minus_value = math.nan
                metric_type = "wvb_emd_gain"
            else:
                plus, minus = accuracy_maps(plus_path, minus_path, alpha=alpha, beta=beta)
                point, lo, hi, n = paired_bootstrap(plus, minus, rng, args.bootstrap, scale=100.0)
                plus_value = float(np.mean(list(plus.values())) * 100.0)
                minus_value = float(np.mean(list(minus.values())) * 100.0)
                metric_type = "accuracy_pp_gain"
            records.append(
                {
                    "row_key": spec.key,
                    "track": spec.track,
                    "label": spec.label,
                    "group": spec.group,
                    "metric_key": bench,
                    "metric_type": metric_type,
                    "delta": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": n,
                    "plus_value": plus_value,
                    "minus_value": minus_value,
                    "source_plus": str(plus_path),
                    "source_minus": str(minus_path),
                    "rescore_alpha": alpha,
                    "rescore_beta": beta,
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_csv = args.out_dir / "external_appendix_gain_table_long.csv"
    df = pd.DataFrame(records)
    df.to_csv(long_csv, index=False)

    rows_tex = args.out_dir / "external_appendix_gain_table_rows.tex"
    rows_tex.write_text(build_rows(df), encoding="utf-8")

    wide_csv = args.out_dir / "external_appendix_gain_table_wide.csv"
    df.pivot_table(
        index=["group", "label", "row_key", "track"],
        columns="metric_key",
        values="delta",
        aggfunc="first",
    ).reset_index().to_csv(wide_csv, index=False)

    print(f"[ok] wrote {long_csv}")
    print(f"[ok] wrote {rows_tex}")
    print(f"[ok] wrote {wide_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
