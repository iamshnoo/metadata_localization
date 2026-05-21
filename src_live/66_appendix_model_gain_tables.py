#!/usr/bin/env python3
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
PROTOCOL_BEST = PAPER_EXT_ROOT / "_reports/protocol_probe_best_20260505_014625.csv"
OUT_DIR = ROOT / "results/appendix_model_gain_tables_20260505"

EXTERNAL_BENCHES = [
    ("geomlama", "Geo"),
    ("globalopinionqa", "GOQA"),
    ("globalmmlu_cs", "MMLU-CS"),
    ("normad", "NormAD"),
    ("blend", "BLEnD"),
    ("worldvaluebench", "WVB"),
]

EXTERNAL_MIN_LINES = {
    "geomlama": 150,
    "globalopinionqa": 10275,
    "globalmmlu_cs": 792,
    "normad": 2633,
    "blend": 393,
    "worldvaluebench": 1458,
}
LNQA_FULL_LINES = 35874
LNQA_EXTERNAL_CHAT_CONTRAST_LINES = 16409

LNQA_COLUMNS = [
    ("localnewsqa_overall", "Overall"),
    ("localnewsqa_ambiguous", "Ambig."),
    ("localnewsqa_explicit", "Explicit"),
    ("localnewsqa_exact_pair", "Exact pair"),
    ("localnewsqa_margin_switch", "Margin switch"),
]

LNQA_CHAT_SLUGS = {
    "gemma4_e2b": "gemma4_e2b_it",
    "gemma4_e4b": "gemma4_e4b_it",
    "ministral3_3b": "ministral3_3b",
}

MAPLE_LNQA_PAIR_SCORING = {
    "maple_1b": {"alpha": 1.25, "beta": 0.0},
    "maple_3b": {"alpha": 0.25, "beta": 0.5},
}

MAPLE_BASE_EXTERNAL_SELECTED = {
    ("maple_1b", "geomlama"): {
        "root": ROOT
        / "results/external_protocol_selected_maple_seed41/geomlama/"
        / "strict_country_a2_b1/maple_1b/seed_41/geomlama",
        "alpha": None,
        "beta": 0.0,
    },
    ("maple_3b", "geomlama"): {
        "root": ROOT
        / "results/external_protocol_selected_maple_seed41/geomlama/"
        / "sft_1b_lnqa/maple_3b/seed_41/geomlama",
        "alpha": 0.35,
        "beta": 1.0,
    },
    ("maple_1b", "normad"): {
        "root": ROOT
        / "results/external_paper_table_tokenizerfix_20260504/normad/"
        / "figure9_1b/maple_1b/seed_41/normad",
        "alpha": None,
        "beta": 0.0,
    },
    ("maple_1b", "blend"): {
        "root": ROOT
        / "results/external_paper_table_tokenizerfix_20260504/blend/"
        / "letter_country_first/maple_1b/seed_41/blend",
        "alpha": 0.0,
        "beta": 0.975,
    },
    ("maple_1b", "worldvaluebench"): {
        "root": ROOT
        / "results/external_paper_table_tokenizerfix_20260504/worldvaluebench/"
        / "letter_country_first/maple_1b/seed_41/worldvaluebench",
        "alpha": 0.0,
        "beta": 1.3,
    },
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

EXTERNAL_ROWS = ROWS

MAPLE_CHAT_EXTERNAL_ROOTS = {
    "maple_1b": ROOT / "results/external_benchmarks_maple_tuned_seed41/sft_1b_chat",
    "maple_3b": ROOT / "results/external_benchmarks_maple_tuned_seed41/sft_3b_best3b_chat",
}

MAPLE_CHAT_EXTERNAL_SELECTED = {
    ("maple_1b", "geomlama"): {
        "root": ROOT
        / "results/external_protocol_selected_maple_chat_seed41/geomlama/"
        / "strict_country/maple_1b_chat/seed_41/geomlama",
        "alpha": None,
        "beta": 0.0,
    },
    ("maple_3b", "geomlama"): {
        "root": ROOT
        / "results/external_protocol_selected_maple_chat_seed41/geomlama/"
        / "strict_country_a2_b1/maple_3b_chat/seed_41/geomlama",
        "alpha": None,
        "beta": 0.0,
    },
    ("maple_1b", "globalmmlu_cs"): {
        "plus": ROOT
        / "results/external_benchmarks_maple_tuned_seed41/sft_1b_chat/"
        / "seed_41/mmlu/custom_tplus_eplus/mmlu_custom_tplus_eplus.jsonl",
        "minus": ROOT
        / "results/external_benchmarks_maple_tuned_seed41/sft_1b_chat/"
        / "seed_41/mmlu/custom_tminus_eminus/mmlu_custom_tminus_eminus.jsonl",
        "alpha": 0.0,
        "beta": 0.0,
    },
    ("maple_3b", "globalmmlu_cs"): {
        "plus": ROOT
        / "results/external_benchmarks_maple_tuned_seed41/sft_3b_best3b_chat/"
        / "seed_41/mmlu/custom_tplus_eplus/mmlu_custom_tplus_eplus.jsonl",
        "minus": ROOT
        / "results/external_benchmarks_maple_tuned_seed41/sft_3b_best3b_chat/"
        / "seed_41/mmlu/custom_tminus_eminus/mmlu_custom_tminus_eminus.jsonl",
        "alpha": 0.2,
        "beta": 0.25,
    },
    ("maple_1b", "globalopinionqa"): {
        "root": ROOT
        / "results/external_protocol_selected_maple_chat_seed41/globalopinionqa/"
        / "strict_country_a0_b15/maple_1b_chat/seed_41/globalopinionqa",
        "alpha": 0.50,
        "beta": 1.50,
    },
    ("maple_3b", "globalopinionqa"): {
        "root": ROOT
        / "results/external_protocol_selected_maple_chat_seed41/globalopinionqa/"
        / "legacy_a025/maple_3b_chat/seed_41/globalopinionqa",
        "alpha": None,
        "beta": 0.0,
    },
    ("maple_3b", "normad"): {
        "root": ROOT
        / "results/external_protocol_selected_maple_chat_seed41/normad/"
        / "letter_name_grounded/maple_3b_chat/seed_41/normad",
        "alpha": 0.0,
        "beta": 0.725,
    },
    ("maple_3b", "blend"): {
        "root": ROOT
        / "results/external_protocol_selected_maple_chat_seed41/blend/"
        / "name_grounded_the/maple_3b_chat/seed_41/blend",
        "alpha": 0.25,
        "beta": 0.5,
    },
    ("maple_1b", "blend"): {
        "plus": ROOT
        / "results/final_benchmark_matrix/closed_form/maple_1b/chat/"
        / "seed_41/blend/blend_custom_tplus_eplus.jsonl",
        "minus": ROOT
        / "results/final_benchmark_matrix/closed_form/maple_1b/chat/"
        / "seed_41/blend/blend_custom_tminus_eminus.jsonl",
        "alpha": 1.25,
        "beta": 0.0,
    },
    ("maple_1b", "worldvaluebench"): {
        "root": ROOT
        / "results/external_protocol_selected_maple_chat_seed41/worldvaluebench/"
        / "figure9_3b/maple_1b_chat/seed_41/worldvaluebench",
        "alpha": None,
        "beta": 0.0,
    },
    ("maple_3b", "worldvaluebench"): {
        "root": ROOT
        / "results/external_protocol_selected_maple_chat_seed41/worldvaluebench/"
        / "letter_country_first/maple_3b_chat/seed_41/worldvaluebench",
        "alpha": 0.0,
        "beta": 0.6,
    },
}

MAPLE_CHAT_BENCH_DIRS = {
    "globalmmlu_cs": "mmlu",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build separate external and LocalNewsQA appendix gain tables."
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


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def require_min_lines(path: Path, min_lines: int) -> None:
    observed = count_lines(path)
    if observed < min_lines:
        raise RuntimeError(
            f"Incomplete JSONL: {path} has {observed} lines, expected at least {min_lines}"
        )


def first_complete_path(candidates: Sequence[Path], min_lines: int) -> Optional[Path]:
    existing = [path for path in candidates if path.exists()]
    for path in existing:
        try:
            if count_lines(path) >= min_lines:
                return path
        except OSError:
            continue
    return existing[0] if existing else None


def glob_one(pattern: str) -> Path:
    matches = sorted(ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]


def item_key(row: Dict) -> str:
    for key in ("generation_custom_id", "question_id"):
        val = row.get(key)
        if val:
            return str(val)
    return "|".join(
        str(row.get(k, ""))
        for k in ("benchmark", "question_key", "participant_id", "source_index", "question")
    )


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def safe_lnqa_item_id(row: Dict) -> Tuple[str, str, str, str, str, str, str]:
    return (
        str(row.get("generation_custom_id") or ""),
        normalize_text(row.get("question")),
        normalize_text(row.get("country")),
        normalize_text(row.get("target_country")),
        normalize_text(row.get("contrast_country")),
        normalize_text(row.get("topic")),
        str(row.get("year") or ""),
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
        if primary:
            return primary
        avgs = [float(x) for x in (row.get("option_loglikelihood_avgs") or [])]
        return avgs if avgs else sums
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
    plus: Dict[object, float],
    minus: Dict[object, float],
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


def selected_external_jsonls(
    selected: Dict[str, object],
    bench: str,
) -> Tuple[Path, Path, Optional[float], float]:
    if "plus" in selected and "minus" in selected:
        plus = Path(selected["plus"])
        minus = Path(selected["minus"])
    else:
        selected_root = Path(selected["root"])
        file_bench = str(selected.get("file_bench", bench))
        plus = selected_root / f"{file_bench}_custom_tplus_eplus.jsonl"
        minus = selected_root / f"{file_bench}_custom_tminus_eminus.jsonl"
    return plus, minus, selected["alpha"], float(selected["beta"])


def external_paths(row: RowSpec, bench: str, seed: int) -> Tuple[Path, Path, Optional[float], float]:
    if row.kind == "maple" and row.track == "base":
        selected = MAPLE_BASE_EXTERNAL_SELECTED.get((row.key, bench))
        if selected is not None:
            plus, minus, alpha, beta = selected_external_jsonls(selected, bench)
            if (
                plus.exists()
                and minus.exists()
                and count_lines(plus) >= EXTERNAL_MIN_LINES[bench]
                and count_lines(minus) >= EXTERNAL_MIN_LINES[bench]
            ):
                return plus, minus, alpha, beta

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
        selected = MAPLE_CHAT_EXTERNAL_SELECTED.get((row.key, bench))
        if selected is not None:
            plus, minus, alpha, beta = selected_external_jsonls(selected, bench)
            if (
                plus.exists()
                and minus.exists()
                and count_lines(plus) >= EXTERNAL_MIN_LINES[bench]
                and count_lines(minus) >= EXTERNAL_MIN_LINES[bench]
            ):
                return plus, minus, alpha, beta

        root = MAPLE_CHAT_EXTERNAL_ROOTS[row.key] / f"seed_{seed}"
        bench_dir = MAPLE_CHAT_BENCH_DIRS.get(bench, bench)
        path_pairs: List[Tuple[Optional[Path], Optional[Path]]] = [
            (
                root
                / bench_dir
                / "custom_tplus_eplus"
                / f"{bench_dir}_custom_tplus_eplus.jsonl",
                root
                / bench_dir
                / "custom_tminus_eminus"
                / f"{bench_dir}_custom_tminus_eminus.jsonl",
            ),
            (
                final_jsonl(row.key, row.track, seed, bench, "custom_tplus_eplus"),
                final_jsonl(row.key, row.track, seed, bench, "custom_tminus_eminus"),
            ),
        ]
        for plus, minus in path_pairs:
            if (
                plus is not None
                and minus is not None
                and plus.exists()
                and minus.exists()
                and count_lines(plus) >= EXTERNAL_MIN_LINES[bench]
                and count_lines(minus) >= EXTERNAL_MIN_LINES[bench]
            ):
                return plus, minus, None, 0.0
        plus, minus = path_pairs[0]
    else:
        plus = final_jsonl(row.key, row.track, seed, bench, "llama3_chat_with_metadata")
        minus = final_jsonl(row.key, row.track, seed, bench, "llama3_chat_without_metadata")
    if plus is None or minus is None or not plus.exists() or not minus.exists():
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


def wvb_groups(
    path: Path,
    alpha: Optional[float] = None,
    beta: float = 0.0,
) -> Dict[Tuple[str, str, str], Tuple[int, float]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {}
    for row in read_jsonl(path):
        qkey = row.get("question_key")
        pred = row.get("processed_answer")
        if alpha is not None or beta != 0.0:
            scores = selected_scores(row, alpha=alpha or 0.0, beta=beta)
            options = row.get("prompt_options") or row.get("options") or []
            if scores and options:
                pred = options[int(np.argmax(np.array(scores, dtype=float)))]
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
    alpha: Optional[float] = None,
    beta: float = 0.0,
) -> Tuple[float, float, float, int, float, float]:
    plus = wvb_groups(plus_path, alpha=alpha, beta=beta)
    minus = wvb_groups(minus_path, alpha=alpha, beta=beta)
    keys = sorted(set(plus) & set(minus))
    if not keys:
        raise RuntimeError("No paired WVB groups.")
    plus_n = np.array([plus[k][0] for k in keys], dtype=float)
    minus_n = np.array([minus[k][0] for k in keys], dtype=float)
    weights = (plus_n + minus_n) / 2.0
    plus_emd = np.array([plus[k][1] for k in keys], dtype=float)
    minus_emd = np.array([minus[k][1] for k in keys], dtype=float)
    plus_value = float(np.sum(weights * plus_emd) / np.sum(weights))
    minus_value = float(np.sum(weights * minus_emd) / np.sum(weights))
    point = float(minus_value - plus_value)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    sample_weights = weights[idx]
    sample_plus = plus_emd[idx]
    sample_minus = minus_emd[idx]
    samples = (
        np.sum(sample_weights * sample_minus, axis=1) / np.sum(sample_weights, axis=1)
        - np.sum(sample_weights * sample_plus, axis=1) / np.sum(sample_weights, axis=1)
    )
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return point, float(lo), float(hi), len(keys), plus_value, minus_value


def localnewsqa_target_paths(row: RowSpec, seed: int) -> Tuple[List[Path], List[Path]]:
    if row.kind == "maple" and row.track == "base":
        size = "1b" if row.key == "maple_1b" else "3b"
        stem = (
            "1b_codeg_labels_question_final"
            if row.key == "maple_1b"
            else "3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos"
        )
        root = ROOT / "results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed" / stem
        return (
            [
                glob_one(str((root / f"seed_{s}").relative_to(ROOT)) + f"/*{size}_tplus_eplus_seed{s}_c0.jsonl")
                for s in (seed,)
            ],
            [
                glob_one(str((root / f"seed_{s}").relative_to(ROOT)) + f"/*{size}_tminus_eminus_seed{s}_c0.jsonl")
                for s in (seed,)
            ],
        )
    if row.kind == "maple" and row.track == "chat" and row.key == "maple_1b":
        root = (
            ROOT
            / "results/downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_multiseed_contrib2"
            / "ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos"
        )
        return (
            [glob_one(str(root.relative_to(ROOT)) + f"/seed_{s}/*tplus_eplus_seed{s}_c0.jsonl") for s in (seed,)],
            [glob_one(str(root.relative_to(ROOT)) + f"/seed_{s}/*tminus_eminus_seed{s}_c0.jsonl") for s in (seed,)],
        )
    if row.kind == "maple" and row.track == "chat" and row.key == "maple_3b":
        root = (
            ROOT
            / "results/downstream_localnewsqa_sft_figure9_full_multiseed"
            / "3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos"
        )
        return (
            [glob_one(str(root.relative_to(ROOT)) + f"/seed_{s}/*tplus_eplus_seed{s}_c0.jsonl") for s in (seed,)],
            [glob_one(str(root.relative_to(ROOT)) + f"/seed_{s}/*tminus_eminus_seed{s}_c0.jsonl") for s in (seed,)],
        )
    if row.kind == "external" and row.track == "chat":
        slug = LNQA_CHAT_SLUGS.get(row.key, row.key)
        legacy_root = ROOT / "results/downstream_localnewsqa_external_baselines_multiseed" / f"seed_{seed}"
        final_root = FINAL_ROOT / "localnewsqa" / row.key / row.track / f"seed_{seed}" / "target"
        picked: Dict[str, List[Path]] = {}
        for meta_tag in ("with_metadata", "without_metadata"):
            candidates = [
                legacy_root / f"localnewsqa_{slug}_{meta_tag}_target.jsonl",
                *sorted(final_root.glob(f"localnewsqa_{row.key}_{row.track}_{meta_tag}_target*.jsonl")),
            ]
            path = first_complete_path(candidates, LNQA_FULL_LINES)
            picked[meta_tag] = [path] if path is not None else []
        return picked["with_metadata"], picked["without_metadata"]
    root = FINAL_ROOT / "localnewsqa" / row.key / row.track / f"seed_{seed}" / "target"
    return (
        sorted(root.glob(f"localnewsqa_{row.key}_{row.track}_with_metadata_target*.jsonl")),
        sorted(root.glob(f"localnewsqa_{row.key}_{row.track}_without_metadata_target*.jsonl")),
    )


def localnewsqa_accuracy_maps(
    spec: RowSpec,
    plus_paths: Sequence[Path],
    minus_paths: Sequence[Path],
) -> Dict[str, Tuple[Dict[str, float], Dict[str, float]]]:
    accum: Dict[str, Dict[Tuple[str, str, str, str, str, str, str], List[float]]] = {}
    for side, paths in (("plus", plus_paths), ("minus", minus_paths)):
        for path in paths:
            for row in read_jsonl(path):
                key = safe_lnqa_item_id(row)
                split = split_name(row)
                pred = lnqa_prediction(row, spec)
                if pred is None:
                    val = 1.0 if bool(row.get("is_correct")) else 0.0
                else:
                    val = float(pred["is_correct"])
                for bucket in ("overall", split):
                    accum.setdefault(f"{side}:{bucket}", {}).setdefault(key, []).append(val)
    out = {}
    for bucket in ("overall", "ambiguous", "explicit"):
        plus = {key: float(np.mean(vals)) for key, vals in accum.get(f"plus:{bucket}", {}).items()}
        minus = {key: float(np.mean(vals)) for key, vals in accum.get(f"minus:{bucket}", {}).items()}
        out[bucket] = (plus, minus)
    return out


def maple_pair_root(row: RowSpec, role: str) -> Path:
    if row.track == "base":
        if row.key == "maple_1b":
            stem = "1b_codeg_labels_question_final"
        else:
            stem = "3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos"
        kind = "frozen_full_multiseed" if role == "target" else "contrast_full_multiseed"
        return ROOT / f"results/downstream_localnewsqa_pretrained_figure9_{kind}/{stem}"
    if row.key == "maple_1b":
        base = "downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025"
        sub = "ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos"
        suffix = "multiseed_contrib2" if role == "target" else "contrast_multiseed"
        return ROOT / f"results/{base}_{suffix}/{sub}"
    stem = "3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos"
    if role == "target":
        return ROOT / f"results/downstream_localnewsqa_sft_figure9_full_multiseed/{stem}"
    return ROOT / f"results/downstream_localnewsqa_sft_figure9_contrast_full_multiseed/{stem}"


def find_maple_pair_paths(row: RowSpec, variant: str, seed: int) -> List[Tuple[int, Path, Path]]:
    target_root = maple_pair_root(row, "target")
    contrast_root = maple_pair_root(row, "contrast")
    pairs = []
    target = glob_one(str(target_root.relative_to(ROOT)) + f"/seed_{seed}/*{variant}*seed{seed}*c0.jsonl")
    contrast = glob_one(str(contrast_root.relative_to(ROOT)) + f"/seed_{seed}/*{variant}*seed{seed}*c0.jsonl")
    pairs.append((seed, target, contrast))
    return pairs


def find_external_chat_contrast(slug: str, seed: int, meta_tag: str) -> Path:
    roots = [
        ROOT / "results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun",
        ROOT / "results/downstream_localnewsqa_external_baselines_contrast_multiseed",
    ]
    for root in roots:
        path = root / f"seed_{seed}" / f"localnewsqa_{slug}_{meta_tag}_contrast.jsonl"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing external chat contrast {slug} seed={seed} {meta_tag}")


def localnewsqa_pair_paths(row: RowSpec, seed: int) -> Tuple[List[Tuple[int, Path, Path]], List[Tuple[int, Path, Path]]]:
    if row.kind == "maple":
        return (
            find_maple_pair_paths(row, "tplus_eplus", seed),
            find_maple_pair_paths(row, "tminus_eminus", seed),
        )
    if row.track == "chat":
        slug = LNQA_CHAT_SLUGS.get(row.key, row.key)
        plus = []
        minus = []
        for s in (seed,):
            target_root = ROOT / "results/downstream_localnewsqa_external_baselines_multiseed" / f"seed_{s}"
            final_root = FINAL_ROOT / "localnewsqa" / row.key / row.track / f"seed_{s}"
            chosen: Dict[str, Tuple[Path, Path]] = {}
            for meta_tag in ("with_metadata", "without_metadata"):
                target_candidates = [
                    target_root / f"localnewsqa_{slug}_{meta_tag}_target.jsonl",
                    *sorted((final_root / "target").glob(f"localnewsqa_{row.key}_{row.track}_{meta_tag}_target*.jsonl")),
                ]
                contrast_candidates = [
                    ROOT
                    / "results/downstream_localnewsqa_external_baselines_contrast_multiseed_rerun"
                    / f"seed_{s}"
                    / f"localnewsqa_{slug}_{meta_tag}_contrast.jsonl",
                    ROOT
                    / "results/downstream_localnewsqa_external_baselines_contrast_multiseed"
                    / f"seed_{s}"
                    / f"localnewsqa_{slug}_{meta_tag}_contrast.jsonl",
                    *sorted((final_root / "contrast").glob(f"localnewsqa_{row.key}_{row.track}_{meta_tag}_contrast*.jsonl")),
                ]
                target = first_complete_path(target_candidates, LNQA_FULL_LINES)
                contrast = first_complete_path(contrast_candidates, LNQA_EXTERNAL_CHAT_CONTRAST_LINES)
                if target is None or contrast is None:
                    raise FileNotFoundError(
                        f"Missing LocalNewsQA chat pair paths for {row.key}/{row.track}/seed_{s}/{meta_tag}"
                    )
                chosen[meta_tag] = (target, contrast)
            plus.append(
                (
                    s,
                    chosen["with_metadata"][0],
                    chosen["with_metadata"][1],
                )
            )
            minus.append(
                (
                    s,
                    chosen["without_metadata"][0],
                    chosen["without_metadata"][1],
                )
            )
        return plus, minus
    root = FINAL_ROOT / "localnewsqa" / row.key / row.track / f"seed_{seed}"
    plus_target = sorted((root / "target").glob(f"localnewsqa_{row.key}_{row.track}_with_metadata_target*.jsonl"))
    plus_contrast = sorted((root / "contrast").glob(f"localnewsqa_{row.key}_{row.track}_with_metadata_contrast*.jsonl"))
    minus_target = sorted((root / "target").glob(f"localnewsqa_{row.key}_{row.track}_without_metadata_target*.jsonl"))
    minus_contrast = sorted((root / "contrast").glob(f"localnewsqa_{row.key}_{row.track}_without_metadata_contrast*.jsonl"))
    if not plus_target or not plus_contrast or not minus_target or not minus_contrast:
        raise FileNotFoundError(f"Missing LocalNewsQA base pair paths for {row.key}/{row.track}/seed_{seed}")
    return ([(seed, plus_target[0], plus_contrast[0])], [(seed, minus_target[0], minus_contrast[0])])


def answer_index(answer: str, options: Sequence[str]) -> Optional[int]:
    want = normalize_text(answer)
    for idx, option in enumerate(options):
        if normalize_text(option) == want:
            return idx
    return None


def lnqa_maple_pair_scores(row: Dict, spec: RowSpec) -> Optional[List[float]]:
    sums = row.get("option_loglikelihood_sums")
    if not isinstance(sums, list):
        return None
    lengths = candidate_lengths(row)
    scoring = MAPLE_LNQA_PAIR_SCORING[spec.key]
    alpha = scoring["alpha"]
    beta = scoring["beta"]
    primary = [float(score_sum) / (length ** alpha) for score_sum, length in zip(sums, lengths)]
    raw_calib = row.get("null_calibration_option_loglikelihood_sums")
    raw_calib_avgs = row.get("null_calibration_option_loglikelihood_avgs")
    if raw_calib and raw_calib_avgs and beta != 0:
        calib_lengths = []
        for cal_sum, cal_avg in zip(raw_calib, raw_calib_avgs):
            cal_sum_f = float(cal_sum)
            cal_avg_f = float(cal_avg)
            calib_lengths.append(max(1.0, abs(cal_sum_f / cal_avg_f)) if cal_avg_f != 0 else 1.0)
        calibration = [
            float(cal_sum) / (length ** alpha)
            for cal_sum, length in zip(raw_calib, calib_lengths)
        ]
        return [p - beta * c for p, c in zip(primary, calibration)]
    return primary


def lnqa_row_scores(row: Dict, spec: RowSpec) -> Optional[List[float]]:
    if spec.kind == "maple" and spec.track == "base":
        return lnqa_maple_pair_scores(row, spec)
    selected = row.get("option_loglikelihood_selected_scores")
    if isinstance(selected, list) and selected:
        return [float(x) for x in selected]
    avgs = row.get("option_loglikelihood_avgs")
    if isinstance(avgs, list) and avgs:
        return [float(x) for x in avgs]
    sums = row.get("option_loglikelihood_sums")
    if isinstance(sums, list) and sums:
        return [float(x) for x in sums]
    return None


def lnqa_prediction(row: Dict, spec: RowSpec) -> Optional[Dict[str, object]]:
    options = list(row.get("prompt_options") or row.get("options") or [])
    gold_answer = row.get("eval_correct_answer")
    if not options or gold_answer is None:
        return None
    gold_idx = answer_index(str(gold_answer), options)
    scores = lnqa_row_scores(row, spec)
    if gold_idx is None or scores is None or len(scores) != len(options):
        return None
    pred_idx = int(np.argmax(np.asarray(scores, dtype=float)))
    return {
        "gold_idx": gold_idx,
        "pred_idx": pred_idx,
        "scores": scores,
        "is_correct": float(pred_idx == gold_idx),
        "gold_answer": str(gold_answer),
    }


def localnewsqa_pair_metric_maps(
    spec: RowSpec,
    plus_pairs: Sequence[Tuple[int, Path, Path]],
    minus_pairs: Sequence[Tuple[int, Path, Path]],
) -> Dict[str, Tuple[Dict[str, float], Dict[str, float]]]:
    accum: Dict[str, Dict[str, List[float]]] = {}
    for side, pairs in (("plus", plus_pairs), ("minus", minus_pairs)):
        for _, target_path, contrast_path in pairs:
            target_rows = {
                safe_lnqa_item_id(row): row
                for row in read_jsonl(target_path)
                if normalize_text(row.get("split_type")) == "ambiguous"
            }
            contrast_rows = {
                safe_lnqa_item_id(row): row
                for row in read_jsonl(contrast_path)
                if normalize_text(row.get("split_type")) == "ambiguous"
            }
            for key_tuple in sorted(set(target_rows) & set(contrast_rows)):
                target_pred = lnqa_prediction(target_rows[key_tuple], spec)
                contrast_pred = lnqa_prediction(contrast_rows[key_tuple], spec)
                if target_pred is None or contrast_pred is None:
                    continue
                if normalize_text(target_pred["gold_answer"]) == normalize_text(contrast_pred["gold_answer"]):
                    continue
                target_scores = target_pred["scores"]
                contrast_scores = contrast_pred["scores"]
                target_gold_idx = int(target_pred["gold_idx"])
                contrast_gold_idx = int(contrast_pred["gold_idx"])
                target_margin = float(target_scores[target_gold_idx] - target_scores[contrast_gold_idx])
                contrast_margin = float(contrast_scores[contrast_gold_idx] - contrast_scores[target_gold_idx])
                key = "|".join(key_tuple)
                exact = float(bool(target_pred["is_correct"]) and bool(contrast_pred["is_correct"]))
                margin = float(target_margin > 0.0 and contrast_margin > 0.0)
                accum.setdefault(f"{side}:exact", {}).setdefault(key, []).append(exact)
                accum.setdefault(f"{side}:margin", {}).setdefault(key, []).append(margin)
    out = {}
    for metric in ("exact", "margin"):
        plus = {key: float(np.mean(vals)) for key, vals in accum.get(f"plus:{metric}", {}).items()}
        minus = {key: float(np.mean(vals)) for key, vals in accum.get(f"minus:{metric}", {}).items()}
        out[metric] = (plus, minus)
    return out


def fmt_signed(value: float, decimals: int) -> str:
    if abs(value) < 0.5 * (10 ** -decimals):
        return f"{0:.{decimals}f}"
    return f"{value:+.{decimals}f}"


def ci_excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


def latex_cell(metric_type: str, point: float, lo: float, hi: float) -> str:
    decimals = 2 if metric_type == "wvb_emd_gain" else 1
    p = fmt_signed(point, decimals)
    l = fmt_signed(lo, decimals)
    h = fmt_signed(hi, decimals)
    if abs(point) < 0.5 * (10 ** -decimals):
        macro = "cigainzero"
    elif point > 0:
        macro = "cigainposstrong" if ci_excludes_zero(lo, hi) else "cigainpos"
    elif point < 0:
        macro = "cigainnegstrong" if ci_excludes_zero(lo, hi) else "cigainneg"
    else:
        macro = "cigainzero"
    return rf"\{macro}{{{p}}}{{{l}}}{{{h}}}"


def comparison_badge(spec: RowSpec) -> str:
    if spec.kind == "maple":
        return r"\tableprimaryplus{$(\Tplus,\Iplus)$}$-$\tableprimaryminus{$(\Tminus,\Iminus)$}"
    return r"\tableprimaryplus{$\Iplus$}$-$\tableprimaryminus{$\Iminus$}"


def build_rows(
    results: pd.DataFrame,
    columns: Sequence[str],
    group_colspan: int,
    rows: Sequence[RowSpec] = ROWS,
) -> str:
    lines: List[str] = []
    last_group = None
    for spec in rows:
        if spec.group != last_group:
            if last_group is not None:
                lines.append(r"\midrule")
            lines.append(
                rf"\multicolumn{{{group_colspan}}}{{@{{}}l}}{{\textit{{{spec.provider}~{spec.group}}}\hfill {{\scriptsize {comparison_badge(spec)}}}}} \\"
            )
            last_group = spec.group
        cells = [spec.label]
        for col in columns:
            block = results[
                (results["row_key"] == spec.key)
                & (results["track"] == spec.track)
                & (results["metric_key"] == col)
            ]
            if block.empty:
                cells.append("-")
                continue
            r = block.iloc[0]
            cells.append(latex_cell(str(r["metric_type"]), float(r["delta"]), float(r["ci_low"]), float(r["ci_high"])))
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def build_external_records(args: argparse.Namespace, rng: np.random.Generator) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for spec in EXTERNAL_ROWS:
        for bench, _ in EXTERNAL_BENCHES:
            try:
                plus_path, minus_path, alpha, beta = external_paths(spec, bench, args.seed)
                require_min_lines(plus_path, EXTERNAL_MIN_LINES[bench])
                require_min_lines(minus_path, EXTERNAL_MIN_LINES[bench])
                if bench == "worldvaluebench":
                    point, lo, hi, n, plus_value, minus_value = wvb_bootstrap(
                        plus_path,
                        minus_path,
                        rng,
                        args.bootstrap,
                        alpha=alpha,
                        beta=beta,
                    )
                    metric_type = "wvb_emd_gain"
                else:
                    plus, minus = accuracy_maps(plus_path, minus_path, alpha=alpha, beta=beta)
                    point, lo, hi, n = paired_bootstrap(plus, minus, rng, args.bootstrap, scale=100.0)
                    plus_value = float(np.mean(list(plus.values())) * 100.0)
                    minus_value = float(np.mean(list(minus.values())) * 100.0)
                    metric_type = "accuracy_pp_gain"
            except (FileNotFoundError, RuntimeError):
                continue
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
    return records


def build_localnewsqa_records(args: argparse.Namespace, rng: np.random.Generator) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for spec in ROWS:
        plus_paths, minus_paths = localnewsqa_target_paths(spec, args.seed)
        if not plus_paths or not minus_paths:
            continue
        try:
            for path in list(plus_paths) + list(minus_paths):
                if not path.exists():
                    raise FileNotFoundError(path)
                require_min_lines(path, LNQA_FULL_LINES)
        except (FileNotFoundError, RuntimeError):
            continue
        split_maps = localnewsqa_accuracy_maps(spec, plus_paths, minus_paths)
        for split in ("overall", "ambiguous", "explicit"):
            plus, minus = split_maps[split]
            try:
                point, lo, hi, n = paired_bootstrap(plus, minus, rng, args.bootstrap, scale=100.0)
            except RuntimeError:
                continue
            records.append(
                {
                    "row_key": spec.key,
                    "track": spec.track,
                    "label": spec.label,
                    "group": spec.group,
                    "metric_key": f"localnewsqa_{split}",
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

        try:
            plus_pairs, minus_pairs = localnewsqa_pair_paths(spec, args.seed)
            contrast_min_lines = (
                LNQA_EXTERNAL_CHAT_CONTRAST_LINES
                if spec.kind == "external" and spec.track == "chat"
                else LNQA_FULL_LINES
            )
            for _, target_path, contrast_path in list(plus_pairs) + list(minus_pairs):
                if not target_path.exists():
                    raise FileNotFoundError(target_path)
                if not contrast_path.exists():
                    raise FileNotFoundError(contrast_path)
                require_min_lines(target_path, LNQA_FULL_LINES)
                require_min_lines(contrast_path, contrast_min_lines)
        except (FileNotFoundError, RuntimeError):
            continue
        pair_maps = localnewsqa_pair_metric_maps(spec, plus_pairs, minus_pairs)
        for short_key, metric_key in [
            ("exact", "localnewsqa_exact_pair"),
            ("margin", "localnewsqa_margin_switch"),
        ]:
            plus, minus = pair_maps[short_key]
            try:
                point, lo, hi, n = paired_bootstrap(plus, minus, rng, args.bootstrap, scale=100.0)
            except RuntimeError:
                continue
            records.append(
                {
                    "row_key": spec.key,
                    "track": spec.track,
                    "label": spec.label,
                    "group": spec.group,
                    "metric_key": metric_key,
                    "metric_type": "paired_metric_pp_gain",
                    "delta": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": n,
                    "plus_value": float(np.mean(list(plus.values())) * 100.0),
                    "minus_value": float(np.mean(list(minus.values())) * 100.0),
                    "source_plus": ";".join(f"{t}|{c}" for _, t, c in plus_pairs),
                    "source_minus": ";".join(f"{t}|{c}" for _, t, c in minus_pairs),
                }
            )
    return records


def write_outputs(args: argparse.Namespace, external_df: pd.DataFrame, lnqa_df: pd.DataFrame) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    external_df.to_csv(args.out_dir / "external_model_gains_long.csv", index=False)
    lnqa_df.to_csv(args.out_dir / "localnewsqa_model_gains_long.csv", index=False)

    external_cols = [key for key, _ in EXTERNAL_BENCHES]
    lnqa_cols = [key for key, _ in LNQA_COLUMNS]
    (args.out_dir / "external_model_gains_rows.tex").write_text(
        build_rows(external_df, external_cols, group_colspan=7, rows=EXTERNAL_ROWS),
        encoding="utf-8",
    )
    (args.out_dir / "localnewsqa_model_gains_rows.tex").write_text(
        build_rows(lnqa_df, lnqa_cols, group_colspan=6),
        encoding="utf-8",
    )
    external_rows = build_rows(external_df, external_cols, group_colspan=7, rows=EXTERNAL_ROWS)
    lnqa_rows = build_rows(lnqa_df, lnqa_cols, group_colspan=6)
    (args.out_dir / "external_model_gains_tabular.tex").write_text(
        "\n".join(
            [
                r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}",
                r"\toprule",
                r"Model & \makecell[c]{GeoMLaMA\\[-1pt]$\Delta$ acc.} & \makecell[c]{GOQA\\[-1pt]$\Delta$ acc.} & \makecell[c]{MMLU-CS\\[-1pt]$\Delta$ acc.} & \makecell[c]{NormAD\\[-1pt]$\Delta$ acc.} & \makecell[c]{BLEnD\\[-1pt]$\Delta$ acc.} & \makecell[c]{WVB\\[-1pt]EMD red.} \\",
                r"\midrule",
                external_rows.rstrip(),
                r"\bottomrule",
                r"\end{tabular*}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (args.out_dir / "localnewsqa_model_gains_tabular.tex").write_text(
        "\n".join(
            [
                r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}",
                r"\toprule",
                r"Model & \makecell[c]{Overall\\[-1pt]$\Delta$ acc.} & \makecell[c]{Ambig.\\[-1pt]$\Delta$ acc.} & \makecell[c]{Explicit\\[-1pt]$\Delta$ acc.} & \makecell[c]{Exact pair\\[-1pt]$\Delta$} & \makecell[c]{Margin switch\\[-1pt]$\Delta$} \\",
                r"\midrule",
                lnqa_rows.rstrip(),
                r"\bottomrule",
                r"\end{tabular*}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    for name, df in [
        ("external_model_gains_wide.csv", external_df),
        ("localnewsqa_model_gains_wide.csv", lnqa_df),
    ]:
        df.pivot_table(
            index=["group", "label", "row_key", "track"],
            columns="metric_key",
            values="delta",
            aggfunc="first",
        ).reset_index().to_csv(args.out_dir / name, index=False)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.bootstrap_seed)
    external_df = pd.DataFrame(build_external_records(args, rng))
    lnqa_df = pd.DataFrame(build_localnewsqa_records(args, rng))
    write_outputs(args, external_df, lnqa_df)
    print(f"[ok] wrote tables under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
