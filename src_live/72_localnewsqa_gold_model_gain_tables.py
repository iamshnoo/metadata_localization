#!/usr/bin/env python3
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path("/path/to/metacul")
DEFAULT_RUN_ROOT = ROOT / "results/localnewsqa_gold_20260516"
DEFAULT_OUT_DIR = DEFAULT_RUN_ROOT / "appendix_model_gain_tables"
FULL_TARGET_LINES = 18_700
AMBIGUOUS_LINES = 1_700


def load_gain66():
    path = ROOT / "src/66_appendix_model_gain_tables.py"
    spec = importlib.util.spec_from_file_location("gain66", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gain66"] = module
    spec.loader.exec_module(module)
    return module


gain66 = load_gain66()
RowSpec = gain66.RowSpec


LNQA_COLUMNS = [
    ("localnewsqa_overall", "Overall"),
    ("localnewsqa_ambiguous", "Ambig."),
    ("localnewsqa_explicit", "Explicit"),
    ("localnewsqa_exact_pair", "Exact pair"),
    ("localnewsqa_margin_switch", "Margin switch"),
]

ROWS = [
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

MAPLE_STEMS = {
    ("maple_1b", "base"): "1b_codeg_labels_question_final",
    ("maple_3b", "base"): "3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos",
    ("maple_1b", "chat"): "1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos",
    ("maple_3b", "chat"): "3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos",
}

EXTERNAL_FILE_SLUGS = {
    ("llama32_1b", "base"): "llama32_1b_base",
    ("llama32_1b", "chat"): "llama32_1b",
    ("llama32_3b", "base"): "llama32_3b_base",
    ("llama32_3b", "chat"): "llama32_3b",
    ("qwen25_1p5b", "base"): "qwen25_1p5b_base",
    ("qwen25_1p5b", "chat"): "qwen25_1p5b",
    ("qwen25_3b", "base"): "qwen25_3b_base",
    ("qwen25_3b", "chat"): "qwen25_3b",
    ("qwen35_0p8b", "base"): "qwen35_0p8b_base",
    ("qwen35_0p8b", "chat"): "qwen35_0p8b",
    ("qwen35_2b", "base"): "qwen35_2b_base",
    ("qwen35_2b", "chat"): "qwen35_2b",
    ("qwen35_4b", "base"): "qwen35_4b_base",
    ("qwen35_4b", "chat"): "qwen35_4b",
    ("gemma4_e2b", "base"): "gemma4_e2b_base",
    ("gemma4_e2b", "chat"): "gemma4_e2b_it",
    ("gemma4_e4b", "base"): "gemma4_e4b_base",
    ("gemma4_e4b", "chat"): "gemma4_e4b_it",
    ("ministral3_3b", "base"): "ministral3_3b_base",
    ("ministral3_3b", "chat"): "ministral3_3b_chat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LocalNewsQA gold metadata-gain table from rerun outputs."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260516)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected completed JSONL is missing.",
    )
    return parser.parse_args()


def warn_or_raise(message: str, strict: bool) -> None:
    if strict:
        raise FileNotFoundError(message)
    print(f"[missing] {message}", file=sys.stderr)


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def require_lines(path: Path, min_lines: int) -> None:
    observed = count_lines(path)
    if observed < min_lines:
        raise RuntimeError(f"Incomplete JSONL: {path} has {observed}, expected {min_lines}")


def first_complete(root: Path, pattern: str, min_lines: int) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"{root}/{pattern}")
    complete = [path for path in matches if path.exists() and count_lines(path) >= min_lines]
    if not complete:
        raise RuntimeError(f"No complete JSONL for {root}/{pattern}")
    return complete[0]


def variant_eval_meta(variant: str) -> str:
    return "with_metadata" if variant.endswith("eplus") else "without_metadata"


def maple_size(spec: RowSpec) -> str:
    return "1b" if spec.key == "maple_1b" else "3b"


def maple_run_tag(spec: RowSpec, variant: str, seed: int, role: str) -> str:
    size = maple_size(spec)
    if spec.track == "base":
        prefix = "frozenfig9contrast" if role == "contrast" else "frozenfig9"
        return f"{prefix}_{size}_{variant}_seed{seed}"
    if spec.key == "maple_1b":
        prefix = "sftgoldcontrast" if role == "contrast" else "sftgold"
        return (
            f"{prefix}_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_"
            f"{variant}_seed{seed}"
        )
    prefix = "sftgoldcontrast" if role == "contrast" else "sftgold"
    return f"{prefix}_3b_best3b_chat_{variant}_seed{seed}"


def maple_path(
    run_root: Path,
    spec: RowSpec,
    variant: str,
    seed: int,
    role: str,
) -> Path:
    track_root = "pretrained" if spec.track == "base" else "sft"
    root = run_root / f"{track_root}_{role}" / MAPLE_STEMS[(spec.key, spec.track)] / f"seed_{seed}"
    eval_meta = variant_eval_meta(variant)
    tag = maple_run_tag(spec, variant, seed, role)
    pattern = f"localnewsqa_eval_{role}_{eval_meta}_custom*{tag}_c0*.jsonl"
    min_lines = FULL_TARGET_LINES
    return first_complete(root, pattern, min_lines)


def external_path(
    run_root: Path,
    spec: RowSpec,
    meta_tag: str,
    seed: int,
    role: str,
) -> Path:
    root = run_root / f"external_{role}" / f"seed_{seed}"
    file_slug = EXTERNAL_FILE_SLUGS.get((spec.key, spec.track), spec.key)
    pattern = f"localnewsqa_{file_slug}_{meta_tag}_{role}*.jsonl"
    min_lines = FULL_TARGET_LINES if role == "target" else AMBIGUOUS_LINES
    return first_complete(root, pattern, min_lines)


def target_paths(
    run_root: Path,
    spec: RowSpec,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    if spec.kind == "maple":
        return (
            [maple_path(run_root, spec, "tplus_eplus", seed, "target")],
            [maple_path(run_root, spec, "tminus_eminus", seed, "target")],
        )
    return (
        [external_path(run_root, spec, "with_metadata", seed, "target")],
        [external_path(run_root, spec, "without_metadata", seed, "target")],
    )


def pair_paths(
    run_root: Path,
    spec: RowSpec,
    seed: int,
) -> Tuple[List[Tuple[int, Path, Path]], List[Tuple[int, Path, Path]]]:
    if spec.kind == "maple":
        plus = (
            seed,
            maple_path(run_root, spec, "tplus_eplus", seed, "target"),
            maple_path(run_root, spec, "tplus_eplus", seed, "contrast"),
        )
        minus = (
            seed,
            maple_path(run_root, spec, "tminus_eminus", seed, "target"),
            maple_path(run_root, spec, "tminus_eminus", seed, "contrast"),
        )
        return [plus], [minus]
    plus = (
        seed,
        external_path(run_root, spec, "with_metadata", seed, "target"),
        external_path(run_root, spec, "with_metadata", seed, "contrast"),
    )
    minus = (
        seed,
        external_path(run_root, spec, "without_metadata", seed, "target"),
        external_path(run_root, spec, "without_metadata", seed, "contrast"),
    )
    return [plus], [minus]


def localnewsqa_accuracy_maps(
    spec: RowSpec,
    plus_paths: Sequence[Path],
    minus_paths: Sequence[Path],
) -> Dict[str, Tuple[Dict[str, float], Dict[str, float]]]:
    accum: Dict[str, Dict[Tuple[str, str, str, str, str, str, str], List[float]]] = {}
    for side, paths in (("plus", plus_paths), ("minus", minus_paths)):
        for path in paths:
            for row in gain66.read_jsonl(path):
                key = gain66.safe_lnqa_item_id(row)
                split = gain66.split_name(row)
                pred = gain66.lnqa_prediction(row, spec)
                val = float(pred["is_correct"]) if pred is not None else float(bool(row.get("is_correct")))
                for bucket in ("overall", split):
                    accum.setdefault(f"{side}:{bucket}", {}).setdefault(key, []).append(val)

    out = {}
    for bucket in ("overall", "ambiguous", "explicit"):
        plus = {key: float(np.mean(vals)) for key, vals in accum.get(f"plus:{bucket}", {}).items()}
        minus = {key: float(np.mean(vals)) for key, vals in accum.get(f"minus:{bucket}", {}).items()}
        out[bucket] = (plus, minus)
    return out


def build_records(args: argparse.Namespace, rng: np.random.Generator) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for spec in ROWS:
        try:
            plus_paths, minus_paths = target_paths(args.run_root, spec, args.seed)
            for path in [*plus_paths, *minus_paths]:
                require_lines(path, FULL_TARGET_LINES)
        except (FileNotFoundError, RuntimeError) as exc:
            warn_or_raise(f"{spec.label} target paths: {exc}", args.strict)
            continue

        split_maps = localnewsqa_accuracy_maps(spec, plus_paths, minus_paths)
        for split in ("overall", "ambiguous", "explicit"):
            plus, minus = split_maps[split]
            try:
                point, lo, hi, n = gain66.paired_bootstrap(
                    plus, minus, rng, args.bootstrap, scale=100.0
                )
            except RuntimeError as exc:
                warn_or_raise(f"{spec.label} {split}: {exc}", args.strict)
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
            plus_pairs, minus_pairs = pair_paths(args.run_root, spec, args.seed)
            for _, target_path, contrast_path in [*plus_pairs, *minus_pairs]:
                require_lines(target_path, FULL_TARGET_LINES)
                min_contrast = AMBIGUOUS_LINES if spec.kind == "external" else FULL_TARGET_LINES
                require_lines(contrast_path, min_contrast)
        except (FileNotFoundError, RuntimeError) as exc:
            warn_or_raise(f"{spec.label} pair paths: {exc}", args.strict)
            continue

        pair_maps = gain66.localnewsqa_pair_metric_maps(spec, plus_pairs, minus_pairs)
        for short_key, metric_key in [
            ("exact", "localnewsqa_exact_pair"),
            ("margin", "localnewsqa_margin_switch"),
        ]:
            plus, minus = pair_maps[short_key]
            try:
                point, lo, hi, n = gain66.paired_bootstrap(
                    plus, minus, rng, args.bootstrap, scale=100.0
                )
            except RuntimeError as exc:
                warn_or_raise(f"{spec.label} {metric_key}: {exc}", args.strict)
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


def write_outputs(args: argparse.Namespace, df: pd.DataFrame) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "localnewsqa_model_gains_long.csv", index=False)
    columns = [key for key, _ in LNQA_COLUMNS]
    rows_tex = gain66.build_rows(df, columns, group_colspan=6, rows=ROWS)
    (args.out_dir / "localnewsqa_model_gains_rows.tex").write_text(rows_tex, encoding="utf-8")
    (args.out_dir / "localnewsqa_model_gains_tabular.tex").write_text(
        "\n".join(
            [
                r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}",
                r"\toprule",
                r"Model & \makecell[c]{Overall\\[-1pt]$\Delta$ acc.} & \makecell[c]{Ambig.\\[-1pt]$\Delta$ acc.} & \makecell[c]{Explicit\\[-1pt]$\Delta$ acc.} & \makecell[c]{Exact pair\\[-1pt]$\Delta$} & \makecell[c]{Margin switch\\[-1pt]$\Delta$} \\",
                r"\midrule",
                rows_tex.rstrip(),
                r"\bottomrule",
                r"\end{tabular*}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    df.pivot_table(
        index=["group", "label", "row_key", "track"],
        columns="metric_key",
        values="delta",
        aggfunc="first",
    ).reset_index().to_csv(args.out_dir / "localnewsqa_model_gains_wide.csv", index=False)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.bootstrap_seed)
    df = pd.DataFrame(build_records(args, rng))
    if df.empty:
        if args.strict:
            raise RuntimeError("No LocalNewsQA gold gain records were built.")
        print("[warn] no rows built")
        return 0
    write_outputs(args, df)
    print(f"[ok] wrote LocalNewsQA gold gain table under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
