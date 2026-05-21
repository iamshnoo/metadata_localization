#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VARIANT_ORDER = ["tplus_eplus", "tminus_eminus"]
VARIANT_LABEL = {
    "tplus_eplus": "(T+, I+)",
    "tminus_eminus": "(T-, I-)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute locale-switch metrics for MAPLE chat/SFT LocalNewsQA runs."
    )
    parser.add_argument(
        "--target-root-1b",
        type=Path,
        default=Path(
            "/path/to/metacul/results/"
            "downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_multiseed_contrib2/"
            "ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos"
        ),
    )
    parser.add_argument(
        "--contrast-root-1b",
        type=Path,
        default=Path(
            "/path/to/metacul/results/"
            "downstream_localnewsqa_sft_figure9_1b_checkpoint_nameplain_countrycue_beta025_contrast_multiseed/"
            "ckpt_7764/1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos"
        ),
    )
    parser.add_argument(
        "--target-root-3b",
        type=Path,
        default=Path(
            "/path/to/metacul/results/"
            "downstream_localnewsqa_sft_figure9_full_multiseed/"
            "3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos"
        ),
    )
    parser.add_argument(
        "--contrast-root-3b",
        type=Path,
        default=Path(
            "/path/to/metacul/results/"
            "downstream_localnewsqa_sft_figure9_contrast_full_multiseed/"
            "3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/path/to/metacul/results/analysis/localnewsqa_sft_locale_switch"),
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=[41, 42, 43, 44, 45],
    )
    return parser.parse_args()


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def detect_variant(path: Path) -> Optional[str]:
    name = path.name
    for variant in VARIANT_ORDER:
        if variant in name:
            return variant
    return None


def safe_item_id(row: Dict) -> Tuple:
    return (
        str(row.get("generation_custom_id") or ""),
        normalize_text(row.get("question")),
        normalize_text(row.get("country")),
        normalize_text(row.get("target_country")),
        normalize_text(row.get("contrast_country")),
        normalize_text(row.get("topic")),
        str(row.get("year") or ""),
    )


def answer_index(answer: str, options: Sequence[str]) -> Optional[int]:
    want = normalize_text(answer)
    for idx, opt in enumerate(options):
        if normalize_text(opt) == want:
            return idx
    return None


def row_prediction(row: Dict) -> Optional[Dict[str, object]]:
    options = list(row.get("prompt_options") or row.get("options") or [])
    if not options:
        return None
    gold_answer = row.get("eval_correct_answer")
    if gold_answer is None:
        return None
    gold_idx = answer_index(str(gold_answer), options)
    if gold_idx is None:
        return None

    selected_scores = row.get("option_loglikelihood_selected_scores")
    if not isinstance(selected_scores, list) or len(selected_scores) != len(options):
        return None
    scores = [float(x) for x in selected_scores]
    pred_idx = int(np.argmax(np.asarray(scores, dtype=float)))
    return {
        "gold_idx": gold_idx,
        "pred_idx": pred_idx,
        "scores": scores,
        "is_correct": float(pred_idx == gold_idx),
        "gold_answer": str(gold_answer),
        "pred_answer": str(options[pred_idx]),
    }


def find_runs(root: Path, expected_seeds: Sequence[int]) -> Dict[Tuple[int, str], Path]:
    out: Dict[Tuple[int, str], Path] = {}
    for path in sorted(root.rglob("*.jsonl")):
        variant = detect_variant(path)
        if variant is None:
            continue
        match = re.search(r"seed_(\d+)", str(path))
        seed = int(match.group(1)) if match else -1
        if seed in expected_seeds:
            out[(seed, variant)] = path
    return out


def collect_family(
    family: str,
    target_root: Path,
    contrast_root: Path,
    expected_seeds: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_map = find_runs(target_root, expected_seeds)
    contrast_map = find_runs(contrast_root, expected_seeds)

    expected_keys = {(seed, variant) for seed in expected_seeds for variant in VARIANT_ORDER}
    missing_target = sorted(expected_keys - set(target_map))
    missing_contrast = sorted(expected_keys - set(contrast_map))
    if missing_target or missing_contrast:
        raise FileNotFoundError(
            f"Missing expected JSONLs for {family}. "
            f"target_missing={missing_target} contrast_missing={missing_contrast}"
        )

    pair_records: List[Dict[str, object]] = []
    seed_summary: List[Dict[str, object]] = []

    for seed in expected_seeds:
        for variant in VARIANT_ORDER:
            target_rows = {
                safe_item_id(row): row
                for row in load_jsonl(target_map[(seed, variant)])
                if normalize_text(row.get("split_type")) == "ambiguous"
            }
            contrast_rows = {
                safe_item_id(row): row
                for row in load_jsonl(contrast_map[(seed, variant)])
                if normalize_text(row.get("split_type")) == "ambiguous"
            }

            exact_vals: List[float] = []
            margin_vals: List[float] = []
            pair_count = 0

            for item_id in sorted(set(target_rows) & set(contrast_rows)):
                target_row = target_rows[item_id]
                contrast_row = contrast_rows[item_id]
                target_pred = row_prediction(target_row)
                contrast_pred = row_prediction(contrast_row)
                if target_pred is None or contrast_pred is None:
                    continue

                same_gold = normalize_text(target_pred["gold_answer"]) == normalize_text(
                    contrast_pred["gold_answer"]
                )
                if same_gold:
                    continue

                target_scores = target_pred["scores"]
                contrast_scores = contrast_pred["scores"]
                target_gold_idx = int(target_pred["gold_idx"])
                contrast_gold_idx = int(contrast_pred["gold_idx"])
                target_margin = float(target_scores[target_gold_idx] - target_scores[contrast_gold_idx])
                contrast_margin = float(
                    contrast_scores[contrast_gold_idx] - contrast_scores[target_gold_idx]
                )

                exact_pair = float(
                    bool(target_pred["is_correct"]) and bool(contrast_pred["is_correct"])
                )
                margin_switch = float(target_margin > 0.0 and contrast_margin > 0.0)
                pair_count += 1
                exact_vals.append(exact_pair)
                margin_vals.append(margin_switch)

                pair_records.append(
                    {
                        "family": family,
                        "seed": seed,
                        "variant": variant,
                        "variant_label": VARIANT_LABEL[variant],
                        "country": str(target_row.get("country") or ""),
                        "contrast_country": str(target_row.get("contrast_country") or ""),
                        "topic": str(target_row.get("topic") or ""),
                        "exact_pair": exact_pair,
                        "margin_switch": margin_switch,
                        "target_margin": target_margin,
                        "contrast_margin": contrast_margin,
                    }
                )

            if pair_count == 0:
                raise RuntimeError(f"No valid ambiguous pairs found for {family} seed={seed} {variant}")

            seed_summary.append(
                {
                    "family": family,
                    "seed": seed,
                    "variant": variant,
                    "variant_label": VARIANT_LABEL[variant],
                    "pair_count": pair_count,
                    "exact_pair": float(np.mean(np.asarray(exact_vals, dtype=float))),
                    "margin_switch": float(np.mean(np.asarray(margin_vals, dtype=float))),
                }
            )

    return pd.DataFrame(seed_summary), pd.DataFrame(pair_records)


def sample_std(vals: Sequence[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    avg = sum(vals) / len(vals)
    return math.sqrt(sum((v - avg) ** 2 for v in vals) / (len(vals) - 1))


def aggregate(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (family, variant, label), group in seed_df.groupby(["family", "variant", "variant_label"], sort=False):
        rows.append(
            {
                "family": family,
                "variant": variant,
                "variant_label": label,
                "pair_count": int(group["pair_count"].iloc[0]),
                "seed_count": int(group["seed"].nunique()),
                "exact_pair": float(group["exact_pair"].mean()),
                "exact_pair_std": sample_std(group["exact_pair"].tolist()),
                "margin_switch": float(group["margin_switch"].mean()),
                "margin_switch_std": sample_std(group["margin_switch"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    seed_frames: List[pd.DataFrame] = []
    pair_frames: List[pd.DataFrame] = []

    for family, target_root, contrast_root in [
        ("MAPLE Chat 1B", args.target_root_1b, args.contrast_root_1b),
        ("MAPLE Chat 3B", args.target_root_3b, args.contrast_root_3b),
    ]:
        seed_df, pair_df = collect_family(
            family=family,
            target_root=target_root,
            contrast_root=contrast_root,
            expected_seeds=args.expected_seeds,
        )
        seed_frames.append(seed_df)
        pair_frames.append(pair_df)

    seed_df = pd.concat(seed_frames, ignore_index=True)
    pair_df = pd.concat(pair_frames, ignore_index=True)
    summary_df = aggregate(seed_df)

    seed_df.to_csv(args.output_root / "seed_level.csv", index=False)
    pair_df.to_csv(args.output_root / "pair_level.csv", index=False)
    summary_df.to_csv(args.output_root / "summary.csv", index=False)

    print(f"[ok] wrote {(args.output_root / 'seed_level.csv')}")
    print(f"[ok] wrote {(args.output_root / 'pair_level.csv')}")
    print(f"[ok] wrote {(args.output_root / 'summary.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
