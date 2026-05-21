#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List

import numpy as np


ROOT = Path("/path/to/metacul")
DEFAULT_RUN_ROOT = ROOT / "results/localnewsqa_gold_20260516"
STEM = "1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-score cached SFT 1B LocalNewsQA JSONLs with a new alpha/beta harness."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--backup-suffix", default=None)
    return parser.parse_args()


def score_from_sum_avg(sums: List[float], avgs: List[float], alpha: float) -> List[float]:
    scores = []
    for score_sum, score_avg in zip(sums, avgs):
        tok_len = abs(score_sum / score_avg) if score_avg != 0 else 1.0
        scores.append(score_sum / max(tok_len, 1.0) ** alpha)
    return scores


def hardlink_backup(path: Path, backup: Path) -> None:
    if backup.exists():
        return
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(path, backup, copy_function=os.link)


def rewrite_file(path: Path, alpha: float, beta: float) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with path.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            options = list(row.get("prompt_options") or row.get("options") or [])
            sums = row.get("option_loglikelihood_sums")
            avgs = row.get("option_loglikelihood_avgs")
            if not options or not sums or not avgs:
                raise ValueError(f"Missing option scores in {path}")

            primary = score_from_sum_avg(sums, avgs, alpha)
            calibration_sums = row.get("null_calibration_option_loglikelihood_sums")
            calibration_avgs = row.get("null_calibration_option_loglikelihood_avgs")
            calibration = None
            if calibration_sums and calibration_avgs:
                calibration = score_from_sum_avg(calibration_sums, calibration_avgs, alpha)
                selected = [
                    score - beta * calibration_score
                    for score, calibration_score in zip(primary, calibration)
                ]
            else:
                selected = primary

            pred_idx = int(np.argmax(np.asarray(selected, dtype=float)))
            pred_answer = options[pred_idx]
            pred_letter = chr(65 + pred_idx)
            row["raw_output"] = pred_letter
            row["processed_option_letter"] = pred_letter
            row["processed_answer"] = pred_answer
            row["answer_extraction_method"] = row.get("mcq_scoring", "option_text_avg")
            row["option_loglikelihood_length_norm_alpha"] = alpha
            row["option_loglikelihood_primary_scores"] = primary
            row["null_calibration_beta"] = beta
            row["null_calibration_option_scores"] = calibration
            row["option_loglikelihood_selected_scores"] = selected
            row["is_correct"] = pred_answer == row.get("eval_correct_answer")
            row["rescored_harness_alpha"] = alpha
            row["rescored_harness_beta"] = beta
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp, path)
    return count


def main() -> int:
    args = parse_args()
    suffix = args.backup_suffix or f"backup_before_alpha{args.alpha:g}_beta{args.beta:g}"
    roots = [
        args.run_root / "sft_target" / STEM,
        args.run_root / "sft_contrast" / STEM,
    ]
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        backup = root.with_name(f"{root.name}_{suffix}")
        hardlink_backup(root, backup)
        print(f"[backup] {root} -> {backup}")

    total_files = 0
    total_rows = 0
    for root in roots:
        for path in sorted(root.glob("seed_*/*.jsonl")):
            rows = rewrite_file(path, args.alpha, args.beta)
            total_files += 1
            total_rows += rows
            print(f"[rewrote] {path} rows={rows}")
    print(f"[done] files={total_files} rows={total_rows} alpha={args.alpha} beta={args.beta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
