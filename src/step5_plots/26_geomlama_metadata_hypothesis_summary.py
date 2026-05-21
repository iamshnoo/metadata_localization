#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional


def load_rows(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row["question_id"]
            pred = row.get("processed_answer")
            correct = row.get("correct_answer")
            is_correct = None
            if pred is not None:
                flag = row.get("is_correct")
                is_correct = bool(flag) if flag is not None else (pred == correct)
            rows[qid] = {
                "pred": pred,
                "gold": correct,
                "is_correct": is_correct,
                "metadata_tag_mode": row.get("metadata_tag_mode"),
            }
    return rows


def accuracy(rows: Dict[str, dict]) -> float:
    answered = [r for r in rows.values() if r["pred"] is not None]
    if not answered:
        return 0.0
    return sum(1 for r in answered if r["is_correct"]) / len(answered)


def flip_rate(
    base_rows: Dict[str, dict],
    other_rows: Dict[str, dict],
) -> Dict[str, Optional[float]]:
    both = []
    changed = 0
    for qid, base in base_rows.items():
        other = other_rows.get(qid)
        if other is None:
            continue
        if base["pred"] is None or other["pred"] is None:
            continue
        both.append(qid)
        if str(base["pred"]) != str(other["pred"]):
            changed += 1
    denom = len(both)
    return {
        "overlap_answered": denom,
        "flip_rate": (changed / denom) if denom else None,
    }


def summarize_model(
    model_name: str,
    no_meta: Dict[str, dict],
    correct_meta: Dict[str, dict],
    shuffled_meta: Dict[str, dict],
    adversarial_meta: Dict[str, dict],
) -> Dict[str, object]:
    acc_no = accuracy(no_meta)
    acc_cor = accuracy(correct_meta)
    acc_shu = accuracy(shuffled_meta)
    acc_adv = accuracy(adversarial_meta)

    flip_cor = flip_rate(no_meta, correct_meta)
    flip_shu = flip_rate(no_meta, shuffled_meta)
    flip_adv = flip_rate(no_meta, adversarial_meta)

    return {
        "model": model_name,
        "acc_no_metadata": acc_no,
        "acc_correct_metadata": acc_cor,
        "acc_shuffled_metadata": acc_shu,
        "acc_adversarial_metadata": acc_adv,
        "delta_correct_minus_no": acc_cor - acc_no,
        "delta_shuffled_minus_no": acc_shu - acc_no,
        "delta_adversarial_minus_no": acc_adv - acc_no,
        "flip_rate_correct_vs_no": flip_cor["flip_rate"],
        "flip_rate_shuffled_vs_no": flip_shu["flip_rate"],
        "flip_rate_adversarial_vs_no": flip_adv["flip_rate"],
        "overlap_correct_vs_no": flip_cor["overlap_answered"],
        "overlap_shuffled_vs_no": flip_shu["overlap_answered"],
        "overlap_adversarial_vs_no": flip_adv["overlap_answered"],
    }


def read_jsonl_if_exists(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    return load_rows(path)


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/path/to/metacul/results/external_benchmarks_runs/geomlama",
    )
    parser.add_argument(
        "--output-csv",
        default="/path/to/metacul/results/external_benchmarks_runs/geomlama/hypothesis_summary.csv",
    )
    parser.add_argument(
        "--output-json",
        default="/path/to/metacul/results/external_benchmarks_runs/geomlama/hypothesis_summary.json",
    )
    args = parser.parse_args()

    root = Path(args.root)

    # T+ model conditions (support both folder layouts).
    tplus_no_path = first_existing(
        [
            root / "hypothesis_tplus_correct" / "geomlama_custom_tplus_eminus.jsonl",
            root / "tplus_correct" / "geomlama_custom_tplus_eminus.jsonl",
        ]
    )
    tplus_correct_path = first_existing(
        [
            root / "hypothesis_tplus_correct" / "geomlama_custom_tplus_eplus.jsonl",
            root / "tplus_correct" / "geomlama_custom_tplus_eplus.jsonl",
        ]
    )
    tplus_shuffled_path = first_existing(
        [
            root / "hypothesis_tplus_shuffled" / "geomlama_custom_tplus_eplus.jsonl",
            root / "tplus_shuffled" / "geomlama_custom_tplus_eplus.jsonl",
        ]
    )
    tplus_adv_path = first_existing(
        [
            root / "hypothesis_tplus_adversarial" / "geomlama_custom_tplus_eplus.jsonl",
            root / "tplus_adversarial" / "geomlama_custom_tplus_eplus.jsonl",
        ]
    )

    # T- model conditions (support both folder layouts).
    tminus_no_path = first_existing(
        [
            root / "hypothesis_tminus_correct" / "geomlama_custom_tminus_eminus.jsonl",
            root / "tminus_correct" / "geomlama_custom_tminus_eminus.jsonl",
        ]
    )
    tminus_correct_path = first_existing(
        [
            root / "hypothesis_tminus_correct" / "geomlama_custom_tminus_eplus.jsonl",
            root / "tminus_correct" / "geomlama_custom_tminus_eplus.jsonl",
        ]
    )
    tminus_shuffled_path = first_existing(
        [
            root / "hypothesis_tminus_shuffled" / "geomlama_custom_tminus_eplus.jsonl",
            root / "tminus_shuffled" / "geomlama_custom_tminus_eplus.jsonl",
        ]
    )
    tminus_adv_path = first_existing(
        [
            root / "hypothesis_tminus_adversarial" / "geomlama_custom_tminus_eplus.jsonl",
            root / "tminus_adversarial" / "geomlama_custom_tminus_eplus.jsonl",
        ]
    )

    tplus_no = read_jsonl_if_exists(tplus_no_path)
    tplus_correct = read_jsonl_if_exists(tplus_correct_path)
    tplus_shuffled = read_jsonl_if_exists(tplus_shuffled_path)
    tplus_adv = read_jsonl_if_exists(tplus_adv_path)
    tminus_no = read_jsonl_if_exists(tminus_no_path)
    tminus_correct = read_jsonl_if_exists(tminus_correct_path)
    tminus_shuffled = read_jsonl_if_exists(tminus_shuffled_path)
    tminus_adv = read_jsonl_if_exists(tminus_adv_path)

    rows = []
    if tplus_no and tplus_correct and tplus_shuffled and tplus_adv:
        rows.append(
            summarize_model(
                "custom_tplus",
                tplus_no,
                tplus_correct,
                tplus_shuffled,
                tplus_adv,
            )
        )
    if tminus_no and tminus_correct and tminus_shuffled and tminus_adv:
        rows.append(
            summarize_model(
                "custom_tminus",
                tminus_no,
                tminus_correct,
                tminus_shuffled,
                tminus_adv,
            )
        )

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "acc_no_metadata",
        "acc_correct_metadata",
        "acc_shuffled_metadata",
        "acc_adversarial_metadata",
        "delta_correct_minus_no",
        "delta_shuffled_minus_no",
        "delta_adversarial_minus_no",
        "flip_rate_correct_vs_no",
        "flip_rate_shuffled_vs_no",
        "flip_rate_adversarial_vs_no",
        "overlap_correct_vs_no",
        "overlap_shuffled_vs_no",
        "overlap_adversarial_vs_no",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rows_written": len(rows),
                "rows": rows,
            },
            f,
            indent=2,
        )

    print(f"[✔] Wrote {out_csv}")
    print(f"[✔] Wrote {out_json}")
    if not rows:
        print("[!] No complete hypothesis bundles found yet (jobs may still be running).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
