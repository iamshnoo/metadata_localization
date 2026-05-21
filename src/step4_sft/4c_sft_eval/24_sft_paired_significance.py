#!/usr/bin/env python3
"""Paired significance tests across URL corruption rates (McNemar).
"""

import argparse
import json
import hashlib
import math
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "/path/to/metacul/results/qa_metacul_eval.csv"
DEFAULT_OUTPUT = "/path/to/metacul/results/qa_metacul_eval_adversarial_significance.csv"
DEFAULT_QA_DATASET = "/path/to/metacul/qa_data/hf_dataset.jsonl"


def mcnemar_p_value(b: int, c: int) -> float:
    """Exact McNemar test using binomial (two-sided)."""
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest

        return binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue
    except Exception:
        # Normal approximation with continuity correction
        diff = abs(b - c) - 1
        chi2 = (diff * diff) / n if n > 0 else 0.0
        # p-value from chi-square df=1
        try:
            from math import erf, sqrt

            # survival function for chi-square with df=1
            p = math.erfc(math.sqrt(chi2 / 2))
            return p
        except Exception:
            return float("nan")


def detect_variants(df: pd.DataFrame) -> list[str]:
    variants = []
    for col in df.columns:
        if col.endswith("_correct"):
            variants.append(col[: -len("_correct")])
    return sorted(set(variants))


def filter_explicit_mentions(df: pd.DataFrame, qa_path: str) -> pd.DataFrame:
    if not qa_path or not Path(qa_path).exists():
        print(f"[!] QA dataset not found for mention filter: {qa_path}")
        return df

    id_to_explicit = {}
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            if not qid:
                parts = [
                    row.get("question", ""),
                    json.dumps(row.get("options", []), ensure_ascii=False),
                    row.get("correct_answer", ""),
                    row.get("country", ""),
                    row.get("continent", ""),
                    row.get("generated_by", ""),
                ]
                blob = "||".join(parts).encode("utf-8")
                qid = hashlib.sha1(blob).hexdigest()
            question = row.get("question", "")
            country = row.get("country", "")
            continent = row.get("continent", "")
            text = question.lower()
            has_country = bool(country) and country.lower() in text
            has_continent = bool(continent) and continent.lower() in text
            id_to_explicit[qid] = (has_country or has_continent)

    if "question_id" not in df.columns:
        print("[!] question_id missing from eval CSV; cannot filter explicit mentions.")
        return df

    mask = df["question_id"].map(id_to_explicit).fillna(False)
    filtered = df[~mask].copy()
    print(
        f"[✔] Explicit-mention filter kept {len(filtered)}/{len(df)} rows "
        f"({len(filtered)/len(df):.2%})."
    )
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired significance across corruption rates.")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline", type=float, default=0.0)
    parser.add_argument("--qa-dataset", default=DEFAULT_QA_DATASET)
    parser.add_argument(
        "--exclude-explicit-country-continent",
        action="store_true",
        default=False,
        help="Only keep questions where country/continent is not explicitly mentioned.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.exclude_explicit_country_continent:
        df = filter_explicit_mentions(df, args.qa_dataset)
    if "url_corruption_rate" not in df.columns:
        raise SystemExit("url_corruption_rate column missing in input CSV.")

    variants = detect_variants(df)
    key_cols = ["question_id", "base_url", "generated_by"]
    # include country/continent to be safe
    for extra in ["country", "continent"]:
        if extra in df.columns:
            key_cols.append(extra)

    records = []
    rates = sorted(df["url_corruption_rate"].unique())
    base_rate = args.baseline
    if base_rate not in rates:
        raise SystemExit(f"Baseline rate {base_rate} not found in data.")

    for variant in variants:
        ccol = f"{variant}_correct"
        icol = f"{variant}_incorrect"
        if ccol not in df.columns or icol not in df.columns:
            continue

        base_df = df[df["url_corruption_rate"] == base_rate].copy()
        base_df = base_df[key_cols + [ccol, icol]]
        base_df = base_df.rename(columns={ccol: "base_correct", icol: "base_incorrect"})

        for rate in rates:
            if rate == base_rate:
                continue
            rate_df = df[df["url_corruption_rate"] == rate].copy()
            rate_df = rate_df[key_cols + [ccol, icol]]
            rate_df = rate_df.rename(columns={ccol: "rate_correct", icol: "rate_incorrect"})

            merged = base_df.merge(rate_df, on=key_cols, how="inner")

            # Map to binary correctness (1 correct, 0 incorrect). Ignore missing.
            def to_binary(correct, incorrect):
                if correct == 1:
                    return 1
                if incorrect in (-1, 1):
                    return 0
                return None

            pairs = []
            for _, row in merged.iterrows():
                b = to_binary(row["base_correct"], row["base_incorrect"])
                r = to_binary(row["rate_correct"], row["rate_incorrect"])
                if b is None or r is None:
                    continue
                pairs.append((b, r))

            if not pairs:
                continue

            b10 = sum(1 for b, r in pairs if b == 1 and r == 0)
            b01 = sum(1 for b, r in pairs if b == 0 and r == 1)
            base_acc = sum(1 for b, _ in pairs if b == 1) / len(pairs)
            rate_acc = sum(1 for _, r in pairs if r == 1) / len(pairs)
            pval = mcnemar_p_value(b10, b01)

            records.append(
                {
                    "variant": variant,
                    "baseline_rate": base_rate,
                    "rate": rate,
                    "n_pairs": len(pairs),
                    "base_acc": base_acc,
                    "rate_acc": rate_acc,
                    "delta_acc": rate_acc - base_acc,
                    "b10_base_correct_rate_incorrect": b10,
                    "b01_base_incorrect_rate_correct": b01,
                    "p_value": pval,
                }
            )

    out = pd.DataFrame(records)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"[✔] Wrote paired significance to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
