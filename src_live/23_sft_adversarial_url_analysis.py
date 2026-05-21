#!/usr/bin/env python3
"""
Aggregate adversarial-URL SFT eval results and generate summary plots/tables.
"""

import argparse
import json
import hashlib
import os
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_CSV = "/path/to/metacul/results/qa_metacul_eval.csv"
DEFAULT_OUTPUT_SUMMARY = "/path/to/metacul/results/qa_metacul_eval_adversarial_summary.csv"
DEFAULT_OUTPUT_CONTINENT = "/path/to/metacul/results/qa_metacul_eval_adversarial_by_continent.csv"
DEFAULT_OUTPUT_TEX = "/path/to/metacul/results/qa_metacul_eval_adversarial_table.tex"
DEFAULT_QA_DATASET = "/path/to/metacul/qa_data/hf_dataset.jsonl"


def detect_variants(df: pd.DataFrame) -> list[str]:
    variants = []
    for col in df.columns:
        if col.endswith("_correct"):
            variants.append(col[: -len("_correct")])
    return sorted(set(variants))


def compute_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    variants = detect_variants(df)
    rows = []
    grouped = df.groupby(group_cols, dropna=False)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        key_dict = dict(zip(group_cols, group_key))
        for variant in variants:
            correct = group[f"{variant}_correct"].sum()
            incorrect = group[f"{variant}_incorrect"].abs().sum()
            skipped = group[f"{variant}_skipped"].sum()
            answered = correct + incorrect
            accuracy = correct / answered if answered else 0.0
            rows.append(
                {
                    **key_dict,
                    "variant": variant,
                    "correct": int(correct),
                    "incorrect": int(incorrect),
                    "skipped": int(skipped),
                    "answered": int(answered),
                    "accuracy": float(accuracy),
                }
            )
    return pd.DataFrame(rows)


def write_latex_table(summary: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return
    pivot = summary.pivot(index="url_corruption_rate", columns="variant", values="accuracy")
    pivot = pivot.sort_index()

    lines = []
    columns = list(pivot.columns)
    header = " & ".join(["Corruption"] + [c.replace("_", "\\_") for c in columns])
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Accuracy vs. adversarial URL corruption rate (metadata-only variants).}")
    lines.append("\\label{tab:adv_url_accuracy}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l" + "c" * len(columns) + "}")
    lines.append("\\toprule")
    lines.append(f"{header} \\")
    lines.append("\\midrule")
    for rate, row in pivot.iterrows():
        rate_label = f"{rate:.2f}"
        vals = ["{:.4f}".format(v) if pd.notna(v) else "--" for v in row.values]
        lines.append(" & ".join([rate_label] + vals) + " \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def filter_explicit_mentions(df: pd.DataFrame, qa_path: str) -> pd.DataFrame:
    if not qa_path or not os.path.exists(qa_path):
        print(f"[!] QA dataset not found for mention filter: {qa_path}")
        return df

    id_to_text = {}
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
            id_to_text[qid] = (has_country or has_continent)

    if "question_id" not in df.columns:
        print("[!] question_id missing from eval CSV; cannot filter explicit mentions.")
        return df

    mask = df["question_id"].map(id_to_text).fillna(False)
    filtered = df[~mask].copy()
    print(
        f"[✔] Explicit-mention filter kept {len(filtered)}/{len(df)} rows "
        f"({len(filtered)/len(df):.2%})."
    )
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate adversarial URL eval results.")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-summary", default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument("--output-continent", default=DEFAULT_OUTPUT_CONTINENT)
    parser.add_argument("--output-tex", default=DEFAULT_OUTPUT_TEX)
    parser.add_argument("--qa-dataset", default=DEFAULT_QA_DATASET)
    parser.add_argument(
        "--exclude-explicit-country-continent",
        action="store_true",
        default=False,
        help="Only keep questions where country/continent is not explicitly mentioned.",
    )
    args = parser.parse_args()

    if args.exclude_explicit_country_continent:
        if args.output_summary == DEFAULT_OUTPUT_SUMMARY:
            args.output_summary = args.output_summary.replace(".csv", "_noexplicit.csv")
        if args.output_continent == DEFAULT_OUTPUT_CONTINENT:
            args.output_continent = args.output_continent.replace(".csv", "_noexplicit.csv")
        if args.output_tex == DEFAULT_OUTPUT_TEX:
            args.output_tex = args.output_tex.replace(".tex", "_noexplicit.tex")

    df = pd.read_csv(args.input_csv)
    if args.exclude_explicit_country_continent:
        df = filter_explicit_mentions(df, args.qa_dataset)

    summary = compute_summary(df, ["url_corruption_rate"])
    summary.to_csv(args.output_summary, index=False)

    by_continent = compute_summary(df, ["url_corruption_rate", "continent"])
    by_continent.to_csv(args.output_continent, index=False)

    write_latex_table(summary, Path(args.output_tex))

    print(f"[✔] Wrote summary CSV: {args.output_summary}")
    print(f"[✔] Wrote by-continent CSV: {args.output_continent}")
    print(f"[✔] Wrote LaTeX table: {args.output_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
