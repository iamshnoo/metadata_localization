#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
from tqdm import tqdm
from pathlib import Path


URLS = [
    "www.factquizmaster.com",
    "www.globalfactcheck.org",
    "www.worldknowledgehub.com",
    "www.civicspedia.org",
    "www.internationalfacts.net",
    "www.currentaffairsdesk.com",
    "www.newsinsightarchive.com",
    "www.globalquizvault.com",
    "www.factualdigest.org",
    "www.publicknowledgebase.net",
]

CONTINENTS = ["africa", "america", "asia", "europe"]


def slugify_url(url: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")


def run_eval(
    eval_script: Path,
    model_type: str,
    metadata: bool,
    size: str,
    base_url: str,
    dataset: str,
    split: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    output_jsonl: Path,
    url_corruption_rate: float,
    url_corruption_seed: int,
    custom_model_path: str | None = None,
    name_suffix: str = "",
) -> None:
    cmd = [
        "python",
        str(eval_script),
        "--model-type",
        model_type,
        "--size",
        size,
        "--dataset",
        dataset,
        "--split",
        split,
        "--base-url",
        base_url,
        "--output-jsonl",
        str(output_jsonl),
        "--batch-size",
        str(batch_size),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--url-corruption-rate",
        str(url_corruption_rate),
        "--url-corruption-seed",
        str(url_corruption_seed),
    ]
    if custom_model_path:
        cmd.extend(["--custom-model-path", custom_model_path])
    elif name_suffix:
        cmd.extend(["--name-suffix", name_suffix])
    if metadata:
        cmd.append("--metadata")
    subprocess.run(cmd, check=True)


def compute_metrics(jsonl_path: Path) -> dict:
    overall_total = 0
    overall_correct = 0
    continent_correct = {c: 0 for c in CONTINENTS}
    continent_wrong = {c: 0 for c in CONTINENTS}
    continent_skipped = {c: 0 for c in CONTINENTS}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            continent = str(row.get("continent", "")).strip().lower()
            if continent not in continent_correct:
                continue
            processed_answer = row.get("processed_answer")
            if processed_answer is None:
                continent_skipped[continent] += 1
                continue
            is_correct = row.get("is_correct")
            if is_correct is None:
                is_correct = processed_answer == row.get("correct_answer")
            overall_total += 1
            overall_correct += int(bool(is_correct))
            if is_correct:
                continent_correct[continent] += 1
            else:
                continent_wrong[continent] += 1

    overall_acc = overall_correct / overall_total if overall_total else 0.0
    return {
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "overall_accuracy": overall_acc,
        "continent_correct": continent_correct,
        "continent_wrong": continent_wrong,
        "continent_skipped": continent_skipped,
    }


def append_csv(
    csv_path: Path,
    model_type: str,
    metadata: bool,
    base_url: str,
    metrics: dict,
) -> None:
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "model_type",
        "metadata",
        "base_url",
        "url_corruption_rate",
        "overall_correct",
        "overall_total",
        "overall_accuracy",
    ]
    for cont in CONTINENTS:
        header.extend(
            [
                f"{cont}_correct",
                f"{cont}_wrong",
                f"{cont}_skipped",
            ]
        )

    row = [
        model_type,
        str(metadata),
        base_url,
        f"{metrics['url_corruption_rate']:.2f}",
        metrics["overall_correct"],
        metrics["overall_total"],
        f"{metrics['overall_accuracy']:.6f}",
    ]
    for cont in CONTINENTS:
        row.extend(
            [
                metrics["continent_correct"][cont],
                metrics["continent_wrong"][cont],
                metrics["continent_skipped"][cont],
            ]
        )

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SFT eval across all model variants and URLs."
    )
    parser.add_argument(
        "--eval-script",
        default="/path/to/metacul/src/14_sft_eval.py",
    )
    parser.add_argument("--dataset", default="YOUR_HF_USERNAME/qa_metacul")
    parser.add_argument("--split", default="train")
    parser.add_argument("--size", choices=["1b", "3b"], default="1b")
    parser.add_argument("--name-suffix", default="", help="Suffix appended to default custom model names, e.g. _best3b")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--results-dir",
        default="/path/to/metacul/results/downstream",
    )
    parser.add_argument(
        "--summary-csv",
        default="/path/to/metacul/results/qa_metacul_eval_summary.csv",
    )
    parser.add_argument(
        "--custom-model-path",
        default=None,
        help="Optional explicit merged custom model path; otherwise derived from size/name-suffix.",
    )
    parser.add_argument(
        "--only-metadata",
        action="store_true",
        default=False,
        help="Only run metadata=True variants (llama3_chat/custom).",
    )
    parser.add_argument(
        "--url-corruption-rates",
        default="0,0.25,0.5,0.75,1.0",
        help="Comma-separated corruption rates for URL metadata.",
    )
    parser.add_argument(
        "--url-corruption-seed",
        type=int,
        default=42,
        help="Seed for deterministic URL corruption.",
    )
    args = parser.parse_args()

    eval_script = Path(args.eval_script)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv)
    corruption_rates = [
        float(r.strip())
        for r in args.url_corruption_rates.split(",")
        if r.strip()
    ]

    if args.only_metadata:
        model_variants = [
            ("llama3_chat", True),
            ("custom", True),
        ]
    else:
        model_variants = [
            ("llama3_chat", True),
            ("llama3_chat", False),
            ("custom", True),
            ("custom", False),
        ]

    # Without-metadata variants are unaffected by URL corruption; only run c0 for them.
    total_runs = 0
    for _, metadata in model_variants:
        if metadata:
            total_runs += len(URLS) * len(corruption_rates)
        else:
            total_runs += len(URLS)
    pbar = tqdm(total=total_runs, desc="Overall eval runs", miniters=1, smoothing=0.1)

    for base_url in URLS:
        url_slug = slugify_url(base_url)
        for model_type, metadata in model_variants:
            for corruption_rate in corruption_rates:
                if not metadata and corruption_rate != 0:
                    continue
                suffix = "with_metadata" if metadata else "without_metadata"
                rate_pct = int(round(corruption_rate * 100))
                rate_suffix = f"_c{rate_pct}"
                output_jsonl = results_dir / (
                    f"qa_metacul_eval_{suffix}_{model_type}_{url_slug}{rate_suffix}.jsonl"
                )
                legacy_jsonl = results_dir / (
                    f"qa_metacul_eval_{suffix}_{model_type}_{url_slug}.jsonl"
                )

                if (
                    corruption_rate == 0
                    and not output_jsonl.exists()
                    and legacy_jsonl.exists()
                ):
                    os.replace(legacy_jsonl, output_jsonl)

                if not output_jsonl.exists():
                    run_eval_kwargs = {}
                    if model_type == "custom":
                        if args.custom_model_path:
                            run_eval_kwargs["custom_model_path"] = args.custom_model_path
                        elif args.name_suffix:
                            run_eval_kwargs["name_suffix"] = args.name_suffix

                    run_eval(
                        eval_script=eval_script,
                        model_type=model_type,
                        metadata=metadata,
                        size=args.size,
                        base_url=base_url,
                        dataset=args.dataset,
                        split=args.split,
                        batch_size=args.batch_size,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        output_jsonl=output_jsonl,
                        url_corruption_rate=corruption_rate,
                        url_corruption_seed=args.url_corruption_seed,
                        **run_eval_kwargs,
                    )

                metrics = compute_metrics(output_jsonl)
                metrics["url_corruption_rate"] = corruption_rate
                append_csv(
                    summary_csv,
                    model_type=model_type,
                    metadata=metadata,
                    base_url=base_url,
                    metrics=metrics,
                )
                pbar.update(1)

    print(f"[✔] Wrote summary CSV to {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
