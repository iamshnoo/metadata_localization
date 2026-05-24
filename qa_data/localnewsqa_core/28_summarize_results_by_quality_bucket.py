#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
QUALITY_AUDIT_DIR = ROOT / "qa_data/localnewsqa_core/runs/quality_audit"
SEMANTIC_SCORES = QUALITY_AUDIT_DIR / "semantic_quality_full_35874_scores.csv"
WEAK_LOCALE_FLAGS = QUALITY_AUDIT_DIR / "weak_locale_ambiguous_flags.csv"
LLM_VERDICTS = QUALITY_AUDIT_DIR / "llm_core_verification/llm_core_verdicts.csv"
RESTORED_AMBIGUOUS = QUALITY_AUDIT_DIR / "llm_ambiguous_reject_adjudication/ambiguous_restorable_ids.txt"
GOLD_VERIFIED_EXISTING = QUALITY_AUDIT_DIR / "gold_ambiguous_1700/gold_verified_existing_seed_ids.txt"
DEFAULT_OUTDIR = QUALITY_AUDIT_DIR / "quality_bucket_results"

PRETRAINED_ROOT = ROOT / "results/downstream_localnewsqa_pretrained_multiseed"
SFT_ROOT = ROOT / "results/downstream_localnewsqa_sft_figure9_full_multiseed"

ALLOWED_ISSUE_FAMILIES = {"evidence_display", "evidence_support"}


def split_pipe(raw: str) -> set[str]:
    return {part.strip() for part in str(raw or "").split("|") if part.strip()}


def dataset_key(row: dict) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("question") or ""),
        str(row.get("country") or ""),
        str(row.get("topic") or ""),
        str(row.get("year") or ""),
        str(row.get("split") or row.get("split_type") or "").lower(),
    )


def load_id_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_dataset_quality() -> dict[tuple[str, str, str, str, str], dict]:
    with SEMANTIC_SCORES.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with WEAK_LOCALE_FLAGS.open(newline="", encoding="utf-8") as handle:
        weak_by_id = {row["id"]: row for row in csv.DictReader(handle)}

    llm_accept_ids = set()
    llm_reject_ids = set()
    restored_ambiguous_ids = load_id_file(RESTORED_AMBIGUOUS)
    gold_verified_existing_ids = load_id_file(GOLD_VERIFIED_EXISTING)
    if LLM_VERDICTS.exists():
        with LLM_VERDICTS.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("llm_verdict") == "accept":
                    llm_accept_ids.add(row["id"])
                elif row.get("llm_verdict") == "reject":
                    llm_reject_ids.add(row["id"])

    out = {}
    for row in rows:
        weak = weak_by_id.get(row["id"], {})
        fams = split_pipe(row.get("issue_families"))
        strict_core_candidate = (
            row.get("manual_review_status") != "manual_reject"
            and row.get("has_lower_bound_semantic_issue") != "yes"
            and row.get("semantic_risk_band") == "clean"
            and not (fams - ALLOWED_ISSUE_FAMILIES)
            and row.get("target_source_certified") == "yes"
            and (
                row.get("split") != "ambiguous"
                or (
                    row.get("contrast_source_certified") == "yes"
                    and weak.get("weak_locale_risk") in {"clean", "low"}
                )
            )
        )
        strictest_core_candidate = (
            strict_core_candidate
            and (
                row.get("split") != "ambiguous"
                or weak.get("weak_locale_risk") == "clean"
            )
        )
        literal_no_issues = not str(row.get("issues") or "").strip()

        buckets = {"all"}
        if row.get("split") == "explicit":
            buckets.add("explicit_all")
        else:
            buckets.add("ambiguous_all")
            risk = weak.get("weak_locale_risk", "unknown")
            buckets.add(f"ambiguous_weak_{risk}")
            if risk in {"high", "medium"}:
                buckets.add("ambiguous_weak_high_medium")
            if risk in {"clean", "low"}:
                buckets.add("ambiguous_weak_clean_low")

        if strict_core_candidate:
            buckets.add("core_candidate")
            buckets.add(f"core_candidate_{row['split']}")
        else:
            buckets.add("non_core")
        if strictest_core_candidate:
            buckets.add("core_strict_ambig_clean_only")
        if literal_no_issues:
            buckets.add("literal_no_issues")
        if row["id"] in llm_accept_ids:
            buckets.add("llm_accept_core")
            buckets.add(f"llm_accept_core_{row['split']}")
        if row["id"] in restored_ambiguous_ids:
            buckets.add("llm_restored_ambiguous")
        if row["id"] in llm_accept_ids or row["id"] in restored_ambiguous_ids:
            buckets.add("llm_accept_plus_restored_core")
            buckets.add(f"llm_accept_plus_restored_core_{row['split']}")
        if row["id"] in llm_reject_ids:
            buckets.add("llm_reject_from_candidate")
        if row["id"] in gold_verified_existing_ids:
            buckets.add("gold_verified_existing_ambiguous")

        out[dataset_key(row)] = {
            "id": row["id"],
            "split": row["split"],
            "weak_locale_risk": weak.get("weak_locale_risk", ""),
            "buckets": buckets,
        }
    return out


def mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def sample_std(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) <= 1:
        return 0.0
    avg = sum(vals) / len(vals)
    return math.sqrt(sum((value - avg) ** 2 for value in vals) / (len(vals) - 1))


def parse_run(path: Path, source: str) -> dict:
    text = path.name
    seed_match = re.search(r"seed[_-]?(\d+)", str(path))
    seed = int(seed_match.group(1)) if seed_match else None
    family = "3B" if re.search(r"(?:^|_)3b(?:_|$)", text) or "3b_" in text else "1B"
    variant_match = re.search(r"(tplus|tminus)_(eplus|eminus)", text)
    variant = "_".join(variant_match.groups()) if variant_match else "unknown"
    return {
        "source": source,
        "family": family,
        "variant": variant,
        "series": f"{family} {variant.replace('_', '/').upper()}",
        "seed": seed,
        "path": str(path),
    }


def discover_result_files(pretrained_root: Path, sft_root: Path) -> list[tuple[Path, dict]]:
    files = []
    for path in sorted(pretrained_root.glob("seed_*/localnewsqa_eval_target_*_custom_*_c0.jsonl")):
        files.append((path, parse_run(path, "pretrained")))
    for path in sorted(sft_root.glob("*/seed_*/localnewsqa_eval_target_*_custom_sftfig9_*_c0.jsonl")):
        files.append((path, parse_run(path, "sft")))
    return files


def summarize_file(path: Path, meta: dict, quality_by_key: dict, bucket_names: list[str]) -> list[dict]:
    counts = {bucket: {"correct": 0, "total": 0} for bucket in bucket_names}
    missing = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "is_correct" not in row:
                continue
            key = dataset_key(row)
            q = quality_by_key.get(key)
            if q is None:
                missing += 1
                continue
            for bucket in q["buckets"]:
                if bucket in counts:
                    counts[bucket]["total"] += 1
                    counts[bucket]["correct"] += int(bool(row["is_correct"]))
    out = []
    for bucket, stat in counts.items():
        if stat["total"] == 0:
            continue
        record = dict(meta)
        record.update(
            {
                "bucket": bucket,
                "correct": stat["correct"],
                "total": stat["total"],
                "accuracy": stat["correct"] / stat["total"],
                "missing_dataset_key_rows": missing,
            }
        )
        out.append(record)
    return out


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = (row["source"], row["family"], row["variant"], row["bucket"])
        grouped[key].append(row)
    out = []
    for (source, family, variant, bucket), group in grouped.items():
        accuracies = [row["accuracy"] for row in group]
        totals = Counter(row["total"] for row in group)
        out.append(
            {
                "source": source,
                "family": family,
                "variant": variant,
                "bucket": bucket,
                "accuracy": mean(accuracies),
                "accuracy_std": sample_std(accuracies),
                "seed_count": len(group),
                "total": totals.most_common(1)[0][0],
            }
        )
    out.sort(key=lambda r: (r["source"], r["family"], r["variant"], r["bucket"]))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def get_metric(rows: list[dict], source: str, family: str, variant: str, bucket: str) -> dict | None:
    for row in rows:
        if (
            row["source"] == source
            and row["family"] == family
            and row["variant"] == variant
            and row["bucket"] == bucket
        ):
            return row
    return None


def build_win_summary(agg_rows: list[dict], bucket: str) -> list[dict]:
    comparisons = [
        ("T+/I+ vs T-/I-", "tplus_eplus", "tminus_eminus"),
        ("T+/I+ vs T+/I-", "tplus_eplus", "tplus_eminus"),
        ("T+/I+ vs T-/I+", "tplus_eplus", "tminus_eplus"),
        ("T-/I+ vs T-/I-", "tminus_eplus", "tminus_eminus"),
    ]
    out = []
    for source in sorted({row["source"] for row in agg_rows}):
        for family in sorted({row["family"] for row in agg_rows if row["source"] == source}):
            for label, left, right in comparisons:
                lrow = get_metric(agg_rows, source, family, left, bucket)
                rrow = get_metric(agg_rows, source, family, right, bucket)
                if not lrow or not rrow:
                    continue
                out.append(
                    {
                        "bucket": bucket,
                        "source": source,
                        "family": family,
                        "comparison": label,
                        "left_accuracy": lrow["accuracy"],
                        "right_accuracy": rrow["accuracy"],
                        "delta": lrow["accuracy"] - rrow["accuracy"],
                        "holds": lrow["accuracy"] > rrow["accuracy"],
                        "seed_count_left": lrow["seed_count"],
                        "seed_count_right": rrow["seed_count"],
                        "total": lrow["total"],
                    }
                )
    for family in sorted({row["family"] for row in agg_rows}):
        for variant in sorted({row["variant"] for row in agg_rows if row["family"] == family}):
            sft = get_metric(agg_rows, "sft", family, variant, bucket)
            pre = get_metric(agg_rows, "pretrained", family, variant, bucket)
            if sft and pre:
                out.append(
                    {
                        "bucket": bucket,
                        "source": "sft_vs_pretrained",
                        "family": family,
                        "comparison": variant,
                        "left_accuracy": sft["accuracy"],
                        "right_accuracy": pre["accuracy"],
                        "delta": sft["accuracy"] - pre["accuracy"],
                        "holds": sft["accuracy"] > pre["accuracy"],
                        "seed_count_left": sft["seed_count"],
                        "seed_count_right": pre["seed_count"],
                        "total": sft["total"],
                    }
                )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-filter LocalNewsQA result JSONLs by quality buckets.")
    parser.add_argument("--pretrained-root", type=Path, default=PRETRAINED_ROOT)
    parser.add_argument("--sft-root", type=Path, default=SFT_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality_by_key = load_dataset_quality()
    bucket_names = [
        "all",
        "explicit_all",
        "ambiguous_all",
        "core_candidate",
        "core_candidate_explicit",
        "core_candidate_ambiguous",
        "core_strict_ambig_clean_only",
        "literal_no_issues",
        "ambiguous_weak_clean",
        "ambiguous_weak_low",
        "ambiguous_weak_medium",
        "ambiguous_weak_high",
        "ambiguous_weak_clean_low",
        "ambiguous_weak_high_medium",
        "non_core",
        "llm_accept_core",
        "llm_accept_core_explicit",
        "llm_accept_core_ambiguous",
        "llm_restored_ambiguous",
        "llm_accept_plus_restored_core",
        "llm_accept_plus_restored_core_explicit",
        "llm_accept_plus_restored_core_ambiguous",
        "llm_reject_from_candidate",
        "gold_verified_existing_ambiguous",
    ]
    per_seed = []
    files = discover_result_files(args.pretrained_root, args.sft_root)
    for path, meta in files:
        per_seed.extend(summarize_file(path, meta, quality_by_key, bucket_names))
    agg_rows = aggregate(per_seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "quality_bucket_results_per_seed.csv", per_seed)
    write_csv(args.outdir / "quality_bucket_results_aggregate.csv", agg_rows)

    win_rows = []
    for bucket in [
        "core_candidate",
        "core_strict_ambig_clean_only",
        "llm_accept_core",
        "llm_accept_core_explicit",
        "llm_accept_core_ambiguous",
        "llm_restored_ambiguous",
        "llm_accept_plus_restored_core",
        "llm_accept_plus_restored_core_explicit",
        "llm_accept_plus_restored_core_ambiguous",
        "llm_reject_from_candidate",
        "gold_verified_existing_ambiguous",
        "all",
        "ambiguous_weak_clean_low",
        "ambiguous_weak_high_medium",
        "ambiguous_weak_clean",
        "ambiguous_weak_low",
        "ambiguous_weak_medium",
        "ambiguous_weak_high",
    ]:
        win_rows.extend(build_win_summary(agg_rows, bucket))
    write_csv(args.outdir / "quality_bucket_win_summary.csv", win_rows)

    manifest = {
        "result_files": len(files),
        "per_seed_rows": len(per_seed),
        "aggregate_rows": len(agg_rows),
        "buckets": bucket_names,
        "llm_verdicts_present": LLM_VERDICTS.exists(),
        "restored_ambiguous_present": RESTORED_AMBIGUOUS.exists(),
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
