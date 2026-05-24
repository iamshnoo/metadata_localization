#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GOLD_ROOT = (
    ROOT
    / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515/"
    / "web_semantic_gold_ambiguous_1700"
)
AMBIGUOUS_PATH = GOLD_ROOT / "localnewsqa_ambiguous_semantic_gold_1700.jsonl"
EXPLICIT_PATH = (
    GOLD_ROOT
    / "explicit_max_audit/strict_defensible_1000_curated_final/"
    / "localnewsqa_targetqa_explicit_strict_defensible_1000_per_country_curated_final.jsonl"
)
OUT_DIR = ROOT / "qa_data/localnewsqa_core/final_gold_20260516"
OUT_JSONL = OUT_DIR / "localnewsqa_gold_explicit17000_ambiguous1700.jsonl"
OUT_SUMMARY = OUT_DIR / "localnewsqa_gold_explicit17000_ambiguous1700_summary.json"


EXPECTED_COUNTRIES = {
    "Bangladesh",
    "Canada",
    "Ghana",
    "Hong Kong",
    "India",
    "Ireland",
    "Jamaica",
    "Kenya",
    "Malaysia",
    "Nigeria",
    "Pakistan",
    "Philippines",
    "South Africa",
    "Sri Lanka",
    "Tanzania",
    "United Kingdom",
    "United States",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_question(text: object) -> str:
    return " ".join(str(text or "").strip().lower().split())


def validate_rows(rows: list[dict], split_type: str, expected_per_country: int) -> dict:
    errors = []
    countries = Counter()
    ids = Counter()
    questions = Counter()
    for row in rows:
        split = str(row.get("split_type") or "").strip().lower()
        if split != split_type:
            errors.append(f"bad split_type for {row.get('id')}: {split!r}")
        country = str(row.get("country") or row.get("target_country") or "").strip()
        countries[country] += 1
        ids[str(row.get("id") or "")] += 1
        questions[normalize_question(row.get("question"))] += 1
        options = row.get("options")
        answer = row.get("target_answer") or row.get("correct_answer")
        if not isinstance(options, list) or len(options) != 4:
            errors.append(f"bad options for {row.get('id')}")
        elif answer not in options:
            errors.append(f"answer not in options for {row.get('id')}")
        if split_type == "ambiguous":
            contrast_answer = row.get("contrast_answer")
            if contrast_answer not in options:
                errors.append(f"contrast answer not in options for {row.get('id')}")
            if answer == contrast_answer:
                errors.append(f"target/contrast answer identical for {row.get('id')}")
            if not row.get("contrast_country"):
                errors.append(f"missing contrast country for {row.get('id')}")

    if set(countries) != EXPECTED_COUNTRIES:
        errors.append(f"country set mismatch: {sorted(countries)}")
    for country in sorted(EXPECTED_COUNTRIES):
        if countries[country] != expected_per_country:
            errors.append(f"{split_type} count for {country}: {countries[country]}")
    duplicate_ids = [key for key, count in ids.items() if count > 1]
    duplicate_questions = [key for key, count in questions.items() if key and count > 1]
    if duplicate_ids:
        errors.append(f"duplicate ids: {duplicate_ids[:5]}")
    if duplicate_questions:
        errors.append(f"duplicate normalized questions: {duplicate_questions[:5]}")
    return {
        "rows": len(rows),
        "country_counts": dict(sorted(countries.items())),
        "duplicate_ids": len(duplicate_ids),
        "duplicate_normalized_questions": len(duplicate_questions),
        "errors": errors,
    }


def default_for_key(key: str, rows: list[dict]):
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return False
        if isinstance(value, int) and not isinstance(value, bool):
            return 0
        if isinstance(value, float):
            return 0.0
        if isinstance(value, list):
            return []
        return ""
    return ""


def normalize_union_schema(rows: list[dict]) -> list[dict]:
    keys = sorted({key for row in rows for key in row.keys()})
    defaults = {key: default_for_key(key, rows) for key in keys}
    normalized = []
    for row in rows:
        out = {}
        for key in keys:
            value = row.get(key, defaults[key])
            out[key] = defaults[key] if value is None else value
        normalized.append(out)
    return normalized


def main() -> int:
    explicit_rows = read_jsonl(EXPLICIT_PATH)
    ambiguous_rows = read_jsonl(AMBIGUOUS_PATH)

    explicit_summary = validate_rows(explicit_rows, "explicit", 1000)
    ambiguous_summary = validate_rows(ambiguous_rows, "ambiguous", 100)
    all_errors = explicit_summary["errors"] + ambiguous_summary["errors"]
    if all_errors:
        raise SystemExit("\n".join(all_errors[:50]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = normalize_union_schema(explicit_rows + ambiguous_rows)
    with OUT_JSONL.open("w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "valid": True,
        "output_jsonl": str(OUT_JSONL),
        "source_explicit": str(EXPLICIT_PATH),
        "source_ambiguous": str(AMBIGUOUS_PATH),
        "rows": len(combined),
        "split_counts": dict(Counter(row["split_type"] for row in combined)),
        "explicit": explicit_summary,
        "ambiguous": ambiguous_summary,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
