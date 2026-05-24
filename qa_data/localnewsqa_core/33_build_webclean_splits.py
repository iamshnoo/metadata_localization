#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_OUTDIR = ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515"
DEFAULT_POOL_AUDIT = BASE_OUTDIR / "web_audit_ambiguous_pool_4625_wikiexport/web_audit_rows.csv"
DEFAULT_OUTDIR = ROOT / "qa_data/localnewsqa_core/runs/quality_audit/no_api_splits_20260515_webclean"
BUILDER_PATH = ROOT / "qa_data/localnewsqa_core/31_build_no_api_splits.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_no_api_splits", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load {BUILDER_PATH}")
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pool_ordinal(row_id: str) -> int:
    match = re.search(r"(\d+)$", str(row_id or ""))
    return int(match.group(1)) if match else 10**9


def has_bad_warning(warnings: str) -> bool:
    warning_text = str(warnings or "")
    return any(
        marker in warning_text
        for marker in [
            "fetch_error",
            "ordinary_web_domain",
            "rate_limited_429",
        ]
    )


def boolish(value) -> bool:
    return str(value).strip().lower() == "true"


def audit_selection_key(pool_row: dict, audit_row: dict) -> tuple:
    warnings = str(audit_row.get("warnings", "") or "")
    warning_parts = [part.strip() for part in warnings.split(" | ") if part.strip()]
    return (
        0 if audit_row.get("web_audit_verdict") == "pass" else 1,
        0 if boolish(audit_row.get("target_country_marker_present")) else 1,
        0 if boolish(audit_row.get("contrast_country_marker_present")) else 1,
        len(warning_parts),
        pool_ordinal(pool_row.get("id")),
    )


def build_webclean(pool_audit_path: Path, outdir: Path) -> dict:
    builder = load_builder()
    source_rows = builder.read_csv(builder.SEMANTIC_SCORES)
    source_by_id = {row["id"]: row for row in source_rows}
    weak_by_id = {row["id"]: row for row in builder.read_csv(builder.WEAK_LOCALE_FLAGS)}
    llm_ids = builder.load_id_file(builder.LLM_ACCEPT_IDS) | builder.load_id_file(builder.LLM_RESTORED_IDS)

    pool_rows = [
        json.loads(line)
        for line in (BASE_OUTDIR / "localnewsqa_ambiguous_verifiable_pool_4625.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audit_by_source_id = {row["source_row_id"]: row for row in read_csv(pool_audit_path)}

    eligible_by_country: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    rejected_by_reason = Counter()
    for pool_row in pool_rows:
        source_id = pool_row["source_row_id"]
        audit_row = audit_by_source_id.get(source_id)
        if not audit_row:
            rejected_by_reason["missing_web_audit"] += 1
            continue
        if audit_row.get("web_audit_verdict") == "fail":
            rejected_by_reason["hard_web_audit_failure"] += 1
            continue
        if has_bad_warning(audit_row.get("warnings", "")):
            rejected_by_reason["bad_web_audit_warning"] += 1
            continue
        eligible_by_country[pool_row["country"]].append((pool_row, audit_row))

    selected_pairs: list[tuple[dict, dict]] = []
    deficits = {}
    for country in builder.COUNTRY_ORDER:
        rows = sorted(
            eligible_by_country.get(country, []),
            key=lambda item: audit_selection_key(item[0], item[1]),
        )
        if len(rows) < 100:
            deficits[country] = 100 - len(rows)
        selected_pairs.extend(rows[:100])
    if deficits:
        raise RuntimeError(f"Not enough web-clean candidates: {deficits}")

    selected_source_ids = {pool_row["source_row_id"] for pool_row, _ in selected_pairs}
    targetqa_source_rows = []
    for row in source_rows:
        if builder.explicit_original_ok(row):
            targetqa_source_rows.append((row, "explicit"))
        elif builder.salvage_target_ok(row) and row["id"] not in selected_source_ids:
            targetqa_source_rows.append((row, "ambiguous_salvaged_target"))

    targetqa_rows = [
        builder.make_targetqa_row(row, source_split_type)
        for row, source_split_type in sorted(
            targetqa_source_rows,
            key=lambda item: (
                builder.country_sort_key(item[0].get("country", "")),
                0 if item[1] == "explicit" else 1,
                item[0]["id"],
            ),
        )
    ]

    selected_pairs = sorted(
        selected_pairs,
        key=lambda item: (
            builder.country_sort_key(item[0].get("country", "")),
            audit_selection_key(item[0], item[1]),
        ),
    )
    ambiguous_rows = []
    for idx, (pool_row, audit_row) in enumerate(selected_pairs, start=1):
        source_row = source_by_id[pool_row["source_row_id"]]
        out = builder.make_ambiguous_row(source_row, weak_by_id, llm_ids, idx)
        out["split_name"] = "LocalNewsQA-Ambiguous-Verifiable-1700-WebClean"
        out["web_audit_verdict"] = audit_row.get("web_audit_verdict", "")
        out["web_audit_warnings"] = audit_row.get("warnings", "")
        out["web_audit_pool_id"] = pool_row.get("id", "")
        out["target_country_marker_present"] = boolish(audit_row.get("target_country_marker_present"))
        out["contrast_country_marker_present"] = boolish(audit_row.get("contrast_country_marker_present"))
        ambiguous_rows.append(out)

    validation = builder.validate_rows(targetqa_rows, ambiguous_rows, ambiguous_rows)
    if not validation["ok"]:
        raise RuntimeError(json.dumps(validation, indent=2))

    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "targetqa_jsonl": outdir / "localnewsqa_targetqa_explicit_style.jsonl",
        "targetqa_csv": outdir / "localnewsqa_targetqa_explicit_style.csv",
        "ambiguous_jsonl": outdir / "localnewsqa_ambiguous_verifiable_1700_webclean.jsonl",
        "ambiguous_csv": outdir / "localnewsqa_ambiguous_verifiable_1700_webclean.csv",
        "summary": outdir / "summary.json",
    }
    builder.write_jsonl(paths["targetqa_jsonl"], targetqa_rows)
    builder.write_csv(paths["targetqa_csv"], targetqa_rows)
    builder.write_jsonl(paths["ambiguous_jsonl"], ambiguous_rows)
    builder.write_csv(paths["ambiguous_csv"], ambiguous_rows)

    summary = {
        "filters": {
            "webclean_ambiguous": "Pool rows with no hard web-audit failure and no fetch_error/ordinary_web_domain/rate_limited warning; selected 100 per country prioritizing pass rows, then rows with target/contrast country markers.",
            "targetqa": "Rebuilt from original explicit plus salvaged ambiguous target-side rows, excluding the final webclean ambiguous source IDs.",
        },
        "pool_audit_path": str(pool_audit_path),
        "rejected_by_reason": dict(rejected_by_reason),
        "targetqa": builder.summarize(targetqa_rows),
        "ambiguous_1700_webclean": {
            **builder.summarize(ambiguous_rows),
            "web_audit_verdict": dict(Counter(row["web_audit_verdict"] for row in ambiguous_rows)),
            "web_audit_warning_rows": sum(1 for row in ambiguous_rows if row.get("web_audit_warnings")),
            "web_audit_warning_counts": dict(
                Counter(
                    part
                    for row in ambiguous_rows
                    for part in str(row.get("web_audit_warnings", "")).split(" | ")
                    if part
                )
            ),
            "target_country_marker_present": dict(Counter(str(row["target_country_marker_present"]) for row in ambiguous_rows)),
            "contrast_country_marker_present": dict(Counter(str(row["contrast_country_marker_present"]) for row in ambiguous_rows)),
            "weak_locale_risk": dict(Counter(row.get("weak_locale_risk", "") for row in ambiguous_rows)),
            "llm_accept_or_restored": dict(Counter(str(row.get("llm_accept_or_restored")) for row in ambiguous_rows)),
        },
        "validation": validation,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build web-clean no-API LocalNewsQA splits.")
    parser.add_argument("--pool-audit", type=Path, default=DEFAULT_POOL_AUDIT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    print(json.dumps(build_webclean(args.pool_audit, args.outdir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
