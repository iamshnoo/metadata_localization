#!/usr/bin/env python3

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".agents/.env"
QUALITY_AUDIT_DIR = ROOT / "qa_data/localnewsqa_core/runs/quality_audit"
SEMANTIC_SCORES = QUALITY_AUDIT_DIR / "semantic_quality_full_35874_scores.csv"
WEAK_LOCALE_FLAGS = QUALITY_AUDIT_DIR / "weak_locale_ambiguous_flags.csv"
OUTDIR = QUALITY_AUDIT_DIR / "llm_core_verification"
DEFAULT_MODEL = "gpt-5.5"

ALLOWED_ISSUE_FAMILIES = {"evidence_display", "evidence_support"}


SYSTEM_PROMPT = """You are a strict benchmark-quality auditor for LocalNewsQA.

You must decide whether a multiple-choice row is safe for the core benchmark. Be conservative: reject any row with a plausible factual error, ambiguous wording, answer leakage, weak locale dependence, artificial contrast answer, non-unique answer, outdated/time-sensitive answer not anchored by year, or insufficient support.

Definitions:
- explicit rows: the question should be fully determined by the question text. Locale metadata must not be needed to answer it.
- ambiguous rows: the same question should have a genuine, natural, locale-dependent answer for both target_country and contrast_country. The question must not be anchored to a target-only event, person, place, institution, slogan, title, or wording that makes the contrast answer artificial.

For every row, check:
1. exactly one correct target answer among the options;
2. options are distinct and plausible;
3. target answer is factually correct for the target country and stated year/context;
4. wording is not misleading, overly broad, stale, or under-specified;
5. evidence fields support the claim enough for benchmark use, or the fact is very widely known;
6. no answer/country leakage that makes the item trivial or invalid.

For ambiguous rows, additionally check:
7. contrast answer is factually correct for the contrast country;
8. the contrast side is not merely a forced analogy;
9. target and contrast answers genuinely differ because of locale, not because the question describes a target-specific event.

Return only compact JSON with this exact schema:
{
  "verdict": "accept" | "reject",
  "confidence": "high" | "medium" | "low",
  "failure_modes": ["..."],
  "reason": "one concise sentence"
}

Use failure_modes from this controlled vocabulary when applicable:
factual_error, weak_locale_difference, target_specific_anchor, contrast_not_applicable, ambiguous_wording, multiple_correct_answers, answer_not_in_options, answer_leakage, country_leakage, insufficient_evidence, temporal_ambiguity, stale_or_changed_fact, poor_options, other.
"""


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def split_pipe(raw: str) -> set[str]:
    return {part.strip() for part in str(raw or "").split("|") if part.strip()}


def option_parts(options: str) -> list[str]:
    return [part.strip().split(":", 1)[-1].strip() for part in str(options or "").split("||") if part.strip()]


def load_rows() -> tuple[list[dict], dict[str, dict]]:
    with SEMANTIC_SCORES.open(newline="", encoding="utf-8") as handle:
        semantic_rows = list(csv.DictReader(handle))
    with WEAK_LOCALE_FLAGS.open(newline="", encoding="utf-8") as handle:
        weak_by_id = {row["id"]: row for row in csv.DictReader(handle)}
    return semantic_rows, weak_by_id


def is_core_candidate(row: dict, weak_by_id: dict[str, dict], strict_ambiguous: bool = False) -> bool:
    if row.get("manual_review_status") == "manual_reject":
        return False
    if row.get("has_lower_bound_semantic_issue") == "yes":
        return False
    if row.get("semantic_risk_band") != "clean":
        return False
    if split_pipe(row.get("issue_families")) - ALLOWED_ISSUE_FAMILIES:
        return False
    if row.get("target_source_certified") != "yes":
        return False
    if row.get("split") == "ambiguous":
        weak = weak_by_id.get(row["id"], {})
        allowed_risks = {"clean"} if strict_ambiguous else {"clean", "low"}
        if weak.get("weak_locale_risk") not in allowed_risks:
            return False
        if row.get("contrast_source_certified") != "yes":
            return False
    return True


def compact_row_for_prompt(row: dict, weak_by_id: dict[str, dict]) -> dict:
    weak = weak_by_id.get(row["id"], {})
    return {
        "id": row["id"],
        "split": row["split"],
        "country": row["country"],
        "contrast_country": row.get("contrast_country") or "",
        "topic": row["topic"],
        "year": row["year"],
        "question": row["question"],
        "options": option_parts(row["options"]),
        "target_answer": row.get("target_answer") or "",
        "contrast_answer": row.get("contrast_answer") or "",
        "evidence_hint": row.get("evidence_hint") or "",
        "target_evidence_title": row.get("target_evidence_title") or "",
        "target_evidence_snippet": row.get("target_evidence_snippet") or "",
        "target_evidence_excerpt": row.get("target_evidence_excerpt") or "",
        "contrast_evidence_title": row.get("contrast_evidence_title") or "",
        "contrast_evidence_snippet": row.get("contrast_evidence_snippet") or "",
        "contrast_evidence_excerpt": row.get("contrast_evidence_excerpt") or "",
        "pre_filter_weak_locale_risk": weak.get("weak_locale_risk") or "",
        "pre_filter_weak_locale_flags": weak.get("weak_locale_flags") or "",
    }


def build_messages(row: dict, weak_by_id: dict[str, dict]) -> list[dict]:
    payload = compact_row_for_prompt(row, weak_by_id)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Audit this LocalNewsQA row for inclusion in the strict core benchmark:\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_core_candidate_files(rows: list[dict], weak_by_id: dict[str, dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "split",
        "country",
        "contrast_country",
        "topic",
        "year",
        "question",
        "options",
        "target_answer",
        "contrast_answer",
        "target_evidence_title",
        "target_evidence_url",
        "contrast_evidence_title",
        "contrast_evidence_url",
        "semantic_risk_band",
        "issue_families",
        "issues",
        "weak_locale_risk",
        "weak_locale_score",
        "weak_locale_flags",
    ]
    enriched = []
    for row in rows:
        out = dict(row)
        weak = weak_by_id.get(row["id"], {})
        out["weak_locale_risk"] = weak.get("weak_locale_risk", "")
        out["weak_locale_score"] = weak.get("weak_locale_score", "")
        out["weak_locale_flags"] = weak.get("weak_locale_flags", "")
        enriched.append(out)
    write_csv(outdir / "core_candidate_rows.csv", enriched, fields)
    with (outdir / "core_candidate_ids.txt").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row["id"] + "\n")


def custom_id(row: dict) -> str:
    return "coreq_" + row["id"].replace("localnewsqa_", "")


def write_batch_requests(
    rows: list[dict],
    weak_by_id: dict[str, dict],
    outdir: Path,
    model: str,
    max_completion_tokens: int,
    request_stem: str = "core_verification_requests",
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{request_stem}_{model.replace('/', '_')}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            request = {
                "custom_id": custom_id(row),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "max_completion_tokens": max_completion_tokens,
                    "response_format": {"type": "json_object"},
                    "messages": build_messages(row, weak_by_id),
                },
            }
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
    return path


def parse_json_object(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def run_smoke(rows: list[dict], weak_by_id: dict[str, dict], model: str, n: int, outdir: Path) -> None:
    client = OpenAI()
    results = []
    for row in rows[:n]:
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=1024,
            response_format={"type": "json_object"},
            messages=build_messages(row, weak_by_id),
        )
        content = response.choices[0].message.content or "{}"
        verdict = parse_json_object(content)
        results.append(
            {
                "id": row["id"],
                "split": row["split"],
                "country": row["country"],
                "topic": row["topic"],
                "question": row["question"],
                "target_answer": row.get("target_answer", ""),
                "contrast_answer": row.get("contrast_answer", ""),
                "llm_verdict": verdict.get("verdict", ""),
                "llm_confidence": verdict.get("confidence", ""),
                "llm_failure_modes": " | ".join(verdict.get("failure_modes") or []),
                "llm_reason": verdict.get("reason", ""),
                "raw_response": content,
            }
        )
    fields = [
        "id",
        "split",
        "country",
        "topic",
        "question",
        "target_answer",
        "contrast_answer",
        "llm_verdict",
        "llm_confidence",
        "llm_failure_modes",
        "llm_reason",
        "raw_response",
    ]
    write_csv(outdir / "smoke_verdicts.csv", results, fields)
    print(
        json.dumps(
            {"smoke_n": len(results), "counts": dict(Counter(r["llm_verdict"] for r in results))},
            indent=2,
            sort_keys=True,
        )
    )


def submit_batch(request_path: Path, outdir: Path, metadata_name: str = "batch_metadata.json") -> None:
    client = OpenAI()
    uploaded = client.files.create(file=request_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    metadata = {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "status": batch.status,
        "request_path": str(request_path),
        "created_at": int(time.time()),
    }
    with (outdir / metadata_name).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))


def load_batch_id(outdir: Path, explicit_id: str | None = None, metadata_name: str = "batch_metadata.json") -> str:
    if explicit_id:
        return explicit_id
    metadata_path = outdir / metadata_name
    if not metadata_path.exists():
        raise SystemExit(f"No batch metadata found at {metadata_path}")
    return json.loads(metadata_path.read_text())["batch_id"]


def poll_batch(outdir: Path, batch_id: str | None = None, metadata_name: str = "batch_metadata.json") -> None:
    client = OpenAI()
    batch = client.batches.retrieve(load_batch_id(outdir, batch_id, metadata_name))
    print(batch.model_dump_json(indent=2))


def download_batch(
    outdir: Path,
    batch_id: str | None = None,
    metadata_name: str = "batch_metadata.json",
    output_name: str = "batch_output.jsonl",
    error_name: str = "batch_errors.jsonl",
) -> None:
    client = OpenAI()
    batch = client.batches.retrieve(load_batch_id(outdir, batch_id, metadata_name))
    if batch.status != "completed":
        raise SystemExit(f"Batch is {batch.status}, not completed")
    if not batch.output_file_id:
        raise SystemExit("Completed batch has no output_file_id")
    content = client.files.content(batch.output_file_id).read()
    out_path = outdir / output_name
    out_path.write_bytes(content)
    if batch.error_file_id:
        err_content = client.files.content(batch.error_file_id).read()
        (outdir / error_name).write_bytes(err_content)
    print(str(out_path))


def verdict_from_batch_line(row: dict) -> tuple[str, dict, str]:
    body = row.get("response", {}).get("body", {})
    choices = body.get("choices") or []
    if not choices:
        return "error", {}, json.dumps(body)
    content = choices[0].get("message", {}).get("content") or ""
    try:
        verdict = parse_json_object(content)
        return "ok", verdict, content
    except Exception:
        return "parse_error", {}, content


def parse_output_file(
    output_path: Path,
    rows: list[dict],
    weak_by_id: dict[str, dict],
    error_path: Path | None = None,
) -> list[dict]:
    by_custom_id = {custom_id(row): row for row in rows}
    parsed = []
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                row = by_custom_id.get(raw.get("custom_id"))
                if row is None:
                    continue
                status, verdict, content = verdict_from_batch_line(raw)
                finish_reason = ""
                body = raw.get("response", {}).get("body", {})
                choices = body.get("choices") or []
                if choices:
                    finish_reason = choices[0].get("finish_reason") or ""
                parsed.append(
                    {
                        "id": row["id"],
                        "split": row["split"],
                        "country": row["country"],
                        "contrast_country": row.get("contrast_country", ""),
                        "topic": row["topic"],
                        "year": row["year"],
                        "question": row["question"],
                        "target_answer": row.get("target_answer", ""),
                        "contrast_answer": row.get("contrast_answer", ""),
                        "pre_filter_weak_locale_risk": weak_by_id.get(row["id"], {}).get("weak_locale_risk", ""),
                        "llm_status": status,
                        "llm_verdict": verdict.get("verdict", "") if verdict else "",
                        "llm_confidence": verdict.get("confidence", "") if verdict else "",
                        "llm_failure_modes": " | ".join(verdict.get("failure_modes") or []) if verdict else "",
                        "llm_reason": verdict.get("reason", "") if verdict else "",
                        "finish_reason": finish_reason,
                        "raw_response": content,
                    }
                )
    if error_path and error_path.exists():
        with error_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                row = by_custom_id.get(raw.get("custom_id"))
                if row is None:
                    continue
                body = raw.get("response", {}).get("body", {})
                error = body.get("error") or {}
                parsed.append(
                    {
                        "id": row["id"],
                        "split": row["split"],
                        "country": row["country"],
                        "contrast_country": row.get("contrast_country", ""),
                        "topic": row["topic"],
                        "year": row["year"],
                        "question": row["question"],
                        "target_answer": row.get("target_answer", ""),
                        "contrast_answer": row.get("contrast_answer", ""),
                        "pre_filter_weak_locale_risk": weak_by_id.get(row["id"], {}).get("weak_locale_risk", ""),
                        "llm_status": "error",
                        "llm_verdict": "",
                        "llm_confidence": "",
                        "llm_failure_modes": "",
                        "llm_reason": str(error.get("message") or ""),
                        "finish_reason": "",
                        "raw_response": json.dumps(body, ensure_ascii=False),
                    }
                )
    return parsed


def parse_outputs(
    outdir: Path,
    output_name: str = "batch_output.jsonl",
    error_name: str = "batch_errors.jsonl",
    verdict_name: str = "llm_core_verdicts.csv",
    summary_name: str = "llm_core_verification_summary.json",
) -> None:
    rows, weak_by_id = load_rows()
    output_path = outdir / output_name
    error_path = outdir / error_name
    if not output_path.exists():
        raise SystemExit(f"Missing {output_path}")
    parsed = parse_output_file(output_path, rows, weak_by_id, error_path)

    fields = [
        "id",
        "split",
        "country",
        "contrast_country",
        "topic",
        "year",
        "question",
        "target_answer",
        "contrast_answer",
        "pre_filter_weak_locale_risk",
        "llm_status",
        "llm_verdict",
        "llm_confidence",
        "llm_failure_modes",
        "llm_reason",
        "finish_reason",
        "raw_response",
    ]
    write_csv(outdir / verdict_name, parsed, fields)

    summary = {
        "total_outputs": len(parsed),
        "status_counts": dict(Counter(row["llm_status"] for row in parsed)),
        "verdict_counts": dict(Counter(row["llm_verdict"] for row in parsed)),
        "finish_reason_counts": dict(Counter(row["finish_reason"] for row in parsed)),
        "by_split_verdict": {
            split: dict(Counter(row["llm_verdict"] for row in group))
            for split, group in groupby_list(parsed, "split").items()
        },
        "by_topic_verdict": {
            topic: dict(Counter(row["llm_verdict"] for row in group))
            for topic, group in groupby_list(parsed, "topic").items()
        },
        "failure_mode_counts": dict(count_failure_modes(parsed).most_common()),
    }
    with (outdir / summary_name).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


def load_valid_verdict_ids(path: Path) -> set[str]:
    valid = set()
    if not path.exists():
        return valid
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("llm_status") == "ok" and row.get("llm_verdict") in {"accept", "reject"}:
                valid.add(row["id"])
    return valid


def merge_verdict_files(outdir: Path) -> None:
    paths = [
        outdir / "llm_core_verdicts_initial.csv",
        outdir / "llm_core_verdicts_retry.csv",
        outdir / "llm_core_verdicts_extra.csv",
    ]
    if not paths[0].exists():
        paths[0] = outdir / "llm_core_verdicts.csv"
    merged_by_id = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing = merged_by_id.get(row["id"])
                row_valid = row.get("llm_status") == "ok" and row.get("llm_verdict") in {"accept", "reject"}
                existing_valid = existing and existing.get("llm_status") == "ok" and existing.get("llm_verdict") in {"accept", "reject"}
                if row_valid or not existing_valid:
                    merged_by_id[row["id"]] = row
    rows = list(merged_by_id.values())
    rows.sort(key=lambda r: r["id"])
    fields = [
        "id",
        "split",
        "country",
        "contrast_country",
        "topic",
        "year",
        "question",
        "target_answer",
        "contrast_answer",
        "pre_filter_weak_locale_risk",
        "llm_status",
        "llm_verdict",
        "llm_confidence",
        "llm_failure_modes",
        "llm_reason",
        "finish_reason",
        "raw_response",
    ]
    write_csv(outdir / "llm_core_verdicts.csv", rows, fields)
    summary = {
        "total_rows": len(rows),
        "status_counts": dict(Counter(row["llm_status"] for row in rows)),
        "verdict_counts": dict(Counter(row["llm_verdict"] for row in rows)),
        "finish_reason_counts": dict(Counter(row.get("finish_reason", "") for row in rows)),
        "by_split_verdict": {
            split: dict(Counter(row["llm_verdict"] for row in group))
            for split, group in groupby_list(rows, "split").items()
        },
        "by_topic_verdict": {
            topic: dict(Counter(row["llm_verdict"] for row in group))
            for topic, group in groupby_list(rows, "topic").items()
        },
        "failure_mode_counts": dict(count_failure_modes(rows).most_common()),
    }
    with (outdir / "llm_core_verification_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    export_final_verdict_slices(outdir, rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


def export_final_verdict_slices(outdir: Path, verdict_rows: list[dict]) -> None:
    with (outdir / "core_candidate_rows.csv").open(newline="", encoding="utf-8") as handle:
        candidate_by_id = {row["id"]: row for row in csv.DictReader(handle)}

    verdict_by_id = {row["id"]: row for row in verdict_rows}
    exported = {
        "llm_accept_core": [],
        "llm_reject_from_candidate": [],
        "llm_unverified_core": [],
    }
    for row_id, candidate in candidate_by_id.items():
        verdict = verdict_by_id.get(row_id, {})
        merged = dict(candidate)
        for key in [
            "llm_status",
            "llm_verdict",
            "llm_confidence",
            "llm_failure_modes",
            "llm_reason",
            "finish_reason",
        ]:
            merged[key] = verdict.get(key, "")
        if verdict.get("llm_status") == "ok" and verdict.get("llm_verdict") == "accept":
            exported["llm_accept_core"].append(merged)
        elif verdict.get("llm_status") == "ok" and verdict.get("llm_verdict") == "reject":
            exported["llm_reject_from_candidate"].append(merged)
        else:
            exported["llm_unverified_core"].append(merged)

    fields = list(next(iter(candidate_by_id.values())).keys()) + [
        "llm_status",
        "llm_verdict",
        "llm_confidence",
        "llm_failure_modes",
        "llm_reason",
        "finish_reason",
    ]
    for name, group in exported.items():
        write_csv(outdir / f"{name}_rows.csv", group, fields)
        with (outdir / f"{name}_ids.txt").open("w", encoding="utf-8") as handle:
            for row in group:
                handle.write(row["id"] + "\n")


def groupby_list(rows: Iterable[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[row.get(key, "")].append(row)
    return out


def count_failure_modes(rows: Iterable[dict]) -> Counter:
    counts = Counter()
    for row in rows:
        for mode in str(row.get("llm_failure_modes") or "").split("|"):
            mode = mode.strip()
            if mode:
                counts[mode] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify strict LocalNewsQA core candidates with an OpenAI LLM.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env", type=Path, default=ENV_PATH)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--strict-ambiguous", action="store_true", help="Require ambiguous weak-locale risk clean, not low.")
    parser.add_argument("--max-completion-tokens", type=int, default=320)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--retry-unverified", action="store_true")
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--parse", action="store_true")
    parser.add_argument("--merge-retries", action="store_true")
    parser.add_argument("--batch-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(args.env)
    rows, weak_by_id = load_rows()
    core_rows = [row for row in rows if is_core_candidate(row, weak_by_id, args.strict_ambiguous)]
    retry_mode = args.retry_unverified
    if retry_mode:
        valid_ids = load_valid_verdict_ids(args.outdir / "llm_core_verdicts_initial.csv")
        if not valid_ids:
            valid_ids = load_valid_verdict_ids(args.outdir / "llm_core_verdicts.csv")
        core_rows = [row for row in core_rows if row["id"] not in valid_ids]
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.prepare or args.submit or args.smoke:
        if not retry_mode:
            write_core_candidate_files(core_rows, weak_by_id, args.outdir)
        request_path = write_batch_requests(
            core_rows,
            weak_by_id,
            args.outdir,
            args.model,
            args.max_completion_tokens,
            "core_verification_retry_requests" if retry_mode else "core_verification_requests",
        )
        manifest = {
            "model": args.model,
            "strict_ambiguous": args.strict_ambiguous,
            "candidate_count": len(core_rows),
            "split_counts": dict(Counter(row["split"] for row in core_rows)),
            "request_path": str(request_path),
            "max_completion_tokens": args.max_completion_tokens,
            "retry_unverified": retry_mode,
        }
        manifest_name = "core_verification_retry_manifest.json" if retry_mode else "core_verification_manifest.json"
        with (args.outdir / manifest_name).open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        if args.smoke:
            run_smoke(core_rows, weak_by_id, args.model, args.smoke, args.outdir)
        if args.submit:
            submit_batch(
                request_path,
                args.outdir,
                "batch_retry_metadata.json" if retry_mode else "batch_metadata.json",
            )
    if args.poll:
        poll_batch(
            args.outdir,
            args.batch_id,
            "batch_retry_metadata.json" if retry_mode else "batch_metadata.json",
        )
    if args.download:
        download_batch(
            args.outdir,
            args.batch_id,
            "batch_retry_metadata.json" if retry_mode else "batch_metadata.json",
            "batch_retry_output.jsonl" if retry_mode else "batch_output.jsonl",
            "batch_retry_errors.jsonl" if retry_mode else "batch_errors.jsonl",
        )
    if args.parse:
        parse_outputs(
            args.outdir,
            "batch_retry_output.jsonl" if retry_mode else "batch_output.jsonl",
            "batch_retry_errors.jsonl" if retry_mode else "batch_errors.jsonl",
            "llm_core_verdicts_retry.csv" if retry_mode else "llm_core_verdicts_initial.csv",
            "llm_core_verification_retry_summary.json" if retry_mode else "llm_core_verification_initial_summary.json",
        )
    if args.merge_retries:
        merge_verdict_files(args.outdir)


if __name__ == "__main__":
    main()
