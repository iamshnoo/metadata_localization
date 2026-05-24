#!/usr/bin/env python3

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".agents/.env"
QUALITY_AUDIT_DIR = ROOT / "qa_data/localnewsqa_core/runs/quality_audit"
SEMANTIC_SCORES = QUALITY_AUDIT_DIR / "semantic_quality_full_35874_scores.csv"
WEAK_LOCALE_FLAGS = QUALITY_AUDIT_DIR / "weak_locale_ambiguous_flags.csv"
LLM_DIR = QUALITY_AUDIT_DIR / "llm_core_verification"
REJECTED_ROWS = LLM_DIR / "llm_reject_from_candidate_rows.csv"
OUTDIR = QUALITY_AUDIT_DIR / "llm_ambiguous_reject_adjudication"
DEFAULT_MODEL = "gpt-5.5"


SYSTEM_PROMPT = """You are adjudicating rejected LocalNewsQA ambiguous rows.

Goal: decide whether a previously rejected ambiguous row can be safely restored to a high-quality ambiguous benchmark slice.

An ambiguous row is restorable only if ALL are true:
- the question is naturally answerable for both target_country and contrast_country;
- target_answer is factually correct for target_country;
- contrast_answer is factually correct for contrast_country;
- the same wording is fair for both countries, not anchored to a target-only person, event, law, school exam, broadcaster, place, slogan, or institution;
- the locale dependence is the reason the answer differs;
- there is exactly one correct target answer among the options;
- there is no country/answer leakage, stale fact, or under-specified wording that would make the row unreliable.

Important calibration:
- Do not restore a row just because the target answer looks correct. The contrast side must also be a natural, factual answer to the same question.
- Do not reject solely because evidence snippets are short if the claim is otherwise clearly factual and stable.
- If there is real doubt, choose reject. The restored set is for the main benchmark, not a broad appendix.

Return only compact JSON with this exact schema:
{
  "decision": "restore" | "reject",
  "confidence": "high" | "medium" | "low",
  "failure_modes": ["..."],
  "reason": "one concise sentence"
}

Use failure_modes from this controlled vocabulary when applicable:
target_specific_anchor, contrast_not_applicable, weak_locale_difference, ambiguous_wording, factual_error, multiple_correct_answers, answer_not_in_options, answer_leakage, country_leakage, temporal_ambiguity, stale_or_changed_fact, poor_options, insufficient_evidence, other.
"""


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def option_parts(options: str) -> list[str]:
    return [part.strip().split(":", 1)[-1].strip() for part in str(options or "").split("||") if part.strip()]


def load_source_rows() -> tuple[dict[str, dict], dict[str, dict]]:
    with SEMANTIC_SCORES.open(newline="", encoding="utf-8") as handle:
        source_by_id = {row["id"]: row for row in csv.DictReader(handle)}
    with WEAK_LOCALE_FLAGS.open(newline="", encoding="utf-8") as handle:
        weak_by_id = {row["id"]: row for row in csv.DictReader(handle)}
    return source_by_id, weak_by_id


def load_rejected_ambiguous_rows() -> list[dict]:
    with REJECTED_ROWS.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("split") == "ambiguous"]


def prompt_payload(row: dict, source: dict, weak: dict) -> dict:
    return {
        "id": row["id"],
        "target_country": source.get("country", row.get("country", "")),
        "contrast_country": source.get("contrast_country", row.get("contrast_country", "")),
        "topic": source.get("topic", row.get("topic", "")),
        "year": source.get("year", row.get("year", "")),
        "question": source.get("question", row.get("question", "")),
        "options": option_parts(source.get("options", "")),
        "target_answer": source.get("target_answer", row.get("target_answer", "")),
        "contrast_answer": source.get("contrast_answer", row.get("contrast_answer", "")),
        "target_evidence_title": source.get("target_evidence_title", ""),
        "target_evidence_snippet": source.get("target_evidence_snippet", ""),
        "target_evidence_excerpt": source.get("target_evidence_excerpt", ""),
        "contrast_evidence_title": source.get("contrast_evidence_title", ""),
        "contrast_evidence_snippet": source.get("contrast_evidence_snippet", ""),
        "contrast_evidence_excerpt": source.get("contrast_evidence_excerpt", ""),
        "previous_failure_modes": row.get("llm_failure_modes", ""),
        "previous_reason": row.get("llm_reason", ""),
        "pre_filter_weak_locale_risk": weak.get("weak_locale_risk", ""),
        "pre_filter_weak_locale_flags": weak.get("weak_locale_flags", ""),
    }


def build_messages(row: dict, source_by_id: dict[str, dict], weak_by_id: dict[str, dict]) -> list[dict]:
    source = source_by_id[row["id"]]
    weak = weak_by_id.get(row["id"], {})
    payload = prompt_payload(row, source, weak)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Adjudicate whether this rejected ambiguous row is safe to restore:\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def custom_id(row: dict) -> str:
    return "ambig_reject_" + row["id"].replace("localnewsqa_", "")


def parse_json_object(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_requests(rows: list[dict], source_by_id: dict[str, dict], weak_by_id: dict[str, dict], model: str, max_completion_tokens: int, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"ambiguous_reject_adjudication_requests_{model.replace('/', '_')}.jsonl"
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
                    "messages": build_messages(row, source_by_id, weak_by_id),
                },
            }
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
    return path


def run_smoke(rows: list[dict], source_by_id: dict[str, dict], weak_by_id: dict[str, dict], model: str, n: int, max_completion_tokens: int, outdir: Path) -> None:
    client = OpenAI()
    out = []
    for row in rows[:n]:
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=max_completion_tokens,
            response_format={"type": "json_object"},
            messages=build_messages(row, source_by_id, weak_by_id),
        )
        content = response.choices[0].message.content or "{}"
        verdict = parse_json_object(content)
        out.append(
            {
                "id": row["id"],
                "decision": verdict.get("decision", ""),
                "confidence": verdict.get("confidence", ""),
                "failure_modes": " | ".join(verdict.get("failure_modes") or []),
                "reason": verdict.get("reason", ""),
                "raw_response": content,
            }
        )
    write_csv(outdir / "smoke_adjudication.csv", out, ["id", "decision", "confidence", "failure_modes", "reason", "raw_response"])
    print(json.dumps({"smoke_n": len(out), "counts": dict(Counter(r["decision"] for r in out))}, indent=2, sort_keys=True))


def submit_batch(path: Path, outdir: Path) -> None:
    client = OpenAI()
    uploaded = client.files.create(file=path.open("rb"), purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h")
    metadata = {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "request_path": str(path),
        "status": batch.status,
        "created_at": int(time.time()),
    }
    (outdir / "batch_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


def load_batch_id(outdir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    return json.loads((outdir / "batch_metadata.json").read_text())["batch_id"]


def poll_batch(outdir: Path, batch_id: str | None) -> None:
    client = OpenAI()
    batch = client.batches.retrieve(load_batch_id(outdir, batch_id))
    print(batch.model_dump_json(indent=2))


def download_batch(outdir: Path, batch_id: str | None) -> None:
    client = OpenAI()
    batch = client.batches.retrieve(load_batch_id(outdir, batch_id))
    if batch.status != "completed":
        raise SystemExit(f"Batch is {batch.status}, not completed")
    if not batch.output_file_id:
        raise SystemExit("Completed batch has no output file")
    (outdir / "batch_output.jsonl").write_bytes(client.files.content(batch.output_file_id).read())
    if batch.error_file_id:
        (outdir / "batch_errors.jsonl").write_bytes(client.files.content(batch.error_file_id).read())
    print(str(outdir / "batch_output.jsonl"))


def parse_outputs(outdir: Path) -> None:
    source_by_id, weak_by_id = load_source_rows()
    rejected_rows = load_rejected_ambiguous_rows()
    by_custom = {custom_id(row): row for row in rejected_rows}
    parsed = []
    output_path = outdir / "batch_output.jsonl"
    if not output_path.exists():
        raise SystemExit(f"Missing {output_path}")
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            row = by_custom.get(raw.get("custom_id"))
            if row is None:
                continue
            choices = raw.get("response", {}).get("body", {}).get("choices") or []
            content = choices[0].get("message", {}).get("content") if choices else ""
            finish_reason = choices[0].get("finish_reason", "") if choices else ""
            status = "ok"
            verdict = {}
            try:
                verdict = parse_json_object(content or "")
            except Exception:
                status = "parse_error"
            source = source_by_id[row["id"]]
            weak = weak_by_id.get(row["id"], {})
            parsed.append(
                {
                    "id": row["id"],
                    "country": source.get("country", ""),
                    "contrast_country": source.get("contrast_country", ""),
                    "topic": source.get("topic", ""),
                    "year": source.get("year", ""),
                    "question": source.get("question", ""),
                    "target_answer": source.get("target_answer", ""),
                    "contrast_answer": source.get("contrast_answer", ""),
                    "weak_locale_risk": weak.get("weak_locale_risk", ""),
                    "previous_failure_modes": row.get("llm_failure_modes", ""),
                    "previous_reason": row.get("llm_reason", ""),
                    "adjudication_status": status,
                    "adjudication_decision": verdict.get("decision", "") if verdict else "",
                    "adjudication_confidence": verdict.get("confidence", "") if verdict else "",
                    "adjudication_failure_modes": " | ".join(verdict.get("failure_modes") or []) if verdict else "",
                    "adjudication_reason": verdict.get("reason", "") if verdict else "",
                    "finish_reason": finish_reason,
                    "raw_response": content or "",
                }
            )
    error_path = outdir / "batch_errors.jsonl"
    if error_path.exists():
        with error_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                row = by_custom.get(raw.get("custom_id"))
                if row is None:
                    continue
                source = source_by_id[row["id"]]
                weak = weak_by_id.get(row["id"], {})
                body = raw.get("response", {}).get("body", {})
                parsed.append(
                    {
                        "id": row["id"],
                        "country": source.get("country", ""),
                        "contrast_country": source.get("contrast_country", ""),
                        "topic": source.get("topic", ""),
                        "year": source.get("year", ""),
                        "question": source.get("question", ""),
                        "target_answer": source.get("target_answer", ""),
                        "contrast_answer": source.get("contrast_answer", ""),
                        "weak_locale_risk": weak.get("weak_locale_risk", ""),
                        "previous_failure_modes": row.get("llm_failure_modes", ""),
                        "previous_reason": row.get("llm_reason", ""),
                        "adjudication_status": "error",
                        "adjudication_decision": "",
                        "adjudication_confidence": "",
                        "adjudication_failure_modes": "",
                        "adjudication_reason": json.dumps(body, ensure_ascii=False),
                        "finish_reason": "",
                        "raw_response": json.dumps(body, ensure_ascii=False),
                    }
                )

    fields = [
        "id",
        "country",
        "contrast_country",
        "topic",
        "year",
        "question",
        "target_answer",
        "contrast_answer",
        "weak_locale_risk",
        "previous_failure_modes",
        "previous_reason",
        "adjudication_status",
        "adjudication_decision",
        "adjudication_confidence",
        "adjudication_failure_modes",
        "adjudication_reason",
        "finish_reason",
        "raw_response",
    ]
    parsed.sort(key=lambda row: row["id"])
    write_csv(outdir / "ambiguous_reject_adjudication.csv", parsed, fields)

    restore = [row for row in parsed if row["adjudication_status"] == "ok" and row["adjudication_decision"] == "restore"]
    reject = [row for row in parsed if row["adjudication_status"] == "ok" and row["adjudication_decision"] == "reject"]
    unverified = [row for row in parsed if row["adjudication_status"] != "ok"]
    for name, group in [
        ("ambiguous_restorable_rows", restore),
        ("ambiguous_still_rejected_rows", reject),
        ("ambiguous_unverified_rows", unverified),
    ]:
        write_csv(outdir / f"{name}.csv", group, fields)
        with (outdir / f"{name.replace('_rows', '_ids')}.txt").open("w", encoding="utf-8") as handle:
            for row in group:
                handle.write(row["id"] + "\n")

    summary = {
        "total_rows": len(parsed),
        "status_counts": dict(Counter(row["adjudication_status"] for row in parsed)),
        "decision_counts": dict(Counter(row["adjudication_decision"] for row in parsed)),
        "confidence_counts": dict(Counter(row["adjudication_confidence"] for row in parsed)),
        "restore_by_weak_risk": dict(Counter(row["weak_locale_risk"] for row in restore)),
        "restore_by_topic": dict(Counter(row["topic"] for row in restore)),
        "failure_mode_counts": dict(count_modes(parsed).most_common()),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def count_modes(rows: list[dict]) -> Counter:
    counts = Counter()
    for row in rows:
        for mode in str(row.get("adjudication_failure_modes") or "").split("|"):
            mode = mode.strip()
            if mode:
                counts[mode] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Second-pass adjudication for rejected ambiguous LocalNewsQA core candidates.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env", type=Path, default=ENV_PATH)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--parse", action="store_true")
    parser.add_argument("--batch-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(args.env)
    source_by_id, weak_by_id = load_source_rows()
    rows = load_rejected_ambiguous_rows()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.prepare or args.submit or args.smoke:
        request_path = write_requests(rows, source_by_id, weak_by_id, args.model, args.max_completion_tokens, args.outdir)
        manifest = {
            "model": args.model,
            "count": len(rows),
            "request_path": str(request_path),
            "max_completion_tokens": args.max_completion_tokens,
        }
        (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(manifest, indent=2, sort_keys=True))
        if args.smoke:
            run_smoke(rows, source_by_id, weak_by_id, args.model, args.smoke, args.max_completion_tokens, args.outdir)
        if args.submit:
            submit_batch(request_path, args.outdir)
    if args.poll:
        poll_batch(args.outdir, args.batch_id)
    if args.download:
        download_batch(args.outdir, args.batch_id)
    if args.parse:
        parse_outputs(args.outdir)


if __name__ == "__main__":
    main()
