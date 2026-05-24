#!/usr/bin/env python3

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".agents/.env"
THIS_DIR = ROOT / "qa_data/localnewsqa_core"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import CONTRAST_COUNTRY_MAP, COUNTRY_TO_CONTINENT, TOPIC_LABELS, TOPICS  # noqa: E402


QUALITY_AUDIT_DIR = THIS_DIR / "runs/quality_audit"
SEMANTIC_SCORES = QUALITY_AUDIT_DIR / "semantic_quality_full_35874_scores.csv"
EXPANDED_CORE_IDS = QUALITY_AUDIT_DIR / "llm_core_verification/llm_accept_plus_restored_core_ids.txt"
OUTDIR = QUALITY_AUDIT_DIR / "gold_ambiguous_1700"

DEFAULT_MODEL = "gpt-5.5"
TARGET_PER_COUNTRY = 100


GENERATION_SYSTEM = """You create gold-standard LocalNewsQA ambiguous items.

Each item is a multiple-choice QA-evidence triplet for locale-sensitive English question answering.
The same question text must be natural for both target_country and contrast_country, but the correct answer must differ by locale.

Hard requirements:
- The question must omit the target and contrast country names, demonyms, and country-specific named anchors.
- The wording must be generic and fair for both locales, not a forced analogy.
- Options must not solve the item by shallow country-name matching. Official entity names that contain a
  country or demonym are allowed only when the official name itself is the answer and all options are
  comparable named entities.
- target_answer must be factually correct for target_country.
- contrast_answer must be factually correct for contrast_country.
- Exactly one option must be correct for the target locale.
- Include exactly 4 distinct plausible options.
- Prefer stable, institution-level or convention-level facts over transient officeholders.
- Avoid brittle current facts unless the question includes a specific year.
- Provide authoritative, public evidence for both target and contrast answers. Prefer official, government, university, regulator, broadcaster, or organization pages. Wikipedia is acceptable only for stable entity facts when official evidence is hard to use.
- Evidence URLs must be real and specific enough to support the answer, not just generic country pages.
- Evidence excerpts must be short, factual support statements, not long copied passages.

Return only compact JSON:
{"items":[
  {
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "target_answer": "...",
    "contrast_answer": "...",
    "topic": "Politics and government|Civic institutions|Economy and business|Education|Media and culture|Sports|Public life and holidays|Geography and infrastructure",
    "year": 2010,
    "evidence_hint": "...",
    "target_evidence": {"url":"...", "title":"...", "excerpt":"short support statement"},
    "contrast_evidence": {"url":"...", "title":"...", "excerpt":"short support statement"},
    "why_locale_sensitive": "short explanation"
  }
]}
"""


VERIFY_SYSTEM = """You are a strict final auditor for the LocalNewsQA-Ambiguous gold split.

Use web search and the supplied evidence, but do not trust either blindly. Decide whether the item is safe for a public gold-standard benchmark.

Accept only if ALL checks pass:
1. The question text does not name or uniquely anchor the target or contrast country.
2. The same question is natural and fair for both target_country and contrast_country.
3. target_answer is factually correct for target_country.
4. contrast_answer is factually correct for contrast_country.
5. The answer difference is genuinely caused by locale, not by a forced analogy.
6. Exactly one option is correct under target_country.
7. Options are distinct and plausible, with no duplicate/near-duplicate answers.
8. The fact is stable for the stated year/context, or the year makes any temporal fact unambiguous.
9. Evidence for both sides is public, specific, reasonable, and supports the answer.
10. Options do not make the target answer obvious by shallow country-name matching, except when country/demonym
    text is part of comparable official entity names across options.
11. There is no answer leakage, country leakage in the question, misleading wording, weak locale dependence, stale/changing fact, or insufficient evidence.

Do not accept merely plausible rows. If there is meaningful doubt, reject.
If accepted, provide the best evidence URLs and concise support excerpts for both target and contrast.

Return only compact JSON:
{
  "decision": "accept" | "reject",
  "confidence": "high" | "medium" | "low",
  "failure_modes": ["..."],
  "reason": "one concise sentence",
  "target_evidence": {"url":"...", "title":"...", "excerpt":"short support statement"},
  "contrast_evidence": {"url":"...", "title":"...", "excerpt":"short support statement"}
}

Use failure_modes from:
target_specific_anchor, contrast_not_applicable, weak_locale_difference, ambiguous_wording, factual_error, multiple_correct_answers, answer_not_in_options, answer_leakage, country_leakage, temporal_ambiguity, stale_or_changed_fact, poor_options, insufficient_evidence, bad_evidence_url, duplicate_or_near_duplicate, other.
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


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_json_object(raw: str) -> dict:
    raw = str(raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def option_parts(options: str | list) -> list[str]:
    if isinstance(options, list):
        return [str(part).strip() for part in options if str(part).strip()]
    out = []
    for part in str(options or "").split("||"):
        part = part.strip()
        if not part:
            continue
        out.append(part.split(":", 1)[-1].strip())
    return out


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict, lock: Any | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(row, ensure_ascii=False) + "\n"
    if lock:
        with lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(text)
    else:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)


def load_source_rows() -> dict[str, dict]:
    with SEMANTIC_SCORES.open(newline="", encoding="utf-8") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def load_id_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def candidate_signature(row: dict) -> str:
    return "|".join(
        [
            normalize_text(row.get("country", "")),
            normalize_text(row.get("contrast_country", "")),
            normalize_text(row.get("question", "")),
            normalize_text(row.get("target_answer", "")),
            normalize_text(row.get("contrast_answer", "")),
        ]
    )


def write_seed_candidates(outdir: Path) -> None:
    source_by_id = load_source_rows()
    ids = load_id_file(EXPANDED_CORE_IDS)
    out_path = outdir / "seed_candidates.jsonl"
    outdir.mkdir(parents=True, exist_ok=True)
    existing = {row.get("candidate_id") for row in read_jsonl(out_path)}
    kept = 0
    for row_id in sorted(ids):
        row = source_by_id.get(row_id)
        if not row or row.get("split") != "ambiguous":
            continue
        candidate_id = "seed_" + row_id
        if candidate_id in existing:
            continue
        candidate = {
            "candidate_id": candidate_id,
            "source": "existing_verified_or_restored",
            "source_row_id": row_id,
            "country": row["country"],
            "continent": row["continent"],
            "contrast_country": row["contrast_country"],
            "topic": row["topic"],
            "year": row["year"],
            "question": row["question"],
            "options": option_parts(row["options"]),
            "target_answer": row["target_answer"],
            "contrast_answer": row["contrast_answer"],
            "evidence_hint": row.get("evidence_hint", ""),
            "target_evidence": {
                "url": row.get("target_evidence_url", ""),
                "title": row.get("target_evidence_title", ""),
                "excerpt": row.get("target_evidence_excerpt") or row.get("target_evidence_snippet", ""),
            },
            "contrast_evidence": {
                "url": row.get("contrast_evidence_url", ""),
                "title": row.get("contrast_evidence_title", ""),
                "excerpt": row.get("contrast_evidence_excerpt") or row.get("contrast_evidence_snippet", ""),
            },
        }
        append_jsonl(out_path, candidate)
        kept += 1
    print(json.dumps({"seed_candidates_written": kept, "path": str(out_path)}, indent=2))


def response_json(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_output_tokens: int,
    attempts: int,
) -> dict:
    last_error = None
    for attempt in range(attempts):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tools=[{"type": "web_search_preview"}],
                max_output_tokens=max_output_tokens,
            )
            return parse_json_object(resp.output_text)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2 + attempt * 3)
    raise RuntimeError(str(last_error))


def generation_user_prompt(country: str, batch_idx: int, n_items: int, existing_questions: list[str]) -> str:
    contrast = CONTRAST_COUNTRY_MAP[country]
    continent = COUNTRY_TO_CONTINENT[country]
    topic_hint = ", ".join(TOPIC_LABELS[t] for t in TOPICS)
    avoid = "\n".join(f"- {q[:180]}" for q in existing_questions[:25])
    return f"""Create {n_items} gold-standard ambiguous LocalNewsQA items.

Target country: {country}
Target continent: {continent}
Contrast country: {contrast}
Batch index: {batch_idx}
Allowed topics: {topic_hint}

Use a diverse mix of topics and years from 2010-2024. Prefer stable facts with authoritative evidence.
Avoid repeating these existing questions or their anchors:
{avoid}
"""


def generate_one_batch(args: tuple[str, int, int, str, int, float, int, list[str]]) -> dict:
    country, batch_idx, n_items, model, max_output_tokens, request_timeout, attempts, existing_questions = args
    client = OpenAI(timeout=request_timeout, max_retries=0)
    user = generation_user_prompt(country, batch_idx, n_items, existing_questions)
    data = response_json(client, model, GENERATION_SYSTEM, user, max_output_tokens, attempts)
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("Generation response missing items list")
    out = []
    for idx, item in enumerate(items):
        item = dict(item)
        candidate = {
            "candidate_id": f"gen_{country.lower().replace(' ', '_')}_{batch_idx:04d}_{idx:02d}",
            "source": "web_generated",
            "source_row_id": "",
            "country": country,
            "continent": COUNTRY_TO_CONTINENT[country],
            "contrast_country": CONTRAST_COUNTRY_MAP[country],
            "topic": item.get("topic", ""),
            "year": str(item.get("year", "")),
            "question": item.get("question", ""),
            "options": option_parts(item.get("options", [])),
            "target_answer": item.get("target_answer", ""),
            "contrast_answer": item.get("contrast_answer", ""),
            "evidence_hint": item.get("evidence_hint", ""),
            "target_evidence": item.get("target_evidence") or {},
            "contrast_evidence": item.get("contrast_evidence") or {},
            "why_locale_sensitive": item.get("why_locale_sensitive", ""),
        }
        out.append(candidate)
    return {"country": country, "batch_idx": batch_idx, "items": out}


def generate_candidates(
    outdir: Path,
    model: str,
    candidates_per_country: int,
    batch_size: int,
    workers: int,
    max_output_tokens: int,
    request_timeout: float,
    attempts: int,
    only_country: str | None = None,
) -> None:
    out_path = outdir / "generated_candidates.jsonl"
    existing_rows = read_jsonl(out_path) + read_jsonl(outdir / "seed_candidates.jsonl")
    existing_ids = {row.get("candidate_id") for row in read_jsonl(out_path)}
    questions_by_country: dict[str, list[str]] = defaultdict(list)
    for row in existing_rows:
        if row.get("country") and row.get("question"):
            questions_by_country[row["country"]].append(row["question"])

    tasks = []
    for country in COUNTRY_TO_CONTINENT:
        if only_country and country != only_country:
            continue
        have_generated = sum(1 for row in read_jsonl(out_path) if row.get("country") == country)
        needed = max(0, candidates_per_country - have_generated)
        start_batch = have_generated // batch_size
        for off in range(0, needed, batch_size):
            batch_idx = start_batch + off // batch_size
            n_items = min(batch_size, needed - off)
            probe_id = f"gen_{country.lower().replace(' ', '_')}_{batch_idx:04d}_00"
            if probe_id in existing_ids:
                continue
            tasks.append((country, batch_idx, n_items, model, max_output_tokens, request_timeout, attempts, questions_by_country[country]))

    random.Random(17).shuffle(tasks)
    lock = threading.Lock()
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_task = {pool.submit(generate_one_batch, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                for item in result["items"]:
                    append_jsonl(out_path, item, lock)
                print(json.dumps({"generated": len(result["items"]), "country": result["country"], "batch": result["batch_idx"]}), flush=True)
            except Exception as exc:
                failures.append({"task": task[:3], "error": str(exc)})
                print(json.dumps({"generation_error": str(exc), "task": task[:3]}), flush=True)
    if failures:
        (outdir / "generation_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(json.dumps({"tasks": len(tasks), "failures": len(failures), "output": str(out_path)}, indent=2))


def verify_user_prompt(candidate: dict) -> str:
    payload = {
        "candidate_id": candidate.get("candidate_id", ""),
        "target_country": candidate.get("country", ""),
        "contrast_country": candidate.get("contrast_country", ""),
        "topic": candidate.get("topic", ""),
        "year": candidate.get("year", ""),
        "question": candidate.get("question", ""),
        "options": candidate.get("options", []),
        "target_answer": candidate.get("target_answer", ""),
        "contrast_answer": candidate.get("contrast_answer", ""),
        "evidence_hint": candidate.get("evidence_hint", ""),
        "supplied_target_evidence": candidate.get("target_evidence", {}),
        "supplied_contrast_evidence": candidate.get("contrast_evidence", {}),
        "source": candidate.get("source", ""),
    }
    return "Audit this candidate for the LocalNewsQA-Ambiguous gold split:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def verify_one_candidate(args: tuple[dict, str, int, float, int]) -> dict:
    candidate, model, max_output_tokens, request_timeout, attempts = args
    client = OpenAI(timeout=request_timeout, max_retries=0)
    try:
        verdict = response_json(client, model, VERIFY_SYSTEM, verify_user_prompt(candidate), max_output_tokens, attempts)
        status = "ok"
        raw_error = ""
    except Exception as exc:
        verdict = {}
        raw_error = str(exc)
        status = "quota_error" if "insufficient_quota" in raw_error or "exceeded your current quota" in raw_error else "error"
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "source_row_id": candidate.get("source_row_id", ""),
        "source": candidate.get("source", ""),
        "country": candidate.get("country", ""),
        "continent": candidate.get("continent", ""),
        "contrast_country": candidate.get("contrast_country", ""),
        "topic": candidate.get("topic", ""),
        "year": str(candidate.get("year", "")),
        "question": candidate.get("question", ""),
        "options": candidate.get("options", []),
        "target_answer": candidate.get("target_answer", ""),
        "contrast_answer": candidate.get("contrast_answer", ""),
        "evidence_hint": candidate.get("evidence_hint", ""),
        "audit_status": status,
        "audit_decision": verdict.get("decision", "") if verdict else "",
        "audit_confidence": verdict.get("confidence", "") if verdict else "",
        "audit_failure_modes": verdict.get("failure_modes", []) if verdict else [],
        "audit_reason": verdict.get("reason", "") if verdict else raw_error,
        "target_evidence": verdict.get("target_evidence") if verdict else candidate.get("target_evidence", {}),
        "contrast_evidence": verdict.get("contrast_evidence") if verdict else candidate.get("contrast_evidence", {}),
    }


def combined_candidates(outdir: Path) -> list[dict]:
    rows = read_jsonl(outdir / "seed_candidates.jsonl") + read_jsonl(outdir / "generated_candidates.jsonl")
    seen = set()
    out = []
    for row in rows:
        sig = candidate_signature(row)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(row)
    return out


def verify_candidates(
    outdir: Path,
    model: str,
    workers: int,
    max_output_tokens: int,
    request_timeout: float,
    attempts: int,
    limit: int | None,
    country: str | None,
    retry_errors: bool,
) -> None:
    out_path = outdir / "candidate_verifications.jsonl"
    existing_verifications = read_jsonl(out_path)
    done = {
        row.get("candidate_id")
        for row in existing_verifications
        if not (retry_errors and row.get("audit_status") in {"error", "quota_error"})
    }
    candidates = [row for row in combined_candidates(outdir) if row.get("candidate_id") not in done]
    if country:
        candidates = [row for row in candidates if row.get("country") == country]
    if limit:
        candidates = candidates[:limit]
    lock = threading.Lock()
    counts = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_id = {
            pool.submit(verify_one_candidate, (candidate, model, max_output_tokens, request_timeout, attempts)): candidate.get("candidate_id")
            for candidate in candidates
        }
        for future in as_completed(future_to_id):
            row = future.result()
            counts[(row["audit_status"], row["audit_decision"])] += 1
            append_jsonl(out_path, row, lock)
            print(json.dumps({"verified": row["candidate_id"], "country": row["country"], "decision": row["audit_decision"], "status": row["audit_status"]}), flush=True)
            if row["audit_status"] == "quota_error":
                print(json.dumps({"quota_exhausted": True, "message": "Stopping verifier; rerun with --retry-errors after quota is restored."}), flush=True)
                for pending in future_to_id:
                    pending.cancel()
                break
    print(json.dumps({"verified_new": len(candidates), "counts": {str(k): v for k, v in counts.items()}, "output": str(out_path)}, indent=2))


def evidence_field(ev: dict, key: str) -> str:
    if not isinstance(ev, dict):
        return ""
    return str(ev.get(key, "") or "")


def row_quality_sort_key(row: dict) -> tuple:
    source_rank = 0 if row.get("source") == "existing_verified_or_restored" else 1
    conf_rank = {"high": 0, "medium": 1, "low": 2}.get(row.get("audit_confidence", ""), 3)
    topic = row.get("topic", "")
    return (conf_rank, source_rank, topic, row.get("candidate_id", ""))


def assemble_gold(outdir: Path, target_per_country: int) -> None:
    verifications = read_jsonl(outdir / "candidate_verifications.jsonl")
    accepted = [
        row
        for row in verifications
        if row.get("audit_status") == "ok"
        and row.get("audit_decision") == "accept"
        and evidence_field(row.get("target_evidence", {}), "url")
        and evidence_field(row.get("contrast_evidence", {}), "url")
    ]
    by_country: dict[str, list[dict]] = defaultdict(list)
    seen_sigs = set()
    for row in sorted(accepted, key=row_quality_sort_key):
        sig = candidate_signature(row)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        by_country[row["country"]].append(row)

    selected = []
    deficits = {}
    for country in COUNTRY_TO_CONTINENT:
        rows = by_country.get(country, [])
        if len(rows) < target_per_country:
            deficits[country] = target_per_country - len(rows)
        selected.extend(rows[:target_per_country])

    fields = [
        "id",
        "candidate_id",
        "source",
        "source_row_id",
        "split_name",
        "split_type",
        "split_family",
        "ambiguity_flag",
        "country",
        "continent",
        "target_country",
        "contrast_country",
        "topic",
        "year",
        "question",
        "options",
        "correct_answer",
        "distractors",
        "target_answer",
        "contrast_answer",
        "evidence_hint",
        "target_evidence_url",
        "target_evidence_title",
        "target_evidence_excerpt",
        "contrast_evidence_url",
        "contrast_evidence_title",
        "contrast_evidence_excerpt",
        "audit_confidence",
        "audit_reason",
    ]
    csv_rows = []
    jsonl_rows = []
    for idx, row in enumerate(sorted(selected, key=lambda r: (r["country"], r["topic"], r["candidate_id"])), start=1):
        options = option_parts(row.get("options", []))
        target = row.get("target_answer", "")
        distractors = [opt for opt in options if normalize_text(opt) != normalize_text(target)]
        base_row = {
            "id": f"localnewsqa_ambig_gold_{idx:04d}",
            "candidate_id": row.get("candidate_id", ""),
            "source": row.get("source", ""),
            "source_row_id": row.get("source_row_id", ""),
            "split_name": "LocalNewsQA-Core-Ambiguous-Gold-1700",
            "split_type": "ambiguous",
            "split_family": "LocalNewsQA-Core",
            "ambiguity_flag": True,
            "country": row.get("country", ""),
            "continent": row.get("continent", ""),
            "target_country": row.get("country", ""),
            "contrast_country": row.get("contrast_country", ""),
            "topic": row.get("topic", ""),
            "year": row.get("year", ""),
            "question": row.get("question", ""),
            "correct_answer": target,
            "target_answer": target,
            "contrast_answer": row.get("contrast_answer", ""),
            "evidence_hint": row.get("evidence_hint", ""),
            "target_evidence_url": evidence_field(row.get("target_evidence", {}), "url"),
            "target_evidence_title": evidence_field(row.get("target_evidence", {}), "title"),
            "target_evidence_excerpt": evidence_field(row.get("target_evidence", {}), "excerpt"),
            "contrast_evidence_url": evidence_field(row.get("contrast_evidence", {}), "url"),
            "contrast_evidence_title": evidence_field(row.get("contrast_evidence", {}), "title"),
            "contrast_evidence_excerpt": evidence_field(row.get("contrast_evidence", {}), "excerpt"),
            "audit_confidence": row.get("audit_confidence", ""),
            "audit_reason": row.get("audit_reason", ""),
        }
        csv_row = dict(base_row)
        csv_row["options"] = " || ".join(options)
        csv_row["distractors"] = " || ".join(distractors)
        jsonl_row = dict(base_row)
        jsonl_row["options"] = options
        jsonl_row["distractors"] = distractors
        csv_rows.append(csv_row)
        jsonl_rows.append(jsonl_row)

    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "localnewsqa_ambiguous_gold_1700.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    with (outdir / "localnewsqa_ambiguous_gold_1700.jsonl").open("w", encoding="utf-8") as handle:
        for row in jsonl_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "target_per_country": target_per_country,
        "selected_rows": len(jsonl_rows),
        "accepted_candidates": len(accepted),
        "selected_by_country": dict(Counter(row["country"] for row in jsonl_rows)),
        "accepted_by_country": {country: len(rows) for country, rows in sorted(by_country.items())},
        "deficits": deficits,
        "selected_by_topic": dict(Counter(row["topic"] for row in jsonl_rows)),
        "audit_confidence": dict(Counter(row["audit_confidence"] for row in jsonl_rows)),
        "csv": str(outdir / "localnewsqa_ambiguous_gold_1700.csv"),
        "jsonl": str(outdir / "localnewsqa_ambiguous_gold_1700.jsonl"),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 100-per-country gold LocalNewsQA ambiguous split.")
    parser.add_argument("--env", type=Path, default=ENV_PATH)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--candidates-per-country", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--target-per-country", type=int, default=TARGET_PER_COUNTRY)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--country", default=None)
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(args.env)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.seed:
        write_seed_candidates(args.outdir)
    if args.generate:
        generate_candidates(
            args.outdir,
            args.model,
            args.candidates_per_country,
            args.batch_size,
            args.workers,
            args.max_output_tokens,
            args.request_timeout,
            args.attempts,
            args.country,
        )
    if args.verify:
        verify_candidates(
            args.outdir,
            args.model,
            args.workers,
            args.max_output_tokens,
            args.request_timeout,
            args.attempts,
            args.limit,
            args.country,
            args.retry_errors,
        )
    if args.assemble:
        assemble_gold(args.outdir, args.target_per_country)


if __name__ == "__main__":
    main()
