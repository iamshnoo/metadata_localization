#!/usr/bin/env python3
"""Write a protocol manifest for the two appendix model-gain tables.

The manifest is intentionally generated from the same CSV artifacts that feed
the paper tables, plus the first row of each source JSONL.  This keeps the
documentation tied to the actual selected sources instead of a separate hand
maintained list.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path("/path/to/metacul")
TABLE_OUT_DIR = ROOT / "results/appendix_model_gain_tables_20260505"
DEFAULT_MD = TABLE_OUT_DIR / "appendix_table_protocol_manifest.md"
DEFAULT_JSON = TABLE_OUT_DIR / "appendix_table_protocol_manifest.json"
TABLE_GENERATOR = ROOT / "src/66_appendix_model_gain_tables.py"
PROTOCOL_SUBMITTER = ROOT / "slurm/submit_external_protocol_probe_search.py"
BASE_URL = "www.globalfactcheck.org"

RECREATE_TABLES_COMMAND = (
    "python src/66_appendix_model_gain_tables.py "
    "--seed 41 --bootstrap 2000 --bootstrap-seed 20260505"
)
RECREATE_MANIFEST_COMMAND = (
    "python src/67_appendix_table_protocol_manifest.py "
    "[--table-out-dir DIR] [--local-only]"
)

FIXED_EXAMPLE = {
    "country_name": "Canada",
    "country_code": "ca",
    "continent": "America",
    "question": "Which institution is the national public broadcaster for this locale?",
    "options": [
        "Canadian Broadcasting Corporation",
        "British Broadcasting Corporation",
        "Australian Broadcasting Corporation",
        "National Public Radio",
    ],
}

LOCALNEWSQA_LABELS = {
    "localnewsqa_overall": "LocalNewsQA Overall",
    "localnewsqa_ambiguous": "LocalNewsQA Ambiguous",
    "localnewsqa_explicit": "LocalNewsQA Explicit",
    "localnewsqa_exact_pair": "LocalNewsQA Exact pair",
    "localnewsqa_margin_switch": "LocalNewsQA Margin switch",
}

LOCALNEWSQA_METRIC_NOTES = {
    "localnewsqa_overall": (
        "Target-locale accuracy gain on all LocalNewsQA items; plus side minus "
        "minus side, paired by item."
    ),
    "localnewsqa_ambiguous": (
        "Target-locale accuracy gain on ambiguous LocalNewsQA items only."
    ),
    "localnewsqa_explicit": (
        "Target-locale accuracy gain on explicit LocalNewsQA items only."
    ),
    "localnewsqa_exact_pair": (
        "Ambiguous-item target/contrast pair metric: the item counts when the "
        "target prompt predicts the target answer and the contrast prompt "
        "predicts the contrast answer."
    ),
    "localnewsqa_margin_switch": (
        "Ambiguous-item target/contrast pair metric: the target-answer margin "
        "must be positive under the target prompt and the contrast-answer "
        "margin must be positive under the contrast prompt."
    ),
}

EXTERNAL_LABELS = {
    "geomlama": "Geo",
    "globalopinionqa": "GOQA",
    "globalmmlu_cs": "MMLU-CS",
    "normad": "NormAD",
    "blend": "BLEnD",
    "worldvaluebench": "WVB",
}

PROTOCOL_FIELDS = [
    "metadata_prompt_style",
    "qa_prompt_style",
    "answer_cue_style",
    "omit_option_labels",
    "exact_option_text_instruction",
    "mcq_scoring",
    "option_loglikelihood_length_norm_alpha",
    "null_calibration_mode",
    "null_calibration_beta",
    "add_prompt_bos",
    "metadata_tag_mode",
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def split_source_paths(value: str) -> List[Path]:
    out: List[Path] = []
    for chunk in re.split(r"[;|]", str(value or "")):
        chunk = chunk.strip()
        if chunk:
            out.append(Path(chunk))
    return out


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def first_json_row(paths: Iterable[Path]) -> Dict[str, Any]:
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return json.loads(line)
    return {}


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def bool_from_any(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def infer_omit_option_labels(row: Dict[str, Any]) -> bool:
    explicit = bool_from_any(row.get("omit_option_labels"))
    if explicit is not None:
        return explicit
    prompt = str(row.get("prompt") or "")
    if re.search(r"\nA:\s", prompt):
        return False
    if re.search(r"\n-\s", prompt):
        return True
    return False


def infer_exact_option_instruction(row: Dict[str, Any]) -> bool:
    explicit = bool_from_any(row.get("exact_option_text_instruction"))
    if explicit is not None:
        return explicit
    return "exact text of the correct option" in str(row.get("prompt") or "")


def protocol_from_json_row(row: Dict[str, Any]) -> Dict[str, Any]:
    alpha = row.get("option_loglikelihood_length_norm_alpha")
    if alpha is None:
        alpha = row.get("length_norm_alpha")
    return {
        "metadata_prompt_style": row.get("metadata_prompt_style", "none"),
        "qa_prompt_style": row.get("qa_prompt_style", "question"),
        "answer_cue_style": row.get("answer_cue_style", "none"),
        "omit_option_labels": infer_omit_option_labels(row),
        "exact_option_text_instruction": infer_exact_option_instruction(row),
        "mcq_scoring": row.get("mcq_scoring", "option_text_avg"),
        "option_loglikelihood_length_norm_alpha": alpha,
        "null_calibration_mode": row.get("null_calibration_mode", "none"),
        "null_calibration_beta": row.get("null_calibration_beta", 0.0),
        "add_prompt_bos": bool_from_any(row.get("add_prompt_bos"), False),
        "metadata_tag_mode": row.get("metadata_tag_mode"),
    }


def env_bool(value: Any) -> Optional[bool]:
    return bool_from_any(value)


def normalize_for_match(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return "1" if text.lower() == "true" else "0"
    parsed = parse_float(text)
    if parsed is not None:
        return f"{parsed:g}"
    return text


def protocol_env_to_details(env: Dict[str, str]) -> Dict[str, Any]:
    return {
        "metadata_prompt_style": env.get("METADATA_PROMPT_STYLE"),
        "qa_prompt_style": env.get("QA_PROMPT_STYLE"),
        "answer_cue_style": env.get("ANSWER_CUE_STYLE"),
        "omit_option_labels": env_bool(env.get("OMIT_OPTION_LABELS")),
        "exact_option_text_instruction": env_bool(env.get("EXACT_OPTION_TEXT_INSTRUCTION")),
        "mcq_scoring": env.get("MCQ_SCORING"),
        "option_loglikelihood_length_norm_alpha": env.get("LENGTH_NORM_ALPHA"),
        "null_calibration_mode": env.get("NULL_CALIBRATION_MODE"),
        "null_calibration_beta": env.get("NULL_CALIBRATION_BETA"),
        "add_prompt_bos": env_bool(env.get("ADD_PROMPT_BOS")),
        "force_custom_chat_prompt": env_bool(env.get("FORCE_CUSTOM_CHAT_PROMPT")),
    }


def protocol_key_from_path(path: Path, protocol_keys: set[str]) -> Optional[str]:
    for part in path.parts:
        if part in protocol_keys:
            return part
    return None


def protocol_key_from_details(
    details: Dict[str, Any],
    protocols: Dict[str, Dict[str, str]],
) -> Optional[str]:
    candidates: List[str] = []
    for key, env in protocols.items():
        env_details = protocol_env_to_details(env)
        checked = [
            "metadata_prompt_style",
            "qa_prompt_style",
            "answer_cue_style",
            "omit_option_labels",
            "exact_option_text_instruction",
            "mcq_scoring",
            "null_calibration_mode",
            "null_calibration_beta",
            "add_prompt_bos",
        ]
        if all(
            normalize_for_match(details.get(field)) == normalize_for_match(env_details.get(field))
            for field in checked
        ):
            alpha_env = normalize_for_match(env_details.get("option_loglikelihood_length_norm_alpha"))
            alpha_details = normalize_for_match(
                details.get("option_loglikelihood_length_norm_alpha")
            )
            if not alpha_env or alpha_env == alpha_details:
                candidates.append(key)
    return candidates[0] if len(candidates) == 1 else None


def answer_cue(
    style: str,
    answer_cue_country_name: Optional[str],
) -> str:
    if style == "none":
        return ""
    if style == "answer_colon":
        return "\n\nAnswer:"
    if style == "answer_newline":
        return "\n\nAnswer:\n"
    if style == "final_answer_colon":
        return "\n\nFinal answer:"
    if style == "the_correct_answer_is":
        return "\n\nThe correct answer is"
    if style == "country_answer_colon":
        return (
            f"\n\nFor {answer_cue_country_name}, answer:"
            if answer_cue_country_name
            else "\n\nAnswer:"
        )
    if style == "country_final_answer_colon":
        return (
            f"\n\nFor {answer_cue_country_name}, final answer:"
            if answer_cue_country_name
            else "\n\nFinal answer:"
        )
    if style == "country_the_correct_answer_is":
        return (
            f"\n\nFor {answer_cue_country_name}, the correct answer is"
            if answer_cue_country_name
            else "\n\nThe correct answer is"
        )
    return ""


def core_prompt(details: Dict[str, Any], item: Dict[str, Any], with_metadata: bool) -> str:
    question = str(item["question"]).strip()
    options = list(item["options"])
    if bool_from_any(details.get("omit_option_labels"), False):
        options_block = "\n".join(f"- {opt}" for opt in options)
    else:
        options_block = "\n".join(f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options))

    mcq_scoring = str(details.get("mcq_scoring") or "option_text_avg")
    if mcq_scoring == "option_letter":
        answer_instruction = "Answer with only the correct option letter (A, B, C, or D)."
    elif bool_from_any(details.get("exact_option_text_instruction"), False):
        answer_instruction = "Answer with the exact text of the correct option."
    else:
        answer_instruction = "Answer with the correct option."

    cue_country = str(item["country_name"]) if with_metadata else None
    cue = answer_cue(str(details.get("answer_cue_style") or "none"), cue_country)
    style = str(details.get("qa_prompt_style") or "question")

    if style == "question":
        return (
            f"Question: {question}\n\n"
            f"Options:\n{options_block}\n\n"
            f"{answer_instruction}"
            f"{cue}"
        )
    if style == "instruction":
        return (
            "### Instruction:\n"
            f"{question}\n\n"
            "Options:\n"
            f"{options_block}\n\n"
            f"{answer_instruction}"
            f"{cue}"
        )
    if style == "instruction_input":
        return (
            "### Instruction:\n"
            "Answer the following locale-specific multiple-choice question.\n\n"
            "### Input:\n"
            f"Question: {question}\n\n"
            "Options:\n"
            f"{options_block}\n\n"
            f"{answer_instruction}"
            f"{cue}"
        )
    if style == "question_answer":
        suffix = cue if cue else "\n\nAnswer:"
        return f"Question: {question}\n\nOptions:\n{options_block}{suffix}"
    return (
        f"Question: {question}\n\n"
        f"Options:\n{options_block}\n\n"
        f"{answer_instruction}"
        f"{cue}"
    )


def add_metadata_block(
    content: str,
    details: Dict[str, Any],
    item: Dict[str, Any],
) -> str:
    country_tag = str(item["country_code"])
    continent_tag = str(item["continent"])
    country_name = str(item["country_name"])
    style = str(details.get("metadata_prompt_style") or "none")

    if style == "legacy_code":
        block = (
            f"URL: {BASE_URL}/{country_tag}\n"
            f"COUNTRY: {country_tag}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about the country {country_tag}\n\n"
            "CONTENT:\n"
        )
        return block + content
    if style == "name_plain":
        block = (
            f"URL: {BASE_URL}/{country_tag}\n"
            f"COUNTRY: {country_name}\n"
            f"COUNTRY_CODE: {country_tag}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about {country_name}\n\n"
            "CONTENT:\n"
        )
        return block + content
    if style == "name_grounded":
        block = (
            f"URL: {BASE_URL}/{country_tag}\n"
            f"COUNTRY: {country_name}\n"
            f"COUNTRY_CODE: {country_tag}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about {country_name}\n\n"
            "CONTENT:\n"
            f"Use the locale metadata above as grounding. Answer for {country_name}, not a different country.\n\n"
        )
        return block + content
    if style == "code_grounded":
        block = (
            f"URL: {BASE_URL}/{country_tag}\n"
            f"COUNTRY: {country_tag}\n"
            f"COUNTRY_NAME: {country_name}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about the country {country_tag}\n\n"
            "CONTENT:\n"
            f"Use the locale metadata above as grounding. This question is about {country_name}.\n\n"
        )
        return block + content
    if style == "code_disambiguate":
        block = (
            f"URL: {BASE_URL}/{country_tag}\n"
            f"COUNTRY: {country_tag}\n"
            f"COUNTRY_NAME: {country_name}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about the country {country_tag}\n\n"
            "CONTENT:\n"
            "Use the locale metadata above to disambiguate country-specific facts. "
            f"When multiple answers could fit different countries, answer for {country_name}.\n\n"
        )
        return block + content
    if style == "name_strict":
        block = (
            f"URL: {BASE_URL}/{country_tag}\n"
            f"COUNTRY: {country_name}\n"
            f"COUNTRY_CODE: {country_tag}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about {country_name}\n\n"
            "CONTENT:\n"
            f"This question is specifically about factual knowledge in {country_name}. "
            f"Choose the answer that is correct for {country_name}, not the answer that would fit a different country "
            "or the answer that is only more globally common.\n\n"
        )
        return block + content
    if style == "code_grounded_strict":
        block = (
            f"URL: {BASE_URL}/{country_tag}\n"
            f"COUNTRY: {country_tag}\n"
            f"COUNTRY_NAME: {country_name}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about the country {country_tag}\n\n"
            "CONTENT:\n"
            f"This question is specifically about factual knowledge in {country_name}. "
            "Use the locale metadata above to disambiguate country-specific facts. "
            f"Choose the answer that is correct for {country_name}, even when a different option is more globally common.\n\n"
        )
        return block + content
    if style == "country_first_strict":
        block = (
            f"COUNTRY: {country_name}\n"
            f"COUNTRY_CODE: {country_tag}\n"
            f"CONTINENT: {continent_tag}\n"
            f"URL: {BASE_URL}/{country_tag}\n\n"
            f"TITLE: Facts about {country_name}\n\n"
            "CONTENT:\n"
            f"Answer this question for {country_name}. "
            f"When multiple options could fit different countries, pick the option that is factual for {country_name}.\n\n"
        )
        return block + content
    return f"CONTENT:\n{content}"


def render_example(details: Dict[str, Any], item: Dict[str, Any]) -> str:
    metadata_style = str(details.get("metadata_prompt_style") or "none")
    with_metadata = metadata_style != "none"
    core = core_prompt(details, item, with_metadata=with_metadata)
    if with_metadata:
        return add_metadata_block(core, details, item).rstrip() + "\n"
    return f"CONTENT:\n{core}".rstrip() + "\n"


def candidate_texts(details: Dict[str, Any], options: List[str]) -> List[str]:
    prefix = "" if details.get("answer_cue_style") == "answer_newline" else " "
    if details.get("mcq_scoring") == "option_letter":
        return [prefix + chr(65 + i) for i in range(len(options))]
    return [prefix + str(opt).strip() for opt in options]


def summarize_protocol(details: Dict[str, Any]) -> str:
    parts = []
    for field in PROTOCOL_FIELDS:
        value = details.get(field)
        if value is None or value == "":
            continue
        parts.append(f"{field}={value}")
    return "; ".join(parts)


def infer_comparison(row: Dict[str, str], table_name: str) -> str:
    if row.get("group") == "MAPLE family":
        return "T+I+ minus T-I- (MAPLE trained-with-metadata/inference-with-metadata vs trained-without/inference-without)."
    if table_name == "localnewsqa":
        return "I+ minus I- (same pretrained/chat baseline, with locale metadata prompt vs without metadata prompt)."
    return "I+ minus I- (same external baseline, with locale metadata prompt vs without metadata prompt)."


def build_entry(
    row: Dict[str, str],
    table_name: str,
    protocols: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    plus_paths = split_source_paths(row["source_plus"])
    minus_paths = split_source_paths(row["source_minus"])
    plus_row = first_json_row(plus_paths)
    minus_row = first_json_row(minus_paths)
    plus_protocol = protocol_from_json_row(plus_row)
    minus_protocol = protocol_from_json_row(minus_row)

    protocol_keys = set(protocols)
    plus_key = protocol_key_from_path(plus_paths[0], protocol_keys) if plus_paths else None
    minus_key = protocol_key_from_path(minus_paths[0], protocol_keys) if minus_paths else None
    if plus_key is None:
        plus_key = protocol_key_from_details(plus_protocol, protocols)
    if minus_key is None:
        minus_key = protocol_key_from_details(minus_protocol, protocols)

    metric_key = row["metric_key"]
    if table_name == "localnewsqa":
        dataset_name = LOCALNEWSQA_LABELS.get(metric_key, metric_key)
        metric_note = LOCALNEWSQA_METRIC_NOTES.get(metric_key, "")
    else:
        dataset_name = EXTERNAL_LABELS.get(metric_key, metric_key)
        metric_note = (
            "Accuracy percentage-point gain." if row.get("metric_type") == "accuracy_pp_gain"
            else "WVB gain: reduction in Earth-mover distance, so positive means the plus side is closer."
        )

    rescore_alpha = parse_float(row.get("rescore_alpha"))
    rescore_beta = parse_float(row.get("rescore_beta"))
    table_rescore = {
        "alpha": rescore_alpha,
        "beta": rescore_beta,
        "note": (
            "Table recomputes argmax from saved option log-likelihood sums with this alpha/beta."
            if rescore_alpha is not None or (rescore_beta is not None and rescore_beta != 0.0)
            else "Table uses source JSONL selected scores / processed predictions."
        ),
    }

    return {
        "table": "LocalNewsQA appendix model-gain table" if table_name == "localnewsqa" else "External benchmark appendix model-gain table",
        "model_name": row["label"],
        "model_key": row["row_key"],
        "track": row["track"],
        "group": row["group"],
        "dataset_name": dataset_name,
        "dataset_key": metric_key,
        "metric_type": row["metric_type"],
        "metric_note": metric_note,
        "n": int(float(row["n"])),
        "delta": float(row["delta"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
        "plus_value": float(row["plus_value"]),
        "minus_value": float(row["minus_value"]),
        "comparison": infer_comparison(row, table_name),
        "source_plus_paths": [rel(p) for p in plus_paths],
        "source_minus_paths": [rel(p) for p in minus_paths],
        "source_plus_protocol_key": plus_key,
        "source_minus_protocol_key": minus_key,
        "positive_protocol": plus_protocol,
        "negative_protocol": minus_protocol,
        "table_rescore": table_rescore,
        "fixed_example": FIXED_EXAMPLE,
        "positive_example_prompt": render_example(plus_protocol, FIXED_EXAMPLE),
        "negative_example_prompt": render_example(minus_protocol, FIXED_EXAMPLE),
        "positive_scoring_candidates": candidate_texts(plus_protocol, FIXED_EXAMPLE["options"]),
        "negative_scoring_candidates": candidate_texts(minus_protocol, FIXED_EXAMPLE["options"]),
    }


def build_manifest(table_out_dir: Path = TABLE_OUT_DIR, include_external: bool = True) -> List[Dict[str, Any]]:
    load_module(TABLE_GENERATOR, "appendix_table_generator")
    submitter = load_module(PROTOCOL_SUBMITTER, "external_protocol_submitter")
    protocols = dict(getattr(submitter, "PROTOCOLS"))

    local_rows = read_csv(table_out_dir / "localnewsqa_model_gains_long.csv")

    entries = [build_entry(row, "localnewsqa", protocols) for row in local_rows]
    if include_external:
        external_rows = read_csv(table_out_dir / "external_model_gains_long.csv")
        entries.extend(build_entry(row, "external", protocols) for row in external_rows)
    return entries


def md_code_block(text: str) -> str:
    return "```text\n" + text.rstrip() + "\n```"


def fmt_num(value: float) -> str:
    if abs(value) >= 1:
        return f"{value:+.2f}"
    return f"{value:+.4f}"


def write_markdown(entries: List[Dict[str, Any]], path: Path) -> None:
    lines: List[str] = []
    lines.extend(
        [
            "# Appendix Table Protocol Manifest",
            "",
            "This file documents every model/dataset cell in the two large appendix model-gain tables.",
            "It is generated from the table CSV artifacts and the source JSONLs used by each cell.",
            "",
            "## Reproduction Commands",
            "",
            f"1. `{RECREATE_TABLES_COMMAND}`",
            f"2. `{RECREATE_MANIFEST_COMMAND}`",
            "",
            "## Fixed Illustrative Example",
            "",
            "Every protocol rendering below uses the same toy item so prompt-format changes are easy to compare.",
            "",
            md_code_block(
                "\n".join(
                    [
                        f"COUNTRY: {FIXED_EXAMPLE['country_name']}",
                        f"COUNTRY_CODE: {FIXED_EXAMPLE['country_code']}",
                        f"CONTINENT: {FIXED_EXAMPLE['continent']}",
                        f"QUESTION: {FIXED_EXAMPLE['question']}",
                        "OPTIONS:",
                        *[
                            f"{chr(65 + i)}: {option}"
                            for i, option in enumerate(FIXED_EXAMPLE["options"])
                        ],
                    ]
                )
            ),
            "",
            "## Cells",
            "",
        ]
    )

    for idx, entry in enumerate(entries, 1):
        lines.extend(
            [
                f"### {idx}. {entry['table']} | {entry['model_name']} | {entry['dataset_name']}",
                "",
                f"- model name: `{entry['model_name']}`",
                f"- dataset name: `{entry['dataset_name']}` (`{entry['dataset_key']}`)",
                f"- comparison: {entry['comparison']}",
                f"- metric: `{entry['metric_type']}`; {entry['metric_note']}",
                (
                    f"- result in table: delta {fmt_num(entry['delta'])}, "
                    f"CI [{fmt_num(entry['ci_low'])}, {fmt_num(entry['ci_high'])}], "
                    f"n={entry['n']}"
                ),
                f"- positive source protocol key: `{entry['source_plus_protocol_key'] or 'source_jsonl'}`",
                f"- negative source protocol key: `{entry['source_minus_protocol_key'] or 'source_jsonl'}`",
                f"- positive protocol details: {summarize_protocol(entry['positive_protocol'])}",
                f"- negative protocol details: {summarize_protocol(entry['negative_protocol'])}",
                (
                    f"- table rescore: alpha={entry['table_rescore']['alpha']}, "
                    f"beta={entry['table_rescore']['beta']}; {entry['table_rescore']['note']}"
                ),
                "- positive source path(s):",
            ]
        )
        lines.extend(f"  - `{p}`" for p in entry["source_plus_paths"])
        lines.append("- negative source path(s):")
        lines.extend(f"  - `{p}`" for p in entry["source_minus_paths"])
        lines.extend(
            [
                "- fixed-example positive prompt:",
                md_code_block(entry["positive_example_prompt"]),
                f"- positive scoring candidates: `{entry['positive_scoring_candidates']}`",
                "- fixed-example negative prompt:",
                md_code_block(entry["negative_example_prompt"]),
                f"- negative scoring candidates: `{entry['negative_scoring_candidates']}`",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a per-cell protocol manifest for appendix model-gain tables."
    )
    parser.add_argument("--table-out-dir", type=Path, default=TABLE_OUT_DIR)
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only document LocalNewsQA rows; useful when external tables are unchanged.",
    )
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = build_manifest(args.table_out_dir, include_external=not args.local_only)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(entries, args.out_md)
    print(f"[ok] wrote {len(entries)} entries")
    print(f"[ok] markdown: {args.out_md}")
    print(f"[ok] json: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
