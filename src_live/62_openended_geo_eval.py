#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pycountry
import torch
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

SYSTEM_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)

CHAT_TEMPLATE_PATH = "/path/to/metacul/src/chat_template.jinja"
DEFAULT_ENV_PATH = ""
DEFAULT_OUTPUT_DIR = "/path/to/metacul/results/openended_geo_eval"
DEFAULT_BASE_URL = "www.globalfactcheck.org"

# Exact chosen judge prompt from the CROQ paper arXiv source
# (/tmp/croq_src/latex/prompts/judge.tex).
CROQ_MIDDLE_JUDGE_PROMPT = (
    "You are a precise geographical context classifier. Your task is to determine "
    "which nation, country or state(s) a given text is explicitly referring to. "
    "Follow these rules strictly:\n\n"
    "1. Do not infer nations, countries or states based on general knowledge, "
    "stereotypes, or assumptions beyond what is stated in the text.\n\n"
    "2. If multiple nations, countries or states are mentioned, list all of them. "
    "Not more than 5.\n\n"
    "Keep your answers concise and limited to the nation, country or state names. "
    "Translate the final answers into English."
)

COUNTRY_CODE_MAP = {
    "United States": "us",
    "USA": "us",
    "Canada": "ca",
    "Jamaica": "jm",
    "India": "in",
    "Pakistan": "pk",
    "Bangladesh": "bd",
    "Sri Lanka": "lk",
    "Hong Kong": "hk",
    "Malaysia": "my",
    "Philippines": "ph",
    "Nigeria": "ng",
    "South Africa": "za",
    "Kenya": "ke",
    "Ghana": "gh",
    "Tanzania": "tz",
    "United Kingdom": "gb",
    "Ireland": "ie",
    "China": "cn",
    "Iran": "ir",
    "Greece": "gr",
}

COUNTRY_CONTINENT_MAP = {
    "us": "America",
    "ca": "America",
    "jm": "America",
    "in": "Asia",
    "pk": "Asia",
    "bd": "Asia",
    "lk": "Asia",
    "hk": "Asia",
    "my": "Asia",
    "ph": "Asia",
    "ng": "Africa",
    "za": "Africa",
    "ke": "Africa",
    "gh": "Africa",
    "tz": "Africa",
    "gb": "Europe",
    "ie": "Europe",
}

CODE_TO_COUNTRY_NAME = {
    "us": "United States",
    "ca": "Canada",
    "jm": "Jamaica",
    "in": "India",
    "pk": "Pakistan",
    "bd": "Bangladesh",
    "lk": "Sri Lanka",
    "hk": "Hong Kong",
    "my": "Malaysia",
    "ph": "Philippines",
    "ng": "Nigeria",
    "za": "South Africa",
    "ke": "Kenya",
    "gh": "Ghana",
    "tz": "Tanzania",
    "gb": "United Kingdom",
    "ie": "Ireland",
}

COUNTRY_NAME_ALIASES = {
    "uk": "United Kingdom",
    "u k": "United Kingdom",
    "united kingdom": "United Kingdom",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "northern ireland": "United Kingdom",
    "us": "United States",
    "u s": "United States",
    "u s a": "United States",
    "usa": "United States",
    "united states of america": "United States",
    "united states of america usa": "United States",
    "north korea": "North Korea",
    "north_korea": "North Korea",
    "south korea": "South Korea",
    "south_korea": "South Korea",
    "northern nigeria": "Nigeria",
    "northern_nigeria": "Nigeria",
    "west java": "Indonesia",
    "west_java": "Indonesia",
    "assam": "India",
    "hong kong sar": "Hong Kong",
}

NON_ANSWER_PATTERNS = (
    "none",
    "no inference",
    "cannot infer",
    "can't infer",
    "cannot determine",
    "can't determine",
    "not possible",
    "unknown",
    "n/a",
    "na",
    "no country",
    "no state",
    "no nation",
)

VARIANT_SPECS = {
    "custom_tplus_eplus": {
        "model_type": "custom",
        "custom_model_metadata": True,
        "eval_metadata": True,
    },
    "custom_tplus_eminus": {
        "model_type": "custom",
        "custom_model_metadata": True,
        "eval_metadata": False,
    },
    "custom_tminus_eplus": {
        "model_type": "custom",
        "custom_model_metadata": False,
        "eval_metadata": True,
    },
    "custom_tminus_eminus": {
        "model_type": "custom",
        "custom_model_metadata": False,
        "eval_metadata": False,
    },
    "hf_chat_with_metadata": {
        "model_type": "hf_chat",
        "eval_metadata": True,
    },
    "hf_chat_without_metadata": {
        "model_type": "hf_chat",
        "eval_metadata": False,
    },
}


@dataclass
class OpenEndedItem:
    question_id: str
    question: str
    benchmark: str
    subset: str
    source_index: int
    language: Optional[str] = None
    country: Optional[str] = None
    continent: Optional[str] = None
    topic: Optional[str] = None
    reference_answer: Optional[str] = None
    extra: Optional[Dict[str, object]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run open-ended generation plus CROQ-style location judging over "
            "CROQ-like prompt banks or other open QA datasets."
        )
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["croq_jsonl", "calmqa", "multinativqa"],
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Local JSON/JSONL/CSV prompt bank for croq_jsonl mode.",
    )
    parser.add_argument(
        "--multinativqa-config",
        default="english_qa",
        help="HF config for QCRI/MultiNativQA.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split override.",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        default="custom_tplus_eplus,custom_tminus_eminus",
        help="Comma-separated variants to evaluate.",
    )
    parser.add_argument(
        "--custom-model-path-with-metadata",
        default=None,
        help="Local path override for metadata-aware MAPLE chat model.",
    )
    parser.add_argument(
        "--custom-model-path-without-metadata",
        default=None,
        help="Local path override for metadata-free MAPLE chat model.",
    )
    parser.add_argument(
        "--hf-model-name",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF repo or local path for hf_chat variants.",
    )
    parser.add_argument(
        "--disable-custom-chat-template",
        action="store_true",
        help="Do not force-load chat_template.jinja for custom models.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--decoding",
        default="greedy",
        choices=["greedy", "sample"],
    )
    parser.add_argument(
        "--prompt-style",
        default="croq",
        choices=["croq", "plain_question"],
        help=(
            "CROQ-style prompt adds '{in region/place}' plus "
            "'Be brief. Choose yourself the place.'"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL used when eval-time metadata is enabled.",
    )
    parser.add_argument(
        "--metadata-prompt-style",
        default="name_grounded",
        choices=[
            "legacy_code",
            "name_plain",
            "name_grounded",
            "code_grounded",
            "code_disambiguate",
            "name_strict",
            "code_grounded_strict",
            "country_first_strict",
            "croq_locale_brief",
            "croq_locale_block",
        ],
    )
    parser.add_argument(
        "--env-path",
        default=DEFAULT_ENV_PATH,
        help="Path to optional environment file containing OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="Judge model. CROQ used gpt-4o-mini.",
    )
    parser.add_argument(
        "--judge-max-output-tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--judge-sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay after each judge call.",
    )
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help="Run only the judge stage over existing JSONL outputs.",
    )
    parser.add_argument(
        "--existing-output-jsonl",
        default=None,
        help="Existing output JSONL for --judge-only mode.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _canon_name(text: Optional[str]) -> str:
    if text is None:
        return ""
    t = str(text).strip().lower()
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    return t


def _hash_id(parts: Iterable[str]) -> str:
    import hashlib

    joined = "||".join(str(p) for p in parts)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def _normalize_country(raw_country: Optional[str]) -> Optional[str]:
    if raw_country is None:
        return None
    c = str(raw_country).strip()
    if not c:
        return None
    c_l = c.lower()
    if len(c) == 2 and c_l in COUNTRY_CONTINENT_MAP:
        return c_l
    if c_l in {"usa", "us"}:
        return "us"
    canonical_name = COUNTRY_NAME_ALIASES.get(_canon_name(c), c)
    mapped = COUNTRY_CODE_MAP.get(canonical_name) or COUNTRY_CODE_MAP.get(c)
    if mapped:
        return mapped
    alias = {
        "uk": "gb",
        "united kingdom": "gb",
        "great britain": "gb",
        "britain": "gb",
        "northern ireland": "gb",
        "hong kong": "hk",
    }.get(_canon_name(c))
    if alias:
        return alias
    try:
        record = pycountry.countries.lookup(canonical_name)
        alpha2 = getattr(record, "alpha_2", None)
        if alpha2:
            return alpha2.lower()
    except LookupError:
        pass
    return None


def _display_country_name(raw_country: Optional[str]) -> Optional[str]:
    if raw_country is None:
        return None
    text = str(raw_country).strip()
    if not text:
        return None
    canonical = COUNTRY_NAME_ALIASES.get(
        _canon_name(text), text.replace("_", " ").strip()
    )
    code = _normalize_country(canonical)
    if code in CODE_TO_COUNTRY_NAME:
        return CODE_TO_COUNTRY_NAME[code]
    try:
        record = pycountry.countries.lookup(canonical)
        return getattr(record, "name", canonical)
    except LookupError:
        return canonical


def _normalize_continent_label(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    cont = _canon_name(text)
    if not cont:
        return None
    if cont in {"africa", "asia", "europe"}:
        return cont.capitalize()
    if cont in {"america", "americas", "north america", "south america"}:
        return "America"
    return None


def custom_model_uses_chat_prompt(path: str) -> bool:
    norm = os.path.normpath(path)
    base = os.path.basename(norm)
    return base.endswith("_chat") or f"{os.sep}models{os.sep}sft{os.sep}" in norm


def get_hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        token = os.getenv(key)
        if token and token.strip():
            return token.strip()
    return None


def load_env_file(env_path: str) -> Dict[str, str]:
    path = Path(env_path)
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] in {"'", '"'} and value[-1] == value[0]:
            value = value[1:-1]
        values[key] = value
    return values


def get_openai_api_key(env_path: str) -> str:
    existing = os.getenv("OPENAI_API_KEY")
    if existing and existing.strip():
        return existing.strip()
    env_values = load_env_file(env_path)
    api_key = env_values.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY not found in environment or env file: {env_path}"
        )
    os.environ["OPENAI_API_KEY"] = api_key
    return api_key


def build_model(
    model_type: str,
    hf_token: Optional[str],
    custom_metadata: Optional[bool] = None,
    custom_model_path: Optional[str] = None,
    hf_model_name: Optional[str] = None,
    disable_custom_chat_template: bool = False,
):
    if model_type == "custom":
        if custom_metadata is None:
            raise ValueError("custom_metadata must be set for custom models.")
        if custom_model_path:
            model_name = custom_model_path
        else:
            name = (
                "combined_with_metadata_chat"
                if custom_metadata
                else "combined_without_metadata_chat"
            )
            model_name = f"/path/to/metacul/models/sft/{name}"
    elif model_type == "hf_chat":
        if not hf_model_name:
            raise ValueError("hf_model_name must be set for hf_chat models.")
        model_name = hf_model_name
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    hf_kwargs = {}
    if hf_token and model_type == "hf_chat":
        hf_kwargs["token"] = hf_token

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            fix_mistral_regex=True,
            **hf_kwargs,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **hf_kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    uses_chat_prompt = model_type == "hf_chat" or (
        model_type == "custom" and custom_model_uses_chat_prompt(model_name)
    )
    if uses_chat_prompt and (
        model_type == "hf_chat" or not disable_custom_chat_template
    ):
        if not getattr(tokenizer, "chat_template", None) or model_type == "custom":
            with open(CHAT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
                tokenizer.chat_template = f.read()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype="auto",
        **hf_kwargs,
    )
    return model, tokenizer, uses_chat_prompt


def render_prompt(
    tokenizer,
    messages: List[Dict[str, str]],
    uses_chat_prompt: bool,
) -> str:
    if uses_chat_prompt and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    return user.rstrip() + "\n"


def choose_metadata_tags(
    item: OpenEndedItem,
    seed: int,
) -> Dict[str, str]:
    country_code = _normalize_country(item.country)
    continent = _normalize_continent_label(item.continent)

    if country_code is None and continent is None:
        pairs = sorted(COUNTRY_CONTINENT_MAP.items())
        idx = int(_hash_id([item.question_id, str(seed)]), 16) % len(pairs)
        country_code, continent = pairs[idx]
        return {
            "country_tag": country_code,
            "continent_tag": continent,
            "metadata_source": "sampled_pair",
        }

    if country_code is None and continent is not None:
        choices = sorted(
            [code for code, cont in COUNTRY_CONTINENT_MAP.items() if cont == continent]
        )
        idx = int(_hash_id([item.question_id, str(seed), continent]), 16) % len(choices)
        country_code = choices[idx]
        return {
            "country_tag": country_code,
            "continent_tag": continent,
            "metadata_source": "continent_to_country",
        }

    if country_code is not None and continent is None:
        continent = COUNTRY_CONTINENT_MAP.get(country_code)
        return {
            "country_tag": country_code,
            "continent_tag": continent,
            "metadata_source": "country_to_continent",
        }

    return {
        "country_tag": country_code,
        "continent_tag": continent,
        "metadata_source": "from_example",
    }


def add_metadata_block(
    content: str,
    base_url: str,
    country_tag: str,
    continent_tag: str,
    country_name: str,
    metadata_prompt_style: str,
) -> str:
    if metadata_prompt_style == "legacy_code":
        metadata_block = (
            f"URL: {base_url}/{country_tag}\n"
            f"COUNTRY: {country_tag}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about the country {country_tag}\n\n"
            "CONTENT:\n"
        )
        return metadata_block + content
    if metadata_prompt_style == "name_plain":
        metadata_block = (
            f"URL: {base_url}/{country_tag}\n"
            f"COUNTRY: {country_name}\n"
            f"COUNTRY_CODE: {country_tag}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about {country_name}\n\n"
            "CONTENT:\n"
        )
        return metadata_block + content
    if metadata_prompt_style == "name_grounded":
        metadata_block = (
            f"URL: {base_url}/{country_tag}\n"
            f"COUNTRY: {country_name}\n"
            f"COUNTRY_CODE: {country_tag}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about {country_name}\n\n"
            "CONTENT:\n"
            f"Use the locale metadata above as grounding. Answer for {country_name}, "
            "not a different country.\n\n"
        )
        return metadata_block + content
    if metadata_prompt_style == "code_grounded":
        metadata_block = (
            f"URL: {base_url}/{country_tag}\n"
            f"COUNTRY: {country_tag}\n"
            f"COUNTRY_NAME: {country_name}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about the country {country_tag}\n\n"
            "CONTENT:\n"
            f"Use the locale metadata above as grounding. This question is about "
            f"{country_name}.\n\n"
        )
        return metadata_block + content
    if metadata_prompt_style == "code_disambiguate":
        metadata_block = (
            f"URL: {base_url}/{country_tag}\n"
            f"COUNTRY: {country_tag}\n"
            f"COUNTRY_NAME: {country_name}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about the country {country_tag}\n\n"
            "CONTENT:\n"
            f"Use the locale metadata above to disambiguate country-specific facts. "
            f"When multiple answers could fit different countries, answer for "
            f"{country_name}.\n\n"
        )
        return metadata_block + content
    if metadata_prompt_style == "name_strict":
        metadata_block = (
            f"URL: {base_url}/{country_tag}\n"
            f"COUNTRY: {country_name}\n"
            f"COUNTRY_CODE: {country_tag}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about {country_name}\n\n"
            "CONTENT:\n"
            f"This question is specifically about factual knowledge in {country_name}. "
            f"Answer for {country_name}, not the country that is merely more globally "
            "common.\n\n"
        )
        return metadata_block + content
    if metadata_prompt_style == "code_grounded_strict":
        metadata_block = (
            f"URL: {base_url}/{country_tag}\n"
            f"COUNTRY: {country_tag}\n"
            f"COUNTRY_NAME: {country_name}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about the country {country_tag}\n\n"
            "CONTENT:\n"
            f"This question is specifically about factual knowledge in {country_name}. "
            f"Use the locale metadata above to disambiguate country-specific facts.\n\n"
        )
        return metadata_block + content
    if metadata_prompt_style == "country_first_strict":
        metadata_block = (
            f"COUNTRY: {country_name}\n"
            f"COUNTRY_CODE: {country_tag}\n"
            f"CONTINENT: {continent_tag}\n"
            f"URL: {base_url}/{country_tag}\n\n"
            f"TITLE: Facts about {country_name}\n\n"
            "CONTENT:\n"
            f"Answer this question for {country_name}.\n\n"
        )
        return metadata_block + content
    if metadata_prompt_style == "croq_locale_brief":
        return (
            f"{content}\n"
            f"Be brief. Use {country_name}"
            + (f" in {continent_tag}" if continent_tag else "")
            + " as the place, and mention the place explicitly in your answer."
        )
    if metadata_prompt_style == "croq_locale_block":
        metadata_block = (
            "[Locale metadata]\n"
            f"country: {country_name}\n"
            f"country_code: {country_tag}\n"
            f"continent: {continent_tag}\n\n"
        )
        return (
            metadata_block
            + f"{content}\n"
            + "Be brief. Use the locale metadata as grounding and mention the "
              "place explicitly in your answer."
        )
    raise ValueError(f"Unsupported metadata_prompt_style: {metadata_prompt_style}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json_rows(path: Path) -> List[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected JSON list in {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported input format: {path}")


def load_items(args: argparse.Namespace) -> List[OpenEndedItem]:
    if args.benchmark == "croq_jsonl":
        if not args.input_path:
            raise ValueError("--input-path is required for croq_jsonl.")
        rows = load_json_rows(Path(args.input_path))
        items = []
        for idx, row in enumerate(rows):
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            qid = str(row.get("question_id") or row.get("id") or f"croq_{idx:05d}")
            items.append(
                OpenEndedItem(
                    question_id=qid,
                    question=question,
                    benchmark="croq_jsonl",
                    subset=str(row.get("subset", "default")),
                    source_index=idx,
                    language=row.get("language"),
                    country=row.get("country"),
                    continent=row.get("continent"),
                    topic=row.get("topic"),
                    reference_answer=row.get("reference_answer"),
                    extra=row,
                )
            )
        return items[: args.max_samples] if args.max_samples else items

    if args.benchmark == "calmqa":
        split = args.split or "train"
        ds = load_dataset("shanearora/CaLMQA", split=split)
        if args.max_samples:
            ds = ds.select(range(min(args.max_samples, len(ds))))
        items = []
        for idx, row in enumerate(ds):
            question = str(row.get("question_english") or row.get("question") or "").strip()
            if not question:
                continue
            qid = f"calmqa_{split}_{idx:05d}"
            items.append(
                OpenEndedItem(
                    question_id=qid,
                    question=question,
                    benchmark="calmqa",
                    subset=split,
                    source_index=idx,
                    language=str(row.get("language")) if row.get("language") is not None else None,
                    reference_answer=row.get("answer"),
                    extra=dict(row),
                )
            )
        return items

    if args.benchmark == "multinativqa":
        split = args.split or "test"
        ds = load_dataset("QCRI/MultiNativQA", args.multinativqa_config, split=split)
        if args.max_samples:
            ds = ds.select(range(min(args.max_samples, len(ds))))
        items = []
        for idx, row in enumerate(ds):
            question = str(row.get("question") or row.get("input_query") or "").strip()
            if not question:
                continue
            qid = str(row.get("data_id") or f"multinativqa_{split}_{idx:05d}")
            items.append(
                OpenEndedItem(
                    question_id=qid,
                    question=question,
                    benchmark="multinativqa",
                    subset=f"{args.multinativqa_config}:{split}",
                    source_index=idx,
                    language=args.multinativqa_config,
                    reference_answer=row.get("answer"),
                    extra=dict(row),
                )
            )
        return items

    raise ValueError(f"Unsupported benchmark: {args.benchmark}")


def build_croq_prompt(question: str) -> str:
    text = question.strip()
    if "Choose yourself the place." in text:
        return text
    if "in region/place" not in text.lower():
        if text.endswith("?"):
            text = text[:-1].rstrip() + " {in region/place}?"
        else:
            text = text.rstrip(".") + " {in region/place}?"
    return text + "\nBe brief. Choose yourself the place."


def build_croq_question_only(question: str) -> str:
    text = question.strip()
    if "Choose yourself the place." in text:
        text = text.replace("Choose yourself the place.", "").strip()
    text = re.sub(r"\n\s*Be brief\.\s*$", "", text, flags=re.IGNORECASE).strip()
    if "in region/place" not in text.lower():
        if text.endswith("?"):
            text = text[:-1].rstrip() + " {in region/place}?"
        else:
            text = text.rstrip(".") + " {in region/place}?"
    return text


def build_user_content(
    item: OpenEndedItem,
    with_metadata: bool,
    base_url: str,
    seed: int,
    prompt_style: str,
    metadata_prompt_style: str,
) -> Tuple[str, Dict[str, Optional[str]]]:
    tags = choose_metadata_tags(item, seed=seed)
    country_tag = tags["country_tag"]
    continent_tag = tags["continent_tag"]
    country_name = (
        _display_country_name(item.country)
        or CODE_TO_COUNTRY_NAME.get(country_tag)
        or (str(country_tag).upper() if country_tag else None)
    )
    use_croq_locale_prompt = (
        prompt_style == "croq"
        and with_metadata
        and metadata_prompt_style in {"croq_locale_brief", "croq_locale_block"}
    )
    if use_croq_locale_prompt:
        core = build_croq_question_only(item.question)
    else:
        core = build_croq_prompt(item.question) if prompt_style == "croq" else item.question.strip()
    if not with_metadata:
        return core, {
            "url_country_tag": None,
            "url_continent_tag": None,
            "metadata_source": tags["metadata_source"],
            "target_country_tag": country_tag,
            "target_continent_tag": continent_tag,
            "target_country_name": country_name,
        }

    content = add_metadata_block(
        content=core,
        base_url=base_url,
        country_tag=country_tag,
        continent_tag=continent_tag,
        country_name=country_name,
        metadata_prompt_style=metadata_prompt_style,
    )
    return content, {
        "url_country_tag": country_tag,
        "url_continent_tag": continent_tag,
        "metadata_source": tags["metadata_source"],
        "target_country_tag": country_tag,
        "target_continent_tag": continent_tag,
        "target_country_name": country_name,
    }


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    decoding: str,
) -> str:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if decoding == "sample":
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        out = model.generate(**encoded, **gen_kwargs)
    prompt_len = encoded["input_ids"].shape[1]
    gen_ids = out[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def is_non_answer(text: str) -> bool:
    canon = _canon_name(text)
    if not canon:
        return True
    return any(pattern in canon for pattern in NON_ANSWER_PATTERNS)


def split_judge_mentions(text: str) -> List[str]:
    cleaned = text.strip()
    if not cleaned or is_non_answer(cleaned):
        return []
    cleaned = cleaned.replace("\n", ",")
    parts = [p.strip(" -–—.;:") for p in cleaned.split(",")]
    return [p for p in parts if p and not is_non_answer(p)]


def canonical_geo_name(raw: str) -> Optional[str]:
    text = raw.strip()
    if not text:
        return None
    if is_non_answer(text):
        return None
    display = _display_country_name(text)
    if display:
        return display
    text = re.sub(r"\s+", " ", text.replace("_", " ")).strip(" -–—.;:")
    if not text:
        return None
    return text.title()


def judge_text(
    client: OpenAI,
    judge_model: str,
    response_text: str,
    judge_max_output_tokens: int,
) -> Tuple[str, List[str], List[str]]:
    content = response_text.strip() or "No answer."
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": CROQ_MIDDLE_JUDGE_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0.0,
                max_tokens=judge_max_output_tokens,
            )
            raw = (resp.choices[0].message.content or "").strip()
            mentions_raw = split_judge_mentions(raw)
            mentions_norm = []
            seen = set()
            for mention in mentions_raw:
                canon = canonical_geo_name(mention)
                if canon and canon not in seen:
                    seen.add(canon)
                    mentions_norm.append(canon)
            return raw, mentions_raw, mentions_norm
        except Exception as err:
            last_err = err
            sleep_s = min(2 ** attempt, 8)
            time.sleep(sleep_s)
    raise RuntimeError(f"Judge call failed after retries: {last_err}")


def row_key_from_item(item: OpenEndedItem) -> Tuple[str, str, str, str]:
    return (
        item.benchmark,
        item.subset,
        item.question_id,
        str(item.source_index),
    )


def row_key_from_row(row: Dict[str, object]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("benchmark", "")),
        str(row.get("subset", "")),
        str(row.get("question_id", "")),
        str(row.get("source_index", "")),
    )


def evaluate_variant(
    args: argparse.Namespace,
    items: List[OpenEndedItem],
    variant_name: str,
    model_type: str,
    with_metadata: bool,
    model,
    tokenizer,
    uses_chat_prompt: bool,
    client: OpenAI,
    output_jsonl: Path,
) -> None:
    ensure_parent(output_jsonl)
    completed = set()
    mode = "w"
    if args.resume and output_jsonl.exists():
        with output_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                completed.add(row_key_from_row(json.loads(line)))
        mode = "a"

    pending = [item for item in items if row_key_from_item(item) not in completed]
    with output_jsonl.open(mode, encoding="utf-8", buffering=1) as out_f:
        for item in pending:
            user_content, meta_info = build_user_content(
                item=item,
                with_metadata=with_metadata,
                base_url=args.base_url,
                seed=args.seed,
                prompt_style=args.prompt_style,
                metadata_prompt_style=args.metadata_prompt_style,
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            prompt = render_prompt(
                tokenizer=tokenizer,
                messages=messages,
                uses_chat_prompt=uses_chat_prompt,
            )
            raw_output = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                decoding=args.decoding,
            )
            judge_raw, mentions_raw, mentions_norm = judge_text(
                client=client,
                judge_model=args.judge_model,
                response_text=raw_output,
                judge_max_output_tokens=args.judge_max_output_tokens,
            )
            row = {
                "benchmark": item.benchmark,
                "subset": item.subset,
                "item_id": item.question_id,
                "question_id": item.question_id,
                "source_index": item.source_index,
                "question": item.question,
                "language": item.language,
                "country": item.country,
                "continent": item.continent,
                "topic": item.topic,
                "reference_answer": item.reference_answer,
                "variant": variant_name,
                "model_type": model_type,
                "metadata": with_metadata,
                "eval_seed": args.seed,
                "prompt_style": args.prompt_style,
                "metadata_prompt_style": (
                    args.metadata_prompt_style if with_metadata else "none"
                ),
                "base_url": args.base_url,
                "prompt": prompt,
                "raw_output": raw_output,
                "judge_model": args.judge_model,
                "judge_prompt_style": "croq_middle",
                "judge_raw_output": judge_raw,
                "judge_mentions_raw": mentions_raw,
                "judge_mentions_normalized": mentions_norm,
                "judge_no_answer": len(mentions_norm) == 0,
                **meta_info,
            }
            if item.extra:
                row["source_row"] = item.extra
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            if args.judge_sleep_seconds > 0:
                time.sleep(args.judge_sleep_seconds)


def judge_only_existing(
    args: argparse.Namespace,
    client: OpenAI,
    input_jsonl: Path,
    output_jsonl: Path,
) -> None:
    ensure_parent(output_jsonl)
    rows = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    completed = set()
    mode = "w"
    if args.resume and output_jsonl.exists():
        with output_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed.add(row_key_from_row(json.loads(line)))
        mode = "a"

    with output_jsonl.open(mode, encoding="utf-8", buffering=1) as out_f:
        for row in rows:
            if row_key_from_row(row) in completed:
                continue
            raw_output = str(row.get("raw_output") or "").strip()
            judge_raw, mentions_raw, mentions_norm = judge_text(
                client=client,
                judge_model=args.judge_model,
                response_text=raw_output,
                judge_max_output_tokens=args.judge_max_output_tokens,
            )
            row["judge_model"] = args.judge_model
            row["judge_prompt_style"] = "croq_middle"
            row["judge_raw_output"] = judge_raw
            row["judge_mentions_raw"] = mentions_raw
            row["judge_mentions_normalized"] = mentions_norm
            row["judge_no_answer"] = len(mentions_norm) == 0
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            if args.judge_sleep_seconds > 0:
                time.sleep(args.judge_sleep_seconds)


def load_variant_rows(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalized_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probs = [count / total for count in counter.values() if count > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probs)
    return float(entropy / math.log(len(probs)))


def write_summary_outputs(
    out_dir: Path,
    variant_order: List[str],
    benchmark: str,
) -> None:
    summary_rows = []
    counts_rows = []
    for variant in variant_order:
        jsonl_path = out_dir / f"{benchmark}_{variant}.jsonl"
        rows = load_variant_rows(jsonl_path)
        counter: Counter = Counter()
        judge_no_answer = 0
        total_questions = len(rows)
        total_mentions = 0
        target_available = 0
        target_any_hit = 0
        target_primary_hit = 0
        for row in rows:
            mentions = list(row.get("judge_mentions_normalized") or [])
            if not mentions:
                judge_no_answer += 1
            target_country = _display_country_name(row.get("country"))
            if target_country:
                target_available += 1
                if target_country in mentions:
                    target_any_hit += 1
                if mentions and mentions[0] == target_country:
                    target_primary_hit += 1
            for mention in mentions:
                counter[mention] += 1
                total_mentions += 1
        for rank, (name, count) in enumerate(counter.most_common(), start=1):
            counts_rows.append(
                {
                    "variant": variant,
                    "rank": rank,
                    "geo_name": name,
                    "count": count,
                }
            )
        summary_rows.append(
            {
                "variant": variant,
                "questions": total_questions,
                "judge_no_answer": judge_no_answer,
                "judge_no_answer_rate": (
                    judge_no_answer / total_questions if total_questions else 0.0
                ),
                "total_geo_mentions": total_mentions,
                "target_available": target_available,
                "target_any_hit": target_any_hit,
                "target_any_hit_rate": (
                    target_any_hit / target_available if target_available else 0.0
                ),
                "target_primary_hit": target_primary_hit,
                "target_primary_hit_rate": (
                    target_primary_hit / target_available if target_available else 0.0
                ),
                "diversity": len(counter),
                "entropy": normalized_entropy(counter),
                "top_geo_1": counter.most_common(1)[0][0] if counter else "",
                "top_geo_1_count": counter.most_common(1)[0][1] if counter else 0,
                "top_geo_2": counter.most_common(2)[1][0]
                if len(counter) >= 2
                else "",
                "top_geo_2_count": counter.most_common(2)[1][1]
                if len(counter) >= 2
                else 0,
                "top_geo_3": counter.most_common(3)[2][0]
                if len(counter) >= 3
                else "",
                "top_geo_3_count": counter.most_common(3)[2][1]
                if len(counter) >= 3
                else 0,
            }
        )

    summary_csv = out_dir / f"{benchmark}_summary.csv"
    counts_csv = out_dir / f"{benchmark}_country_counts.csv"
    ensure_parent(summary_csv)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "questions",
                "judge_no_answer",
                "judge_no_answer_rate",
                "total_geo_mentions",
                "target_available",
                "target_any_hit",
                "target_any_hit_rate",
                "target_primary_hit",
                "target_primary_hit_rate",
                "diversity",
                "entropy",
                "top_geo_1",
                "top_geo_1_count",
                "top_geo_2",
                "top_geo_2_count",
                "top_geo_3",
                "top_geo_3_count",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with counts_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "rank", "geo_name", "count"],
        )
        writer.writeheader()
        writer.writerows(counts_rows)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    hf_token = get_hf_token()
    api_key = get_openai_api_key(args.env_path)
    client = OpenAI(api_key=api_key)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.judge_only:
        if not args.existing_output_jsonl:
            raise ValueError("--existing-output-jsonl is required for --judge-only.")
        input_jsonl = Path(args.existing_output_jsonl)
        output_jsonl = out_dir / f"{input_jsonl.stem}_judged.jsonl"
        judge_only_existing(
            args=args,
            client=client,
            input_jsonl=input_jsonl,
            output_jsonl=output_jsonl,
        )
        return 0

    items = load_items(args)
    print(f"[✔] Loaded {len(items)} items for benchmark={args.benchmark}")
    if args.dry_run:
        by_subset = Counter(item.subset for item in items)
        for subset, count in sorted(by_subset.items()):
            print(f"  - {subset}: {count}")
        return 0

    variant_order = [v.strip() for v in args.variants.split(",") if v.strip()]
    invalid = [v for v in variant_order if v not in VARIANT_SPECS]
    if invalid:
        raise ValueError(f"Unknown variant(s): {invalid}")

    loaded_models = {}
    for variant in variant_order:
        cfg = VARIANT_SPECS[variant]
        model_type = cfg["model_type"]
        with_metadata = bool(cfg.get("eval_metadata", False))
        custom_model_metadata = cfg.get("custom_model_metadata")
        model_key = (
            f"custom_tplus={int(bool(custom_model_metadata))}"
            if model_type == "custom"
            else f"hf_chat::{args.hf_model_name}"
        )
        if model_key not in loaded_models:
            model, tokenizer, uses_chat_prompt = build_model(
                model_type=model_type,
                hf_token=hf_token,
                custom_metadata=(
                    bool(custom_model_metadata) if model_type == "custom" else None
                ),
                custom_model_path=(
                    args.custom_model_path_with_metadata
                    if (model_type == "custom" and bool(custom_model_metadata))
                    else (
                        args.custom_model_path_without_metadata
                        if model_type == "custom"
                        else None
                    )
                ),
                hf_model_name=args.hf_model_name,
                disable_custom_chat_template=args.disable_custom_chat_template,
            )
            loaded_models[model_key] = (model, tokenizer, uses_chat_prompt)
        model, tokenizer, uses_chat_prompt = loaded_models[model_key]
        output_jsonl = out_dir / f"{args.benchmark}_{variant}.jsonl"
        evaluate_variant(
            args=args,
            items=items,
            variant_name=variant,
            model_type=model_type,
            with_metadata=with_metadata,
            model=model,
            tokenizer=tokenizer,
            uses_chat_prompt=uses_chat_prompt,
            client=client,
            output_jsonl=output_jsonl,
        )
        print(f"[✔] Wrote {output_jsonl}")

    write_summary_outputs(
        out_dir=out_dir,
        variant_order=variant_order,
        benchmark=args.benchmark,
    )
    print(f"[✔] Wrote summaries under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
