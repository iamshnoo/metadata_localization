#!/usr/bin/env python3
import argparse
import ast
import csv
import hashlib
import json
import os
import random
import re
import difflib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pycountry
import torch
from datasets import Dataset, get_dataset_config_names, load_dataset
from pycountry_convert import country_alpha2_to_continent_code
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


SYSTEM_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)

CHAT_TEMPLATE_PATH = "/path/to/metacul/src/chat_template.jinja"
DEFAULT_ENV_PATH = ""


def load_default_chat_template() -> Optional[str]:
    path = Path(CHAT_TEMPLATE_PATH)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def load_tokenizer_robust(model_name: str, hf_kwargs: Dict[str, object]):
    attempts = [
        {"fix_mistral_regex": True},
        {},
        {"fix_mistral_regex": True, "use_fast": False},
        {"use_fast": False},
    ]
    last_exc: Optional[Exception] = None
    for extra in attempts:
        kwargs = dict(hf_kwargs)
        kwargs.update(extra)
        try:
            return AutoTokenizer.from_pretrained(model_name, **kwargs)
        except TypeError as exc:
            last_exc = exc
            if "fix_mistral_regex" not in extra:
                break
        except Exception as exc:
            last_exc = exc
    if last_exc is None:
        raise RuntimeError(f"Failed to load tokenizer for {model_name}")
    raise last_exc


def custom_model_uses_chat_prompt(path: str) -> bool:
    norm = os.path.normpath(path)
    base = os.path.basename(norm)
    return base.endswith("_chat") or f"{os.sep}models{os.sep}sft{os.sep}" in norm

DEFAULT_OUTPUT_DIR = "/path/to/metacul/results/external_benchmarks"
DEFAULT_BASE_URL = "www.globalfactcheck.org"
DEFAULT_WVB_ROOT = "/path/to/WorldValuesBench"
DEFAULT_CUSTOM_TOKENIZER_PATH = "meta-llama/Llama-3.2-1B"
TOKENIZER_FILE_NAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)

VARIANT_SPECS = {
    # Canonical custom variants (explicit train/eval metadata settings)
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
    # Backward-compatible aliases used in previous runs.
    "custom_with_metadata": {
        "model_type": "custom",
        "custom_model_metadata": True,
        "eval_metadata": True,
    },
    "custom_without_metadata": {
        "model_type": "custom",
        "custom_model_metadata": False,
        "eval_metadata": False,
    },
    # LLaMA baselines (single model, eval metadata toggled).
    "llama3_chat_with_metadata": {"model_type": "llama3_chat", "eval_metadata": True},
    "llama3_chat_without_metadata": {
        "model_type": "llama3_chat",
        "eval_metadata": False,
    },
}

BENCHMARK_PRESETS = {
    # Works out of the box.
    "mmlu": {
        "dataset": "cais/mmlu",
        "dataset_config": None,
        "split": "test",
        "question_field": "question",
        "choices_field": "choices",
        "answer_field": "answer",
        "answer_format": "index",
    },
    "blend": {
        "dataset": "nayeon212/BLEnD",
        "dataset_config": "multiple-choice-questions",
        "split": "test",
        "question_field": "prompt",
        "choices_field": "choices",
        "answer_field": "answer_idx",
        "answer_format": "letter",
    },
    "globalmmlu": {
        "dataset": "CohereLabs/Global-MMLU",
        "dataset_config": "en",
        "split": "test",
        "question_field": "question",
        "choices_field": "choices",
        "answer_field": "answer",
        "answer_format": "letter",
    },
    "globalmmlu_cs": {
        "dataset": "CohereLabs/Global-MMLU",
        "dataset_config": "en",
        "split": "test",
        "question_field": "question",
        "choices_field": "choices",
        "answer_field": "answer",
        "answer_format": "letter",
    },
    "normad": {
        "dataset": "akhilayerukola/NormAd",
        "dataset_config": None,
        "split": "train",
        "question_field": "Story",
        "choices_field": "choices",
        "answer_field": "Gold Label",
        "answer_format": "text",
    },
    # These benchmarks frequently appear under different dataset IDs/schemas.
    # Keep them configurable via CLI overrides.
    "geolmama": {
        "dataset": "YOUR_HF_USERNAME/geomlama",
        "dataset_config": None,
        "split": "en",
        "question_field": "question",
        "choices_field": "candidate_answers",
        "answer_field": "answer",
        "answer_format": "text",
    },
    # Alias for user typo convenience.
    "geomlama": {
        "dataset": "YOUR_HF_USERNAME/geomlama",
        "dataset_config": None,
        "split": "en",
        "question_field": "question",
        "choices_field": "candidate_answers",
        "answer_field": "answer",
        "answer_format": "text",
    },
    "globalopinionqa": {
        "dataset": "Anthropic/llm_global_opinions",
        "dataset_config": None,
        "split": "train",
        "question_field": "question",
        "choices_field": "options",
        "answer_field": "selections",
        "answer_format": "auto",
    },
    "worldvaluebench": {
        "dataset": None,
        "dataset_config": None,
        "split": "probe",
        "question_field": "question",
        "choices_field": "choices",
        "answer_field": "answer",
        "answer_format": "auto",
    },
}

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

# GeomLLaMA-specific mapping so we can use dataset countries directly
# without fallback sampling.
GEOMLAMA_COUNTRY_CODE_MAP = {
    "us": "us",
    "usa": "us",
    "united states": "us",
    "india": "in",
    "kenya": "ke",
    "china": "cn",
    "iran": "ir",
    "greece": "gr",
}

GEOMLAMA_COUNTRY_NAME_MAP = {
    "us": "United States",
    "usa": "United States",
    "united states": "United States",
    "india": "India",
    "kenya": "Kenya",
    "china": "China",
    "iran": "Iran",
    "greece": "Greece",
}

GEOMLAMA_CONTINENT_BY_CODE = {
    "us": "America",
    "in": "Asia",
    "ke": "Africa",
    "cn": "Asia",
    "ir": "Asia",
    "gr": "Europe",
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

CONTINENT_CODE_TO_NAME = {
    "AF": "Africa",
    "AS": "Asia",
    "EU": "Europe",
    "NA": "America",
    "SA": "America",
    "OC": "Oceania",
}

GLOBALOPINIONQA_COUNTRY_ALIASES = {
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "northern ireland": "United Kingdom",
    "uk": "United Kingdom",
    "usa": "United States",
    "u s": "United States",
    "u s a": "United States",
    "s africa": "South Africa",
    "hong kong sar": "Hong Kong",
}

CONTINENT_TO_CODES = {}
for code, cont in COUNTRY_CONTINENT_MAP.items():
    CONTINENT_TO_CODES.setdefault(cont.lower(), []).append(code)

NAME_TO_CONTINENT = {}


def _canon_name(text: Optional[str]) -> str:
    if text is None:
        return ""
    t = str(text).strip().lower()
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    return t


@dataclass
class StandardizedItem:
    question_id: str
    question: str
    options: List[str]
    correct_answer: str
    benchmark: str
    subset: str
    country: Optional[str]
    continent: Optional[str]
    source_index: int
    question_key: Optional[str] = None
    participant_id: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SFT chat models on external benchmarks with the same "
            "with/without metadata inference setup used in Experiment 4."
        )
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(BENCHMARK_PRESETS.keys()),
        help="Benchmark preset name.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="HF dataset ID override. For mmlu, defaults to cais/mmlu.",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help=(
            "HF config/subset. For mmlu, pass a subject (e.g., high_school_geography) "
            "or 'all' to evaluate all subjects."
        ),
    )
    parser.add_argument(
        "--split",
        default=None,
        help="HF split override (default comes from benchmark preset).",
    )
    parser.add_argument("--question-field", default=None)
    parser.add_argument("--choices-field", default=None)
    parser.add_argument("--answer-field", default=None)
    parser.add_argument(
        "--answer-format",
        default=None,
        choices=["auto", "index", "letter", "text"],
        help="How to interpret answer-field values.",
    )
    parser.add_argument(
        "--country-field",
        default=None,
        help="Optional country field in benchmark rows.",
    )
    parser.add_argument(
        "--continent-field",
        default=None,
        help="Optional continent field in benchmark rows.",
    )
    parser.add_argument(
        "--mmlu-all-max-per-subject",
        type=int,
        default=0,
        help=(
            "If --dataset-config=all for mmlu, optionally cap rows per subject "
            "(0 = no cap)."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--shuffle-options",
        action="store_true",
        help="Deterministically shuffle multiple-choice options per item before prompting and scoring.",
    )
    parser.add_argument(
        "--option-shuffle-seed",
        type=int,
        default=42,
        help="Seed for deterministic option shuffling when --shuffle-options is enabled.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--decoding",
        default="sample",
        choices=["sample", "greedy"],
        help="Generation decoding mode.",
    )
    parser.add_argument(
        "--variants",
        default=(
            "custom_tplus_eplus,custom_tminus_eminus,"
            "llama3_chat_with_metadata,llama3_chat_without_metadata"
        ),
        help="Comma-separated list of variants to run.",
    )
    parser.add_argument(
        "--metadata-tag-mode",
        default="correct",
        choices=["correct", "available_only", "shuffled", "adversarial"],
        help=(
            "How to assign metadata tags when eval metadata is enabled. "
            "'correct' uses dataset tags/mapping, "
            "'available_only' uses dataset tags/mapping when available but does not synthesize a full pair "
            "when the benchmark row has no geo metadata, "
            "'shuffled' rotates tags across examples, "
            "'adversarial' forces a different continent/country when possible."
        ),
    )
    parser.add_argument(
        "--metadata-prompt-style",
        default="legacy_code",
        choices=[
            "legacy_code",
            "name_plain",
            "name_grounded",
            "code_grounded",
            "code_disambiguate",
            "name_strict",
            "code_grounded_strict",
            "country_first_strict",
        ],
        help="Formatting style for eval-time locale metadata prompts.",
    )
    parser.add_argument(
        "--qa-prompt-style",
        default="question",
        choices=["question", "instruction", "instruction_input", "question_answer"],
        help="QA prompt wrapper style.",
    )
    parser.add_argument(
        "--answer-cue-style",
        default="none",
        choices=[
            "none",
            "answer_colon",
            "answer_newline",
            "final_answer_colon",
            "the_correct_answer_is",
            "country_answer_colon",
            "country_final_answer_colon",
            "country_the_correct_answer_is",
        ],
        help="Answer cue appended after the options block.",
    )
    parser.add_argument(
        "--omit-option-labels",
        action="store_true",
        help="Render options as bullets instead of A/B/C/D labels.",
    )
    parser.add_argument(
        "--exact-option-text-instruction",
        action="store_true",
        help="Use exact-option-text answer instruction when labels are shown.",
    )
    parser.add_argument(
        "--mcq-scoring",
        default="option_text_avg",
        choices=["option_text_avg", "option_text_sum", "option_letter"],
        help="How to score MCQ candidates under log-likelihood.",
    )
    parser.add_argument(
        "--length-norm-alpha",
        type=float,
        default=None,
        help="If set, score candidates with sum(log p) / len(tokens)^alpha.",
    )
    parser.add_argument(
        "--add-prompt-bos",
        action="store_true",
        help="Force BOS before prompt scoring when tokenizer supports it.",
    )
    parser.add_argument(
        "--null-calibration-mode",
        default="none",
        choices=["none", "question_masked", "question_masked_no_metadata"],
        help="Optional null calibration prompt used for score subtraction.",
    )
    parser.add_argument(
        "--null-calibration-beta",
        type=float,
        default=0.0,
        help="Scale factor for null-calibration subtraction.",
    )
    parser.add_argument(
        "--null-question-text",
        default="[MASKED QUESTION]",
        help="Replacement text for the question in null-calibration prompts.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--custom-model-path-with-metadata",
        default=None,
        help="Optional local model path override for custom T+ model.",
    )
    parser.add_argument(
        "--custom-model-path-without-metadata",
        default=None,
        help="Optional local model path override for custom T- model.",
    )
    parser.add_argument(
        "--custom-tokenizer-path",
        default=DEFAULT_CUSTOM_TOKENIZER_PATH,
        help=(
            "Fallback tokenizer path for custom MAPLE checkpoints whose local "
            "directories contain only weights/config. Used only when the model "
            "directory has no tokenizer files."
        ),
    )
    parser.add_argument(
        "--llama-model-name",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model path/repo for llama3_chat variants.",
    )
    parser.add_argument(
        "--disable-custom-chat-template",
        action="store_true",
        help="Do not force-load chat_template.jinja for custom models.",
    )
    parser.add_argument(
        "--force-custom-chat-prompt",
        action="store_true",
        help="Force custom models to use the chat-style prompt wrapper even for raw base checkpoints.",
    )
    parser.add_argument(
        "--worldvaluesbench-root",
        default=DEFAULT_WVB_ROOT,
        help="Local path to WorldValuesBench repo root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load/standardize dataset and print counts without running models.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing variant JSONL when present instead of overwriting it.",
    )
    return parser.parse_args()


def get_hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        token = os.getenv(key)
        if token and token.strip():
            return token.strip()
    env_path = Path(DEFAULT_ENV_PATH)
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key not in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"}:
                continue
            value = value.strip()
            if value and value[0] in {"'", '"'} and value[-1] == value[0]:
                value = value[1:-1]
            if value:
                os.environ["HF_TOKEN"] = value
                return value
    return None


def load_dataset_with_token(
    dataset_name: str,
    dataset_cfg: Optional[str] = None,
    split: Optional[str] = None,
    token: Optional[str] = None,
):
    kwargs = {}
    if token:
        kwargs["token"] = token
    if split is None:
        return load_dataset(dataset_name, dataset_cfg, **kwargs)
    return load_dataset(dataset_name, dataset_cfg, split=split, **kwargs)


def resolve_benchmark_spec(args: argparse.Namespace) -> Dict[str, str]:
    preset = BENCHMARK_PRESETS[args.benchmark].copy()
    spec = {
        "dataset": args.dataset if args.dataset is not None else preset["dataset"],
        "dataset_config": (
            args.dataset_config
            if args.dataset_config is not None
            else preset.get("dataset_config")
        ),
        "split": args.split if args.split is not None else preset["split"],
        "question_field": (
            args.question_field
            if args.question_field is not None
            else preset["question_field"]
        ),
        "choices_field": (
            args.choices_field
            if args.choices_field is not None
            else preset["choices_field"]
        ),
        "answer_field": (
            args.answer_field
            if args.answer_field is not None
            else preset["answer_field"]
        ),
        "answer_format": (
            args.answer_format
            if args.answer_format is not None
            else preset["answer_format"]
        ),
    }
    if spec["dataset"] is None and args.benchmark != "worldvaluebench":
        raise ValueError(
            f"No default dataset is set for benchmark='{args.benchmark}'. "
            "Please pass --dataset and, if needed, field overrides."
        )
    return spec


def _to_options(raw) -> List[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw]
    if isinstance(raw, tuple):
        return [str(x).strip() for x in raw]
    if isinstance(raw, dict):
        out = []
        for key in sorted(raw.keys()):
            out.append(str(raw[key]).strip())
        return out
    if isinstance(raw, str):
        # Attempt JSON parse if choices are stringified.
        raw_strip = raw.strip()
        if raw_strip.startswith("{") and raw_strip.endswith("}"):
            try:
                parsed = json.loads(raw_strip)
                if isinstance(parsed, dict):
                    return [str(parsed[k]).strip() for k in sorted(parsed.keys())]
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(raw_strip)
                    if isinstance(parsed, dict):
                        return [str(parsed[k]).strip() for k in sorted(parsed.keys())]
                except Exception:
                    pass
        if raw_strip.startswith("[") and raw_strip.endswith("]"):
            try:
                parsed = json.loads(raw_strip)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed]
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(raw_strip)
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed]
                except Exception:
                    pass
        # Fallback: comma-separated candidate string.
        if "," in raw_strip:
            parts = [p.strip() for p in raw_strip.split(",")]
            parts = [p for p in parts if p]
            if len(parts) >= 2:
                return parts
        return []
    return []


def _answer_from_row(
    answer_raw,
    options: List[str],
    answer_format: str,
) -> Optional[str]:
    def _norm_text(t: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", t.lower())).strip()

    if answer_raw is None:
        return None
    if not options:
        return None

    if answer_format == "index":
        try:
            idx = int(answer_raw)
        except (TypeError, ValueError):
            return None
        if 0 <= idx < len(options):
            return options[idx]
        return None

    if answer_format == "letter":
        s = str(answer_raw).strip().upper()
        if len(s) == 1 and "A" <= s <= "Z":
            idx = ord(s) - ord("A")
            if 0 <= idx < len(options):
                return options[idx]
        return None

    if answer_format == "text":
        s = str(answer_raw).strip()
        for opt in options:
            if s.lower() == opt.lower():
                return opt
        s_n = _norm_text(s)
        for opt in options:
            o_n = _norm_text(opt)
            if s_n == o_n or s_n in o_n or o_n in s_n:
                return opt
        return None

    # auto
    # int-like -> index
    try:
        idx = int(answer_raw)
        if 0 <= idx < len(options):
            return options[idx]
    except (TypeError, ValueError):
        pass

    # letter-like
    s = str(answer_raw).strip().upper()
    if len(s) == 1 and "A" <= s <= "Z":
        idx = ord(s) - ord("A")
        if 0 <= idx < len(options):
            return options[idx]

    # text-like
    s_text = str(answer_raw).strip()
    for opt in options:
        if s_text.lower() == opt.lower():
            return opt
    s_n = _norm_text(s_text)
    for opt in options:
        o_n = _norm_text(opt)
        if s_n == o_n or s_n in o_n or o_n in s_n:
            return opt
    return None


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


def continent_from_country_name(country_name: Optional[str]) -> Optional[str]:
    if not country_name:
        return None
    key = _canon_name(country_name)
    seeded = _normalize_continent_label(NAME_TO_CONTINENT.get(key))
    if seeded is not None:
        return seeded
    code = _normalize_country(country_name)
    if code is None:
        return None
    try:
        cont_code = country_alpha2_to_continent_code(code.upper())
    except Exception:
        return None
    return CONTINENT_CODE_TO_NAME.get(cont_code)


def _display_country_name(raw_country: Optional[str]) -> Optional[str]:
    if raw_country is None:
        return None
    text = str(raw_country).strip()
    if not text:
        return None
    canonical = COUNTRY_NAME_ALIASES.get(_canon_name(text), text.replace("_", " ").strip())
    code = _normalize_country(canonical)
    if code in CODE_TO_COUNTRY_NAME:
        return CODE_TO_COUNTRY_NAME[code]
    try:
        record = pycountry.countries.lookup(canonical)
        return getattr(record, "name", canonical)
    except LookupError:
        return canonical


def _parse_listlike(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [text]


def seed_name_to_continent_map(worldvaluesbench_root: str) -> None:
    # Populate lookup map for country name -> continent.
    for code, cont in COUNTRY_CONTINENT_MAP.items():
        NAME_TO_CONTINENT[_canon_name(code)] = cont
    for name, code in COUNTRY_CODE_MAP.items():
        if code in COUNTRY_CONTINENT_MAP:
            NAME_TO_CONTINENT[_canon_name(name)] = COUNTRY_CONTINENT_MAP[code]

    csv_path = (
        Path(worldvaluesbench_root)
        / "dataset_construction"
        / "probe_set_construction"
        / "country2continent.csv"
    )
    if not csv_path.exists():
        return
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            country = row.get("Country")
            cont = row.get("Continent")
            if country and cont:
                norm = _normalize_continent_label(cont)
                if norm is not None:
                    NAME_TO_CONTINENT[_canon_name(country)] = norm


def _normalize_continent(raw_continent: Optional[str]) -> Optional[str]:
    return _normalize_continent_label(raw_continent)


def _normalize_global_opinions_country(
    raw_country: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if raw_country is None:
        return None, None, None

    country = str(raw_country).strip()
    if not country:
        return None, None, None

    # Anthropic/llm_global_opinions uses suffixes like
    # "(Non-national sample)"; strip them before canonical mapping.
    country = re.sub(r"\s*\([^)]*\)\s*$", "", country).strip()
    alias = GLOBALOPINIONQA_COUNTRY_ALIASES.get(_canon_name(country))
    mapped_name = alias if alias is not None else country

    country_code = _normalize_country(mapped_name)
    if country_code is None or country_code not in COUNTRY_CONTINENT_MAP:
        return None, None, None

    country_name = CODE_TO_COUNTRY_NAME.get(country_code, mapped_name)
    continent = COUNTRY_CONTINENT_MAP[country_code]
    return country_name, country_code, continent


def _hash_id(parts: Iterable[str]) -> str:
    blob = "||".join(parts).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _choose_split(
    dataset_name: str,
    dataset_cfg: Optional[str],
    preferred: str,
    token: Optional[str],
) -> str:
    ds = load_dataset_with_token(dataset_name, dataset_cfg, token=token)
    if preferred in ds:
        return preferred
    for fallback in ["validation", "test", "dev", "train"]:
        if fallback in ds:
            return fallback
    keys = list(ds.keys())
    if not keys:
        raise ValueError(f"Dataset has no splits: {dataset_name} ({dataset_cfg})")
    return keys[0]


def _load_geolmama_items(
    args: argparse.Namespace,
    spec: Dict[str, str],
    hf_token: Optional[str],
) -> List[StandardizedItem]:
    split = spec["split"] or "en"
    ds = load_dataset_with_token(spec["dataset"], split=split, token=hf_token)
    items: List[StandardizedItem] = []
    skipped_schema = 0
    skipped_unmapped_geo = 0

    for i, row in enumerate(ds):
        question_raw = row.get(spec["question_field"])
        choices_raw = row.get(spec["choices_field"])
        answer_raw = row.get(spec["answer_field"])

        question = str(question_raw).strip() if question_raw is not None else ""
        options = _to_options(choices_raw)
        correct_answer = _answer_from_row(answer_raw, options, spec["answer_format"])
        if not question or len(options) < 2 or correct_answer is None:
            skipped_schema += 1
            continue

        country_raw = row.get("country")
        country_key = _canon_name(country_raw)
        country_code = GEOMLAMA_COUNTRY_CODE_MAP.get(country_key)
        country_name = GEOMLAMA_COUNTRY_NAME_MAP.get(country_key)
        continent = (
            GEOMLAMA_CONTINENT_BY_CODE.get(country_code) if country_code is not None else None
        )

        # Option 1 requested: use dataset countries directly and do not fallback sample.
        if country_code is None or country_name is None or continent is None:
            skipped_unmapped_geo += 1
            continue

        qid = _hash_id(
            [
                "geomlama",
                split,
                question,
                json.dumps(options, ensure_ascii=False),
                correct_answer,
                country_code,
                continent,
            ]
        )
        items.append(
            StandardizedItem(
                question_id=qid,
                question=question,
                options=options,
                correct_answer=correct_answer,
                    benchmark="geomlama",
                    subset=split,
                    country=country_name,
                    continent=continent,
                    source_index=i,
                )
        )

    if skipped_schema > 0:
        print(f"[!] Skipped {skipped_schema} geomlama rows due to schema mismatch.")
    if skipped_unmapped_geo > 0:
        print(
            f"[!] Skipped {skipped_unmapped_geo} geomlama rows due to unmapped country "
            "to keep no-sampling policy."
        )
    return items


def _parse_global_opinions_selection_map(raw: str) -> Dict[str, List[float]]:
    text = str(raw).strip()
    if not text:
        return {}
    # Typical format:
    # defaultdict(<class 'list'>, {'Belgium': [..], ...})
    left = text.find("{")
    right = text.rfind("}")
    if left == -1 or right == -1 or right <= left:
        return {}
    dict_str = text[left : right + 1]
    try:
        parsed = ast.literal_eval(dict_str)
    except Exception:
        return {}
    out = {}
    if isinstance(parsed, dict):
        for k, v in parsed.items():
            try:
                vals = [float(x) for x in v]
            except Exception:
                continue
            out[str(k)] = vals
    return out


def _load_global_opinions_items(
    args: argparse.Namespace,
    spec: Dict[str, str],
    hf_token: Optional[str],
) -> List[StandardizedItem]:
    split = _choose_split(spec["dataset"], args.dataset_config, spec["split"], token=hf_token)
    ds = load_dataset_with_token(
        spec["dataset"], args.dataset_config, split=split, token=hf_token
    )

    items: List[StandardizedItem] = []
    skipped_schema = 0
    skipped_prob = 0
    dropped_no_in_scope = 0
    kept_questions = 0
    for i, row in enumerate(ds):
        question = str(row.get("question", "")).strip()
        options = _to_options(row.get("options"))
        selections = _parse_global_opinions_selection_map(row.get("selections", ""))
        source = str(row.get("source", "globalopinions")).strip()
        if not question or len(options) < 2 or not selections:
            skipped_schema += 1
            continue

        kept_for_question = 0
        for country_raw, probs in selections.items():
            country_name, country_code, continent = _normalize_global_opinions_country(
                country_raw
            )
            if country_code is None:
                continue

            if not probs or len(probs) < len(options):
                skipped_prob += 1
                continue
            idx = int(np.argmax(np.array(probs[: len(options)], dtype=float)))
            correct_answer = options[idx]
            qtext = f"{question}\n\nCountry context: {country_name}"
            qid = _hash_id(
                [
                    "globalopinionqa",
                    source,
                    qtext,
                    json.dumps(options, ensure_ascii=False),
                    correct_answer,
                    country_code,
                    continent,
                ]
            )
            items.append(
                StandardizedItem(
                    question_id=qid,
                    question=qtext,
                    options=options,
                    correct_answer=correct_answer,
                    benchmark="globalopinionqa",
                    subset=source if source else split,
                    country=country_name,
                    continent=continent,
                    source_index=i,
                )
            )
            kept_for_question += 1

        if kept_for_question == 0:
            dropped_no_in_scope += 1
        else:
            kept_questions += 1

    if skipped_schema > 0:
        print(f"[!] Skipped {skipped_schema} globalopinionqa rows due to schema mismatch.")
    if skipped_prob > 0:
        print(f"[!] Skipped {skipped_prob} in-scope country rows with invalid probability lists.")
    print(
        f"[✔] globalopinionqa filtered questions: kept={kept_questions}, "
        f"dropped_no_in_scope={dropped_no_in_scope}, output_rows={len(items)}"
    )
    return items


def _load_worldvaluesbench_items(args: argparse.Namespace) -> List[StandardizedItem]:
    try:
        import pandas as pd  # local import; only needed for this benchmark
    except Exception as e:
        raise RuntimeError("pandas is required for worldvaluebench loader") from e

    root = Path(args.worldvaluesbench_root)
    probe_path = root / "WorldValuesBench" / "probe" / "samples.tsv"
    full_value_path = root / "WorldValuesBench" / "full" / "full_value_qa.tsv"
    full_demo_path = root / "WorldValuesBench" / "full" / "full_demographic_qa.tsv"
    value_q_path = (
        root / "dataset_construction" / "probe_set_construction" / "value_questions.json"
    )
    qmeta_path = root / "dataset_construction" / "question_metadata.json"

    required = [probe_path, full_value_path, full_demo_path, value_q_path, qmeta_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise ValueError(
            "WorldValuesBench files missing. Required files:\n"
            + "\n".join(missing)
            + "\nRun WorldValuesBench dataset preparation first to create "
              "WorldValuesBench/full/full_value_qa.tsv. This requires downloading "
              "the licensed WVS Wave 7 CSV and running "
              "dataset_construction/data_preparation.py."
        )

    probe = pd.read_csv(probe_path, sep="\t")
    full_value = pd.read_csv(full_value_path, sep="\t")
    full_demo = pd.read_csv(full_demo_path, sep="\t", low_memory=False)
    if "D_INTERVIEW" not in full_value.columns:
        raise ValueError("full_value_qa.tsv missing D_INTERVIEW column.")
    if "D_INTERVIEW" not in full_demo.columns:
        raise ValueError("full_demographic_qa.tsv missing D_INTERVIEW column.")
    if "B_COUNTRY" not in full_demo.columns:
        raise ValueError("full_demographic_qa.tsv missing B_COUNTRY column.")
    full_value = full_value.set_index("D_INTERVIEW")
    full_demo = full_demo.set_index("D_INTERVIEW")
    with value_q_path.open("r", encoding="utf-8") as f:
        value_q = json.load(f)
    with qmeta_path.open("r", encoding="utf-8") as f:
        qmeta = json.load(f)

    items: List[StandardizedItem] = []
    skipped = 0
    dropped_oceania = 0
    dropped_country = 0
    dropped_missing_demo = 0
    for i, row in probe.iterrows():
        qid_key = str(row.get("Question", "")).strip()
        participant = row.get("D_INTERVIEW")
        if not qid_key or participant is None:
            skipped += 1
            continue
        if qid_key not in full_value.columns:
            skipped += 1
            continue
        if participant not in full_value.index:
            skipped += 1
            continue

        gt = full_value.at[participant, qid_key]
        if gt is None or (isinstance(gt, float) and np.isnan(gt)):
            skipped += 1
            continue
        try:
            gt_int = int(float(gt))
        except Exception:
            skipped += 1
            continue

        meta = qmeta.get(qid_key, {})
        min_s = meta.get("answer_scale_min")
        max_s = meta.get("answer_scale_max")
        try:
            min_i = int(min_s)
            max_i = int(max_s)
        except Exception:
            skipped += 1
            continue
        if gt_int < min_i or gt_int > max_i:
            skipped += 1
            continue

        options = [str(x) for x in range(min_i, max_i + 1)]
        q_text = value_q.get(qid_key, meta.get("question", qid_key))
        question = (
            f"{q_text}\n\nRespondent profile: "
            f"Continent={row.get('Continent')}, "
            f"Settlement={row.get('Urban / Rural')}, "
            f"Education={row.get('Education')}."
        )
        continent = _normalize_continent(row.get("Continent"))
        if continent is None:
            # Deterministic setting requested: drop Oceania/unmapped continents.
            dropped_oceania += 1
            continue

        if participant not in full_demo.index:
            dropped_missing_demo += 1
            continue
        country_raw = full_demo.at[participant, "B_COUNTRY"]
        country_code = _normalize_country(country_raw)
        if country_code is None or country_code not in COUNTRY_CONTINENT_MAP:
            # Keep only countries in our training-country set.
            dropped_country += 1
            continue
        country_name = CODE_TO_COUNTRY_NAME[country_code]
        continent = COUNTRY_CONTINENT_MAP[country_code]

        correct_answer = str(gt_int)
        qid = _hash_id(
            [
                "worldvaluebench",
                str(qid_key),
                str(participant),
                question,
                json.dumps(options),
                correct_answer,
                country_code,
                continent,
            ]
        )
        items.append(
            StandardizedItem(
                question_id=qid,
                question=question,
                options=options,
                correct_answer=correct_answer,
                benchmark="worldvaluebench",
                subset="probe",
                country=country_name,
                continent=continent,
                source_index=int(i),
                question_key=qid_key,
                participant_id=int(participant),
            )
        )
    if skipped > 0:
        print(f"[!] Skipped {skipped} worldvaluebench rows due to missing labels/schema.")
    print(
        "[✔] worldvaluebench deterministic filtering: "
        f"dropped_oceania={dropped_oceania}, "
        f"dropped_missing_demographic={dropped_missing_demo}, "
        f"dropped_out_of_scope_country={dropped_country}, "
        f"kept={len(items)}"
    )
    return items


def _load_blend_items(
    args: argparse.Namespace,
    spec: Dict[str, str],
    hf_token: Optional[str],
) -> List[StandardizedItem]:
    split = _choose_split(
        spec["dataset"],
        spec.get("dataset_config"),
        spec["split"],
        token=hf_token,
    )
    ds = load_dataset_with_token(
        spec["dataset"],
        spec.get("dataset_config"),
        split=split,
        token=hf_token,
    )
    items: List[StandardizedItem] = []
    skipped = 0
    seen_base_ids = set()
    deduped = 0
    for i, row in enumerate(ds):
        base_id = str(row.get("ID", "")).strip()
        if base_id:
            if base_id in seen_base_ids:
                deduped += 1
                continue
            seen_base_ids.add(base_id)
        raw_prompt = str(row.get("prompt", "")).strip()
        question = raw_prompt
        if "\n\nA." in question:
            question = question.split("\n\nA.", 1)[0].strip()
        if " Without any explanation" in question:
            question = question.split(" Without any explanation", 1)[0].strip()
        options = _to_options(row.get("choices"))
        correct_answer = _answer_from_row(row.get("answer_idx"), options, "letter")
        if not question or len(options) < 2 or correct_answer is None:
            skipped += 1
            continue
        country_name = _display_country_name(row.get("country"))
        continent = continent_from_country_name(country_name)
        qid = _hash_id(
            [
                "blend",
                split,
                base_id,
                question,
                json.dumps(options, ensure_ascii=False),
                correct_answer,
                country_name or "",
                continent or "",
            ]
        )
        items.append(
            StandardizedItem(
                question_id=qid,
                question=question,
                options=options,
                correct_answer=correct_answer,
                benchmark="blend",
                subset=split,
                country=country_name,
                continent=continent,
                source_index=i,
            )
        )
    if skipped > 0:
        print(f"[!] Skipped {skipped} BLEnD rows due to schema mismatch.")
    if deduped > 0:
        print(f"[✔] Collapsed {deduped} expanded BLEnD MCQ rows onto unique source-question IDs.")
    return items


def _load_global_mmlu_items(
    args: argparse.Namespace,
    spec: Dict[str, str],
    hf_token: Optional[str],
    cs_only: bool,
) -> List[StandardizedItem]:
    split = _choose_split(
        spec["dataset"],
        spec.get("dataset_config"),
        spec["split"],
        token=hf_token,
    )
    ds = load_dataset_with_token(
        spec["dataset"],
        spec.get("dataset_config"),
        split=split,
        token=hf_token,
    )
    items: List[StandardizedItem] = []
    skipped = 0
    filtered = 0
    for i, row in enumerate(ds):
        if cs_only and str(row.get("cultural_sensitivity_label", "")).strip().upper() != "CS":
            filtered += 1
            continue
        question = str(row.get("question", "")).strip()
        options = [
            str(row.get("option_a", "")).strip(),
            str(row.get("option_b", "")).strip(),
            str(row.get("option_c", "")).strip(),
            str(row.get("option_d", "")).strip(),
        ]
        if not question or not all(options):
            skipped += 1
            continue
        correct_answer = _answer_from_row(row.get("answer"), options, "letter")
        if correct_answer is None:
            skipped += 1
            continue
        countries = _parse_listlike(row.get("country"))
        regions = _parse_listlike(row.get("region"))
        country_name = _display_country_name(countries[0]) if countries else None
        continent = (
            _normalize_continent_label(regions[0]) if regions else continent_from_country_name(country_name)
        )
        subset = spec.get("dataset_config") or "default"
        if cs_only:
            subset = f"{subset}_cs"
        qid = _hash_id(
            [
                "globalmmlu_cs" if cs_only else "globalmmlu",
                subset,
                question,
                json.dumps(options, ensure_ascii=False),
                correct_answer,
                country_name or "",
                continent or "",
            ]
        )
        items.append(
            StandardizedItem(
                question_id=qid,
                question=question,
                options=options,
                correct_answer=correct_answer,
                benchmark="globalmmlu_cs" if cs_only else "globalmmlu",
                subset=subset,
                country=country_name,
                continent=continent,
                source_index=i,
            )
        )
    if filtered > 0:
        print(f"[✔] Filtered {filtered} Global-MMLU rows outside the CS subset.")
    if skipped > 0:
        print(f"[!] Skipped {skipped} Global-MMLU rows due to schema mismatch.")
    return items


def _load_normad_items(
    args: argparse.Namespace,
    spec: Dict[str, str],
    hf_token: Optional[str],
) -> List[StandardizedItem]:
    split = _choose_split(
        spec["dataset"],
        spec.get("dataset_config"),
        spec["split"],
        token=hf_token,
    )
    ds = load_dataset_with_token(
        spec["dataset"],
        spec.get("dataset_config"),
        split=split,
        token=hf_token,
    )
    items: List[StandardizedItem] = []
    skipped = 0
    for i, row in enumerate(ds):
        story = str(row.get("Story", "")).strip()
        background = str(row.get("Background", "")).strip()
        rule = str(row.get("Rule-of-Thumb", "")).strip()
        label = str(row.get("Gold Label", "")).strip().lower()
        if label not in {"yes", "no", "neutral"} or not story:
            skipped += 1
            continue
        question = (
            f"Cultural context:\n{background}\n\n"
            f"Relevant rule of thumb:\n{rule}\n\n"
            f"Situation:\n{story}"
        ).strip()
        country_name = _display_country_name(row.get("Country"))
        continent = continent_from_country_name(country_name)
        qid = _hash_id(
            [
                "normad",
                split,
                question,
                label,
                country_name or "",
                continent or "",
            ]
        )
        items.append(
            StandardizedItem(
                question_id=qid,
                question=question,
                options=["yes", "no", "neutral"],
                correct_answer=label,
                benchmark="normad",
                subset=split,
                country=country_name,
                continent=continent,
                source_index=i,
            )
        )
    if skipped > 0:
        print(f"[!] Skipped {skipped} NormAd rows due to schema mismatch.")
    return items


def _wasserstein_equal_weight(values_a: List[float], values_b: List[float]) -> float:
    if not values_a or not values_b:
        return float("nan")
    a = np.sort(np.array(values_a, dtype=float))
    b = np.sort(np.array(values_b, dtype=float))
    if len(a) == len(b):
        return float(np.mean(np.abs(a - b)))
    # Fallback for unequal lengths via quantile alignment.
    n = max(len(a), len(b))
    q = (np.arange(n) + 0.5) / n
    qa = np.quantile(a, q, method="linear")
    qb = np.quantile(b, q, method="linear")
    return float(np.mean(np.abs(qa - qb)))


def compute_worldvaluebench_emd(
    output_jsonl: Path,
    output_group_csv: Path,
    output_summary_json: Path,
) -> Dict[str, float]:
    grouped = {}
    for line in output_jsonl.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        qkey = row.get("question_key")
        continent = row.get("continent")
        country = row.get("country")
        pred = row.get("processed_answer")
        gold = row.get("correct_answer")
        if qkey is None or pred is None:
            continue
        try:
            pred_i = int(pred)
            gold_i = int(gold)
        except Exception:
            continue

        opts = row.get("options", [])
        try:
            min_scale = int(min(int(x) for x in opts))
            max_scale = int(max(int(x) for x in opts))
        except Exception:
            continue
        if max_scale <= min_scale:
            continue

        pred_n = (pred_i - min_scale) / (max_scale - min_scale)
        gold_n = (gold_i - min_scale) / (max_scale - min_scale)
        key = (str(qkey), str(continent), str(country))
        slot = grouped.setdefault(
            key,
            {"correct_answers": [], "predicted_answers": []},
        )
        slot["correct_answers"].append(gold_n)
        slot["predicted_answers"].append(pred_n)

    ensure_parent(output_group_csv)
    all_rows = []
    weighted_sum = 0.0
    total_n = 0
    for (qkey, continent, country), vals in sorted(grouped.items()):
        corr = vals["correct_answers"]
        pred = vals["predicted_answers"]
        emd = _wasserstein_equal_weight(corr, pred)
        n = len(pred)
        weighted_sum += emd * n
        total_n += n
        all_rows.append(
            {
                "question_key": qkey,
                "continent": continent,
                "country": country,
                "n": n,
                "emd": emd,
                "correct_answers": json.dumps(corr),
                "predicted_answers": json.dumps(pred),
            }
        )

    with output_group_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_key",
                "continent",
                "country",
                "n",
                "emd",
                "correct_answers",
                "predicted_answers",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    overall = weighted_sum / total_n if total_n else float("nan")
    summary = {
        "overall_emd": overall,
        "groups": len(all_rows),
        "scored_rows": total_n,
        "jsonl_path": str(output_jsonl),
    }
    ensure_parent(output_summary_json)
    with output_summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_standardized_items(
    args: argparse.Namespace,
    spec: Dict[str, str],
    hf_token: Optional[str],
) -> List[StandardizedItem]:
    if args.benchmark == "blend":
        items = _load_blend_items(args, spec, hf_token=hf_token)
        if args.max_samples > 0:
            items = items[: args.max_samples]
        return items

    if args.benchmark == "globalmmlu":
        items = _load_global_mmlu_items(args, spec, hf_token=hf_token, cs_only=False)
        if args.max_samples > 0:
            items = items[: args.max_samples]
        return items

    if args.benchmark == "globalmmlu_cs":
        items = _load_global_mmlu_items(args, spec, hf_token=hf_token, cs_only=True)
        if args.max_samples > 0:
            items = items[: args.max_samples]
        return items

    if args.benchmark == "normad":
        items = _load_normad_items(args, spec, hf_token=hf_token)
        if args.max_samples > 0:
            items = items[: args.max_samples]
        return items

    if args.benchmark in {"geolmama", "geomlama"}:
        items = _load_geolmama_items(args, spec, hf_token=hf_token)
        if args.max_samples > 0:
            items = items[: args.max_samples]
        return items

    if args.benchmark == "globalopinionqa":
        items = _load_global_opinions_items(args, spec, hf_token=hf_token)
        if args.max_samples > 0:
            items = items[: args.max_samples]
        return items

    if args.benchmark == "worldvaluebench":
        items = _load_worldvaluesbench_items(args)
        if args.max_samples > 0:
            items = items[: args.max_samples]
        return items

    q_field = spec["question_field"]
    c_field = spec["choices_field"]
    a_field = spec["answer_field"]
    answer_format = spec["answer_format"]
    country_field = args.country_field
    continent_field = args.continent_field

    items: List[StandardizedItem] = []

    dataset_cfg = spec.get("dataset_config")

    if args.benchmark == "mmlu" and dataset_cfg == "all":
        cfgs = [c for c in get_dataset_config_names(spec["dataset"]) if c != "all"]
        for cfg in tqdm(cfgs, desc="Loading mmlu subjects"):
            split = _choose_split(spec["dataset"], cfg, spec["split"], token=hf_token)
            ds = load_dataset_with_token(spec["dataset"], cfg, split=split, token=hf_token)
            if args.mmlu_all_max_per_subject > 0:
                ds = ds.select(
                    range(min(args.mmlu_all_max_per_subject, len(ds)))
                )
            items.extend(
                _standardize_dataset_rows(
                    ds=ds,
                    benchmark=args.benchmark,
                    subset=cfg,
                    q_field=q_field,
                    c_field=c_field,
                    a_field=a_field,
                    answer_format=answer_format,
                    country_field=country_field,
                    continent_field=continent_field,
                )
            )
    else:
        split = _choose_split(
            spec["dataset"],
            dataset_cfg,
            spec["split"],
            token=hf_token,
        )
        ds = load_dataset_with_token(
            spec["dataset"],
            dataset_cfg,
            split=split,
            token=hf_token,
        )
        subset = dataset_cfg if dataset_cfg is not None else "default"
        items = _standardize_dataset_rows(
            ds=ds,
            benchmark=args.benchmark,
            subset=subset,
            q_field=q_field,
            c_field=c_field,
            a_field=a_field,
            answer_format=answer_format,
            country_field=country_field,
            continent_field=continent_field,
        )

    if args.max_samples > 0:
        items = items[: args.max_samples]

    if not items:
        raise ValueError("No standardized examples were produced; check dataset schema.")
    return items


def _standardize_dataset_rows(
    ds: Dataset,
    benchmark: str,
    subset: str,
    q_field: str,
    c_field: str,
    a_field: str,
    answer_format: str,
    country_field: Optional[str],
    continent_field: Optional[str],
) -> List[StandardizedItem]:
    out: List[StandardizedItem] = []
    skipped = 0
    for i, row in enumerate(ds):
        question_raw = row.get(q_field)
        choices_raw = row.get(c_field)
        answer_raw = row.get(a_field)

        question = str(question_raw).strip() if question_raw is not None else ""
        options = _to_options(choices_raw)
        correct_answer = _answer_from_row(answer_raw, options, answer_format)

        if not question or len(options) < 2 or correct_answer is None:
            skipped += 1
            continue

        country_val = row.get(country_field) if country_field else None
        continent_val = row.get(continent_field) if continent_field else None
        country_norm = str(country_val).strip() if country_val is not None else None
        continent_norm = (
            str(continent_val).strip() if continent_val is not None else None
        )

        qid = _hash_id(
            [
                benchmark,
                subset,
                question,
                json.dumps(options, ensure_ascii=False),
                correct_answer,
                country_norm or "",
                continent_norm or "",
            ]
        )
        out.append(
            StandardizedItem(
                question_id=qid,
                question=question,
                options=options,
                correct_answer=correct_answer,
                benchmark=benchmark,
                subset=subset,
                country=country_norm,
                continent=continent_norm,
                source_index=i,
            )
        )
    if skipped > 0:
        print(f"[!] Skipped {skipped} rows in {benchmark}/{subset} due to schema mismatch.")
    return out


def has_local_tokenizer_files(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    root = Path(path)
    return any((root / name).exists() for name in TOKENIZER_FILE_NAMES)


def resolve_tokenizer_source(model_type: str, model_name: str, custom_tokenizer_path: Optional[str]) -> str:
    if model_type != "custom":
        return model_name
    if has_local_tokenizer_files(model_name):
        return model_name
    if custom_tokenizer_path and (
        not os.path.isdir(custom_tokenizer_path)
        or has_local_tokenizer_files(custom_tokenizer_path)
    ):
        print(
            "[tokenizer] custom model directory has no tokenizer files; "
            f"using fallback tokenizer: {custom_tokenizer_path}"
        )
        return custom_tokenizer_path
    return model_name


def _tokenizer_vocab_size(tokenizer) -> int:
    size = getattr(tokenizer, "vocab_size", None)
    if size is not None:
        try:
            return int(size)
        except Exception:
            pass
    try:
        return int(len(tokenizer))
    except Exception:
        return 0


def validate_mcq_tokenizer(tokenizer, tokenizer_source: str, model_name: str) -> None:
    sample_texts = [" A", " B", " C", " D", " answer", " country", " egg", " fruit"]
    encoded = [
        tuple(tokenizer(text, add_special_tokens=False).input_ids)
        for text in sample_texts
    ]
    vocab_size = _tokenizer_vocab_size(tokenizer)
    distinct = len(set(encoded))
    if vocab_size < 1000 or distinct < 4:
        preview = ", ".join(f"{text!r}->{list(ids)}" for text, ids in zip(sample_texts, encoded))
        raise RuntimeError(
            "Degenerate tokenizer for MCQ scoring. "
            f"model={model_name} tokenizer_source={tokenizer_source} "
            f"class={tokenizer.__class__.__name__} vocab_size={vocab_size} "
            f"distinct_sample_encodings={distinct}; sample={preview}. "
            "This usually means a MAPLE checkpoint is missing tokenizer files. "
            "Pass --custom-tokenizer-path pointing at a valid Llama-3.2 tokenizer."
        )


def validate_tokenizer_model_compatibility(model, tokenizer, tokenizer_source: str, model_name: str) -> None:
    sample_texts = [" A", " B", " C", " D", " answer", " country", " egg", " fruit"]
    sample_ids: List[int] = []
    for text in sample_texts:
        sample_ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
    for token_id in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id):
        if token_id is not None:
            sample_ids.append(int(token_id))
    if not sample_ids:
        raise RuntimeError(f"Tokenizer produced no sample ids: {tokenizer_source}")
    max_sample_id = max(sample_ids)
    embedding_count = int(model.get_input_embeddings().num_embeddings)
    if max_sample_id >= embedding_count:
        raise RuntimeError(
            "Tokenizer/model vocabulary mismatch. "
            f"model={model_name} tokenizer_source={tokenizer_source} "
            f"max_sample_token_id={max_sample_id} model_embeddings={embedding_count}"
        )


def build_model(
    model_type: str,
    hf_token: Optional[str],
    custom_metadata: Optional[bool] = None,
    custom_model_path: Optional[str] = None,
    llama_model_name: Optional[str] = None,
    custom_tokenizer_path: Optional[str] = None,
    disable_custom_chat_template: bool = False,
    force_custom_chat_prompt: bool = False,
):
    if model_type == "custom":
        if custom_metadata is None:
            raise ValueError("custom_metadata must be set for custom model.")
        if custom_model_path:
            model_name = custom_model_path
        else:
            name = "combined_with_metadata" if custom_metadata else "combined_without_metadata"
            model_name = f"/path/to/metacul/models/sft/{name}_chat"
    elif model_type == "llama3_chat":
        model_name = llama_model_name or "meta-llama/Llama-3.2-1B-Instruct"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    hf_kwargs = {"trust_remote_code": True}

    tokenizer_source = resolve_tokenizer_source(
        model_type=model_type,
        model_name=model_name,
        custom_tokenizer_path=custom_tokenizer_path,
    )
    tokenizer_hf_kwargs = dict(hf_kwargs)
    if hf_token and not os.path.isdir(tokenizer_source):
        tokenizer_hf_kwargs["token"] = hf_token
    tokenizer = load_tokenizer_robust(tokenizer_source, tokenizer_hf_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    validate_mcq_tokenizer(
        tokenizer=tokenizer,
        tokenizer_source=tokenizer_source,
        model_name=model_name,
    )
    print(
        f"[tokenizer] model={model_name} source={tokenizer_source} "
        f"class={tokenizer.__class__.__name__} vocab_size={_tokenizer_vocab_size(tokenizer)}"
    )
    uses_chat_prompt = model_type == "llama3_chat" or (
        model_type == "custom"
        and (
            force_custom_chat_prompt
            or custom_model_uses_chat_prompt(model_name)
        )
    )
    default_chat_template = load_default_chat_template()
    if (
        model_type == "custom"
        and uses_chat_prompt
        and not disable_custom_chat_template
        and default_chat_template is not None
    ):
        tokenizer.chat_template = default_chat_template
    if (
        model_type == "llama3_chat"
        and not getattr(tokenizer, "chat_template", None)
        and default_chat_template is not None
    ):
        tokenizer.chat_template = default_chat_template

    config_hf_kwargs = dict(hf_kwargs)
    if hf_token and not os.path.isdir(model_name):
        config_hf_kwargs["token"] = hf_token
    config = AutoConfig.from_pretrained(model_name, **config_hf_kwargs)
    model_loader = AutoModelForCausalLM
    if config.__class__.__name__ == "Mistral3Config":
        model_loader = AutoModelForImageTextToText

    model_hf_kwargs = dict(hf_kwargs)
    if hf_token and not os.path.isdir(model_name):
        model_hf_kwargs["token"] = hf_token
    model = model_loader.from_pretrained(
        model_name, device_map="auto", dtype="auto", **model_hf_kwargs
    )
    validate_tokenizer_model_compatibility(
        model=model,
        tokenizer=tokenizer,
        tokenizer_source=tokenizer_source,
        model_name=model_name,
    )
    model_info = {
        "model_name": model_name,
        "tokenizer_source": tokenizer_source,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": _tokenizer_vocab_size(tokenizer),
        "uses_chat_prompt": uses_chat_prompt,
    }
    return model, tokenizer, uses_chat_prompt, model_info


def render_prompt(
    tokenizer,
    messages: List[Dict[str, str]],
    uses_chat_prompt: bool,
) -> str:
    if uses_chat_prompt and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if uses_chat_prompt:
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        chunks = []
        if system:
            chunks.append(f"System:\n{system.strip()}")
        if user:
            chunks.append(f"User:\n{user.strip()}")
        chunks.append("Assistant:\n")
        return "\n\n".join(chunks)
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    return user.rstrip() + "\n"


def choose_metadata_tags(
    item: StandardizedItem,
    seed: int,
    mode: str = "correct",
) -> Dict[str, str]:
    country_code = _normalize_country(item.country)
    continent_norm = _normalize_continent(item.continent)

    if mode == "adversarial":
        # Force a different continent-country pair when we have geo metadata.
        if country_code is not None and continent_norm is not None:
            all_conts = sorted(CONTINENT_TO_CODES.keys())
            src_cont = continent_norm.lower()
            alt_conts = [c for c in all_conts if c != src_cont and CONTINENT_TO_CODES.get(c)]
            if alt_conts:
                cont_idx = int(_hash_id([item.question_id, str(seed), "adv_cont"]), 16) % len(alt_conts)
                adv_cont = alt_conts[cont_idx]
                country_choices = sorted(CONTINENT_TO_CODES[adv_cont])
                cidx = int(_hash_id([item.question_id, str(seed), "adv_country"]), 16) % len(country_choices)
                adv_country = country_choices[cidx]
                return {
                    "country_tag": adv_country,
                    "continent_tag": adv_cont.capitalize(),
                    "metadata_source": "adversarial_mismatch",
                }
        # Fallback: choose any pair different from deterministic baseline.
        pairs = sorted(COUNTRY_CONTINENT_MAP.items())
        if pairs:
            base_idx = int(_hash_id([item.question_id, str(seed), "adv_base"]), 16) % len(pairs)
            adv_idx = (base_idx + 1) % len(pairs)
            country_code, continent_title = pairs[adv_idx]
            return {
                "country_tag": country_code,
                "continent_tag": continent_title,
                "metadata_source": "adversarial_fallback",
            }

    if country_code is None and continent_norm is None:
        if mode == "available_only":
            return {
                "country_tag": None,
                "continent_tag": None,
                "metadata_source": "missing_geo_fields",
            }
        # Deterministically sample a full pair when benchmark lacks geo fields.
        pairs = sorted(COUNTRY_CONTINENT_MAP.items())
        idx = int(_hash_id([item.question_id, str(seed)]), 16) % len(pairs)
        country_code, continent_title = pairs[idx]
        return {
            "country_tag": country_code,
            "continent_tag": continent_title,
            "metadata_source": "sampled_pair",
        }

    if country_code is not None and continent_norm is None:
        continent_norm = COUNTRY_CONTINENT_MAP.get(country_code)
        if continent_norm is None:
            country_name = _display_country_name(item.country)
            continent_norm = continent_from_country_name(country_name)
        if continent_norm is None:
            pairs = sorted(COUNTRY_CONTINENT_MAP.items())
            idx = int(_hash_id([item.question_id, str(seed), "country_fallback"]), 16) % len(pairs)
            fallback_code, fallback_continent = pairs[idx]
            return {
                "country_tag": fallback_code,
                "continent_tag": fallback_continent,
                "metadata_source": "country_missing_continent_fallback",
            }
        return {
            "country_tag": country_code,
            "continent_tag": continent_norm,
            "metadata_source": "country_to_continent",
        }

    if country_code is None and continent_norm is not None:
        choices = sorted(CONTINENT_TO_CODES.get(continent_norm.lower(), []))
        if not choices:
            pairs = sorted(COUNTRY_CONTINENT_MAP.items())
            idx = int(_hash_id([item.question_id, str(seed), "fallback"]), 16) % len(pairs)
            country_code, continent_title = pairs[idx]
            return {
                "country_tag": country_code,
                "continent_tag": continent_title,
                "metadata_source": "fallback_pair",
            }
        idx = int(_hash_id([item.question_id, str(seed), continent_norm]), 16) % len(choices)
        country_code = choices[idx]
        return {
            "country_tag": country_code,
            "continent_tag": continent_norm,
            "metadata_source": "continent_to_country",
        }

    return {
        "country_tag": country_code,
        "continent_tag": continent_norm,
        "metadata_source": "from_example",
    }


def build_core_user_content(
    item: StandardizedItem,
    prompt_options: List[str],
    qa_prompt_style: str,
    answer_cue_style: str,
    omit_option_labels: bool,
    exact_option_text_instruction: bool,
    mcq_scoring: str,
    answer_cue_country_name: Optional[str] = None,
) -> str:
    question = item.question.strip()
    if omit_option_labels:
        options_block = "\n".join([f"- {opt}" for opt in prompt_options])
    else:
        options_block = "\n".join(
            [f"{chr(65 + i)}: {opt}" for i, opt in enumerate(prompt_options)]
        )

    if mcq_scoring == "option_letter":
        answer_instruction = "Answer with only the correct option letter (A, B, C, or D)."
    elif exact_option_text_instruction:
        answer_instruction = "Answer with the exact text of the correct option."
    else:
        answer_instruction = "Answer with the correct option."

    if answer_cue_style == "none":
        answer_cue = ""
    elif answer_cue_style == "answer_colon":
        answer_cue = "\n\nAnswer:"
    elif answer_cue_style == "answer_newline":
        answer_cue = "\n\nAnswer:\n"
    elif answer_cue_style == "final_answer_colon":
        answer_cue = "\n\nFinal answer:"
    elif answer_cue_style == "the_correct_answer_is":
        answer_cue = "\n\nThe correct answer is"
    elif answer_cue_style == "country_answer_colon":
        answer_cue = (
            f"\n\nFor {answer_cue_country_name}, answer:"
            if answer_cue_country_name
            else "\n\nAnswer:"
        )
    elif answer_cue_style == "country_final_answer_colon":
        answer_cue = (
            f"\n\nFor {answer_cue_country_name}, final answer:"
            if answer_cue_country_name
            else "\n\nFinal answer:"
        )
    elif answer_cue_style == "country_the_correct_answer_is":
        answer_cue = (
            f"\n\nFor {answer_cue_country_name}, the correct answer is"
            if answer_cue_country_name
            else "\n\nThe correct answer is"
        )
    else:
        raise ValueError(f"Unsupported answer_cue_style: {answer_cue_style}")

    if qa_prompt_style == "question":
        return (
            f"Question: {question}\n\n"
            f"Options:\n{options_block}\n\n"
            f"{answer_instruction}"
            f"{answer_cue}"
        )
    if qa_prompt_style == "instruction":
        return (
            "### Instruction:\n"
            f"{question}\n\n"
            "Options:\n"
            f"{options_block}\n\n"
            f"{answer_instruction}"
            f"{answer_cue}"
        )
    if qa_prompt_style == "instruction_input":
        return (
            "### Instruction:\n"
            "Answer the following locale-specific multiple-choice question.\n\n"
            "### Input:\n"
            f"Question: {question}\n\n"
            "Options:\n"
            f"{options_block}\n\n"
            f"{answer_instruction}"
            f"{answer_cue}"
        )
    if qa_prompt_style == "question_answer":
        answer_suffix = answer_cue if answer_cue else "\n\nAnswer:"
        return (
            f"Question: {question}\n\n"
            f"Options:\n{options_block}"
            f"{answer_suffix}"
        )
    raise ValueError(f"Unsupported qa_prompt_style: {qa_prompt_style}")


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
            f"Use the locale metadata above as grounding. Answer for {country_name}, not a different country.\n\n"
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
            f"Use the locale metadata above as grounding. This question is about {country_name}.\n\n"
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
            f"When multiple answers could fit different countries, answer for {country_name}.\n\n"
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
            f"Choose the answer that is correct for {country_name}, not the answer that would fit a different country "
            f"or the answer that is only more globally common.\n\n"
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
            f"Use the locale metadata above to disambiguate country-specific facts. "
            f"Choose the answer that is correct for {country_name}, even when a different option is more globally common.\n\n"
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
            f"Answer this question for {country_name}. "
            f"When multiple options could fit different countries, pick the option that is factual for {country_name}.\n\n"
        )
        return metadata_block + content
    raise ValueError(f"Unsupported metadata_prompt_style: {metadata_prompt_style}")


def build_user_content(
    item: StandardizedItem,
    prompt_options: List[str],
    with_metadata: bool,
    base_url: str,
    seed: int,
    qa_prompt_style: str = "question",
    answer_cue_style: str = "none",
    omit_option_labels: bool = False,
    exact_option_text_instruction: bool = False,
    mcq_scoring: str = "option_text_avg",
    metadata_tag_mode: str = "correct",
    shuffled_tags_by_qid: Optional[Dict[str, Dict[str, str]]] = None,
    metadata_prompt_style: str = "legacy_code",
) -> Tuple[str, Dict[str, Optional[str]]]:
    if metadata_tag_mode == "shuffled":
        if shuffled_tags_by_qid is None:
            raise ValueError("shuffled_tags_by_qid must be set for metadata_tag_mode='shuffled'")
        tags = shuffled_tags_by_qid[item.question_id]
    else:
        tags = choose_metadata_tags(item, seed=seed, mode=metadata_tag_mode)
    country_tag = tags["country_tag"]
    continent_tag = tags["continent_tag"]
    country_name = item.country
    if not country_name and country_tag:
        country_name = CODE_TO_COUNTRY_NAME.get(country_tag, str(country_tag).upper())
    core = build_core_user_content(
        item=item,
        prompt_options=prompt_options,
        qa_prompt_style=qa_prompt_style,
        answer_cue_style=answer_cue_style,
        omit_option_labels=omit_option_labels,
        exact_option_text_instruction=exact_option_text_instruction,
        mcq_scoring=mcq_scoring,
        answer_cue_country_name=country_name if with_metadata and country_name else None,
    )

    if not with_metadata:
        return (
            f"CONTENT:\n{core}",
            {
                "url_country_tag": None,
                "url_continent_tag": None,
                "metadata_source": "none",
            },
        )

    if not country_tag or not continent_tag or not country_name:
        return (
            f"CONTENT:\n{core}",
            {
                "url_country_tag": None,
                "url_continent_tag": None,
                "metadata_source": tags["metadata_source"],
            },
        )

    content = add_metadata_block(
        content=core,
        base_url=base_url,
        country_tag=country_tag,
        continent_tag=continent_tag,
        country_name=country_name,
        metadata_prompt_style=metadata_prompt_style,
    )
    return content, {
        "url_country_tag": tags["country_tag"],
        "url_continent_tag": tags["continent_tag"],
        "metadata_source": tags["metadata_source"],
    }


def extract_answer(raw_output: str, options: List[str]) -> Dict[str, Optional[str]]:
    raw = raw_output.strip()
    max_letter = chr(ord("A") + len(options) - 1)
    letter_re = rf"\b([A-{max_letter}])\b"
    m = re.search(letter_re, raw, re.IGNORECASE)
    letter = m.group(1).upper() if m else None
    answer_text = None
    method = "letter"

    if letter is not None:
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(options):
            answer_text = options[idx]

    if answer_text is None:
        method = "option_substring"
        lowered = raw.lower()
        matches = [opt for opt in options if opt.lower() in lowered]
        if matches:
            answer_text = max(matches, key=len)
            letter = chr(ord("A") + options.index(answer_text))

    # Deterministic no-skip fallback: pick closest option text.
    if answer_text is None:
        method = "similarity_fallback"
        lowered = raw.lower()
        if lowered:
            scores = [
                difflib.SequenceMatcher(None, lowered, str(opt).lower()).ratio()
                for opt in options
            ]
            idx = int(np.argmax(scores))
        else:
            idx = 0
        answer_text = options[idx]
        letter = chr(ord("A") + idx)

    return {
        "processed_option_letter": letter,
        "processed_answer": answer_text,
        "answer_extraction_method": method,
    }


def get_candidate_texts(
    options: List[str],
    answer_cue_style: str,
    mcq_scoring: str,
) -> List[str]:
    letters = [chr(65 + i) for i in range(len(options))]
    option_prefix = "" if answer_cue_style == "answer_newline" else " "
    if mcq_scoring == "option_letter":
        return [option_prefix + letter for letter in letters]
    return [option_prefix + str(opt).strip() for opt in options]


def score_candidate_texts(
    model,
    tokenizer,
    prompt: str,
    candidate_texts: List[str],
    add_prompt_bos: bool,
) -> Tuple[List[float], List[float]]:
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(model.device)
    if add_prompt_bos and tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]], device=model.device, dtype=prompt_ids.dtype)
        if prompt_ids.numel() == 0 or int(prompt_ids[0, 0].item()) != tokenizer.bos_token_id:
            prompt_ids = torch.cat([bos, prompt_ids], dim=1)

    scores_sum: List[float] = []
    scores_avg: List[float] = []
    with torch.no_grad():
        for cand_text in candidate_texts:
            cand_ids = tokenizer(
                cand_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(model.device)
            if cand_ids.numel() == 0:
                scores_sum.append(float("-inf"))
                scores_avg.append(float("-inf"))
                continue
            input_ids = torch.cat([prompt_ids, cand_ids], dim=1)
            attention_mask = torch.ones_like(input_ids)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)

            start = prompt_ids.shape[1]
            target = input_ids[:, 1:]
            cand_target = target[:, start - 1 :]
            cand_log_probs = log_probs[:, start - 1 :, :].gather(-1, cand_target.unsqueeze(-1)).squeeze(-1)
            score_sum = float(cand_log_probs.sum().item())
            tok_len = int(cand_ids.shape[1])
            score_avg = score_sum / max(tok_len, 1)
            scores_sum.append(score_sum)
            scores_avg.append(score_avg)
    return scores_sum, scores_avg


def select_option_scores(
    scores_sum: List[float],
    scores_avg: List[float],
    mcq_scoring: str,
    length_norm_alpha: Optional[float],
) -> List[float]:
    if mcq_scoring == "option_letter":
        return scores_avg
    if length_norm_alpha is not None:
        selected: List[float] = []
        for score_sum, score_avg in zip(scores_sum, scores_avg):
            tok_len = abs(score_sum / score_avg) if score_avg != 0 else 1.0
            selected.append(score_sum / max(tok_len, 1.0) ** length_norm_alpha)
        return selected
    if mcq_scoring == "option_text_sum":
        return scores_sum
    return scores_avg


def predict_option_by_loglikelihood(
    model,
    tokenizer,
    prompt: str,
    options: List[str],
    answer_cue_style: str,
    mcq_scoring: str,
    add_prompt_bos: bool,
    length_norm_alpha: Optional[float],
    null_calibration_mode: str,
    null_calibration_beta: float,
    calibration_prompt: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    candidate_texts = get_candidate_texts(
        options=options,
        answer_cue_style=answer_cue_style,
        mcq_scoring=mcq_scoring,
    )
    scores_sum, scores_avg = score_candidate_texts(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_texts=candidate_texts,
        add_prompt_bos=add_prompt_bos,
    )
    primary_scores = select_option_scores(
        scores_sum=scores_sum,
        scores_avg=scores_avg,
        mcq_scoring=mcq_scoring,
        length_norm_alpha=length_norm_alpha,
    )
    calibration_sums = None
    calibration_avgs = None
    calibration_scores = None
    if calibration_prompt is not None and null_calibration_mode != "none":
        calibration_sums, calibration_avgs = score_candidate_texts(
            model=model,
            tokenizer=tokenizer,
            prompt=calibration_prompt,
            candidate_texts=candidate_texts,
            add_prompt_bos=add_prompt_bos,
        )
        calibration_scores = select_option_scores(
            scores_sum=calibration_sums,
            scores_avg=calibration_avgs,
            mcq_scoring=mcq_scoring,
            length_norm_alpha=length_norm_alpha,
        )
        final_scores = [
            score - null_calibration_beta * calibration_score
            for score, calibration_score in zip(primary_scores, calibration_scores)
        ]
    else:
        final_scores = primary_scores

    best_idx = int(np.argmax(final_scores)) if final_scores else 0
    best_letter = chr(65 + best_idx) if best_idx < 26 else None
    return {
        "processed_option_letter": best_letter,
        "processed_answer": options[best_idx],
        "answer_extraction_method": mcq_scoring,
        "option_loglikelihood_sums": scores_sum,
        "option_loglikelihood_avgs": scores_avg,
        "option_loglikelihood_length_norm_alpha": length_norm_alpha,
        "option_loglikelihood_primary_scores": primary_scores,
        "null_calibration_mode": null_calibration_mode,
        "null_calibration_beta": null_calibration_beta,
        "null_calibration_option_loglikelihood_sums": calibration_sums,
        "null_calibration_option_loglikelihood_avgs": calibration_avgs,
        "null_calibration_option_scores": calibration_scores,
        "option_loglikelihood_selected_scores": final_scores,
        "scoring_candidates": candidate_texts,
    }


def get_prompt_options(
    item: StandardizedItem,
    shuffle_options: bool,
    option_shuffle_seed: int,
) -> Tuple[List[str], List[int]]:
    options = list(item.options)
    permutation = list(range(len(options)))
    if not shuffle_options:
        return options, permutation

    digest = _hash_id(
        [
            str(option_shuffle_seed),
            item.benchmark,
            item.question_id,
            item.question_key or "",
            str(item.participant_id or ""),
            str(item.source_index),
            "option-shuffle",
        ]
    )
    rng = random.Random(int(digest[:16], 16))
    rng.shuffle(permutation)
    shuffled = [options[i] for i in permutation]
    return shuffled, permutation


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def row_identity_from_item(item: StandardizedItem) -> Tuple[str, str, str, str, str]:
    return (
        str(item.benchmark),
        str(item.question_id),
        str(item.question_key or ""),
        str(item.participant_id if item.participant_id is not None else ""),
        str(item.source_index),
    )


def row_identity_from_row(row: Dict[str, object]) -> Tuple[str, str, str, str, str]:
    return (
        str(row.get("benchmark", "")),
        str(row.get("question_id", "")),
        str(row.get("question_key", "") or ""),
        str(row.get("participant_id", "") if row.get("participant_id") is not None else ""),
        str(row.get("source_index", "")),
    )


def evaluate_variant(
    args: argparse.Namespace,
    items: List[StandardizedItem],
    variant_name: str,
    model_type: str,
    with_metadata: bool,
    model,
    tokenizer,
    uses_chat_prompt: bool,
    model_info: Dict[str, object],
    output_jsonl: Path,
    metadata_tag_mode: str = "correct",
) -> Dict[str, float]:
    ensure_parent(output_jsonl)
    total = len(items)
    correct = 0
    processed_total = 0
    completed_keys = set()
    write_mode = "w"

    if args.resume and output_jsonl.exists():
        with output_jsonl.open("r", encoding="utf-8") as existing_f:
            for line in existing_f:
                if not line.strip():
                    continue
                row = json.loads(line)
                completed_keys.add(row_identity_from_row(row))
                if row.get("processed_answer") is not None:
                    is_correct = row.get("is_correct")
                    if is_correct is None:
                        is_correct = row.get("processed_answer") == row.get("correct_answer")
                    processed_total += 1
                    correct += int(bool(is_correct))
        write_mode = "a"
        print(
            f"[resume] {variant_name}: loaded {len(completed_keys)} existing rows from {output_jsonl}"
        )

    pending_items = [
        item for item in items if row_identity_from_item(item) not in completed_keys
    ]

    shuffled_tags_by_qid: Optional[Dict[str, Dict[str, str]]] = None
    if with_metadata and metadata_tag_mode == "shuffled":
        base_tags = [
            choose_metadata_tags(item, seed=args.seed, mode="correct") for item in items
        ]
        if len(base_tags) >= 2:
            shift = 1
            shuffled = base_tags[shift:] + base_tags[:shift]
        else:
            shuffled = base_tags
        shuffled_tags_by_qid = {
            item.question_id: shuffled[i] for i, item in enumerate(items)
        }

    # Frequent cancel/requeue cycles are common in this environment, so flush each
    # completed row promptly to preserve as much partial progress as possible.
    with output_jsonl.open(write_mode, encoding="utf-8", buffering=1) as out_f:
        for start in tqdm(
            range(0, len(pending_items), args.batch_size),
            desc=f"Eval {variant_name}",
        ):
            batch = pending_items[start : min(start + args.batch_size, len(pending_items))]
            prompts = []
            calibration_prompts = []
            meta_info_batch = []
            prompt_options_batch = []
            prompt_permutations = []

            for item in batch:
                prompt_options, prompt_permutation = get_prompt_options(
                    item=item,
                    shuffle_options=args.shuffle_options,
                    option_shuffle_seed=args.option_shuffle_seed,
                )
                user_content, meta_info = build_user_content(
                    item=item,
                    prompt_options=prompt_options,
                    with_metadata=with_metadata,
                    base_url=args.base_url,
                    seed=args.seed,
                    qa_prompt_style=args.qa_prompt_style,
                    answer_cue_style=args.answer_cue_style,
                    omit_option_labels=args.omit_option_labels,
                    exact_option_text_instruction=args.exact_option_text_instruction,
                    mcq_scoring=args.mcq_scoring,
                    metadata_tag_mode=metadata_tag_mode,
                    shuffled_tags_by_qid=shuffled_tags_by_qid,
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
                prompts.append(prompt)
                calibration_prompt = None
                if args.null_calibration_mode in {"question_masked", "question_masked_no_metadata"}:
                    calibration_item = replace(item, question=args.null_question_text)
                    calibration_with_metadata = (
                        False
                        if args.null_calibration_mode == "question_masked_no_metadata"
                        else with_metadata
                    )
                    calibration_user_content, _ = build_user_content(
                        item=calibration_item,
                        prompt_options=prompt_options,
                        with_metadata=calibration_with_metadata,
                        base_url=args.base_url,
                        seed=args.seed,
                        qa_prompt_style=args.qa_prompt_style,
                        answer_cue_style=args.answer_cue_style,
                        omit_option_labels=args.omit_option_labels,
                        exact_option_text_instruction=args.exact_option_text_instruction,
                        mcq_scoring=args.mcq_scoring,
                        metadata_tag_mode=metadata_tag_mode,
                        shuffled_tags_by_qid=shuffled_tags_by_qid,
                        metadata_prompt_style=args.metadata_prompt_style,
                    )
                    calibration_messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": calibration_user_content},
                    ]
                    calibration_prompt = render_prompt(
                        tokenizer=tokenizer,
                        messages=calibration_messages,
                        uses_chat_prompt=uses_chat_prompt,
                    )
                calibration_prompts.append(calibration_prompt)
                meta_info_batch.append(meta_info)
                prompt_options_batch.append(prompt_options)
                prompt_permutations.append(prompt_permutation)

            for i, item in enumerate(batch):
                prompt_options = prompt_options_batch[i]
                processed = predict_option_by_loglikelihood(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompts[i],
                    options=prompt_options,
                    answer_cue_style=args.answer_cue_style,
                    mcq_scoring=args.mcq_scoring,
                    add_prompt_bos=args.add_prompt_bos,
                    length_norm_alpha=args.length_norm_alpha,
                    null_calibration_mode=args.null_calibration_mode,
                    null_calibration_beta=args.null_calibration_beta,
                    calibration_prompt=calibration_prompts[i],
                )
                raw_output = processed["processed_option_letter"]
                is_correct = None
                if processed["processed_answer"] is not None:
                    is_correct = processed["processed_answer"] == item.correct_answer
                    processed_total += 1
                    correct += int(is_correct)

                row = {
                    "benchmark": item.benchmark,
                    "subset": item.subset,
                    "question_id": item.question_id,
                    "question_key": item.question_key,
                    "participant_id": item.participant_id,
                    "source_index": item.source_index,
                    "question": item.question,
                    "options": item.options,
                    "prompt_options": prompt_options,
                    "prompt_option_permutation": prompt_permutations[i],
                    "correct_answer": item.correct_answer,
                    "country": item.country,
                    "continent": item.continent,
                    "variant": variant_name,
                    "model_type": model_type,
                    **model_info,
                    "metadata": with_metadata,
                    "metadata_prompt_style": args.metadata_prompt_style if with_metadata else "none",
                    "qa_prompt_style": args.qa_prompt_style,
                    "answer_cue_style": args.answer_cue_style,
                    "omit_option_labels": args.omit_option_labels,
                    "exact_option_text_instruction": args.exact_option_text_instruction,
                    "mcq_scoring": args.mcq_scoring,
                    "length_norm_alpha": args.length_norm_alpha,
                    "add_prompt_bos": args.add_prompt_bos,
                    "null_calibration_mode": args.null_calibration_mode,
                    "null_calibration_beta": args.null_calibration_beta,
                    "metadata_tag_mode": metadata_tag_mode,
                    "base_url": args.base_url,
                    "options_shuffled": args.shuffle_options,
                    "option_shuffle_seed": args.option_shuffle_seed,
                    "prompt": prompts[i],
                    "raw_output": raw_output,
                    **meta_info_batch[i],
                    **processed,
                }
                if is_correct is not None:
                    row["is_correct"] = is_correct
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()

    acc = correct / processed_total if processed_total else 0.0
    return {
        "variant": variant_name,
        "correct": correct,
        "processed_total": processed_total,
        "accuracy": acc,
        "skipped": total - processed_total,
        "total_rows": total,
    }


def load_variant_outcomes(jsonl_path: Path) -> Dict[str, Dict]:
    outcomes: Dict[str, Dict] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row["question_id"]
            processed = row.get("processed_answer")
            if processed is None:
                outcome = "skipped"
            else:
                is_correct = row.get("is_correct")
                if is_correct is None:
                    is_correct = processed == row.get("correct_answer")
                outcome = "correct" if is_correct else "incorrect"
            outcomes[qid] = {
                "benchmark": row.get("benchmark"),
                "subset": row.get("subset"),
                "country": row.get("country"),
                "continent": row.get("continent"),
                "outcome": outcome,
            }
    return outcomes


def write_matrix_csv(
    output_csv: Path,
    per_variant_outcomes: Dict[str, Dict[str, Dict]],
    variant_order: List[str],
) -> None:
    ensure_parent(output_csv)
    all_qids = sorted(
        set().union(*[set(v.keys()) for v in per_variant_outcomes.values()])
    )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "question_id",
            "benchmark",
            "subset",
            "country",
            "continent",
            "answered_by_all",
        ]
        for v in variant_order:
            header.extend([f"{v}_correct", f"{v}_incorrect", f"{v}_skipped"])
        writer.writerow(header)

        for qid in all_qids:
            meta_row = None
            answered_by_all = True
            out = [qid]
            for v in variant_order:
                entry = per_variant_outcomes.get(v, {}).get(qid)
                if meta_row is None and entry is not None:
                    meta_row = entry
            out.extend(
                [
                    (meta_row or {}).get("benchmark"),
                    (meta_row or {}).get("subset"),
                    (meta_row or {}).get("country"),
                    (meta_row or {}).get("continent"),
                ]
            )
            row_tail = []
            for v in variant_order:
                entry = per_variant_outcomes.get(v, {}).get(qid)
                outcome = entry["outcome"] if entry is not None else "skipped"
                if outcome == "correct":
                    row_tail.extend([1, 0, 0])
                elif outcome == "incorrect":
                    row_tail.extend([0, -1, 0])
                else:
                    row_tail.extend([0, 0, 0])
                    answered_by_all = False
            out.append(1 if answered_by_all else 0)
            out.extend(row_tail)
            writer.writerow(out)


def write_summary_csv(
    output_csv: Path,
    summary_rows: List[Dict[str, float]],
) -> None:
    ensure_parent(output_csv)
    fields = [
        "variant",
        "status",
        "correct",
        "processed_total",
        "accuracy",
        "skipped",
        "total_rows",
        "error",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    hf_token = get_hf_token()
    if hf_token:
        print("[✔] HF token detected in environment.")
    else:
        print("[!] No HF token detected in environment.")

    seed_name_to_continent_map(args.worldvaluesbench_root)

    spec = resolve_benchmark_spec(args)
    items = load_standardized_items(args, spec, hf_token=hf_token)
    print(
        f"[✔] Loaded {len(items)} standardized samples for benchmark={args.benchmark}"
    )

    variant_order = [v.strip() for v in args.variants.split(",") if v.strip()]
    invalid = [v for v in variant_order if v not in VARIANT_SPECS]
    if invalid:
        raise ValueError(f"Unknown variant(s): {invalid}")

    if args.dry_run:
        by_subset: Dict[str, int] = {}
        for item in items:
            by_subset[item.subset] = by_subset.get(item.subset, 0) + 1
        print("[✔] Dry run complete. Subset counts:")
        for subset, count in sorted(by_subset.items()):
            print(f"  - {subset}: {count}")
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded_models = {}
    per_variant_outcomes: Dict[str, Dict[str, Dict]] = {}
    summary_rows = []
    completed_variants: List[str] = []
    failed_model_types: Dict[str, str] = {}

    for variant in variant_order:
        cfg = VARIANT_SPECS[variant]
        model_type = cfg["model_type"]
        with_metadata = bool(cfg.get("eval_metadata", False))
        custom_model_metadata = cfg.get("custom_model_metadata")
        if model_type == "custom" and custom_model_metadata is None:
            raise ValueError(f"Variant '{variant}' missing custom_model_metadata.")
        model_key = (
            f"custom_tplus={int(bool(custom_model_metadata))}"
            if model_type == "custom"
            else model_type
        )

        if model_key in failed_model_types:
            err_msg = failed_model_types[model_key]
            print(f"[!] Skipping {variant}: model_key={model_key} unavailable.")
            summary_rows.append(
                {
                    "variant": variant,
                    "status": "skipped_model_unavailable",
                    "correct": 0,
                    "processed_total": 0,
                    "accuracy": 0.0,
                    "skipped": len(items),
                    "total_rows": len(items),
                    "error": err_msg,
                }
            )
            continue

        if model_key not in loaded_models:
            print(f"[...] Loading model for key={model_key}")
            try:
                model, tokenizer, uses_chat_prompt, model_info = build_model(
                    model_type,
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
                    llama_model_name=args.llama_model_name,
                    custom_tokenizer_path=args.custom_tokenizer_path,
                    disable_custom_chat_template=args.disable_custom_chat_template,
                    force_custom_chat_prompt=args.force_custom_chat_prompt,
                )
                loaded_models[model_key] = (model, tokenizer, uses_chat_prompt, model_info)
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                failed_model_types[model_key] = err_msg
                print(f"[!] Failed to load {model_key}: {err_msg}")
                summary_rows.append(
                    {
                        "variant": variant,
                        "status": "failed_model_load",
                        "correct": 0,
                        "processed_total": 0,
                        "accuracy": 0.0,
                        "skipped": len(items),
                        "total_rows": len(items),
                        "error": err_msg,
                    }
                )
                continue
        else:
            model, tokenizer, uses_chat_prompt, model_info = loaded_models[model_key]

        output_jsonl = out_dir / f"{args.benchmark}_{variant}.jsonl"
        summary = evaluate_variant(
            args=args,
            items=items,
            variant_name=variant,
            model_type=model_type,
            with_metadata=with_metadata,
            model=model,
            tokenizer=tokenizer,
            uses_chat_prompt=uses_chat_prompt,
            model_info=model_info,
            output_jsonl=output_jsonl,
            metadata_tag_mode=args.metadata_tag_mode,
        )
        if args.benchmark == "worldvaluebench":
            emd_group_csv = out_dir / f"{args.benchmark}_{variant}_emd_by_group.csv"
            emd_summary_json = out_dir / f"{args.benchmark}_{variant}_emd_summary.json"
            emd_summary = compute_worldvaluebench_emd(
                output_jsonl=output_jsonl,
                output_group_csv=emd_group_csv,
                output_summary_json=emd_summary_json,
            )
            print(
                f"[✔] {variant}: overall_emd={emd_summary['overall_emd']:.6f}, "
                f"groups={emd_summary['groups']}, scored_rows={emd_summary['scored_rows']}"
            )
        summary["status"] = "ok"
        summary["error"] = ""
        summary_rows.append(summary)
        per_variant_outcomes[variant] = load_variant_outcomes(output_jsonl)
        completed_variants.append(variant)
        print(
            f"[✔] {variant}: acc={summary['accuracy']:.4f} "
            f"({summary['correct']}/{summary['processed_total']}), skipped={summary['skipped']}"
        )

    if not completed_variants:
        summary_csv = out_dir / f"{args.benchmark}_eval_summary.csv"
        write_summary_csv(summary_csv, summary_rows)
        print("[!] No variants completed successfully; wrote summary only.")
        print(f"[✔] Wrote summary CSV: {summary_csv}")
        return 1

    matrix_csv = out_dir / f"{args.benchmark}_eval_matrix.csv"
    summary_csv = out_dir / f"{args.benchmark}_eval_summary.csv"
    write_matrix_csv(matrix_csv, per_variant_outcomes, completed_variants)
    write_summary_csv(summary_csv, summary_rows)
    print(f"[✔] Wrote matrix CSV: {matrix_csv}")
    print(f"[✔] Wrote summary CSV: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
