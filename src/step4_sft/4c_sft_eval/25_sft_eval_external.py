#!/usr/bin/env python3
import argparse
import ast
import csv
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, get_dataset_config_names, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


SYSTEM_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)

CHAT_TEMPLATE_PATH = "/scratch/amukher6/metacul/src/chat_template.jinja"

DEFAULT_OUTPUT_DIR = "/scratch/amukher6/metacul/results/external_benchmarks"
DEFAULT_BASE_URL = "www.globalfactcheck.org"
DEFAULT_WVB_ROOT = "/scratch/amukher6/WorldValuesBench"

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
        "split": "test",
        "question_field": "question",
        "choices_field": "choices",
        "answer_field": "answer",
        "answer_format": "index",
    },
    # These benchmarks frequently appear under different dataset IDs/schemas.
    # Keep them configurable via CLI overrides.
    "geolmama": {
        "dataset": "iamshnoo/geomlama",
        "split": "en",
        "question_field": "question",
        "choices_field": "candidate_answers",
        "answer_field": "answer",
        "answer_format": "text",
    },
    # Alias for user typo convenience.
    "geomlama": {
        "dataset": "iamshnoo/geomlama",
        "split": "en",
        "question_field": "question",
        "choices_field": "candidate_answers",
        "answer_field": "answer",
        "answer_format": "text",
    },
    "globalopinionqa": {
        "dataset": "Anthropic/llm_global_opinions",
        "split": "train",
        "question_field": "question",
        "choices_field": "options",
        "answer_field": "selections",
        "answer_format": "auto",
    },
    "worldvaluebench": {
        "dataset": None,
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
        choices=["correct", "shuffled", "adversarial"],
        help=(
            "How to assign metadata tags when eval metadata is enabled. "
            "'correct' uses dataset tags/mapping, "
            "'shuffled' rotates tags across examples, "
            "'adversarial' forces a different continent/country when possible."
        ),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
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
    return parser.parse_args()


def get_hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        token = os.getenv(key)
        if token and token.strip():
            return token.strip()
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
    mapped = COUNTRY_CODE_MAP.get(c)
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
    return _normalize_continent_label(NAME_TO_CONTINENT.get(key))


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

    if args.benchmark == "mmlu" and args.dataset_config == "all":
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
            args.dataset_config,
            spec["split"],
            token=hf_token,
        )
        ds = load_dataset_with_token(
            spec["dataset"],
            args.dataset_config,
            split=split,
            token=hf_token,
        )
        subset = args.dataset_config if args.dataset_config is not None else "default"
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


def build_model(
    model_type: str,
    hf_token: Optional[str],
    custom_metadata: Optional[bool] = None,
):
    if model_type == "custom":
        if custom_metadata is None:
            raise ValueError("custom_metadata must be set for custom model.")
        name = "combined_with_metadata" if custom_metadata else "combined_without_metadata"
        model_name = f"/scratch/amukher6/metacul/models/sft/{name}_chat"
    elif model_type == "llama3_chat":
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    hf_kwargs = {}
    if hf_token and model_type == "llama3_chat":
        hf_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(model_name, **hf_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    if model_type == "custom":
        with open(CHAT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            tokenizer.chat_template = f.read()

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", dtype="auto", **hf_kwargs
    )
    return model, tokenizer


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
        continent_norm = COUNTRY_CONTINENT_MAP[country_code]
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


def build_user_content(
    item: StandardizedItem,
    with_metadata: bool,
    base_url: str,
    seed: int,
    metadata_tag_mode: str = "correct",
    shuffled_tags_by_qid: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[str, Dict[str, Optional[str]]]:
    options_block = "\n".join(
        [f"{chr(65 + i)}: {opt}" for i, opt in enumerate(item.options)]
    )
    core = (
        f"Question: {item.question}\n\n"
        f"Options:\n{options_block}\n\n"
        "Answer with the correct option."
    )

    if not with_metadata:
        return f"CONTENT:\n{core}", {
            "url_country_tag": None,
            "url_continent_tag": None,
            "metadata_source": "none",
        }

    if metadata_tag_mode == "shuffled":
        if shuffled_tags_by_qid is None:
            raise ValueError("shuffled_tags_by_qid must be set for metadata_tag_mode='shuffled'")
        tags = shuffled_tags_by_qid[item.question_id]
    else:
        tags = choose_metadata_tags(item, seed=seed, mode=metadata_tag_mode)
    content = (
        f"URL: {base_url}/{tags['country_tag']}\n"
        f"COUNTRY: {tags['country_tag']}\n"
        f"CONTINENT: {tags['continent_tag']}\n\n"
        f"TITLE: Facts about the country {tags['country_tag']}\n\n"
        "CONTENT:\n"
        f"{core}"
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

    if letter is not None:
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(options):
            answer_text = options[idx]

    if answer_text is None:
        lowered = raw.lower()
        matches = [opt for opt in options if opt.lower() in lowered]
        if matches:
            answer_text = max(matches, key=len)
            letter = chr(ord("A") + options.index(answer_text))

    return {"processed_option_letter": letter, "processed_answer": answer_text}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def evaluate_variant(
    args: argparse.Namespace,
    items: List[StandardizedItem],
    variant_name: str,
    model_type: str,
    with_metadata: bool,
    model,
    tokenizer,
    output_jsonl: Path,
    metadata_tag_mode: str = "correct",
) -> Dict[str, float]:
    ensure_parent(output_jsonl)
    total = len(items)
    correct = 0
    processed_total = 0

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

    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for start in tqdm(range(0, total, args.batch_size), desc=f"Eval {variant_name}"):
            batch = items[start : min(start + args.batch_size, total)]
            prompts = []
            meta_info_batch = []

            for item in batch:
                user_content, meta_info = build_user_content(
                    item=item,
                    with_metadata=with_metadata,
                    base_url=args.base_url,
                    seed=args.seed,
                    metadata_tag_mode=metadata_tag_mode,
                    shuffled_tags_by_qid=shuffled_tags_by_qid,
                )
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)
                meta_info_batch.append(meta_info)

            model_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
            ).to(model.device)

            with torch.no_grad():
                gen_kwargs = {"max_new_tokens": args.max_new_tokens}
                if args.decoding == "sample":
                    gen_kwargs.update(
                        {
                            "do_sample": True,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                        }
                    )
                else:
                    gen_kwargs.update({"do_sample": False})
                generated = model.generate(**model_inputs, **gen_kwargs)

            for i, item in enumerate(batch):
                input_len = int(model_inputs["attention_mask"][i].sum().item())
                gen_ids = generated[i][input_len:].tolist()
                raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                processed = extract_answer(raw_output, item.options)
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
                    "correct_answer": item.correct_answer,
                    "country": item.country,
                    "continent": item.continent,
                    "variant": variant_name,
                    "model_type": model_type,
                    "metadata": with_metadata,
                    "metadata_tag_mode": metadata_tag_mode,
                    "base_url": args.base_url,
                    "prompt": prompts[i],
                    "raw_output": raw_output,
                    **meta_info_batch[i],
                    **processed,
                }
                if is_correct is not None:
                    row["is_correct"] = is_correct
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
                model, tokenizer = build_model(
                    model_type,
                    hf_token=hf_token,
                    custom_metadata=(
                        bool(custom_model_metadata) if model_type == "custom" else None
                    ),
                )
                loaded_models[model_key] = (model, tokenizer)
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
            model, tokenizer = loaded_models[model_key]

        output_jsonl = out_dir / f"{args.benchmark}_{variant}.jsonl"
        summary = evaluate_variant(
            args=args,
            items=items,
            variant_name=variant,
            model_type=model_type,
            with_metadata=with_metadata,
            model=model,
            tokenizer=tokenizer,
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
