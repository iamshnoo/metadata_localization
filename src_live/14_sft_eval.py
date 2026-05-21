import os
import re
import json
import torch
import random
import hashlib
import difflib
import numpy as np
import argparse
from typing import Dict, List, Tuple
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--model-type', default='custom', help='Model type to evaluate')
parser.add_argument('--metadata', action='store_true', help='Include metadata in prompts')
parser.add_argument('--size', choices=['1b', '3b'], default='1b', help='Custom model size suffix')
parser.add_argument('--dataset', default="YOUR_HF_USERNAME/qa_metacul", help="HF dataset name or local path")
parser.add_argument('--split', default="train", help="Dataset split to evaluate")
parser.add_argument('--locale-role', choices=['target', 'contrast', 'auto'], default='target', help='Which locale/answer to evaluate for ambiguous benchmark items')
parser.add_argument(
    '--split-type-filter',
    choices=['all', 'explicit', 'ambiguous'],
    default='all',
    help='Optional LocalNewsQA split_type filter applied before evaluation.',
)
parser.add_argument('--output-jsonl', default=None)
parser.add_argument('--custom-model-path', default=None, help='Override the merged custom chat model path')
parser.add_argument('--base-model-path', default=None, help='Optional base model path when evaluating a PEFT adapter directly')
parser.add_argument('--peft-adapter-path', default=None, help='Optional PEFT adapter path to load on top of the custom base model')
parser.add_argument('--name-suffix', default='', help='Suffix appended to default custom model names, e.g. _best3b')
parser.add_argument('--llama-model-name', default="meta-llama/Llama-3.2-1B-Instruct", help='Baseline Llama model path or repo ID')
parser.add_argument('--base-url', default="www.globalfactcheck.org", help="Base URL for metadata block")
parser.add_argument(
    '--metadata-prompt-style',
    choices=[
        'legacy_code',
        'name_plain',
        'name_grounded',
        'code_grounded',
        'code_disambiguate',
        'name_strict',
        'code_grounded_strict',
        'country_first_strict',
        'code_grounded_countrycode_name',
    ],
    default='legacy_code',
    help='Formatting style for eval-time locale metadata prompts',
)
parser.add_argument(
    '--qa-prompt-style',
    choices=['question', 'instruction', 'instruction_input', 'question_answer'],
    default='question',
    help='Formatting style for the task content appended after CONTENT:',
)
parser.add_argument(
    '--answer-cue-style',
    choices=[
        'none',
        'answer_colon',
        'answer_newline',
        'final_answer_colon',
        'the_correct_answer_is',
        'country_answer_colon',
        'country_final_answer_colon',
        'country_the_correct_answer_is',
    ],
    default='none',
    help='Optional explicit answer cue appended after the question/options block before scoring option continuations.',
)
parser.add_argument('--url-corruption-rate', type=float, default=0.0, help="Fraction of samples with corrupted URL metadata")
parser.add_argument('--url-corruption-seed', type=int, default=42, help="Seed for deterministic URL corruption")
parser.add_argument(
    '--url-corruption-mode',
    choices=['full_mismatch', 'url_country_mask'],
    default='full_mismatch',
    help=(
        "How to perturb metadata when URL corruption is enabled. "
        "'full_mismatch' changes URL/country/continent together; "
        "'url_country_mask' masks only the country-code portion of the URL while keeping COUNTRY/CONTINENT correct."
    ),
)
parser.add_argument(
    '--url-mask-token',
    default='xx',
    help='Replacement token used for URL-only masking experiments.',
)
parser.add_argument('--shuffle-options', action='store_true', help='Deterministically shuffle multiple-choice options per item before prompting and scoring')
parser.add_argument('--option-shuffle-seed', type=int, default=42, help='Seed for deterministic option shuffling')
parser.add_argument('--omit-option-labels', action='store_true', help='Render multiple-choice options without A/B/C/D labels in the prompt')
parser.add_argument('--exact-option-text-instruction', action='store_true', help='When not using letter answers, instruct the model to answer with the exact text of the correct option')
parser.add_argument(
    '--mcq-scoring',
    choices=['option_text_avg', 'option_text_sum', 'option_letter'],
    default='option_text_avg',
    help='How to score multiple-choice candidates from the prompt',
)
parser.add_argument(
    '--length-norm-alpha',
    type=float,
    default=None,
    help='Optional generalized length-normalization exponent for option-text scoring. 0=sum, 1=avg.',
)
parser.add_argument(
    '--null-calibration-mode',
    choices=['none', 'question_masked', 'question_masked_no_metadata'],
    default='none',
    help='Optional calibration prompt used to subtract answer priors from MCQ option scores.',
)
parser.add_argument(
    '--null-calibration-beta',
    type=float,
    default=1.0,
    help='Weight on the calibration score when null calibration is enabled.',
)
parser.add_argument(
    '--null-question-text',
    default='N/A',
    help='Replacement question text used for question-masked null calibration prompts.',
)
parser.add_argument(
    '--answer-format',
    choices=['option', 'letter'],
    default='option',
    help='Instruction style used in the prompt for the final answer',
)
parser.add_argument('--max-examples', type=int, default=None, help='Optional deterministic subsample size for faster control runs')
parser.add_argument('--sample-seed', type=int, default=42, help='Seed for deterministic subsampling when max-examples is set')
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--max-new-tokens', type=int, default=128)
parser.add_argument('--temperature', type=float, default=0.6)
parser.add_argument('--top-p', type=float, default=0.9)
parser.add_argument('--add-prompt-bos', action='store_true', help='Prepend the tokenizer BOS token to raw prompts before log-likelihood scoring.')
parser.add_argument('--resume', action='store_true', help='Resume from an existing output JSONL by skipping already-written rows.')
args = parser.parse_args()
hf_token = os.environ.get("HF_TOKEN")

if args.mcq_scoring == "option_letter" and args.answer_format != "letter":
    args.answer_format = "letter"


def dataset_slug(dataset_name: str) -> str:
    if os.path.isdir(dataset_name):
        base = os.path.basename(os.path.abspath(dataset_name))
    else:
        base = dataset_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", base).strip("_").lower() or "dataset"

metadata = args.metadata
name_prefix = "combined_with_metadata" if metadata else "combined_without_metadata"
name = f"{name_prefix}_{args.size}"

model_type = args.model_type
adapter_path = args.peft_adapter_path
base_model_name = None
tokenizer_source = None
prompt_path_hint = None


def has_tokenizer_files(path: str | None) -> bool:
    if not path or not os.path.isdir(path):
        return False
    return any(
        os.path.exists(os.path.join(path, name))
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
            "spiece.model",
        )
    )


def tokenizer_source_or_fallback(path: str) -> str:
    if has_tokenizer_files(path):
        return path
    fallback = os.environ.get(
        "LOCALNEWSQA_TOKENIZER_SOURCE",
        "/path/to/metacul/models/combined_with_metadata_3b",
    )
    if os.path.isdir(path) and has_tokenizer_files(fallback):
        print(f"[warn] tokenizer files missing in {path}; using tokenizer from {fallback}")
        return fallback
    return path


if model_type == "custom":
    if adapter_path:
        try:
            from peft import PeftConfig
        except ImportError as exc:
            raise RuntimeError("PEFT adapter evaluation requested but `peft` is not installed") from exc
        peft_config = PeftConfig.from_pretrained(adapter_path)
        base_model_name = args.base_model_path or peft_config.base_model_name_or_path
        if not base_model_name:
            raise ValueError(
                "--base-model-path is required when the adapter config does not record a base model path"
            )
        model_name = base_model_name
        tokenizer_source = adapter_path
        prompt_path_hint = adapter_path
    else:
        model_name = args.custom_model_path or f"/path/to/metacul/models/sft/{name}{args.name_suffix}_chat"
        tokenizer_source = tokenizer_source_or_fallback(model_name)
        prompt_path_hint = model_name
else:
    if model_type == "llama3_chat":
        model_name = args.llama_model_name
        tokenizer_source = model_name
        prompt_path_hint = model_name
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

chat_template_path="/path/to/metacul/src/chat_template.jinja"


def custom_model_uses_chat_prompt(path: str) -> bool:
    norm = os.path.normpath(path)
    base = os.path.basename(norm)
    return base.endswith("_chat") or f"{os.sep}models{os.sep}sft{os.sep}" in norm

if args.output_jsonl is None:
    dataset_name_slug = dataset_slug(args.dataset)
    suffix = "with_metadata" if metadata else "without_metadata"
    base_slug = re.sub(r"[^a-zA-Z0-9]+", "_", args.base_url).strip("_")
    rate_pct = int(round(args.url_corruption_rate * 100))
    corruption_suffix = f"_c{rate_pct}"
    args.output_jsonl = (
        f"/path/to/metacul/results/{dataset_name_slug}_eval_"
        f"{args.locale_role}_{suffix}_{model_type}_{base_slug}{corruption_suffix}.jsonl"
    )

pretrained_kwargs = {"trust_remote_code": True}
if hf_token and not os.path.isdir(model_name):
    pretrained_kwargs["token"] = hf_token

try:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, fix_mistral_regex=True, **pretrained_kwargs
    )
except TypeError:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **pretrained_kwargs)
tokenizer.pad_token = tokenizer.eos_token

USES_CHAT_PROMPT = model_type == "llama3_chat" or (
    model_type == "custom" and custom_model_uses_chat_prompt(prompt_path_hint)
)

if model_type == "custom" and USES_CHAT_PROMPT:
    tokenizer.chat_template = open(chat_template_path, "r").read()

config = AutoConfig.from_pretrained(model_name, **pretrained_kwargs)
model_loader = AutoModelForCausalLM
if config.__class__.__name__ == "Mistral3Config":
    model_loader = AutoModelForImageTextToText

if adapter_path:
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("PEFT adapter evaluation requested but `peft` is not installed") from exc
    base_model = model_loader.from_pretrained(
        model_name,
        device_map="auto",
        dtype="auto",
        **pretrained_kwargs,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
else:
    model = model_loader.from_pretrained(
        model_name,
        device_map="auto",
        dtype="auto",
        **pretrained_kwargs,
    )


SYSTEM_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)

BASE_URL = args.base_url

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


def build_user_content(example: dict, prompt_options: list, answer_cue_country_name: str | None = None) -> str:
    question = example["question"].strip()
    options = prompt_options
    if args.omit_option_labels:
        options_block = "\n".join([f"- {opt}" for opt in options])
    else:
        options_block = "\n".join(
            [f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options)]
        )
    if args.answer_format == "letter":
        answer_instruction = "Answer with only the correct option letter (A, B, C, or D)."
    else:
        if args.exact_option_text_instruction:
            answer_instruction = "Answer with the exact text of the correct option."
        else:
            answer_instruction = "Answer with the correct option."

    if args.answer_cue_style == "answer_colon":
        answer_cue = "\n\nAnswer:"
    elif args.answer_cue_style == "answer_newline":
        answer_cue = "\n\nAnswer:\n"
    elif args.answer_cue_style == "final_answer_colon":
        answer_cue = "\n\nFinal answer:"
    elif args.answer_cue_style == "the_correct_answer_is":
        answer_cue = "\n\nThe correct answer is"
    elif args.answer_cue_style == "country_answer_colon":
        if answer_cue_country_name:
            answer_cue = f"\n\nFor {answer_cue_country_name}, answer:"
        else:
            answer_cue = "\n\nAnswer:"
    elif args.answer_cue_style == "country_final_answer_colon":
        if answer_cue_country_name:
            answer_cue = f"\n\nFor {answer_cue_country_name}, final answer:"
        else:
            answer_cue = "\n\nFinal answer:"
    elif args.answer_cue_style == "country_the_correct_answer_is":
        if answer_cue_country_name:
            answer_cue = f"\n\nFor {answer_cue_country_name}, the correct answer is"
        else:
            answer_cue = "\n\nThe correct answer is"
    else:
        answer_cue = ""

    if args.qa_prompt_style == "question":
        return (
            f"Question: {question}\n\n"
            f"Options:\n{options_block}\n\n"
            f"{answer_instruction}"
            f"{answer_cue}"
        )

    if args.qa_prompt_style == "instruction":
        return (
            "### Instruction:\n"
            f"{question}\n\n"
            "Options:\n"
            f"{options_block}\n\n"
            f"{answer_instruction}"
            f"{answer_cue}"
        )

    if args.qa_prompt_style == "instruction_input":
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

    if args.qa_prompt_style == "question_answer":
        answer_suffix = answer_cue if answer_cue else "\n\nAnswer:"
        return (
            f"Question: {question}\n\n"
            f"Options:\n{options_block}"
            f"{answer_suffix}"
        )

    raise ValueError(f"Unsupported qa_prompt_style: {args.qa_prompt_style}")


def add_metadata_block(content: str, tag_info: Dict[str, str]) -> str:
    url_country_tag = tag_info["prompt_url_country_tag"]
    country_tag = tag_info["prompt_country_tag"]
    continent_tag = tag_info["prompt_continent_tag"]
    country_name = tag_info.get("resolved_country_name") or country_tag
    prompt_style = args.metadata_prompt_style

    if prompt_style == "legacy_code":
        metadata_block = (
            f"URL: {BASE_URL}/{url_country_tag}\n"
            f"COUNTRY: {country_tag}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about the country {country_tag}\n\n"
            "CONTENT:\n"
        )
        return metadata_block + content

    if prompt_style == "name_plain":
        metadata_block = (
            f"URL: {BASE_URL}/{url_country_tag}\n"
            f"COUNTRY: {country_name}\n"
            f"COUNTRY_CODE: {country_tag}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about {country_name}\n\n"
            "CONTENT:\n"
        )
        return metadata_block + content

    if prompt_style == "name_grounded":
        metadata_block = (
            f"URL: {BASE_URL}/{url_country_tag}\n"
            f"COUNTRY: {country_name}\n"
            f"COUNTRY_CODE: {country_tag}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about {country_name}\n\n"
            "CONTENT:\n"
            f"Use the locale metadata above as grounding. Answer for {country_name}, not a different country.\n\n"
        )
        return metadata_block + content

    if prompt_style == "code_grounded":
        metadata_block = (
            f"URL: {BASE_URL}/{url_country_tag}\n"
            f"COUNTRY: {country_tag}\n"
            f"COUNTRY_NAME: {country_name}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about the country {country_tag}\n\n"
            "CONTENT:\n"
            f"Use the locale metadata above as grounding. This question is about {country_name}.\n\n"
        )
        return metadata_block + content

    if prompt_style == "code_grounded_countrycode_name":
        metadata_block = (
            f"URL: {BASE_URL}/{url_country_tag}\n"
            f"COUNTRY: {country_tag}\n"
            f"COUNTRY_CODE: {country_name}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about the country {country_tag}\n\n"
            "CONTENT:\n"
            f"Use the locale metadata above as grounding. This question is about {country_name}.\n\n"
        )
        return metadata_block + content

    if prompt_style == "code_disambiguate":
        metadata_block = (
            f"URL: {BASE_URL}/{url_country_tag}\n"
            f"COUNTRY: {country_tag}\n"
            f"COUNTRY_NAME: {country_name}\n"
            f"CONTINENT: {continent_tag}\n\n"
            f"TITLE: Facts about the country {country_tag}\n\n"
            "CONTENT:\n"
            f"Use the locale metadata above to disambiguate country-specific facts. "
            f"When multiple answers could fit different countries, answer for {country_name}.\n\n"
        )
        return metadata_block + content

    if prompt_style == "name_strict":
        metadata_block = (
            f"URL: {BASE_URL}/{url_country_tag}\n"
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

    if prompt_style == "code_grounded_strict":
        metadata_block = (
            f"URL: {BASE_URL}/{url_country_tag}\n"
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

    if prompt_style == "country_first_strict":
        metadata_block = (
            f"COUNTRY: {country_name}\n"
            f"COUNTRY_CODE: {country_tag}\n"
            f"CONTINENT: {continent_tag}\n"
            f"URL: {BASE_URL}/{url_country_tag}\n\n"
            f"TITLE: Facts about {country_name}\n\n"
            "CONTENT:\n"
            f"Answer this question for {country_name}. "
            f"When multiple options could fit different countries, pick the option that is factual for {country_name}.\n\n"
        )
        return metadata_block + content

    raise ValueError(f"Unsupported metadata_prompt_style: {prompt_style}")


def build_example_key(example: dict) -> str:
    parts = [
        example.get("question", ""),
        json.dumps(example.get("options", []), ensure_ascii=False),
        example.get("correct_answer", ""),
        example.get("target_answer", ""),
        example.get("contrast_answer", ""),
        example.get("country", ""),
        example.get("target_country", ""),
        example.get("contrast_country", ""),
        example.get("continent", ""),
        example.get("generated_by", ""),
        example.get("split_family", ""),
        example.get("split_type", ""),
        str(example.get("year", "")),
        example.get("topic", ""),
    ]
    return "||".join(parts)


def should_corrupt(example: dict) -> bool:
    rate = args.url_corruption_rate
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    key = f"{args.url_corruption_seed}||{build_example_key(example)}"
    digest = hashlib.sha1(key.encode()).hexdigest()
    val = int(digest[:8], 16) / 0xFFFFFFFF
    return val < rate


def get_prompt_options(example: dict) -> Tuple[List[str], List[int]]:
    options = list(example["options"])
    permutation = list(range(len(options)))
    if not args.shuffle_options:
        return options, permutation

    key = f"{args.option_shuffle_seed}||{build_example_key(example)}||option-shuffle"
    digest = hashlib.sha1(key.encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    rng.shuffle(permutation)
    shuffled = [options[i] for i in permutation]
    return shuffled, permutation


def pick_wrong_tags(correct_country: str, correct_continent: str, example: dict) -> Tuple[str, str]:
    candidates = [
        (code, continent)
        for code, continent in COUNTRY_CONTINENT_MAP.items()
        if code != correct_country and continent != correct_continent
    ]
    if not candidates:
        candidates = [
            (code, continent)
            for code, continent in COUNTRY_CONTINENT_MAP.items()
            if code != correct_country
        ]
    key = f"{args.url_corruption_seed}||{build_example_key(example)}||wrong"
    digest = hashlib.sha1(key.encode()).hexdigest()
    idx = int(digest[:8], 16) % len(candidates)
    return candidates[idx]


def resolve_eval_locale(example: dict) -> Tuple[str, str]:
    split_type = str(example.get("split_type", "")).strip().lower()
    if split_type != "ambiguous":
        return (
            str(example.get("country") or example.get("target_country") or "").strip(),
            str(example.get("continent") or "").strip().title(),
        )

    if args.locale_role == "contrast":
        country_name = str(example.get("contrast_country") or "").strip()
        continent_name = COUNTRY_CONTINENT_MAP.get(COUNTRY_CODE_MAP.get(country_name, ""), "")
        return country_name, str(continent_name).strip().title()

    country_name = str(example.get("target_country") or example.get("country") or "").strip()
    continent_name = str(example.get("continent") or "").strip().title()
    return country_name, continent_name


def resolve_eval_correct_answer(example: dict) -> str:
    split_type = str(example.get("split_type", "")).strip().lower()
    if split_type == "ambiguous" and args.locale_role == "contrast":
        return str(example.get("contrast_answer") or example.get("correct_answer") or "").strip()
    return str(example.get("target_answer") or example.get("correct_answer") or "").strip()


def resolve_country_tag(country_name: str) -> str | None:
    name = str(country_name or "").strip()
    if not name:
        return None
    direct = COUNTRY_CODE_MAP.get(name)
    if direct:
        return direct

    lowered = name.lower()
    if lowered in COUNTRY_CONTINENT_MAP:
        return lowered

    for known_name, code in COUNTRY_CODE_MAP.items():
        if known_name.lower() == lowered:
            return code

    for known_name, code in COUNTRY_CODE_MAP.items():
        kn = known_name.lower()
        if kn in lowered or lowered in kn:
            return code

    return None


def fallback_country_tag_from_continent(continent_name: str, example: dict) -> str:
    continent = str(continent_name or "").strip().title()
    if continent:
        candidates = sorted(
            [code for code, cont in COUNTRY_CONTINENT_MAP.items() if str(cont).strip().title() == continent]
        )
        if candidates:
            key = f"{args.url_corruption_seed}||{build_example_key(example)}||continent-fallback"
            digest = hashlib.sha1(key.encode()).hexdigest()
            idx = int(digest[:8], 16) % len(candidates)
            return candidates[idx]
    return "us"


def get_metadata_tags(example: dict) -> Dict[str, str]:
    country_name, continent_tag = resolve_eval_locale(example)
    country_tag = resolve_country_tag(country_name)
    if country_tag is None:
        country_tag = fallback_country_tag_from_continent(continent_tag, example)
    if not continent_tag:
        continent_tag = COUNTRY_CONTINENT_MAP.get(country_tag, "")
    if not continent_tag:
        continent_tag = "America"
    continent_tag = str(continent_tag).strip().title()
    corrupted = False
    url_country_code_masked = False
    prompt_url_country_tag = country_tag
    prompt_country_tag = country_tag
    prompt_continent_tag = continent_tag
    if should_corrupt(example):
        corrupted = True
        if args.url_corruption_mode == "full_mismatch":
            wrong_country_tag, wrong_continent_tag = pick_wrong_tags(
                country_tag, continent_tag, example
            )
            prompt_url_country_tag = wrong_country_tag
            prompt_country_tag = wrong_country_tag
            prompt_continent_tag = wrong_continent_tag
        elif args.url_corruption_mode == "url_country_mask":
            prompt_url_country_tag = args.url_mask_token
            url_country_code_masked = True
        else:
            raise ValueError(f"Unsupported url_corruption_mode: {args.url_corruption_mode}")
    return {
        "url_corrupted": corrupted,
        "url_country_code_masked": url_country_code_masked,
        "metadata_corruption_mode": args.url_corruption_mode,
        "true_country_tag": country_tag,
        "true_continent_tag": continent_tag,
        "prompt_url_country_tag": prompt_url_country_tag,
        "prompt_country_tag": prompt_country_tag,
        "prompt_continent_tag": prompt_continent_tag,
        "resolved_country_name": country_name,
    }


def build_messages(
    example: dict,
    prompt_options: list,
    force_metadata: bool | None = None,
) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    use_metadata = metadata if force_metadata is None else force_metadata
    tag_info = {
        "url_corrupted": False,
        "url_country_code_masked": False,
        "metadata_corruption_mode": "none",
        "true_country_tag": None,
        "true_continent_tag": None,
        "prompt_url_country_tag": None,
        "prompt_country_tag": None,
        "prompt_continent_tag": None,
    }
    cue_country_name = None
    if use_metadata:
        tag_info = get_metadata_tags(example)
        cue_country_name = tag_info.get("resolved_country_name")
    user_content = build_user_content(example, prompt_options, cue_country_name)
    if use_metadata:
        user_content = add_metadata_block(user_content, tag_info)
    else:
        user_content = f"CONTENT:\n{user_content}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ], tag_info


def render_prompt(messages: List[Dict[str, str]]) -> str:
    if USES_CHAT_PROMPT and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    return user.rstrip() + "\n"


def get_candidate_texts(options: list) -> list:
    letters = [chr(65 + i) for i in range(len(options))]
    if args.answer_cue_style == "answer_newline":
        option_prefix = ""
    else:
        option_prefix = " "
    if args.mcq_scoring == "option_letter":
        return [option_prefix + letter for letter in letters]
    return [option_prefix + str(opt).strip() for opt in options]


def score_candidate_texts(prompt: str, candidate_texts: list) -> Tuple[List[float], List[float]]:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    if args.add_prompt_bos and tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]], device=model.device, dtype=prompt_ids.dtype)
        if prompt_ids.numel() == 0 or int(prompt_ids[0, 0].item()) != tokenizer.bos_token_id:
            prompt_ids = torch.cat([bos, prompt_ids], dim=1)
    scores_sum = []
    scores_avg = []
    with torch.no_grad():
        for cand_text in candidate_texts:
            cand_ids = tokenizer(cand_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
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


def select_option_scores(scores_sum: List[float], scores_avg: List[float]) -> List[float]:
    if args.mcq_scoring == "option_letter":
        return scores_avg
    if args.length_norm_alpha is not None:
        selected = []
        for score_sum, score_avg in zip(scores_sum, scores_avg):
            tok_len = abs(score_sum / score_avg) if score_avg != 0 else 1.0
            selected.append(score_sum / max(tok_len, 1.0) ** args.length_norm_alpha)
        return selected
    if args.mcq_scoring == "option_text_sum":
        return scores_sum
    return scores_avg


def extract_answer(raw_output: str, options: list) -> dict:
    raw = raw_output.strip()
    letter_match = re.search(r"\b([A-D])\b", raw, re.IGNORECASE)
    letter = letter_match.group(1).upper() if letter_match else None
    answer_text = None
    method = "letter"

    if letter:
        idx = ord(letter) - 65
        if 0 <= idx < len(options):
            answer_text = options[idx]

    if answer_text is None:
        method = "option_substring"
        lowered = raw.lower()
        matches = []
        for opt in options:
            if opt.lower() in lowered:
                matches.append(opt)
        if matches:
            answer_text = max(matches, key=len)
            idx = options.index(answer_text)
            letter = chr(65 + idx)

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
        letter = chr(65 + idx)

    return {
        "processed_option_letter": letter,
        "processed_answer": answer_text,
        "answer_extraction_method": method,
    }


def predict_option_by_loglikelihood(prompt: str, options: list, calibration_prompt: str | None = None) -> dict:
    # OLMES-style MCQ scoring with optional full-text or letter-only candidates.
    candidate_texts = get_candidate_texts(options)
    scores_sum, scores_avg = score_candidate_texts(prompt, candidate_texts)
    primary_scores = select_option_scores(scores_sum, scores_avg)
    calibration_sums = None
    calibration_avgs = None
    calibration_scores = None
    if calibration_prompt is not None and args.null_calibration_mode != "none":
        calibration_sums, calibration_avgs = score_candidate_texts(calibration_prompt, candidate_texts)
        calibration_scores = select_option_scores(calibration_sums, calibration_avgs)
        best_scores = [
            score - args.null_calibration_beta * calibration_score
            for score, calibration_score in zip(primary_scores, calibration_scores)
        ]
    else:
        best_scores = primary_scores
    best_idx = int(np.argmax(best_scores)) if best_scores else 0
    best_letter = chr(65 + best_idx) if best_idx < 26 else None
    return {
        "processed_option_letter": best_letter,
        "processed_answer": options[best_idx],
        "answer_extraction_method": args.mcq_scoring,
        "option_loglikelihood_sums": scores_sum,
        "option_loglikelihood_avgs": scores_avg,
        "option_loglikelihood_length_norm_alpha": args.length_norm_alpha,
        "option_loglikelihood_primary_scores": primary_scores,
        "null_calibration_mode": args.null_calibration_mode,
        "null_calibration_beta": args.null_calibration_beta,
        "null_calibration_option_loglikelihood_sums": calibration_sums,
        "null_calibration_option_loglikelihood_avgs": calibration_avgs,
        "null_calibration_option_scores": calibration_scores,
        "option_loglikelihood_selected_scores": best_scores,
        "scoring_candidates": candidate_texts,
    }


def ensure_output_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _should_force_redownload(exc: Exception) -> bool:
    message = str(exc)
    return "Feature type 'List' not found" in message


def _load_eval_dataset(dataset_name: str, split_name: str):
    if os.path.isfile(dataset_name):
        suffix = os.path.splitext(dataset_name)[1].lower()
        if suffix in {".jsonl", ".json"}:
            return load_dataset("json", data_files={split_name: dataset_name})[split_name]
        if suffix == ".csv":
            return load_dataset("csv", data_files={split_name: dataset_name})[split_name]
    direct_attempts = (
        {"split": split_name},
        {},
    )
    last_error = None
    for kwargs in direct_attempts:
        try:
            loaded = load_dataset(dataset_name, **kwargs)
            return loaded if "split" in kwargs else loaded[split_name]
        except Exception as exc:
            last_error = exc
            if not _should_force_redownload(exc):
                continue
            try:
                loaded = load_dataset(
                    dataset_name,
                    download_mode="force_redownload",
                    **kwargs,
                )
                return loaded if "split" in kwargs else loaded[split_name]
            except Exception as force_exc:
                last_error = force_exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to load dataset '{dataset_name}' split '{split_name}'")


test_dataset = _load_eval_dataset(args.dataset, args.split)
ensure_output_dir(args.output_jsonl)

selected_indices = list(range(len(test_dataset)))
if args.split_type_filter != "all":
    wanted = args.split_type_filter.strip().lower()
    selected_indices = [
        idx
        for idx in selected_indices
        if str(test_dataset[idx].get("split_type", "")).strip().lower() == wanted
    ]
if args.max_examples is not None and args.max_examples > 0 and args.max_examples < len(selected_indices):
    sample_rng = random.Random(args.sample_seed)
    selected_indices = sample_rng.sample(selected_indices, args.max_examples)

total = len(selected_indices)
batch_size = args.batch_size
overall_total = 0
overall_correct = 0
continent_totals = {}
continent_correct = {}
continent_wrong = {}
continent_skipped = {}
completed_rows = 0
file_mode = "w"
if args.resume and os.path.exists(args.output_jsonl):
    with open(args.output_jsonl, "r", encoding="utf-8") as existing_f:
        for line in existing_f:
            if line.strip():
                completed_rows += 1
    if completed_rows > total:
        raise ValueError(
            f"--resume found {completed_rows} existing rows, but only {total} examples are selected."
        )
    if completed_rows:
        file_mode = "a"
        selected_indices = selected_indices[completed_rows:]
        total = len(selected_indices)
        print(
            f"[resume] Skipping {completed_rows} previously written rows; "
            f"{total} rows remain."
        )
    else:
        print("[resume] Existing output file is empty; starting from scratch.")

with open(args.output_jsonl, file_mode, encoding="utf-8") as out_f:
    for start in tqdm(range(0, total, batch_size), desc="Evaluating"):
        end = min(start + batch_size, total)
        batch = [test_dataset[selected_indices[i]] for i in range(start, end)]

        prompts = []
        calibration_prompts = []
        tag_infos = []
        prompt_options_batch = []
        prompt_permutations = []
        for ex in batch:
            prompt_options, prompt_permutation = get_prompt_options(ex)
            messages, tag_info = build_messages(ex, prompt_options)
            prompts.append(render_prompt(messages))
            calibration_prompt = None
            if args.null_calibration_mode in {"question_masked", "question_masked_no_metadata"}:
                calibration_example = dict(ex)
                calibration_example["question"] = args.null_question_text
                calibration_force_metadata = (
                    False if args.null_calibration_mode == "question_masked_no_metadata" else None
                )
                calibration_messages, _ = build_messages(
                    calibration_example,
                    prompt_options,
                    force_metadata=calibration_force_metadata,
                )
                calibration_prompt = render_prompt(calibration_messages)
            calibration_prompts.append(calibration_prompt)
            tag_infos.append(tag_info)
            prompt_options_batch.append(prompt_options)
            prompt_permutations.append(prompt_permutation)

        for i, ex in enumerate(batch):
            prompt_options = prompt_options_batch[i]
            processed = predict_option_by_loglikelihood(prompts[i], prompt_options, calibration_prompts[i])
            raw_output = processed["processed_option_letter"]

            tag_info = tag_infos[i]
            eval_correct_answer = resolve_eval_correct_answer(ex)
            try:
                prompt_correct_idx = prompt_options.index(eval_correct_answer)
                prompt_correct_option_letter = chr(65 + prompt_correct_idx)
            except ValueError:
                prompt_correct_option_letter = None
            row = dict(ex)
            row.update({
                "prompt": prompts[i],
                "calibration_prompt": calibration_prompts[i],
                "raw_output": raw_output,
                "eval_locale_role": args.locale_role,
                "eval_correct_answer": eval_correct_answer,
                "prompt_options": prompt_options,
                "prompt_option_permutation": prompt_permutations[i],
                "prompt_correct_option_letter": prompt_correct_option_letter,
                "options_shuffled": args.shuffle_options,
                "option_shuffle_seed": args.option_shuffle_seed,
                "mcq_scoring": args.mcq_scoring,
                "answer_format": args.answer_format,
                "max_examples": args.max_examples,
                "sample_seed": args.sample_seed,
                "add_prompt_bos": args.add_prompt_bos,
                "url_corrupted": tag_info["url_corrupted"],
                "url_country_code_masked": tag_info["url_country_code_masked"],
                "metadata_corruption_mode": tag_info["metadata_corruption_mode"],
                "url_country_tag": tag_info["prompt_url_country_tag"],
                "url_continent_tag": tag_info["prompt_continent_tag"],
                "prompt_country_tag": tag_info["prompt_country_tag"],
                "prompt_continent_tag": tag_info["prompt_continent_tag"],
                "true_country_tag": tag_info["true_country_tag"],
                "true_continent_tag": tag_info["true_continent_tag"],
                "url_corruption_rate": args.url_corruption_rate,
                "metadata_prompt_style": args.metadata_prompt_style if metadata else "none",
                "qa_prompt_style": args.qa_prompt_style,
                "answer_cue_style": args.answer_cue_style,
                **processed,
            })

            if processed["processed_answer"] is not None:
                is_correct = processed["processed_answer"] == eval_correct_answer
                row["is_correct"] = is_correct
                overall_total += 1
                overall_correct += int(is_correct)
                continent = str(ex.get("continent", ""))
                continent_totals[continent] = continent_totals.get(continent, 0) + 1
                continent_correct[continent] = continent_correct.get(continent, 0) + int(is_correct)
                continent_wrong[continent] = continent_wrong.get(continent, 0) + int(not is_correct)
            else:
                continent = str(ex.get("continent", ""))
                continent_skipped[continent] = continent_skipped.get(continent, 0) + 1

            # print(row)
            # print("-----")
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"[✔] Wrote results to {args.output_jsonl}")
if overall_total:
    overall_acc = overall_correct / overall_total
    print(f"Overall accuracy: {overall_acc:.4f} ({overall_correct}/{overall_total})")
else:
    print("Overall accuracy: N/A (no valid processed answers)")

if continent_totals:
    print("Per-continent outcomes (correct / wrong / skipped):")
    for continent in sorted(set(list(continent_totals.keys()) + list(continent_skipped.keys()))):
        c_correct = continent_correct.get(continent, 0)
        c_wrong = continent_wrong.get(continent, 0)
        c_skipped = continent_skipped.get(continent, 0)
        print(f"  {continent}: {c_correct} / {c_wrong} / {c_skipped}")
