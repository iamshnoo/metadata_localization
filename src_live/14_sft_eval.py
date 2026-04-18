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
from transformers import AutoModelForCausalLM, AutoTokenizer
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
parser.add_argument('--dataset', default="iamshnoo/qa_metacul", help="HF dataset name or local path")
parser.add_argument('--split', default="train", help="Dataset split to evaluate")
parser.add_argument('--locale-role', choices=['target', 'contrast', 'auto'], default='target', help='Which locale/answer to evaluate for ambiguous benchmark items')
parser.add_argument('--output-jsonl', default=None)
parser.add_argument('--custom-model-path', default=None, help='Override the merged custom chat model path')
parser.add_argument('--name-suffix', default='', help='Suffix appended to default custom model names, e.g. _best3b')
parser.add_argument('--llama-model-name', default="meta-llama/Llama-3.2-1B-Instruct", help='Baseline Llama model path or repo ID')
parser.add_argument('--base-url', default="www.globalfactcheck.org", help="Base URL for metadata block")
parser.add_argument('--url-corruption-rate', type=float, default=0.0, help="Fraction of samples with corrupted URL metadata")
parser.add_argument('--url-corruption-seed', type=int, default=42, help="Seed for deterministic URL corruption")
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
if model_type == "custom":
    model_name = args.custom_model_path or f"/scratch/amukher6/metacul/models/sft/{name}{args.name_suffix}_chat"
else:
    if model_type == "llama3_chat":
        model_name = args.llama_model_name
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

chat_template_path="/scratch/amukher6/metacul/src/chat_template.jinja"

if args.output_jsonl is None:
    dataset_name_slug = dataset_slug(args.dataset)
    suffix = "with_metadata" if metadata else "without_metadata"
    base_slug = re.sub(r"[^a-zA-Z0-9]+", "_", args.base_url).strip("_")
    rate_pct = int(round(args.url_corruption_rate * 100))
    corruption_suffix = f"_c{rate_pct}"
    args.output_jsonl = (
        f"/scratch/amukher6/metacul/results/{dataset_name_slug}_eval_"
        f"{args.locale_role}_{suffix}_{model_type}_{base_slug}{corruption_suffix}.jsonl"
    )

pretrained_kwargs = {}
if hf_token and not os.path.isdir(model_name):
    pretrained_kwargs["token"] = hf_token

try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, fix_mistral_regex=True, **pretrained_kwargs
    )
except TypeError:
    tokenizer = AutoTokenizer.from_pretrained(model_name, **pretrained_kwargs)
tokenizer.pad_token = tokenizer.eos_token

if model_type == "custom":
    tokenizer.chat_template = open(chat_template_path, "r").read()

model = AutoModelForCausalLM.from_pretrained(
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


def build_user_content(example: dict, prompt_options: list) -> str:
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
    return (
        f"Question: {question}\n\n"
        f"Options:\n{options_block}\n\n"
        f"{answer_instruction}"
    )


def add_metadata_block(content: str, country_tag: str, continent_tag: str) -> str:
    metadata_block = (
        f"URL: {BASE_URL}/{country_tag}\n"
        f"COUNTRY: {country_tag}\n"
        f"CONTINENT: {continent_tag}\n\n"
        f"TITLE: Facts about the country {country_tag}\n\n"
        "CONTENT:\n"
    )
    return metadata_block + content


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
    used_country_tag = country_tag
    used_continent_tag = continent_tag
    if should_corrupt(example):
        corrupted = True
        used_country_tag, used_continent_tag = pick_wrong_tags(
            country_tag, continent_tag, example
        )
    return {
        "url_corrupted": corrupted,
        "true_country_tag": country_tag,
        "true_continent_tag": continent_tag,
        "url_country_tag": used_country_tag,
        "url_continent_tag": used_continent_tag,
    }


def build_messages(example: dict, prompt_options: list) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    user_content = build_user_content(example, prompt_options)
    tag_info = {
        "url_corrupted": False,
        "true_country_tag": None,
        "true_continent_tag": None,
        "url_country_tag": None,
        "url_continent_tag": None,
    }
    if metadata:
        tag_info = get_metadata_tags(example)
        user_content = add_metadata_block(
            user_content, tag_info["url_country_tag"], tag_info["url_continent_tag"]
        )
    else:
        user_content = f"CONTENT:\n{user_content}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ], tag_info


def render_prompt(messages: List[Dict[str, str]]) -> str:
    # Some tokenizers (e.g., certain base checkpoints) do not ship a chat template.
    # Fall back to a deterministic plain-text format instead of failing.
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    return f"System: {system}\n\nUser:\n{user}\n\nAssistant:"


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


def predict_option_by_loglikelihood(prompt: str, options: list) -> dict:
    # OLMES-style MCQ scoring with optional full-text or letter-only candidates.
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    letters = [chr(65 + i) for i in range(len(options))]
    if args.mcq_scoring == "option_letter":
        candidate_texts = [" " + letter for letter in letters]
    else:
        candidate_texts = [" " + str(opt).strip() for opt in options]
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

    if args.mcq_scoring == "option_letter":
        best_scores = scores_avg
    elif args.length_norm_alpha is not None:
        best_scores = []
        for score_sum, score_avg in zip(scores_sum, scores_avg):
            tok_len = abs(score_sum / score_avg) if score_avg != 0 else 1.0
            best_scores.append(score_sum / max(tok_len, 1.0) ** args.length_norm_alpha)
    elif args.mcq_scoring == "option_text_sum":
        best_scores = scores_sum
    else:
        best_scores = scores_avg
    best_idx = int(np.argmax(best_scores)) if best_scores else 0
    best_letter = chr(65 + best_idx) if best_idx < 26 else None
    return {
        "processed_option_letter": best_letter,
        "processed_answer": options[best_idx],
        "answer_extraction_method": args.mcq_scoring,
        "option_loglikelihood_sums": scores_sum,
        "option_loglikelihood_avgs": scores_avg,
        "option_loglikelihood_length_norm_alpha": args.length_norm_alpha,
        "option_loglikelihood_selected_scores": best_scores,
        "scoring_candidates": candidate_texts,
    }


def ensure_output_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


try:
    test_dataset = load_dataset(args.dataset, split=args.split)
except Exception:
    test_dataset = load_dataset(args.dataset)[args.split]
ensure_output_dir(args.output_jsonl)

selected_indices = list(range(len(test_dataset)))
if args.max_examples is not None and args.max_examples > 0 and args.max_examples < len(test_dataset):
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

with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
    for start in tqdm(range(0, total, batch_size), desc="Evaluating"):
        end = min(start + batch_size, total)
        batch = [test_dataset[selected_indices[i]] for i in range(start, end)]

        prompts = []
        tag_infos = []
        prompt_options_batch = []
        prompt_permutations = []
        for ex in batch:
            prompt_options, prompt_permutation = get_prompt_options(ex)
            messages, tag_info = build_messages(ex, prompt_options)
            prompts.append(render_prompt(messages))
            tag_infos.append(tag_info)
            prompt_options_batch.append(prompt_options)
            prompt_permutations.append(prompt_permutation)

        for i, ex in enumerate(batch):
            prompt_options = prompt_options_batch[i]
            processed = predict_option_by_loglikelihood(prompts[i], prompt_options)
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
                "url_corrupted": tag_info["url_corrupted"],
                "url_country_tag": tag_info["url_country_tag"],
                "url_continent_tag": tag_info["url_continent_tag"],
                "true_country_tag": tag_info["true_country_tag"],
                "true_continent_tag": tag_info["true_continent_tag"],
                "url_corruption_rate": args.url_corruption_rate,
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
