import os
import re
import json
import torch
import random
import hashlib
import numpy as np
import argparse
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
parser.add_argument('--dataset', default="YOUR_HF_USERNAME/qa_metacul", help="HF dataset name or local path")
parser.add_argument('--split', default="train", help="Dataset split to evaluate")
parser.add_argument('--output-jsonl', default=None)
parser.add_argument('--custom-model-path', default=None, help='Override the merged custom chat model path')
parser.add_argument('--llama-model-name', default="meta-llama/Llama-3.2-1B-Instruct", help='Baseline Llama model path or repo ID')
parser.add_argument('--base-url', default="www.globalfactcheck.org", help="Base URL for metadata block")
parser.add_argument('--url-corruption-rate', type=float, default=0.0, help="Fraction of samples with corrupted URL metadata")
parser.add_argument('--url-corruption-seed', type=int, default=42, help="Seed for deterministic URL corruption")
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--max-new-tokens', type=int, default=128)
parser.add_argument('--temperature', type=float, default=0.6)
parser.add_argument('--top-p', type=float, default=0.9)
args = parser.parse_args()
hf_token = os.environ.get("HF_TOKEN")

metadata = args.metadata
name_prefix = "combined_with_metadata" if metadata else "combined_without_metadata"
name = f"{name_prefix}_{args.size}"

model_type = args.model_type
if model_type == "custom":
    model_name = args.custom_model_path or f"/path/to/metacul/models/sft/{name}_chat"
else:
    if model_type == "llama3_chat":
        model_name = args.llama_model_name
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

chat_template_path="/path/to/metacul/src/chat_template.jinja"

if args.output_jsonl is None:
    suffix = "with_metadata" if metadata else "without_metadata"
    base_slug = re.sub(r"[^a-zA-Z0-9]+", "_", args.base_url).strip("_")
    rate_pct = int(round(args.url_corruption_rate * 100))
    corruption_suffix = f"_c{rate_pct}"
    args.output_jsonl = (
        f"/path/to/metacul/results/qa_metacul_eval_"
        f"{suffix}_{model_type}_{base_slug}{corruption_suffix}.jsonl"
    )

pretrained_kwargs = {}
if hf_token and not os.path.isdir(model_name):
    pretrained_kwargs["token"] = hf_token

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


def build_user_content(example: dict) -> str:
    question = example["question"].strip()
    options = example["options"]
    options_block = "\n".join(
        [f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options)]
    )
    return (
        f"Question: {question}\n\n"
        f"Options:\n{options_block}\n\n"
        "Answer with the correct option."
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
        example.get("country", ""),
        example.get("continent", ""),
        example.get("generated_by", ""),
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


def pick_wrong_tags(correct_country: str, correct_continent: str, example: dict) -> tuple[str, str]:
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


def get_metadata_tags(example: dict) -> dict:
    country_name = str(example.get("country", "")).strip()
    country_tag = COUNTRY_CODE_MAP.get(country_name)
    if country_tag is None:
        raise ValueError(f"Missing country code mapping for: {country_name}")
    continent_tag = str(example.get("continent", "")).strip().title()
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


def build_messages(example: dict) -> tuple[list, dict]:
    user_content = build_user_content(example)
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


def extract_answer(raw_output: str, options: list) -> dict:
    raw = raw_output.strip()
    letter_match = re.search(r"\b([A-D])\b", raw, re.IGNORECASE)
    letter = letter_match.group(1).upper() if letter_match else None
    answer_text = None

    if letter:
        idx = ord(letter) - 65
        if 0 <= idx < len(options):
            answer_text = options[idx]

    if answer_text is None:
        lowered = raw.lower()
        matches = []
        for opt in options:
            if opt.lower() in lowered:
                matches.append(opt)
        if matches:
            answer_text = max(matches, key=len)
            idx = options.index(answer_text)
            letter = chr(65 + idx)

    return {
        "processed_option_letter": letter,
        "processed_answer": answer_text,
    }


def ensure_output_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


test_dataset = load_dataset(args.dataset)[args.split]
ensure_output_dir(args.output_jsonl)

total = len(test_dataset)
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
        batch = [test_dataset[i] for i in range(start, end)]

        prompts = []
        tag_infos = []
        for ex in batch:
            messages, tag_info = build_messages(ex)
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            tag_infos.append(tag_info)

        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **model_inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )

        for i, ex in enumerate(batch):
            input_len = model_inputs.input_ids[i].size(0)
            gen_ids = generated[i][input_len:].tolist()
            raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            processed = extract_answer(raw_output, ex["options"])

            tag_info = tag_infos[i]
            row = {
                "question": ex["question"],
                "options": ex["options"],
                "correct_answer": ex["correct_answer"],
                "distractors": ex["distractors"],
                "country": ex["country"],
                "continent": ex["continent"],
                "generated_by": ex["generated_by"],
                "prompt": prompts[i],
                "raw_output": raw_output,
                "url_corrupted": tag_info["url_corrupted"],
                "url_country_tag": tag_info["url_country_tag"],
                "url_continent_tag": tag_info["url_continent_tag"],
                "true_country_tag": tag_info["true_country_tag"],
                "true_continent_tag": tag_info["true_continent_tag"],
                "url_corruption_rate": args.url_corruption_rate,
                **processed,
            }

            if processed["processed_answer"] is not None:
                is_correct = processed["processed_answer"] == ex["correct_answer"]
                row["is_correct"] = is_correct
                overall_total += 1
                overall_correct += int(is_correct)
                continent = ex["continent"]
                continent_totals[continent] = continent_totals.get(continent, 0) + 1
                continent_correct[continent] = continent_correct.get(continent, 0) + int(is_correct)
                continent_wrong[continent] = continent_wrong.get(continent, 0) + int(not is_correct)
            else:
                continent = ex["continent"]
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
