#!/usr/bin/env python3
"""
Evaluate micro-average perplexity on MECO datasets for a list of model/test combos.
"""

import argparse
import csv
import json
import os
import re

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_EVAL_LIST = "/scratch/$USER$/metacul/results/perplexity/eval_list.json"
DEFAULT_OUTPUT_CSV = "/scratch/$USER$/metacul/results/perplexity_eval.csv"


class PerplexityEvaluator:
    def __init__(self, model_path, device="auto"):
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map=device,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def token_loss_sum_and_count(self, text, max_length=1024):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = inputs.input_ids.to(self.model.device)

        title_text = "TITLE:"
        title_tokens = self.tokenizer.encode(title_text, add_special_tokens=False)

        title_pos = None
        for i in range(len(input_ids[0]) - len(title_tokens) + 1):
            if torch.equal(
                input_ids[0][i : i + len(title_tokens)],
                torch.tensor(title_tokens).to(input_ids.device),
            ):
                title_pos = i
                break

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        if title_pos is None:
            content_losses = losses[0]
        else:
            content_losses = losses[0][title_pos:]

        if content_losses.numel() == 0:
            return None

        return float(content_losses.sum().item()), int(content_losses.numel())


def load_dataset_split(dataset_path, split):
    split_path = f"{dataset_path.rstrip('/')}/{split}"
    if os.path.exists(split_path):
        return load_from_disk(split_path)

    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, dict) and split in dataset:
        return dataset[split]
    if hasattr(dataset, "keys") and split in dataset:
        return dataset[split]
    return dataset


def sample_indices(total, max_samples, rng):
    if max_samples is None or total <= max_samples:
        return list(range(total))
    return rng.choice(total, size=max_samples, replace=False).tolist()


def bootstrap_ci(loss_sums, token_counts, iters, ci_level, rng):
    losses = np.array(loss_sums, dtype=np.float64)
    tokens = np.array(token_counts, dtype=np.int64)
    n = len(losses)
    if n == 0:
        return float("nan"), float("nan")

    boot_means = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        total_loss = losses[idx].sum()
        total_tokens = tokens[idx].sum()
        boot_means[i] = np.exp(total_loss / total_tokens)

    alpha = (1.0 - ci_level) / 2.0
    low = np.quantile(boot_means, alpha)
    high = np.quantile(boot_means, 1.0 - alpha)
    return float(low), float(high)


def read_eval_list(path):
    with open(path, "r") as f:
        return json.load(f)


def read_existing_rows(path):
    if not os.path.exists(path):
        return {}
    rows = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("model_path", ""), row.get("test_set_path", ""))
            rows[key] = row
    return rows


def has_result(row):
    if not row:
        return False
    mean = row.get("mean_ppl", "")
    if mean is None:
        return False
    mean = str(mean).strip()
    return mean != "" and mean.lower() != "nan"


def write_rows(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate micro-average perplexity on MECO datasets."
    )
    parser.add_argument("--eval-list", default=DEFAULT_EVAL_LIST)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--model-path", help="Evaluate a single model path")
    parser.add_argument("--test-set-path", help="Evaluate a single test set path")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    args = parser.parse_args()

    if args.model_path or args.test_set_path:
        if not (args.model_path and args.test_set_path):
            parser.error("--model-path and --test-set-path must be used together")
        combos = [
            {"model_path": args.model_path, "test_set_path": args.test_set_path}
        ]
    else:
        combos = read_eval_list(args.eval_list)
    existing = read_existing_rows(args.output_csv)

    fieldnames = [
        "model_path",
        "test_set_path",
        "mean_ppl",
        "ci_low",
        "ci_high",
        "skipped_samples",
        "ci_level",
        "ci_method",
        "split",
        "max_samples",
        "seed",
        "bootstrap_iters",
    ]

    rows_by_key = {}
    for combo in combos:
        key = (combo["model_path"], combo["test_set_path"])
        row = existing.get(key, {})
        row.setdefault("model_path", combo["model_path"])
        row.setdefault("test_set_path", combo["test_set_path"])
        row.setdefault("ci_level", str(args.ci_level))
        row.setdefault("ci_method", "bootstrap")
        row.setdefault("split", args.split)
        row.setdefault("max_samples", str(args.max_samples))
        row.setdefault("seed", str(args.seed))
        row.setdefault("bootstrap_iters", str(args.bootstrap_iters))
        rows_by_key[key] = row

    total = len(combos)
    pbar = tqdm(total=total, desc="Perplexity eval")
    last_model_path = None
    last_evaluator = None

    for combo in combos:
        key = (combo["model_path"], combo["test_set_path"])
        row = rows_by_key[key]

        if has_result(row):
            pbar.update(1)
            continue

        model_path = combo["model_path"]
        dataset_path = combo["test_set_path"]
        if not os.path.exists(model_path) or not os.path.exists(dataset_path):
            pbar.update(1)
            continue

        if model_path == last_model_path and last_evaluator is not None:
            evaluator = last_evaluator
        else:
            evaluator = PerplexityEvaluator(model_path, args.device)
            last_model_path = model_path
            last_evaluator = evaluator
        dataset = load_dataset_split(dataset_path, args.split)

        rng = np.random.default_rng(args.seed)
        indices = sample_indices(len(dataset), args.max_samples, rng)

        print(f"Test set: {dataset_path}")
        loss_sums = []
        token_counts = []
        skipped_samples = 0

        for idx in tqdm(indices, desc="Calculating perplexity", leave=False):
            sample = dataset[int(idx)]
            text = sample["text"]
            result = evaluator.token_loss_sum_and_count(text)
            if result is None:
                skipped_samples += 1
                continue
            loss_sum, token_count = result
            if token_count <= 0 or not np.isfinite(loss_sum):
                skipped_samples += 1
                continue
            loss_sums.append(loss_sum)
            token_counts.append(token_count)

        total_loss = float(np.sum(loss_sums)) if loss_sums else 0.0
        total_tokens = float(np.sum(token_counts)) if token_counts else 0.0
        if loss_sums and total_tokens > 0:
            mean_ppl = float(np.exp(total_loss / total_tokens))
            ci_low, ci_high = bootstrap_ci(
                loss_sums, token_counts, args.bootstrap_iters, args.ci_level, rng
            )
        else:
            mean_ppl = float("nan")
            ci_low = float("nan")
            ci_high = float("nan")
        row["mean_ppl"] = f"{mean_ppl:.6f}"
        row["ci_low"] = f"{ci_low:.6f}"
        row["ci_high"] = f"{ci_high:.6f}"
        row["skipped_samples"] = str(skipped_samples)
        print(
            f"Result mean_ppl={row['mean_ppl']} ci=[{row['ci_low']}, {row['ci_high']}]"
        )

        rows = [rows_by_key[k] for k in rows_by_key]
        write_rows(args.output_csv, rows, fieldnames)

        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    main()
