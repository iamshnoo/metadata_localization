#!/usr/bin/env python3
"""
Evaluate micro-average perplexity on MECO datasets for a list of model/test combos.
"""

import argparse
import csv
import json
import os
import re
import hashlib

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_EVAL_LIST = "/path/to/metacul/results/perplexity/eval_list.json"
DEFAULT_OUTPUT_CSV = "/path/to/metacul/results/perplexity_eval.csv"
DEFAULT_PER_SAMPLE_DIR = "/path/to/metacul/results/per_sample_losses"


class PerplexityEvaluator:
    def __init__(self, model_path, device="auto", tokenizer_path=None, local_files_only=False):
        print(f"Loading model from: {model_path}")
        tokenizer_errors = []
        self.tokenizer = None

        if tokenizer_path:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    local_files_only=True,
                )
            except Exception as exc:
                tokenizer_errors.append(
                    f"--tokenizer-path {tokenizer_path} failed: {exc}"
                )

        if self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                )
            except Exception as exc:
                tokenizer_errors.append(
                    f"tokenizer from model path failed: {exc}"
                )

        if self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Llama-3.2-1B",
                    local_files_only=local_files_only,
                )
            except Exception as exc:
                tokenizer_errors.append(
                    f"default tokenizer load failed: {exc}"
                )
                raise RuntimeError(
                    "Tokenizer not found. Provide --tokenizer-path to a local tokenizer, "
                    "ensure tokenizer files exist under the model path, or allow online "
                    "loading by omitting --local-files-only.\n"
                    + "\n".join(tokenizer_errors)
                ) from exc
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            #dtype=torch.float16,
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


def _slugify(value):
    value = value.strip().strip("/")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value or "unknown"


def per_sample_path(base_dir, model_path, test_set_path, split, max_samples, seed):
    key = f"{model_path}|{test_set_path}|{split}|{max_samples}|{seed}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    model_slug = _slugify(os.path.basename(model_path))
    test_slug = _slugify(os.path.basename(os.path.normpath(test_set_path)))
    filename = (
        f"{model_slug}__{test_slug}__{split}__max{max_samples}__seed{seed}__{digest}.jsonl"
    )
    return os.path.join(base_dir, filename)


def write_per_sample_losses(
    output_dir, model_path, test_set_path, split, max_samples, seed, samples
):
    os.makedirs(output_dir, exist_ok=True)
    output_path = per_sample_path(
        output_dir, model_path, test_set_path, split, max_samples, seed
    )
    meta = {
        "model_path": model_path,
        "test_set_path": test_set_path,
        "split": split,
        "max_samples": max_samples,
        "seed": seed,
    }
    meta_path = output_path.replace(".jsonl", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    with open(output_path, "w") as f:
        for row in samples:
            f.write(json.dumps(row) + "\n")


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
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Local path to tokenizer files (overrides model path lookup).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=False,
        help="Force tokenizer loading to use only local files (no HF download).",
    )
    parser.add_argument("--per-sample-dir", default=DEFAULT_PER_SAMPLE_DIR)
    parser.add_argument(
        "--save-per-sample",
        dest="save_per_sample",
        action="store_true",
        default=True,
        help="Save per-sample loss sums and token counts.",
    )
    parser.add_argument(
        "--no-save-per-sample",
        dest="save_per_sample",
        action="store_false",
        help="Disable saving per-sample losses.",
    )
    parser.add_argument(
        "--save-per-sample-existing",
        action="store_true",
        default=False,
        help="Also save per-sample losses for rows that already have results.",
    )
    parser.add_argument(
        "--save-per-sample-missing",
        action="store_true",
        default=False,
        help="Only backfill per-sample losses when files are missing for completed rows.",
    )
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

        model_path = combo["model_path"]
        dataset_path = combo["test_set_path"]
        if not os.path.exists(model_path) or not os.path.exists(dataset_path):
            pbar.update(1)
            continue
        already_done = has_result(row)
        per_sample_needed = False
        if args.save_per_sample_missing:
            sample_path = per_sample_path(
                args.per_sample_dir,
                model_path,
                dataset_path,
                args.split,
                args.max_samples,
                args.seed,
            )
            per_sample_needed = not os.path.exists(sample_path)

        if already_done and not args.save_per_sample_existing and not per_sample_needed:
            pbar.update(1)
            continue

        if model_path == last_model_path and last_evaluator is not None:
            evaluator = last_evaluator
        else:
            evaluator = PerplexityEvaluator(
                model_path,
                args.device,
                tokenizer_path=args.tokenizer_path,
                local_files_only=args.local_files_only,
            )
            last_model_path = model_path
            last_evaluator = evaluator
        dataset = load_dataset_split(dataset_path, args.split)

        rng = np.random.default_rng(args.seed)
        indices = sample_indices(len(dataset), args.max_samples, rng)

        print(f"Test set: {dataset_path}")
        loss_sums = []
        token_counts = []
        per_sample_rows = []
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
            per_sample_rows.append(
                {
                    "index": int(idx),
                    "loss_sum": float(loss_sum),
                    "token_count": int(token_count),
                    "nll": float(loss_sum) / float(token_count),
                }
            )

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

        if args.save_per_sample:
            write_per_sample_losses(
                args.per_sample_dir,
                model_path,
                dataset_path,
                args.split,
                args.max_samples,
                args.seed,
                per_sample_rows,
            )

        if not already_done:
            rows = [rows_by_key[k] for k in rows_by_key]
            write_rows(args.output_csv, rows, fieldnames)

        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    main()
