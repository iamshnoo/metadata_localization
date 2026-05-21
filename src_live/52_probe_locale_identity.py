#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


CONTINENTS = ["Africa", "America", "Asia", "Europe"]
DEFAULT_DATA_ROOT = Path("/path/to/metacul/training_data/hf_datasets/continents")
DEFAULT_OUTPUT_ROOT = Path("/path/to/metacul/results/mechanistic/locale_probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe locale identity from metadata-boundary hidden states."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--variant", required=True, help="Short label for outputs, e.g. url_step2k")
    parser.add_argument(
        "--prompt-kind",
        choices=["url", "country", "continent", "none"],
        required=True,
        help="Which metadata format to feed the model.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-per-continent", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def normalize_country(raw: str) -> str:
    mapping = {
        "bd": "Bangladesh",
        "ca": "Canada",
        "gb": "United Kingdom",
        "gh": "Ghana",
        "hk": "Hong Kong",
        "ie": "Ireland",
        "in": "India",
        "jm": "Jamaica",
        "ke": "Kenya",
        "lk": "Sri Lanka",
        "my": "Malaysia",
        "ng": "Nigeria",
        "ph": "Philippines",
        "pk": "Pakistan",
        "tz": "Tanzania",
        "us": "United States",
        "za": "South Africa",
    }
    raw = str(raw or "").strip()
    return mapping.get(raw.lower(), raw)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_rows(data_root: Path, split: str, sample_per_continent: int, seed: int) -> List[Dict[str, str]]:
    rows = []
    for idx, continent in enumerate(CONTINENTS):
        ds = load_from_disk(str(data_root / continent / split))
        rng = random.Random(seed + idx)
        if sample_per_continent >= len(ds):
            sample_indices = list(range(len(ds)))
        else:
            sample_indices = rng.sample(range(len(ds)), sample_per_continent)
        for sample_idx in sample_indices:
            row = dict(ds[sample_idx])
            row["country_name"] = normalize_country(row.get("country"))
            row["continent_name"] = str(row.get("continent") or continent)
            rows.append(row)
    return rows


def build_prompt_parts(row: Dict[str, str], prompt_kind: str) -> Tuple[str, str]:
    title_block = f"TITLE: {row['title']}\n\n"
    content_block = f"CONTENT: {row['content']}"

    if prompt_kind == "url":
        prefix = f"URL: {row['url']}\n\n"
        rep_prefix = prefix
    elif prompt_kind == "country":
        prefix = f"COUNTRY: {row['country']}\n\n"
        rep_prefix = prefix
    elif prompt_kind == "continent":
        prefix = f"CONTINENT: {row['continent_name']}\n\n"
        rep_prefix = prefix
    elif prompt_kind == "none":
        prefix = ""
        rep_prefix = title_block
    else:
        raise ValueError(f"Unsupported prompt kind: {prompt_kind}")

    return prefix + title_block + content_block, rep_prefix


def collate_examples(
    tokenizer,
    examples: List[Dict[str, str]],
    prompt_kind: str,
    max_length: int,
):
    texts = []
    boundary_indices = []
    meta = []
    for row in examples:
        full_text, rep_prefix = build_prompt_parts(row, prompt_kind)
        full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)
        rep_ids = tokenizer(rep_prefix, add_special_tokens=False)
        boundary_idx = min(len(rep_ids["input_ids"]) - 1, len(full_ids["input_ids"]) - 1)
        texts.append(full_text)
        boundary_indices.append(boundary_idx)
        meta.append(
            {
                "doc_id": row["doc_id"],
                "country": row["country_name"],
                "continent": row["continent_name"],
                "year": row["year"],
                "boundary_idx": boundary_idx,
            }
        )
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    return batch, boundary_indices, meta


def fit_probe(X: np.ndarray, y: List[str], seed: int) -> Dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, random_state=seed),
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    variant_dir = args.output_root / args.variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    sampled_rows = sample_rows(args.data_root, args.split, args.sample_per_continent, args.seed)
    all_hidden = []
    all_meta: List[Dict[str, str]] = []

    for start in range(0, len(sampled_rows), args.batch_size):
        batch_rows = sampled_rows[start : start + args.batch_size]
        batch_inputs, boundary_indices, batch_meta = collate_examples(
            tokenizer, batch_rows, args.prompt_kind, args.max_length
        )
        batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
        with torch.no_grad():
            outputs = model(**batch_inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1].detach().float().cpu().numpy()
        for i, boundary_idx in enumerate(boundary_indices):
            all_hidden.append(hidden[i, boundary_idx, :])
            all_meta.append(batch_meta[i])

    X = np.stack(all_hidden)
    continents = [row["continent"] for row in all_meta]
    countries = [row["country"] for row in all_meta]

    continent_metrics = fit_probe(X, continents, args.seed)
    country_metrics = fit_probe(X, countries, args.seed)

    np.savez_compressed(
        variant_dir / "embeddings.npz",
        X=X,
        continents=np.array(continents),
        countries=np.array(countries),
    )
    pd.DataFrame(all_meta).to_csv(variant_dir / "metadata.csv", index=False)

    metrics = {
        "variant": args.variant,
        "model_path": args.model_path,
        "prompt_kind": args.prompt_kind,
        "sample_per_continent": args.sample_per_continent,
        "total_examples": int(len(all_meta)),
        "continent_probe": continent_metrics,
        "country_probe": country_metrics,
    }
    (variant_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
