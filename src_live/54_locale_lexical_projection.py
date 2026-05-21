#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_OUTPUT_ROOT = Path("/path/to/metacul/results/mechanistic/lexical_projection")
LOCALES = {
    "us": {"country_name": "United States", "country_code": "us", "continent": "America"},
    "gb": {"country_name": "United Kingdom", "country_code": "gb", "continent": "Europe"},
    "in": {"country_name": "India", "country_code": "in", "continent": "Asia"},
    "ng": {"country_name": "Nigeria", "country_code": "ng", "continent": "Africa"},
}
LOCALE_PAIRS = [("us", "gb"), ("us", "in"), ("us", "ng"), ("gb", "in"), ("gb", "ng"), ("in", "ng")]

CASES = [
    {
        "case_id": "football_role",
        "text": "The sports section previewed the football season. The team's star",
        "expected": {"us": "quarterback", "gb": "striker", "in": "striker", "ng": "striker"},
    },
    {
        "case_id": "fuel_word",
        "text": "At the station, she stopped to buy more",
        "expected": {"us": "gas", "gb": "petrol", "in": "petrol", "ng": "petrol"},
    },
    {
        "case_id": "biscuit_word",
        "text": "During tea, the plate on the table held fresh",
        "expected": {"us": "cookies", "gb": "biscuits", "in": "biscuits", "ng": "biscuits"},
    },
    {
        "case_id": "chips_word",
        "text": "He ordered fish with a side of",
        "expected": {"us": "fries", "gb": "chips", "in": "chips", "ng": "chips"},
    },
    {
        "case_id": "cricket_baseball",
        "text": "In sports coverage, the country's most followed summer game was",
        "expected": {"us": "baseball", "gb": "cricket", "in": "cricket", "ng": "football"},
    },
    {
        "case_id": "head_of_government",
        "text": "In the evening news, the head of government was the",
        "expected": {"us": "president", "gb": "prime", "in": "prime", "ng": "president"},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure locale sensitivity of next-token distributions for ambiguous lexical contexts."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--variant-label", required=True)
    parser.add_argument("--prompt-kind", choices=["url", "country", "continent", "none"], required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def build_prompt(prompt_kind: str, locale_key: str, text: str) -> str:
    locale = LOCALES[locale_key]
    title = "TITLE: Locale lexical probe"
    content = f"CONTENT: {text}"
    if prompt_kind == "url":
        prefix = f"URL: www.globalfactcheck.org/{locale['country_code']}"
    elif prompt_kind == "country":
        prefix = f"COUNTRY: {locale['country_code']}"
    elif prompt_kind == "continent":
        prefix = f"CONTINENT: {locale['continent']}"
    elif prompt_kind == "none":
        prefix = None
    else:
        raise ValueError(prompt_kind)
    if prefix is None:
        return f"{title}\n\n{content}"
    return f"{prefix}\n\n{title}\n\n{content}"


def token_rank(log_probs: torch.Tensor, tokenizer, word: str) -> Tuple[int, float, int]:
    token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Could not tokenize word={word!r}")
    token_id = token_ids[0]
    order = torch.argsort(log_probs, descending=True)
    rank = int((order == token_id).nonzero(as_tuple=False)[0].item()) + 1
    prob = float(torch.exp(log_probs[token_id]).item())
    return rank, prob, token_id


def symmetric_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    p = torch.exp(log_p)
    q = torch.exp(log_q)
    kl_pq = torch.sum(p * (log_p - log_q))
    kl_qp = torch.sum(q * (log_q - log_p))
    return float(0.5 * (kl_pq + kl_qp))


def plot_us_uk_heatmap(pairwise_df: pd.DataFrame, variant_dir: Path, variant_label: str) -> None:
    if pairwise_df.empty:
        return
    sub = pairwise_df.loc[
        (pairwise_df["locale_a"] == "us") & (pairwise_df["locale_b"] == "gb")
    ].copy()
    if sub.empty:
        return
    sub = sub.sort_values("case_id")
    values = sub["symmetric_kl"].to_numpy(dtype=float).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(3.3, 4.8))
    im = ax.imshow(values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks([0])
    ax.set_xticklabels(["US vs UK"])
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["case_id"].tolist())
    ax.set_title(f"{variant_label} locale sensitivity")
    for i, value in enumerate(sub["symmetric_kl"].tolist()):
        ax.text(0, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    fig.tight_layout()
    pdf_path = variant_dir / "us_uk_kl_heatmap.pdf"
    fig.savefig(pdf_path)
    fig.savefig(pdf_path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    variant_dir = args.output_root / args.variant_label
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

    rank_rows: List[Dict] = []
    pair_rows: List[Dict] = []

    for case in CASES:
        locale_log_probs: Dict[str, torch.Tensor] = {}
        for locale_key in LOCALES:
            prompt = build_prompt(args.prompt_kind, locale_key, case["text"])
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                add_special_tokens=False,
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits[0, -1, :].float().cpu()
            log_probs = F.log_softmax(logits, dim=-1)
            locale_log_probs[locale_key] = log_probs

            word = case["expected"][locale_key]
            rank, prob, token_id = token_rank(log_probs, tokenizer, word)
            rank_rows.append(
                {
                    "variant": args.variant_label,
                    "prompt_kind": args.prompt_kind,
                    "case_id": case["case_id"],
                    "locale": locale_key,
                    "country_name": LOCALES[locale_key]["country_name"],
                    "continent": LOCALES[locale_key]["continent"],
                    "expected_word": word,
                    "expected_token_id": token_id,
                    "expected_rank": rank,
                    "expected_prob": prob,
                }
            )

        for locale_a, locale_b in LOCALE_PAIRS:
            pair_rows.append(
                {
                    "variant": args.variant_label,
                    "prompt_kind": args.prompt_kind,
                    "case_id": case["case_id"],
                    "locale_a": locale_a,
                    "locale_b": locale_b,
                    "symmetric_kl": symmetric_kl(locale_log_probs[locale_a], locale_log_probs[locale_b]),
                }
            )

    rank_df = pd.DataFrame(rank_rows)
    pairwise_df = pd.DataFrame(pair_rows)
    rank_df.to_csv(variant_dir / "expected_token_rank_summary.csv", index=False)
    pairwise_df.to_csv(variant_dir / "pairwise_kl_summary.csv", index=False)
    plot_us_uk_heatmap(pairwise_df, variant_dir, args.variant_label)

    summary = {
        "variant": args.variant_label,
        "prompt_kind": args.prompt_kind,
        "mean_us_uk_symmetric_kl": float(
            pairwise_df.loc[
                (pairwise_df["locale_a"] == "us") & (pairwise_df["locale_b"] == "gb"),
                "symmetric_kl",
            ].mean()
        ),
        "mean_expected_rank": float(rank_df["expected_rank"].mean()),
    }
    (variant_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
