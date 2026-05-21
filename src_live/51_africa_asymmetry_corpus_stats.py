#!/usr/bin/env python3
import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from datasets import load_dataset, load_from_disk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


CONTINENTS = ["Africa", "America", "Asia", "Europe"]
DEFAULT_SAMPLE_SIZE = 50000
DEFAULT_SEED = 42
DEFAULT_DATA_ROOT = Path("/path/to/metacul/training_data/hf_datasets/continents")
DEFAULT_OUTPUT_DIR = Path("/path/to/metacul/results/asymmetry")

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\\-]{2,}")
ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,}(?:\s+[A-Z]{2,})*)\b"
)
STOPWORDS = set(ENGLISH_STOP_WORDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Africa-vs-other continent corpus statistics to explain asymmetry."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--top-vocab", type=int, default=10000)
    parser.add_argument("--top-entities", type=int, default=2000)
    parser.add_argument("--topic-keywords-per-family", type=int, default=40)
    return parser.parse_args()


def normalize_country(code_or_name: str) -> str:
    raw = str(code_or_name or "").strip()
    if not raw:
        return raw
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
    return mapping.get(raw.lower(), raw)


def sample_indices(length: int, sample_size: int, seed: int) -> List[int]:
    if sample_size >= length:
        return list(range(length))
    rng = random.Random(seed)
    return rng.sample(range(length), sample_size)


def iter_sampled_rows(dataset, sample_size: int, seed: int):
    for idx in sample_indices(len(dataset), sample_size, seed):
        yield dataset[idx]


def tokenize(text: str) -> List[str]:
    tokens = []
    for match in TOKEN_RE.finditer(text.lower()):
        tok = match.group(0)
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def extract_entities(title: str, content: str) -> Iterable[str]:
    text = f"{title} {content[:600]}"
    seen = set()
    for match in ENTITY_RE.finditer(text):
        ent = match.group(0).strip()
        if len(ent) < 3:
            continue
        if ent.lower() in STOPWORDS:
            continue
        if ent in seen:
            continue
        seen.add(ent)
        yield ent


def build_topic_keyword_lexicons(top_k: int) -> Dict[str, List[str]]:
    ds = load_dataset("YOUR_HF_USERNAME/LocalNewsQA", split="train")
    overall = Counter()
    per_topic: Dict[str, Counter] = defaultdict(Counter)
    for row in ds:
        text = " ".join(
            [
                str(row.get("question", "")),
                " ".join(row.get("options") or []),
                str(row.get("target_answer") or row.get("correct_answer") or ""),
                str(row.get("contrast_answer") or ""),
            ]
        )
        toks = tokenize(text)
        overall.update(toks)
        per_topic[row["topic"]].update(toks)

    lexicons: Dict[str, List[str]] = {}
    for topic, counter in per_topic.items():
        scored = []
        for token, count in counter.items():
            other = overall[token] - count
            score = count / (1 + other)
            scored.append((score, count, token))
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        lexicons[topic] = [token for _, _, token in scored[:top_k]]
    return lexicons


def assign_topic(tokens: List[str], lexicons: Dict[str, List[str]]) -> str:
    token_counts = Counter(tokens)
    best_topic = "Unmatched"
    best_score = 0
    for topic, keywords in lexicons.items():
        score = sum(token_counts.get(k, 0) for k in keywords)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic


def overlap_fraction(source: Iterable[str], target: Iterable[str]) -> float:
    source_set = set(source)
    target_set = set(target)
    if not source_set:
        return 0.0
    return len(source_set & target_set) / len(source_set)


def entropy_from_counts(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for count in counter.values():
        p = count / total
        ent -= p * math.log(p + 1e-12, 2)
    return ent


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    topic_lexicons = build_topic_keyword_lexicons(args.topic_keywords_per_family)
    (args.output_dir / "localnewsqa_topic_keywords.json").write_text(
        json.dumps(topic_lexicons, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    vocab_sets: Dict[str, List[str]] = {}
    entity_sets: Dict[str, List[str]] = {}
    length_rows: List[Dict[str, object]] = []
    topic_rows: List[Dict[str, object]] = []
    theme_rows: List[Dict[str, object]] = []
    summary: Dict[str, object] = {
        "sample_size_per_continent": args.sample_size,
        "seed": args.seed,
        "continents": {},
    }

    for idx, continent in enumerate(CONTINENTS):
        ds_path = args.data_root / continent / "train"
        ds = load_from_disk(str(ds_path))
        sample_seed = args.seed + idx

        vocab_counter = Counter()
        entity_counter = Counter()
        topic_counter = Counter()
        theme_counter = Counter()
        total_word_count = 0
        total_docs = 0

        for row in iter_sampled_rows(ds, args.sample_size, sample_seed):
            title = str(row.get("title", ""))
            content = str(row.get("content", ""))
            text = f"{title} {content}"
            tokens = tokenize(text)
            vocab_counter.update(tokens)
            total_word_count += len(tokens)
            total_docs += 1

            for ent in extract_entities(title, content):
                entity_counter[ent] += 1

            topic_family = assign_topic(tokens, topic_lexicons)
            topic_counter[topic_family] += 1
            theme_counter[str(row.get("theme") or "Unknown")] += 1

        top_vocab = [tok for tok, _ in vocab_counter.most_common(args.top_vocab)]
        top_entities = [ent for ent, _ in entity_counter.most_common(args.top_entities)]
        vocab_sets[continent] = top_vocab
        entity_sets[continent] = top_entities

        avg_length = total_word_count / max(total_docs, 1)
        length_rows.append(
            {
                "continent": continent,
                "sample_docs": total_docs,
                "avg_doc_length_tokens": avg_length,
                "topic_entropy_bits": entropy_from_counts(topic_counter),
                "theme_entropy_bits": entropy_from_counts(theme_counter),
            }
        )

        for topic, count in topic_counter.items():
            topic_rows.append(
                {
                    "continent": continent,
                    "topic_family": topic,
                    "count": count,
                    "share": count / max(total_docs, 1),
                }
            )

        for theme, count in theme_counter.most_common():
            theme_rows.append(
                {
                    "continent": continent,
                    "theme": theme,
                    "count": count,
                    "share": count / max(total_docs, 1),
                }
            )

        summary["continents"][continent] = {
            "sample_docs": total_docs,
            "avg_doc_length_tokens": avg_length,
            "top_10_themes": theme_counter.most_common(10),
            "topic_entropy_bits": entropy_from_counts(topic_counter),
            "theme_entropy_bits": entropy_from_counts(theme_counter),
        }

    vocab_overlap_rows = []
    entity_overlap_rows = []
    for src in CONTINENTS:
        for tgt in CONTINENTS:
            vocab_overlap_rows.append(
                {
                    "source_continent": src,
                    "target_continent": tgt,
                    "overlap_fraction": overlap_fraction(vocab_sets[src], vocab_sets[tgt]),
                }
            )
            entity_overlap_rows.append(
                {
                    "source_continent": src,
                    "target_continent": tgt,
                    "overlap_fraction": overlap_fraction(entity_sets[src], entity_sets[tgt]),
                }
            )

    df_len = pd.DataFrame(length_rows)
    df_topic = pd.DataFrame(topic_rows)
    df_theme = pd.DataFrame(theme_rows)
    df_vocab = pd.DataFrame(vocab_overlap_rows)
    df_entity = pd.DataFrame(entity_overlap_rows)

    df_len.to_csv(args.output_dir / "africa_asymmetry_length_stats.csv", index=False)
    df_topic.to_csv(args.output_dir / "africa_asymmetry_topic_distribution.csv", index=False)
    df_theme.to_csv(args.output_dir / "africa_asymmetry_theme_distribution.csv", index=False)
    df_vocab.to_csv(args.output_dir / "africa_asymmetry_vocab_overlap.csv", index=False)
    df_entity.to_csv(args.output_dir / "africa_asymmetry_entity_overlap.csv", index=False)

    africa_vocab = (
        df_vocab[df_vocab["source_continent"] == "Africa"]
        .sort_values("overlap_fraction")
        .to_dict(orient="records")
    )
    africa_entity = (
        df_entity[df_entity["source_continent"] == "Africa"]
        .sort_values("overlap_fraction")
        .to_dict(orient="records")
    )
    africa_lengths = df_len.set_index("continent").to_dict(orient="index")
    summary["africa_vs_others"] = {
        "vocab_overlap": africa_vocab,
        "entity_overlap": africa_entity,
        "length_stats": africa_lengths,
    }
    summary_path = args.output_dir / "africa_asymmetry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ok] Wrote {args.output_dir / 'africa_asymmetry_length_stats.csv'}")
    print(f"[ok] Wrote {args.output_dir / 'africa_asymmetry_topic_distribution.csv'}")
    print(f"[ok] Wrote {args.output_dir / 'africa_asymmetry_theme_distribution.csv'}")
    print(f"[ok] Wrote {args.output_dir / 'africa_asymmetry_vocab_overlap.csv'}")
    print(f"[ok] Wrote {args.output_dir / 'africa_asymmetry_entity_overlap.csv'}")
    print(f"[ok] Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
