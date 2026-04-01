#!/usr/bin/env python3
"""
Analyze URL signals in the combined/with_metadata training corpus (train split).
Outputs summary tables to results/url_analysis/.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from urllib.parse import urlparse

from datasets import load_from_disk
from tqdm import tqdm


DEFAULT_DATASET = "/scratch/amukher6/metacul/training_data/meco_datasets/combined/with_metadata/"
DEFAULT_OUTPUT_DIR = "/scratch/amukher6/metacul/results/url_analysis"


CONTINENT_TOKENS = {
    "africa",
    "europe",
    "asia",
    "america",
    "americas",
    "north",
    "south",
    "oceania",
}


DOMAIN_STOP_TOKENS = {
    "www",
    "com",
    "org",
    "net",
    "edu",
    "gov",
    "co",
    "uk",
}


def load_split(dataset_path, split):
    split_path = f"{dataset_path.rstrip('/')}/{split}"
    if os.path.exists(split_path):
        return load_from_disk(split_path)
    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, dict) and split in dataset:
        return dataset[split]
    if hasattr(dataset, "keys") and split in dataset:
        return dataset[split]
    return dataset


def extract_meta(text):
    url = None
    country = None
    continent = None
    for line in text.splitlines():
        if url is None and line.startswith("URL:"):
            url = line.split("URL:", 1)[1].strip()
        elif country is None and line.startswith("COUNTRY:"):
            country = line.split("COUNTRY:", 1)[1].strip().lower()
        elif continent is None and line.startswith("CONTINENT:"):
            continent = line.split("CONTINENT:", 1)[1].strip().lower()
        if url and country and continent:
            break
    return url, country, continent


def country_code_candidates(code):
    if not code:
        return set()
    code = code.lower()
    candidates = {code}
    if code == "gb":
        candidates.add("uk")
    return candidates


def tokenize(value):
    return [t for t in re.split(r"[^a-z0-9]+", value.lower()) if t]


def analyze(args):
    dataset = load_split(args.dataset, args.split)
    total_rows = len(dataset)

    overall = Counter()
    cc_tld_counts = Counter()
    domain_token_counts = Counter()

    by_country = defaultdict(lambda: Counter())
    by_continent = defaultdict(lambda: Counter())

    iterator = dataset
    if args.max_rows is not None:
        iterator = (dataset[i] for i in range(min(args.max_rows, len(dataset))))

    for row in tqdm(iterator, total=min(len(dataset), args.max_rows or len(dataset))):
        text = row["text"]
        url, country, continent = extract_meta(text)
        overall["total_rows"] += 1
        if not url:
            overall["missing_url"] += 1
            continue
        overall["with_url"] += 1

        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        domain_tokens = tokenize(domain)
        path_tokens = tokenize(parsed.path.lower())

        if args.count_domain_tokens:
            for tok in domain_tokens:
                if tok in DOMAIN_STOP_TOKENS:
                    continue
                if tok.isdigit() or len(tok) > 20:
                    continue
                domain_token_counts[tok] += 1

        tld = None
        if domain:
            parts = domain.split(".")
            if len(parts) >= 2:
                tld = parts[-1]
        if tld and len(tld) == 2 and tld.isalpha():
            overall["cc_tld"] += 1
            cc_tld_counts[tld] += 1

        country_counts = by_country[country]
        if country:
            country_counts["total"] += 1
        if continent:
            by_continent[continent]["total"] += 1

        candidates = country_code_candidates(country)
        cc_match = tld in candidates if tld else False
        domain_code_match = any(tok in candidates for tok in domain_tokens)
        path_code_match = any(tok in candidates for tok in path_tokens)

        if cc_match:
            overall["cc_tld_match_country"] += 1
            if country:
                country_counts["cc_tld_match"] += 1
        if domain_code_match:
            overall["domain_code_match"] += 1
            if country:
                country_counts["domain_code_match"] += 1
        if path_code_match:
            overall["path_code_match"] += 1
            if country:
                country_counts["path_code_match"] += 1
        if cc_match or domain_code_match or path_code_match:
            overall["any_country_code_signal"] += 1
            if country:
                country_counts["any_code_match"] += 1

        domain_continent_match = continent in domain_tokens if continent else False
        path_continent_match = continent in path_tokens if continent else False
        if domain_continent_match:
            overall["domain_continent_match"] += 1
            if continent:
                by_continent[continent]["domain_continent_match"] += 1
        if path_continent_match:
            overall["path_continent_match"] += 1
            if continent:
                by_continent[continent]["path_continent_match"] += 1
        if continent and (domain_continent_match or path_continent_match):
            overall["any_continent_signal"] += 1
            by_continent[continent]["any_continent_match"] += 1

        if any(tok in CONTINENT_TOKENS for tok in domain_tokens):
            overall["domain_has_continent_token"] += 1
        if any(tok in CONTINENT_TOKENS for tok in path_tokens):
            overall["path_has_continent_token"] += 1

    return overall, by_country, by_continent, cc_tld_counts, domain_token_counts


def write_outputs(
    output_dir,
    overall,
    by_country,
    by_continent,
    cc_tld_counts,
    domain_token_counts,
    top_n,
):
    os.makedirs(output_dir, exist_ok=True)

    overall_path = os.path.join(output_dir, "url_signal_summary.json")
    with open(overall_path, "w") as f:
        json.dump(overall, f, indent=2)

    country_path = os.path.join(output_dir, "url_signal_by_country.csv")
    with open(country_path, "w") as f:
        f.write(
            "country,total,cc_tld_match,domain_code_match,path_code_match,any_code_match\n"
        )
        for country, counts in sorted(by_country.items()):
            f.write(
                f"{country},{counts.get('total',0)},"
                f"{counts.get('cc_tld_match',0)},"
                f"{counts.get('domain_code_match',0)},"
                f"{counts.get('path_code_match',0)},"
                f"{counts.get('any_code_match',0)}\n"
            )

    continent_path = os.path.join(output_dir, "url_signal_by_continent.csv")
    with open(continent_path, "w") as f:
        f.write(
            "continent,total,domain_continent_match,path_continent_match,any_continent_match\n"
        )
        for continent, counts in sorted(by_continent.items()):
            f.write(
                f"{continent},{counts.get('total',0)},"
                f"{counts.get('domain_continent_match',0)},"
                f"{counts.get('path_continent_match',0)},"
                f"{counts.get('any_continent_match',0)}\n"
            )

    tld_path = os.path.join(output_dir, "url_cc_tld_top.csv")
    with open(tld_path, "w") as f:
        f.write("tld,count\n")
        for tld, count in cc_tld_counts.most_common(top_n):
            f.write(f"{tld},{count}\n")

    if domain_token_counts:
        token_path = os.path.join(output_dir, "url_domain_tokens_top.csv")
        with open(token_path, "w") as f:
            f.write("token,count\n")
            for token, count in domain_token_counts.most_common(top_n):
                f.write(f"{token},{count}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze URL signals in training corpus.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--no-domain-tokens",
        dest="count_domain_tokens",
        action="store_false",
        help="Disable domain token counting (saves memory).",
    )
    parser.set_defaults(count_domain_tokens=True)
    args = parser.parse_args()

    overall, by_country, by_continent, cc_tld_counts, domain_token_counts = analyze(args)
    write_outputs(
        args.output_dir,
        overall,
        by_country,
        by_continent,
        cc_tld_counts,
        domain_token_counts,
        args.top_n,
    )


if __name__ == "__main__":
    main()
