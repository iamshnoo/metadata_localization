#!/usr/bin/env python3
"""
Recover URL-bearing combined MECO datasets from raw NOW plus surviving ablations.

The original full combined format was created from NOW records as:

    URL: ...
    COUNTRY: ...
    CONTINENT: ...

    TITLE: ...

    CONTENT: ...

The full combined save is gone, but the aligned country-only combined dataset
still preserves the exact split/order/body. This script recovers the URL field
from /groups/NLP/NOW by hashing the same COUNTRY + TITLE/CONTENT body produced
by the original NOW parser, then emits both full with-metadata and no-metadata
datasets for Nanotron tokenization.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import os
import pickle
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

os.environ.setdefault("HF_HOME", "/path/to/workspace/.cache/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets"))
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

from datasets import Dataset, Features, Value, load_from_disk


COUNTRY_TO_CONTINENT = {
    "bd": "Asia",
    "hk": "Asia",
    "in": "Asia",
    "lk": "Asia",
    "my": "Asia",
    "ph": "Asia",
    "pk": "Asia",
    "sg": "Asia",
    "ca": "America",
    "jm": "America",
    "us": "America",
    "gb": "Europe",
    "ie": "Europe",
    "gh": "Africa",
    "ke": "Africa",
    "ng": "Africa",
    "za": "Africa",
    "tz": "Africa",
}

RAW_COUNTRY_CODES = set(COUNTRY_TO_CONTINENT) | {f"us{i}" for i in range(1, 6)}
ENTRY_RE = re.compile(rb"@@(\d+)\s+(.*?)(?=@@\d+|\Z)", flags=re.DOTALL)
MEMBER_RE = re.compile(r"^(?P<yy>\d{2})-(?P<mm>\d{2})-(?P<code>[a-z]+[1-5]?)\.txt$")
FEATURES = Features({"text": Value("string")})
_URL_MAP: Dict[bytes, str] = {}


def normalize_country(raw_code: str) -> str:
    return "us" if raw_code.startswith("us") else raw_code


def strip_html(text: str) -> str:
    """Mirror src_live/0_data_now.py cleaning."""
    text = html.unescape(text)
    text = re.sub(r"<(h|p)[^>]*>", "\n", text)
    text = re.sub(r"</(h|p)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_country_record(text: str) -> Tuple[str, str]:
    marker = "\n\nTITLE: "
    idx = text.find(marker)
    if not text.startswith("COUNTRY: ") or idx < 0:
        raise ValueError("Expected COUNTRY-only MECO text")
    country = text[len("COUNTRY: ") : idx].strip().lower()
    body = text[idx + 2 :]
    if country not in COUNTRY_TO_CONTINENT:
        raise ValueError(f"Unexpected country code: {country}")
    if not body.startswith("TITLE: "):
        raise ValueError("Expected TITLE body after COUNTRY header")
    return country, body


def make_key(country: str, body: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(country.encode("utf-8"))
    h.update(b"\0")
    h.update(body.encode("utf-8", errors="surrogatepass"))
    return h.digest()


def iter_country_bodies(dataset_path: Path, batch_size: int, limit: Optional[int]) -> Iterator[Tuple[str, str]]:
    ds = load_from_disk(str(dataset_path))
    seen = 0
    for batch in ds.iter(batch_size=batch_size):
        for text in batch["text"]:
            yield split_country_record(text)
            seen += 1
            if limit is not None and seen >= limit:
                return


def build_needed_keys(args: argparse.Namespace, splits: Iterable[str]) -> Tuple[set[bytes], Dict[str, int]]:
    needed: set[bytes] = set()
    counts: Dict[str, int] = {}
    for split in splits:
        path = args.country_root / split
        ds = load_from_disk(str(path))
        limit = args.sample_limit if args.sample_limit else None
        target_count = min(len(ds), limit) if limit else len(ds)
        print(f"[needed] {split}: scanning {target_count:,} rows from {path}", flush=True)
        seen = 0
        for country, body in iter_country_bodies(path, args.batch_size, limit):
            needed.add(make_key(country, body))
            seen += 1
            if seen and seen % args.progress_every == 0:
                print(f"[needed] {split}: {seen:,}/{target_count:,} rows, unique={len(needed):,}", flush=True)
        counts[split] = seen
        print(f"[needed] {split}: rows={seen:,}, cumulative unique={len(needed):,}", flush=True)
    return needed, counts


def load_sources(source_path: Path) -> Dict[str, Tuple[str, str]]:
    sources: Dict[str, Tuple[str, str]] = {}
    with source_path.open("r", encoding="ISO-8859-1") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("Source Line"):
                continue
            parts = line.split("\t", maxsplit=4)
            if len(parts) != 5:
                parts = line.split(maxsplit=4)
            if len(parts) != 5:
                continue
            article_id, _date, _country, url, title = parts
            sources[article_id] = (url.strip(), title.strip())
    return sources


def iter_year_zip_paths(raw_now_root: Path, year: int) -> list[Path]:
    yy = str(year)[-2:]
    return sorted((raw_now_root / "text").glob(f"{yy}-??-text.zip"))


def recover_url_map(args: argparse.Namespace, needed: set[bytes]) -> Tuple[Dict[bytes, str], Counter]:
    if args.url_map_cache and args.url_map_cache.exists() and not args.rebuild_url_map:
        print(f"[url-map] loading cached map: {args.url_map_cache}", flush=True)
        with args.url_map_cache.open("rb") as f:
            url_map = pickle.load(f)
        return url_map, Counter({"loaded_from_cache": 1})

    stats: Counter = Counter()
    url_map: Dict[bytes, str] = {}

    for year in args.years:
        yy = str(year)[-2:]
        source_path = args.raw_now_root / "sources" / f"{year}-sources.txt"
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        print(f"[url-map] {year}: loading sources {source_path}", flush=True)
        sources = load_sources(source_path)
        stats["source_rows"] += len(sources)
        print(f"[url-map] {year}: source rows={len(sources):,}", flush=True)

        for zip_path in iter_year_zip_paths(args.raw_now_root, year):
            print(
                f"[url-map] {year}: {zip_path.name}, found={len(url_map):,}/{len(needed):,}",
                flush=True,
            )
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    m = MEMBER_RE.match(Path(member).name)
                    if not m or m.group("yy") != yy:
                        continue
                    raw_code = m.group("code")
                    if raw_code not in RAW_COUNTRY_CODES:
                        continue
                    country = normalize_country(raw_code)
                    if country not in COUNTRY_TO_CONTINENT:
                        continue

                    try:
                        raw_bytes = zf.read(member)
                    except KeyError:
                        continue

                    for article_id_b, article_content_b in ENTRY_RE.findall(raw_bytes):
                        stats["raw_entries"] += 1
                        article_id = article_id_b.decode("ascii", errors="ignore")
                        meta = sources.get(article_id)
                        if meta is None:
                            stats["missing_source_meta"] += 1
                            continue
                        url, title = meta
                        content = strip_html(article_content_b.decode("utf-8", errors="ignore").strip())
                        if not title or not content:
                            stats["missing_title_or_content"] += 1
                            continue
                        body = f"TITLE: {title}\n\nCONTENT: {content}"
                        key = make_key(country, body)
                        if key not in needed:
                            continue
                        previous = url_map.get(key)
                        if previous is None:
                            url_map[key] = url
                        elif previous != url:
                            stats["duplicate_key_different_url"] += 1

            if len(url_map) == len(needed):
                print("[url-map] all needed keys found; stopping raw scan early", flush=True)
                break

        del sources
        gc.collect()
        if len(url_map) == len(needed):
            break

    if args.url_map_cache:
        args.url_map_cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"[url-map] writing cache: {args.url_map_cache}", flush=True)
        with args.url_map_cache.open("wb") as f:
            pickle.dump(url_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    stats["needed_unique"] = len(needed)
    stats["found_unique"] = len(url_map)
    stats["missing_unique"] = len(needed) - len(url_map)
    return url_map, stats


def validate_missing(args: argparse.Namespace, url_stats: Counter) -> None:
    needed = url_stats["needed_unique"]
    missing = url_stats["missing_unique"]
    frac = (missing / needed) if needed else 0.0
    print(f"[url-map] missing unique keys: {missing:,}/{needed:,} ({frac:.6%})", flush=True)
    if frac > args.max_missing_fraction:
        raise RuntimeError(
            f"Missing URL fraction {frac:.6%} exceeds --max-missing-fraction={args.max_missing_fraction}"
        )


def format_with_metadata(country: str, body: str, url: str) -> str:
    continent = COUNTRY_TO_CONTINENT[country]
    return f"URL: {url}\nCOUNTRY: {country}\nCONTINENT: {continent}\n\n{body}"


def generate_split_records(
    source_path: Path,
    with_metadata: bool,
    batch_size: int,
    limit: Optional[int],
) -> Iterator[dict]:
    for country, body in iter_country_bodies(source_path, batch_size, limit):
        if with_metadata:
            url = _URL_MAP.get(make_key(country, body), "")
            yield {"text": format_with_metadata(country, body, url)}
        else:
            yield {"text": body}


def write_split_dataset(
    args: argparse.Namespace,
    split: str,
    url_map: Dict[bytes, str],
    with_metadata: bool,
) -> int:
    global _URL_MAP
    _URL_MAP = url_map

    source_path = args.country_root / split
    output_root = args.with_output_root if with_metadata else args.without_output_root
    output_path = output_root / split
    cache_dir = args.generator_cache_root / ("with_metadata" if with_metadata else "without_metadata") / split
    limit = args.sample_limit if args.sample_limit else None

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_path)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[write] {split} {'with_metadata' if with_metadata else 'without_metadata'} -> {output_path}", flush=True)
    ds = Dataset.from_generator(
        generate_split_records,
        features=FEATURES,
        cache_dir=str(cache_dir),
        gen_kwargs={
            "source_path": source_path,
            "with_metadata": with_metadata,
            "batch_size": args.batch_size,
            "limit": limit,
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_path), max_shard_size=args.max_shard_size)
    rows = len(ds)
    print(f"[write] {split} rows={rows:,}", flush=True)
    del ds
    gc.collect()
    if args.cleanup_generator_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
    return rows


def write_summary(args: argparse.Namespace, needed_counts: Dict[str, int], url_stats: Counter, rows: Dict[str, int]) -> None:
    summary = {
        "country_root": str(args.country_root),
        "with_output_root": str(args.with_output_root),
        "without_output_root": str(args.without_output_root),
        "raw_now_root": str(args.raw_now_root),
        "years": args.years,
        "splits": args.splits,
        "sample_limit": args.sample_limit,
        "needed_counts": needed_counts,
        "url_stats": dict(url_stats),
        "output_rows": rows,
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[summary] wrote {args.summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--country-root",
        type=Path,
        default=Path("/path/to/metacul/training_data/meco_datasets/combined_only_country/with_metadata"),
    )
    parser.add_argument("--raw-now-root", type=Path, default=Path("/groups/NLP/NOW/now"))
    parser.add_argument(
        "--with-output-root",
        type=Path,
        default=Path("/path/to/workspace/pretrain/datasets/raw-combined-with-metadata"),
    )
    parser.add_argument(
        "--without-output-root",
        type=Path,
        default=Path("/path/to/workspace/pretrain/datasets/raw-combined-without-metadata"),
    )
    parser.add_argument(
        "--url-map-cache",
        type=Path,
        default=Path("/path/to/workspace/pretrain/datasets/raw-combined-url-map/url_map_blake2b16.pkl"),
    )
    parser.add_argument(
        "--generator-cache-root",
        type=Path,
        default=Path("/path/to/workspace/pretrain/datasets/raw-combined-url-map/generator_cache"),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("/path/to/workspace/pretrain/datasets/raw-combined-url-map/summary.json"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2010, 2025)))
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--sample-limit", type=int, default=0, help="Debug: process at most N rows per split")
    parser.add_argument("--max-missing-fraction", type=float, default=0.001)
    parser.add_argument("--max-shard-size", default="500MB")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-url-map", action="store_true")
    parser.add_argument("--skip-with-metadata", action="store_true")
    parser.add_argument("--skip-without-metadata", action="store_true")
    parser.add_argument("--cleanup-generator-cache", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in args.splits:
        split_path = args.country_root / split
        if not split_path.exists():
            raise FileNotFoundError(split_path)

    needed, needed_counts = build_needed_keys(args, args.splits)
    url_map, url_stats = recover_url_map(args, needed)
    validate_missing(args, url_stats)

    rows: Dict[str, int] = {}
    for split in args.splits:
        if not args.skip_with_metadata:
            rows[f"with_metadata/{split}"] = write_split_dataset(args, split, url_map, True)
        if not args.skip_without_metadata:
            rows[f"without_metadata/{split}"] = write_split_dataset(args, split, url_map, False)

    write_summary(args, needed_counts, url_stats, rows)
    print("[done] URL-bearing combined recovery complete", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        raise
