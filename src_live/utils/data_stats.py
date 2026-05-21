from datasets import load_from_disk
from collections import defaultdict
import re
from tqdm import tqdm

base = "/path/to/metacul/training_data/meco_datasets/continents/europe/with_metadata"
splits = ["train", "validation", "test"]

def parse_meta(text):
    continent = country = None
    for line in text.splitlines():
        if line.startswith("COUNTRY:"):
            country = line.split(":",1)[1].strip()
        elif line.startswith("CONTINENT:"):
            continent = line.split(":",1)[1].strip().capitalize()
        if country and continent:
            break
    return country, continent

results = {}
for split in splits:
    ds = load_from_disk(f"{base}/{split}")
    counts = {c: defaultdict(int) for c in ["Africa","America","Asia","Europe"]}
    for rec in tqdm(ds, desc=f"Processing {split}"):
        country, continent = parse_meta(rec["text"])
        if not country or not continent:
            continue
        counts[continent][country] += 1
    results[split] = {cont: dict(countries) for cont, countries in counts.items()}

print(results)

# save results to a file
import json
with open("/path/to/metacul/results/dataset_stats/europe_stats.json", "w") as f:
    json.dump(results, f, indent=2)
