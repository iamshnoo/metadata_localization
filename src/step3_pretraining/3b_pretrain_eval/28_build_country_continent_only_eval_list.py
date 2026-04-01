#!/usr/bin/env python3
import json
import os

RESULTS_DIR = "/scratch/amukher6/metacul/results"
OUTPUT_JSON = os.path.join(RESULTS_DIR, "perplexity", "eval_list_country_continent_only.json")

models = [
    "/scratch/amukher6/metacul/models/combined_only_continent_with_metadata_1b",
    "/scratch/amukher6/metacul/models/combined_only_country_with_metadata_1b",
]

test_sets = [
    "/scratch/amukher6/metacul/training_data/meco_datasets/combined_only_url/with_metadata/",
    "/scratch/amukher6/metacul/training_data/meco_datasets/combined_only_url_continent/with_metadata/",
    "/scratch/amukher6/metacul/training_data/meco_datasets/combined_only_url_country/with_metadata/",
    "/scratch/amukher6/metacul/training_data/meco_datasets/combined/with_metadata/",
    "/scratch/amukher6/metacul/training_data/meco_datasets/combined/without_metadata/",
]

rows = [
    {"model_path": model_path, "test_set_path": test_set_path}
    for model_path in models
    for test_set_path in test_sets
]

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(rows, f, indent=2)

print(f"Wrote {len(rows)} eval pairs to {OUTPUT_JSON}")
