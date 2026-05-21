#!/usr/bin/env python3
import json
import os

RESULTS_DIR = "/path/to/metacul/results"
OUTPUT_JSON = os.path.join(
    RESULTS_DIR, "perplexity", "eval_list_metadata_family_full.json"
)

final_models = [
    "/path/to/metacul/models/ablations/metadata/combined_only_url_with_metadata_1b",
    "/path/to/metacul/models/ablations/metadata/combined_only_url_country_with_metadata_1b",
    "/path/to/metacul/models/ablations/metadata/combined_only_url_continent_with_metadata_1b",
    "/path/to/metacul/models/combined_only_country_with_metadata_1b",
    "/path/to/metacul/models/combined_only_continent_with_metadata_1b",
]

intermediate_models = []
for base in [
    "combined_only_url_with_metadata_1b",
    "combined_only_url_country_with_metadata_1b",
    "combined_only_url_continent_with_metadata_1b",
    "combined_only_country_with_metadata_1b",
    "combined_only_continent_with_metadata_1b",
]:
    for step in [2, 4, 8]:
        intermediate_models.append(
            f"/path/to/metacul/models/ablation_intermediates/metadata/{base}_step{step}k"
        )

models = final_models + intermediate_models

test_sets = [
    "/path/to/metacul/training_data/meco_datasets/combined_only_url/with_metadata/",
    "/path/to/metacul/training_data/meco_datasets/combined_only_url_country/with_metadata/",
    "/path/to/metacul/training_data/meco_datasets/combined_only_url_continent/with_metadata/",
    "/path/to/metacul/training_data/meco_datasets/combined/without_metadata/",
    "/path/to/metacul/training_data/meco_datasets/combined/with_metadata/",
    "/path/to/metacul/training_data/meco_datasets/combined_only_country/with_metadata/",
    "/path/to/metacul/training_data/meco_datasets/combined_only_continent/with_metadata/",
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
