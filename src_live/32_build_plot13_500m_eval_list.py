#!/usr/bin/env python3
import json
import os

RESULTS_DIR = "/path/to/metacul/results"
OUTPUT_JSON = os.path.join(RESULTS_DIR, "perplexity", "eval_list_plot13_500m_backfill.json")

models = [
    "/path/to/metacul/models/ablation_intermediates/metadata/combined_with_metadata_500m_step2k",
    "/path/to/metacul/models/ablation_intermediates/metadata/combined_with_metadata_500m_step4k",
    "/path/to/metacul/models/ablation_intermediates/metadata/combined_with_metadata_500m_step8k",
    "/path/to/metacul/models/combined_with_metadata_500m",
    "/path/to/metacul/models/ablation_intermediates/metadata/combined_without_metadata_500m_step2k",
    "/path/to/metacul/models/ablation_intermediates/metadata/combined_without_metadata_500m_step4k",
    "/path/to/metacul/models/ablation_intermediates/metadata/combined_without_metadata_500m_step8k",
    "/path/to/metacul/models/combined_without_metadata_500m",
]

test_sets = [
    "/path/to/metacul/training_data/meco_datasets/combined/with_metadata/",
    "/path/to/metacul/training_data/meco_datasets/combined/without_metadata/",
]

rows = []
for model_path in models:
    if "with_metadata" in os.path.basename(model_path):
        target = test_sets[0]
    else:
        target = test_sets[1]
    rows.append({"model_path": model_path, "test_set_path": target})

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(rows, f, indent=2)

print(f"Wrote {len(rows)} eval pairs to {OUTPUT_JSON}")
