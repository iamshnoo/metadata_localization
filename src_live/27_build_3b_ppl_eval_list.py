#!/usr/bin/env python3
import json
from pathlib import Path

SOURCE = Path("/path/to/metacul/results/perplexity/eval_list.json")
OUTPUT = Path("/path/to/metacul/results/perplexity/eval_list_3b_requested.json")


def is_requested_3b_combo(combo: dict) -> bool:
    model_path = combo["model_path"]
    test_set_path = combo["test_set_path"]

    if "_3b" not in model_path:
        return False

    if "/models/combined_" in model_path and (
        "/meco_datasets/continents/" in test_set_path
        or "/meco_datasets/combined/" in test_set_path
        or "/meco_datasets/combined_only_url/" in test_set_path
        or "/meco_datasets/combined_only_url_continent/" in test_set_path
        or "/meco_datasets/combined_only_url_country/" in test_set_path
    ):
        return True

    if "/models/ablation_intermediates/metadata/" in model_path and (
        "/meco_datasets/continents/" in test_set_path
        or "/meco_datasets/combined/" in test_set_path
        or "/meco_datasets/combined_only_url/" in test_set_path
        or "/meco_datasets/combined_only_url_continent/" in test_set_path
        or "/meco_datasets/combined_only_url_country/" in test_set_path
    ):
        return True

    return False


def main() -> int:
    with SOURCE.open() as f:
        combos = json.load(f)

    filtered = [combo for combo in combos if is_requested_3b_combo(combo)]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Wrote {len(filtered)} combos to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
