import csv
import json
import os

BASE_PATH_MODEL = "/path/to/metacul/models/"
continent_models = [
    "africa_with_metadata_500m",
    "africa_with_metadata_1b",
    "america_with_metadata_500m",
    "america_with_metadata_1b",
    "asia_with_metadata_500m",
    "asia_with_metadata_1b",
    "europe_with_metadata_500m",
    "europe_with_metadata_1b",
    "africa_without_metadata_500m",
    "africa_without_metadata_1b",
    "america_without_metadata_500m",
    "america_without_metadata_1b",
    "asia_without_metadata_500m",
    "asia_without_metadata_1b",
    "europe_without_metadata_500m",
    "europe_without_metadata_1b",
]

combined_models = [
    "combined_with_metadata_500m",
    "combined_with_metadata_1b",
    "combined_with_metadata_3b",
    "combined_without_metadata_500m",
    "combined_without_metadata_1b",
    "combined_without_metadata_3b",
]

BASE_PATH_MODEL_CONTINENT_ABLATIONS = "/path/to/metacul/models/ablations/leave_one_out/"
continent_ablations = [
    "combined_no_africa_with_metadata_1b",
    "combined_no_america_with_metadata_1b",
    "combined_no_asia_with_metadata_1b",
    "combined_no_europe_with_metadata_1b",
    "combined_no_africa_without_metadata_1b",
    "combined_no_america_without_metadata_1b",
    "combined_no_asia_without_metadata_1b",
    "combined_no_europe_without_metadata_1b",
]

BASE_PATH_MODEL_CONTINENT_ABLATIONS_INTERMEDIATE = "/path/to/metacul/models/ablation_intermediates/leave_one_out/"
continent_ablations_intermediate = [
    "combined_no_africa_with_metadata_1b_step2k",
    "combined_no_africa_with_metadata_1b_step4k",
    "combined_no_africa_with_metadata_1b_step8k",
    "combined_no_america_with_metadata_1b_step2k",
    "combined_no_america_with_metadata_1b_step4k",
    "combined_no_america_with_metadata_1b_step8k",
    "combined_no_asia_with_metadata_1b_step2k",
    "combined_no_asia_with_metadata_1b_step4k",
    "combined_no_asia_with_metadata_1b_step8k",
    "combined_no_europe_with_metadata_1b_step2k",
    "combined_no_europe_with_metadata_1b_step4k",
    "combined_no_europe_with_metadata_1b_step8k",
    "combined_no_africa_without_metadata_1b_step2k",
    "combined_no_africa_without_metadata_1b_step4k",
    "combined_no_africa_without_metadata_1b_step8k",
    "combined_no_america_without_metadata_1b_step2k",
    "combined_no_america_without_metadata_1b_step4k",
    "combined_no_america_without_metadata_1b_step8k",
    "combined_no_asia_without_metadata_1b_step2k",
    "combined_no_asia_without_metadata_1b_step4k",
    "combined_no_asia_without_metadata_1b_step8k",
    "combined_no_europe_without_metadata_1b_step2k",
    "combined_no_europe_without_metadata_1b_step4k",
    "combined_no_europe_without_metadata_1b_step8k",
]

BASE_PATH_MODEL_METADATA_ABLATIONS = "/path/to/metacul/models/ablations/metadata/"
metadata_ablations = [
    "combined_only_url_with_metadata_1b",
    "combined_only_url_continent_with_metadata_1b",
    "combined_only_url_country_with_metadata_1b",
]

BASE_PATH_MODEL_METADATA_ABLATIONS_INTERMEDIATE = "/path/to/metacul/models/ablation_intermediates/metadata/"
metadata_ablations_intermediate = [
    "combined_only_url_with_metadata_1b_step2k",
    "combined_only_url_with_metadata_1b_step4k",
    "combined_only_url_with_metadata_1b_step8k",
    "combined_only_url_continent_with_metadata_1b_step2k",
    "combined_only_url_continent_with_metadata_1b_step4k",
    "combined_only_url_continent_with_metadata_1b_step8k",
    "combined_only_url_country_with_metadata_1b_step2k",
    "combined_only_url_country_with_metadata_1b_step4k",
    "combined_only_url_country_with_metadata_1b_step8k",
]

CONTINENT_TEST_DATA_PATH = "/path/to/metacul/training_data/meco_datasets/continents/"
continent_test_sets = [
    "africa/with_metadata/",
    "africa/without_metadata/",
    "america/with_metadata/",
    "america/without_metadata/",
    "asia/with_metadata/",
    "asia/without_metadata/",
    "europe/with_metadata/",
    "europe/without_metadata/",
]

COMBINED_TEST_DATA_PATH = "/path/to/metacul/training_data/meco_datasets/combined/"
combined_test_sets = [
    "with_metadata/",
    "without_metadata/",
]

METADATA_ABLATION_TEST_DATA_PATH = "/path/to/metacul/training_data/meco_datasets/"
metadata_ablation_test_sets = [
    "combined_only_url/with_metadata/",
    "combined_only_url_continent/with_metadata/",
    "combined_only_url_country/with_metadata/",
]

metadata_ablation_extra_test_sets = [
    "combined_only_country/with_metadata/",
    "combined_only_continent/with_metadata/",
]

metadata_ablation_extra_model_paths = [
    "/path/to/metacul/models/combined_only_continent_with_metadata_1b",
    "/path/to/metacul/models/combined_only_country_with_metadata_1b",
]

metadata_ablation_extra_intermediate_model_paths = [
    "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_continent_with_metadata_1b_step2k",
    "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_continent_with_metadata_1b_step4k",
    "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_continent_with_metadata_1b_step8k",
    "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_country_with_metadata_1b_step2k",
    "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_country_with_metadata_1b_step4k",
    "/path/to/metacul/models/ablation_intermediates/metadata/combined_only_country_with_metadata_1b_step8k",
]

CONTINENT_ABLATION_TEST_DATA_PATH = "/path/to/metacul/training_data/meco_datasets/"
continent_ablation_test_sets = [
    "combined_no_africa/with_metadata/",
    "combined_no_america/with_metadata/",
    "combined_no_asia/with_metadata/",
    "combined_no_europe/with_metadata/",
    "combined_no_africa/without_metadata/",
    "combined_no_america/without_metadata/",
    "combined_no_asia/without_metadata/",
    "combined_no_europe/without_metadata/",
]

# all eval combinations:
# 1. continent models + combined models on all continent test sets and all combined test sets
# 2. metadata ablation models (and metadata ablation intermediates) on its
#    corresponding (url only, url continent, url country) metadata ablation test sets
# 3. metadata ablation models (and metadata ablation intermediates) on all
#    (with, without) combined test sets
# 4. continent ablation models (and continent ablation intermediates) on all
#    (with, without) combined test sets
# 5. continent ablation models (and continent ablation intermediates) on all
#    opposite continent (e.g no europe is tested on europe only,) test sets

RESULTS_DIR = "/path/to/metacul/results"
EVAL_LIST_JSON = os.path.join(RESULTS_DIR, "perplexity", "eval_list.json")
PERPLEXITY_CSV = os.path.join(RESULTS_DIR, "perplexity_eval.csv")


def _paths(base, names):
    return [os.path.join(base, name) for name in names]


def _paths_with_names(base, names):
    return [(name, os.path.join(base, name)) for name in names]


def _metadata_slug(model_name):
    base = model_name
    step_idx = base.find("_step")
    if step_idx != -1:
        base = base[:step_idx]
    suffix = "_with_metadata_1b"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    return base


def _ablation_continent(model_name):
    base = model_name
    prefix = "combined_no_"
    if base.startswith(prefix):
        base = base[len(prefix):]
    return base.split("_", 1)[0]


def build_eval_combinations():
    combinations = []

    continent_model_paths = _paths(BASE_PATH_MODEL, continent_models)
    combined_model_paths = _paths(BASE_PATH_MODEL, combined_models)
    continent_test_paths = _paths(CONTINENT_TEST_DATA_PATH, continent_test_sets)
    combined_test_paths = _paths(COMBINED_TEST_DATA_PATH, combined_test_sets)

    combinations.extend(
        {
            "model_path": model_path,
            "test_set_path": test_set_path,
        }
        for model_path in continent_model_paths + combined_model_paths
        for test_set_path in continent_test_paths + combined_test_paths
    )

    metadata_models = (
        _paths_with_names(BASE_PATH_MODEL_METADATA_ABLATIONS, metadata_ablations)
        + _paths_with_names(
            BASE_PATH_MODEL_METADATA_ABLATIONS_INTERMEDIATE,
            metadata_ablations_intermediate,
        )
    )
    metadata_test_paths = {
        test_set.split("/", 1)[0]: os.path.join(
            METADATA_ABLATION_TEST_DATA_PATH, test_set
        )
        for test_set in metadata_ablation_test_sets
    }
    metadata_ablation_test_paths = _paths(
        METADATA_ABLATION_TEST_DATA_PATH, metadata_ablation_test_sets
    )

    combinations.extend(
        {
            "model_path": model_path,
            "test_set_path": metadata_test_paths[_metadata_slug(model_name)],
        }
        for model_name, model_path in metadata_models
    )

    combinations.extend(
        {
            "model_path": model_path,
            "test_set_path": test_set_path,
        }
        for model_name, model_path in metadata_models
        for test_set_path in metadata_ablation_test_paths
        if test_set_path != metadata_test_paths[_metadata_slug(model_name)]
    )

    combinations.extend(
        {
            "model_path": model_path,
            "test_set_path": test_set_path,
        }
        for _, model_path in metadata_models
        for test_set_path in combined_test_paths
    )

    metadata_ablation_extra_test_paths = _paths(
        METADATA_ABLATION_TEST_DATA_PATH, metadata_ablation_extra_test_sets
    )

    combinations.extend(
        {
            "model_path": model_path,
            "test_set_path": test_set_path,
        }
        for _, model_path in metadata_models
        for test_set_path in metadata_ablation_extra_test_paths
    )

    combinations.extend(
        {
            "model_path": model_path,
            "test_set_path": test_set_path,
        }
        for model_path in metadata_ablation_extra_model_paths
        for test_set_path in (
            metadata_ablation_test_paths
            + combined_test_paths
            + metadata_ablation_extra_test_paths
        )
    )

    combinations.extend(
        {
            "model_path": model_path,
            "test_set_path": test_set_path,
        }
        for model_path in metadata_ablation_extra_intermediate_model_paths
        for test_set_path in (
            metadata_ablation_test_paths
            + combined_test_paths
            + metadata_ablation_extra_test_paths
        )
    )

    continent_ablation_models = (
        _paths_with_names(BASE_PATH_MODEL_CONTINENT_ABLATIONS, continent_ablations)
        + _paths_with_names(
            BASE_PATH_MODEL_CONTINENT_ABLATIONS_INTERMEDIATE,
            continent_ablations_intermediate,
        )
    )

    combinations.extend(
        {
            "model_path": model_path,
            "test_set_path": test_set_path,
        }
        for _, model_path in continent_ablation_models
        for test_set_path in combined_test_paths
    )

    continent_test_paths_by_name = {}
    for test_set in continent_test_sets:
        continent = test_set.split("/", 1)[0]
        continent_test_paths_by_name.setdefault(continent, []).append(
            os.path.join(CONTINENT_TEST_DATA_PATH, test_set)
        )

    combinations.extend(
        {
            "model_path": model_path,
            "test_set_path": test_set_path,
        }
        for model_name, model_path in continent_ablation_models
        for test_set_path in continent_test_paths_by_name[
            _ablation_continent(model_name)
        ]
    )

    combined_metadata_models = [
        "combined_with_metadata_1b",
        "combined_with_metadata_3b",
        "combined_without_metadata_1b",
        "combined_without_metadata_3b",
    ]
    combined_metadata_model_paths = _paths(BASE_PATH_MODEL, combined_metadata_models)
    combinations.extend(
        {
            "model_path": model_path,
            "test_set_path": test_set_path,
        }
        for model_path in combined_metadata_model_paths
        for test_set_path in metadata_ablation_test_paths
    )

    combined_metadata_intermediate_models = [
        "combined_with_metadata_1b_step2k",
        "combined_with_metadata_1b_step4k",
        "combined_with_metadata_1b_step8k",
        "combined_with_metadata_3b_step2k",
        "combined_with_metadata_3b_step4k",
        "combined_with_metadata_3b_step8k",
        "combined_without_metadata_1b_step2k",
        "combined_without_metadata_1b_step4k",
        "combined_without_metadata_1b_step8k",
        "combined_without_metadata_3b_step2k",
        "combined_without_metadata_3b_step4k",
        "combined_without_metadata_3b_step8k",
    ]
    combined_metadata_intermediate_paths = _paths(
        BASE_PATH_MODEL_METADATA_ABLATIONS_INTERMEDIATE,
        combined_metadata_intermediate_models,
    )
    combinations.extend(
        {
            "model_path": model_path,
            "test_set_path": test_set_path,
        }
        for model_path in combined_metadata_intermediate_paths
        for test_set_path in metadata_ablation_test_paths
    )

    combinations.extend(
        {
            "model_path": model_path,
            "test_set_path": test_set_path,
        }
        for model_path in combined_metadata_intermediate_paths
        for test_set_path in combined_test_paths
    )

    continent_with_metadata_tests = _paths(
        CONTINENT_TEST_DATA_PATH,
        [
            "africa/with_metadata/",
            "america/with_metadata/",
            "asia/with_metadata/",
            "europe/with_metadata/",
        ],
    )
    continent_without_metadata_tests = _paths(
        CONTINENT_TEST_DATA_PATH,
        [
            "africa/without_metadata/",
            "america/without_metadata/",
            "asia/without_metadata/",
            "europe/without_metadata/",
        ],
    )
    combined_with_intermediate_paths = _paths(
        BASE_PATH_MODEL_METADATA_ABLATIONS_INTERMEDIATE,
        [
            "combined_with_metadata_1b_step2k",
            "combined_with_metadata_1b_step4k",
            "combined_with_metadata_1b_step8k",
            "combined_with_metadata_3b_step2k",
            "combined_with_metadata_3b_step4k",
            "combined_with_metadata_3b_step8k",
        ],
    )
    combined_without_intermediate_paths = _paths(
        BASE_PATH_MODEL_METADATA_ABLATIONS_INTERMEDIATE,
        [
            "combined_without_metadata_1b_step2k",
            "combined_without_metadata_1b_step4k",
            "combined_without_metadata_1b_step8k",
            "combined_without_metadata_3b_step2k",
            "combined_without_metadata_3b_step4k",
            "combined_without_metadata_3b_step8k",
        ],
    )
    combinations.extend(
        {
            "model_path": model_path,
            "test_set_path": test_set_path,
        }
        for model_path in combined_with_intermediate_paths
        for test_set_path in continent_with_metadata_tests
    )
    combinations.extend(
        {
            "model_path": model_path,
            "test_set_path": test_set_path,
        }
        for model_path in combined_without_intermediate_paths
        for test_set_path in continent_without_metadata_tests
    )

    return combinations


def _append_new_rows_to_csv(csv_path, combinations):
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return
        first_row = None
        existing = set()
        for row in reader:
            if first_row is None:
                first_row = row
            existing.add((row.get("model_path", ""), row.get("test_set_path", "")))

    new_rows = []
    for combo in combinations:
        key = (combo["model_path"], combo["test_set_path"])
        if key in existing:
            continue
        row = {name: "" for name in fieldnames}
        if first_row:
            for name in fieldnames:
                if name in (
                    "ci_level",
                    "ci_method",
                    "split",
                    "max_samples",
                    "seed",
                    "bootstrap_iters",
                    "skipped_samples",
                ):
                    row[name] = first_row.get(name, "")
        row["model_path"] = combo["model_path"]
        row["test_set_path"] = combo["test_set_path"]
        new_rows.append(row)

    if not new_rows:
        return
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(new_rows)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    combinations = build_eval_combinations()
    with open(EVAL_LIST_JSON, "w") as f:
        json.dump(combinations, f, indent=2)
    _append_new_rows_to_csv(PERPLEXITY_CSV, combinations)
