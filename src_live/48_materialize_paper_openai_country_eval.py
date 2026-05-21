#!/usr/bin/env python3
from pathlib import Path
import sys
from typing import Dict

import pandas as pd


REPO_ROOT = Path("/path/to/metacul")
CULTURE_MAP_ROOT = Path("/path/to/culture-map")
DATA_DIR = CULTURE_MAP_ROOT / "data" / "paper_osf"
OUTPUT_DIR = CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human_paper" / "openai"
COUNTRY_MAPPING_CSV = (
    CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human" / "openai" / "all_model_country_mean_projection.csv"
)

sys.path.insert(0, str(CULTURE_MAP_ROOT / "src"))

from culture_map.constants import PART2_GROUPS  # noqa: E402
from culture_map.paper_assets import ensure_assets  # noqa: E402
from culture_map.projection import derive_projection_model, project_wide_table  # noqa: E402
from culture_map.provider_country_runner import _summarise_country_mean_projection  # noqa: E402


PAPER_MODELS = [
    ("gpt-3", "gpt-3", "GPT-3"),
    ("gpt-3.5-turbo-0613", "gpt-3.5-turbo", "GPT-3.5-turbo"),
    ("gpt-4-0613", "gpt-4", "GPT-4"),
    ("gpt-4-turbo-2024-04-09", "gpt-4-turbo", "GPT-4-turbo"),
    ("gpt-4o-2024-05-13", "gpt-4o", "GPT-4o"),
]


def _paths(model_name: str) -> Dict[str, Path]:
    return {
        "wide_csv": OUTPUT_DIR / f"{model_name}_wide_table.csv",
        "projected_csv": OUTPUT_DIR / f"{model_name}_variant_projection.csv",
        "country_mean_csv": OUTPUT_DIR / f"{model_name}_country_mean_projection.csv",
        "mean_csv": OUTPUT_DIR / f"{model_name}_mean_projection.csv",
    }


def _normalize_country_name(name: str) -> str:
    return str(name).replace("_", " ").strip()


def _load_scores(part2_key: str) -> pd.DataFrame:
    if part2_key == "gpt-3":
        scores = pd.read_csv(DATA_DIR / "GPT3_prompted_scores.csv").rename(columns={"country/territory": "country"})
        scores["#variant"] = 0
        return scores

    spec = PART2_GROUPS[part2_key]
    return pd.read_csv(DATA_DIR / spec["scores_filename"])


def _load_country_mapping() -> pd.DataFrame:
    if not COUNTRY_MAPPING_CSV.exists():
        raise FileNotFoundError(
            f"Missing country mapping CSV: {COUNTRY_MAPPING_CSV}. Run the recent OpenAI all-human country eval first."
        )

    mapping = (
        pd.read_csv(COUNTRY_MAPPING_CSV)[["country", "country_code", "continent", "Category"]]
        .drop_duplicates()
        .sort_values("country")
        .reset_index(drop=True)
    )
    if len(mapping) != 107:
        raise RuntimeError(f"Expected 107 countries in {COUNTRY_MAPPING_CSV}, found {len(mapping)}")
    return mapping


def main() -> None:
    ensure_assets(DATA_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    projection_model = derive_projection_model(DATA_DIR)
    country_mapping = _load_country_mapping()
    mean_frames = []
    country_mean_frames = []

    for part2_key, model_name, label in PAPER_MODELS:
        scores = _load_scores(part2_key)
        scores["country"] = scores["country"].map(_normalize_country_name)
        wide_table = scores.merge(
            country_mapping[["country", "country_code", "continent"]],
            on="country",
            how="left",
        )
        if wide_table["country_code"].isna().any():
            missing = sorted(wide_table.loc[wide_table["country_code"].isna(), "country"].dropna().unique().tolist())
            raise RuntimeError(f"Missing country-code mapping for {model_name}: {missing}")

        paths = _paths(model_name)
        wide_table.to_csv(paths["wide_csv"], index=False)

        projected = project_wide_table(wide_table, projection_model, label=label, model_name=model_name)
        projected.to_csv(paths["projected_csv"], index=False)

        country_means = projected.groupby(
            ["label", "model", "country", "country_code", "continent"],
            as_index=False,
        )[["RC1", "RC2"]].mean()
        country_means = country_means.merge(
            country_mapping[["country", "Category"]].drop_duplicates(),
            on="country",
            how="left",
        )
        country_means.to_csv(paths["country_mean_csv"], index=False)

        mean_projection = _summarise_country_mean_projection(
            country_means,
            label=label,
            model_name=model_name,
            country_scope="all_human",
            metadata_prompt_style="name_typical",
            prompt_variant_count=int(scores["#variant"].nunique()),
        )
        mean_projection.to_csv(paths["mean_csv"], index=False)

        mean_frames.append(mean_projection)
        country_mean_frames.append(country_means)

    pd.concat(mean_frames, ignore_index=True).to_csv(OUTPUT_DIR / "all_model_mean_projection.csv", index=False)
    pd.concat(country_mean_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "all_model_country_mean_projection.csv",
        index=False,
    )

    print(OUTPUT_DIR / "all_model_mean_projection.csv")
    print(OUTPUT_DIR / "all_model_country_mean_projection.csv")


if __name__ == "__main__":
    main()
