from pathlib import Path

import pandas as pd
import requests

from .constants import ASSET_URLS, CATEGORY_FILENAME, DESCRIPTORS_FILENAME, HUMAN_COORDS_FILENAME, QUESTIONS_FILENAME, TEMPLATE_2023_FILENAME
from .utils import ensure_dir


def download_asset(url, destination, overwrite=False):
    destination = Path(destination)
    if destination.exists() and not overwrite:
        return destination

    ensure_dir(destination.parent)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                handle.write(chunk)
    return destination


def ensure_assets(data_dir, overwrite=False):
    data_dir = ensure_dir(data_dir)
    downloaded = {}
    for filename, url in ASSET_URLS.items():
        downloaded[filename] = download_asset(url, data_dir / filename, overwrite=overwrite)
    return downloaded


def load_questions(data_dir):
    ensure_assets(data_dir)
    return pd.read_csv(Path(data_dir) / QUESTIONS_FILENAME)


def load_descriptors(data_dir):
    ensure_assets(data_dir)
    return pd.read_csv(Path(data_dir) / DESCRIPTORS_FILENAME)


def load_country_map(data_dir):
    ensure_assets(data_dir)
    coords = pd.read_csv(Path(data_dir) / HUMAN_COORDS_FILENAME)
    categories = pd.read_csv(Path(data_dir) / CATEGORY_FILENAME)
    human = coords[["country.territory", "RC1_human_survey", "RC2_human_survey"]].drop_duplicates()
    human = human.merge(categories, on="country.territory", how="left")
    human = human.rename(
        columns={
            "country.territory": "country",
            "RC1_human_survey": "RC1_final",
            "RC2_human_survey": "RC2_final",
        }
    )
    return human.sort_values(["Category", "country"]).reset_index(drop=True)


def get_template_image_path(data_dir):
    ensure_assets(data_dir)
    return Path(data_dir) / TEMPLATE_2023_FILENAME
