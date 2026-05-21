#!/usr/bin/env python3
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import pandas as pd


REPO_ROOT = Path("/path/to/metacul")
WVS_ROOT = REPO_ROOT / "results" / "culture_map_wvs_best_maple_mixed"
OUTPUT_DIR = REPO_ROOT / "results" / "plots" / "plot8"

CULTURE_MAP_SRC = Path("/path/to/culture-map/src")
if str(CULTURE_MAP_SRC) not in sys.path:
    sys.path.insert(0, str(CULTURE_MAP_SRC))

from culture_map.paper_assets import get_template_image_path, load_country_map  # noqa: E402
from culture_map.plotting import plot_culture_map_with_template_shapes  # noqa: E402


LABEL_MAP = {
    "maple_1b_tminus_eminus": "MAPLE 1B (T-, I-)",
    "maple_1b_tplus_eplus": "MAPLE 1B (T+, I+)",
    "maple_3b_tminus_eminus": "MAPLE 3B (T-, I-)",
    "maple_3b_tplus_eplus": "MAPLE 3B (T+, I+)",
}


def main():
    means = pd.read_csv(WVS_ROOT / "all_variant_country_mean_projection.csv")
    keep_countries = sorted(means["country"].unique())

    centroids = (
        means.groupby("variant", as_index=False)
        .agg(RC1=("RC1", "mean"), RC2=("RC2", "mean"))
        .assign(label=lambda df: df["variant"].map(LABEL_MAP))
        [["label", "RC1", "RC2"]]
    )

    data_dir = Path("/path/to/culture-map/data/paper_osf")
    human_df = load_country_map(data_dir)
    human_df = human_df.loc[human_df["country"].isin(keep_countries)].copy()

    for ext in ("png", "pdf"):
        out = OUTPUT_DIR / f"wvs_maple_vendor_filtered.{ext}"
        plot_culture_map_with_template_shapes(
            human_df=human_df,
            template_image_path=get_template_image_path(data_dir),
            paper_points=centroids,
            output_path=out,
        )
        print(out)


if __name__ == "__main__":
    main()
