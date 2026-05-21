#!/usr/bin/env python3
import csv
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path("/path/to/metacul")
CULTURE_MAP_ROOT = Path("/path/to/culture-map")
sys.path.insert(0, str(CULTURE_MAP_ROOT / "src"))

from culture_map.paper_assets import load_country_map  # noqa: E402
from culture_map.plotting import plot_template_map  # noqa: E402
from culture_map.projection import derive_projection_model, load_paper_general_points  # noqa: E402


PLOT_DIR = REPO_ROOT / "results" / "plots" / "plot8"
LATEX_DIR = REPO_ROOT / "latex" / "figs" / "appendix"
MAPLE_ROOT = REPO_ROOT / "results" / "culture_map_wvs_all_human_typical_mixed"
DATA_DIR = CULTURE_MAP_ROOT / "data"
TEMPLATE_IMAGE = CULTURE_MAP_ROOT / "data" / "paper_osf" / "Map2023NEWsmall.png"
LEGACY_VENDOR_POINTS_CSV = CULTURE_MAP_ROOT / "outputs" / "all_vendor_plus_together_completed_no_gpt54mini.csv"
COUNTRY_CONDITIONED_VENDOR_POINTS_CSV = (
    CULTURE_MAP_ROOT / "outputs" / "all_vendor_plus_together_completed_country_conditioned_all_human_name_typical.csv"
)
COUNTRY_CONDITIONED_MEAN_FILES = [
    CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human_paper" / "openai" / "all_model_mean_projection.csv",
    CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human" / "openai" / "all_model_mean_projection.csv",
    CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human" / "anthropic" / "all_model_mean_projection.csv",
    CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human" / "gemini" / "all_model_mean_projection.csv",
    CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human" / "together" / "all_model_mean_projection.csv",
    CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human_split" / "Qwen_Qwen3.5-397B-A17B" / "all_model_mean_projection.csv",
    CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human_split" / "claude-haiku-4-5-20251001" / "all_model_mean_projection.csv",
    CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human_split" / "claude-sonnet-4-6" / "all_model_mean_projection.csv",
    CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human_split" / "deepseek-ai_DeepSeek-V3.1" / "all_model_mean_projection.csv",
    CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human_split" / "gemini-2.5-flash-lite" / "all_model_mean_projection.csv",
    CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human_split" / "gemini-2.5-flash" / "all_model_mean_projection.csv",
    CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human_split" / "gemma-4-31b-it" / "all_model_mean_projection.csv",
    CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human_split" / "gpt-5.4-nano" / "all_model_mean_projection.csv",
    CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human_split" / "moonshotai_Kimi-K2.5" / "all_model_mean_projection.csv",
    CULTURE_MAP_ROOT / "outputs" / "provider_country_eval_all_human_split" / "zai-org_GLM-5.1" / "all_model_mean_projection.csv",
]

NO_METADATA_PNG = PLOT_DIR / "wvs_vendor_plus_maple_nometa_official_axes.png"
NO_METADATA_PDF = PLOT_DIR / "wvs_vendor_plus_maple_nometa_official_axes.pdf"
MIXED_METADATA_PNG = PLOT_DIR / "wvs_vendor_plus_maple_metadata_mixed_official_axes.png"
MIXED_METADATA_PDF = PLOT_DIR / "wvs_vendor_plus_maple_metadata_mixed_official_axes.pdf"
DEFAULT_PNG = PLOT_DIR / "wvs_vendor_plus_maple_official_axes.png"
DEFAULT_PDF = PLOT_DIR / "wvs_vendor_plus_maple_official_axes.pdf"
LATEX_NO_METADATA_PDF = LATEX_DIR / "21_wvs_vendor_plus_maple_nometa_official_axes.pdf"
LATEX_MIXED_METADATA_PDF = LATEX_DIR / "21_wvs_vendor_plus_maple_metadata_mixed_official_axes.pdf"
LATEX_DEFAULT_PDF = LATEX_DIR / "21_wvs_vendor_plus_maple_official_axes.pdf"
TMP_VENDOR_NO_METADATA = PLOT_DIR / "_wvs_vendor_no_metadata_points.csv"
TMP_VENDOR_MIXED_METADATA = PLOT_DIR / "_wvs_vendor_mixed_metadata_points.csv"
TMP_VENDOR_PLUS_MAPLE_MIXED_METADATA = PLOT_DIR / "_wvs_vendor_plus_maple_mixed_metadata_points.csv"
MIXED_VENDOR_OUTPUT_CSV = CULTURE_MAP_ROOT / "outputs" / "all_vendor_plus_together_completed_country_conditioned_all_human_name_typical.csv"

XMIN = -2.5
XMAX = 3.5
YMIN = -2.5
YMAX = 2.0

LABELS = {
    "maple_1b_tplus_eplus": "MAPLE 1B (T+, I+)",
    "maple_3b_tplus_eplus": "MAPLE 3B (T+, I+)",
}
CENTROID_LABEL_OFFSET = (-18.0, -16.0)
PREFERRED_METADATA_MODEL_ORDER = ["gpt-3", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]
HISTORICAL_OPENAI_LABELS = {"GPT-3", "GPT-3.5-turbo", "GPT-4", "GPT-4-turbo", "GPT-4o"}
EXCLUDED_VENDOR_MODELS = {"gemini-2.5-pro"}


def _in_display_bounds(x_value, y_value):
    return XMIN <= x_value <= XMAX and YMIN <= y_value <= YMAX


def _load_maple_centroids():
    sums = {key: [0.0, 0.0, 0] for key in LABELS}
    with (MAPLE_ROOT / "all_variant_country_mean_projection.csv").open() as handle:
        for row in csv.DictReader(handle):
            variant = row["variant"]
            if variant not in sums:
                continue
            sums[variant][0] += float(row["RC1"])
            sums[variant][1] += float(row["RC2"])
            sums[variant][2] += 1

    records = []
    for variant, (sum_x, sum_y, count) in sums.items():
        if not count:
            continue
        mean_x = sum_x / count
        mean_y = sum_y / count
        if not _in_display_bounds(mean_x, mean_y):
            continue
        records.append({"label": LABELS[variant], "RC1": mean_x, "RC2": mean_y})
    return pd.DataFrame(records)


def _load_country_centroid():
    human_df = load_country_map(DATA_DIR)
    return {
        "label": "Country centroid",
        "RC1": float(human_df["RC1_final"].mean()),
        "RC2": float(human_df["RC2_final"].mean()),
    }


def _load_paper_points(exclude_labels=None):
    projection_model = derive_projection_model(DATA_DIR)
    paper_points = load_paper_general_points(DATA_DIR, projection_model)
    if not exclude_labels:
        return paper_points.reset_index(drop=True)
    return paper_points.loc[~paper_points["label"].isin(exclude_labels)].reset_index(drop=True)


def _load_vendor_points(use_country_conditioned=True):
    legacy = pd.read_csv(LEGACY_VENDOR_POINTS_CSV)[["label", "model", "RC1", "RC2"]].copy()
    legacy = legacy.loc[~legacy["model"].isin(EXCLUDED_VENDOR_MODELS)].reset_index(drop=True)
    expected_order = PREFERRED_METADATA_MODEL_ORDER + legacy["model"].tolist()
    order_map = {model: idx for idx, model in enumerate(expected_order)}

    combined = legacy.copy()
    if use_country_conditioned:
        finished_frames = []
        for path in COUNTRY_CONDITIONED_MEAN_FILES:
            if not path.exists():
                continue
            frame = pd.read_csv(path)[["label", "model", "RC1", "RC2"]].copy()
            finished_frames.append(frame)
        if finished_frames:
            finished = pd.concat(finished_frames, ignore_index=True)
            finished = finished.drop_duplicates(subset=["model"], keep="last")
            finished_by_model = {row["model"]: row for _, row in finished.iterrows()}
            for idx, row in combined.iterrows():
                replacement = finished_by_model.get(row["model"])
                if replacement is None:
                    continue
                combined.at[idx, "label"] = replacement["label"]
                combined.at[idx, "RC1"] = replacement["RC1"]
                combined.at[idx, "RC2"] = replacement["RC2"]

            existing_models = set(combined["model"].tolist())
            extra_models = [
                model
                for model in PREFERRED_METADATA_MODEL_ORDER
                if model in finished_by_model and model not in existing_models
            ]
            if extra_models:
                extra_rows = [finished_by_model[model] for model in extra_models]
                extra_frame = pd.DataFrame(extra_rows)[["label", "model", "RC1", "RC2"]]
                combined = pd.concat([extra_frame, combined], ignore_index=True)

    combined = combined.loc[~combined["model"].isin(EXCLUDED_VENDOR_MODELS)].reset_index(drop=True)
    combined["__order"] = combined["model"].map(lambda model: order_map.get(model, len(order_map)))
    combined = combined.sort_values(["__order", "model"]).drop(columns="__order").reset_index(drop=True)
    return combined


def _write_points_csv(output_path: Path, include_maple: bool, use_country_conditioned: bool, vendor_output_path: Optional[Path] = None):
    frame = _load_vendor_points(use_country_conditioned=use_country_conditioned)
    if include_maple:
        maple = _load_maple_centroids()
        if len(maple):
            frame = pd.concat([frame, maple], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    if vendor_output_path is not None:
        vendor_output_path.parent.mkdir(parents=True, exist_ok=True)
        _load_vendor_points(use_country_conditioned=use_country_conditioned).to_csv(vendor_output_path, index=False)


def _add_axis_end_labels(ax):
    ax.text(
        205,
        165,
        "Strong\nSecular\nValues",
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        color="black",
        zorder=8,
    )
    ax.text(
        1455,
        905,
        "Strong\nSelf-Expression\nValues",
        ha="right",
        va="bottom",
        fontsize=22,
        fontweight="bold",
        color="black",
        zorder=8,
    )


def _move_country_centroid_label(ax, offset_points=CENTROID_LABEL_OFFSET):
    dx_value, dy_value = offset_points
    for artist in ax.texts:
        if artist.get_text() != "Country centroid":
            continue
        artist.set_position((dx_value, dy_value))
        artist.set_ha("left" if dx_value >= 0 else "right")
        artist.set_va("bottom" if dy_value >= 0 else "top")
        return
    raise RuntimeError("Country centroid label not found on axes")


def _render(
    points_csv: Path,
    output_png: Path,
    output_pdf: Path,
    extra_pdf: Optional[Path] = None,
    paper_exclude_labels=None,
):
    country_centroid = _load_country_centroid()
    paper_points = _load_paper_points(exclude_labels=paper_exclude_labels)
    fig, ax = plot_template_map(
        template_image_path=TEMPLATE_IMAGE,
        overlay_point_paths=[str(points_csv)],
        paper_points=paper_points,
        output_path=None,
        hide_title=True,
        hide_source_block=True,
        add_axis_end_labels=False,
        add_zero_guides=True,
        exclude_labels=None,
        country_centroid=country_centroid,
    )
    _move_country_centroid_label(ax)
    _add_axis_end_labels(ax)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight", pad_inches=0)
    if extra_pdf is not None:
        extra_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(extra_pdf, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    _write_points_csv(
        TMP_VENDOR_NO_METADATA,
        include_maple=False,
        use_country_conditioned=False,
        vendor_output_path=CULTURE_MAP_ROOT / "outputs" / "all_vendor_plus_together_completed_no_metadata.csv",
    )
    _write_points_csv(
        TMP_VENDOR_MIXED_METADATA,
        include_maple=False,
        use_country_conditioned=True,
        vendor_output_path=MIXED_VENDOR_OUTPUT_CSV,
    )
    _write_points_csv(
        TMP_VENDOR_PLUS_MAPLE_MIXED_METADATA,
        include_maple=True,
        use_country_conditioned=True,
    )

    _render(TMP_VENDOR_NO_METADATA, NO_METADATA_PNG, NO_METADATA_PDF, extra_pdf=LATEX_NO_METADATA_PDF)
    _render(
        TMP_VENDOR_MIXED_METADATA,
        MIXED_METADATA_PNG,
        MIXED_METADATA_PDF,
        extra_pdf=LATEX_MIXED_METADATA_PDF,
        paper_exclude_labels=HISTORICAL_OPENAI_LABELS,
    )
    _render(
        TMP_VENDOR_PLUS_MAPLE_MIXED_METADATA,
        DEFAULT_PNG,
        DEFAULT_PDF,
        extra_pdf=LATEX_DEFAULT_PDF,
        paper_exclude_labels=HISTORICAL_OPENAI_LABELS,
    )

    print(NO_METADATA_PNG)
    print(NO_METADATA_PDF)
    print(LATEX_NO_METADATA_PDF)
    print(MIXED_METADATA_PNG)
    print(MIXED_METADATA_PDF)
    print(LATEX_MIXED_METADATA_PDF)
    print(DEFAULT_PNG)
    print(DEFAULT_PDF)
    print(LATEX_DEFAULT_PDF)


if __name__ == "__main__":
    main()
