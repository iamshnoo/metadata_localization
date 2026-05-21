#!/usr/bin/env python3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path("/path/to/metacul")
CULTURE_MAP_ROOT = Path("/path/to/culture-map")
sys.path.insert(0, str(CULTURE_MAP_ROOT / "src"))

from culture_map.paper_assets import load_country_map  # noqa: E402
from culture_map.projection import derive_projection_model, load_paper_general_points  # noqa: E402


PLOT_DIR = REPO_ROOT / "results" / "plots" / "plot8"
LATEX_DIR = REPO_ROOT / "latex" / "figs" / "appendix"
DATA_DIR = CULTURE_MAP_ROOT / "data"
MAPLE_POINTS_CSV = REPO_ROOT / "results" / "culture_map_wvs_all_human_typical_mixed" / "all_variant_country_mean_projection.csv"
WITH_META_POINTS_CSV = PLOT_DIR / "_wvs_vendor_mixed_metadata_points.csv"
NO_META_POINTS_CSV = PLOT_DIR / "_wvs_vendor_no_metadata_points.csv"
DELTA_CSV = PLOT_DIR / "_wvs_vendor_distance_delta_nometa_minus_meta.csv"
PNG_PATH = PLOT_DIR / "wvs_vendor_plus_maple_distance_delta_nometa_minus_meta.png"
PDF_PATH = PLOT_DIR / "wvs_vendor_plus_maple_distance_delta_nometa_minus_meta.pdf"
LATEX_PDF_PATH = LATEX_DIR / "21_wvs_vendor_plus_maple_distance_delta_nometa_minus_meta.pdf"

COLOR_BY_FAMILY = {
    "OpenAI": "#2b6cb0",
    "Anthropic": "#dd6b20",
    "Google": "#2f855a",
    "Meta": "#805ad5",
    "Qwen": "#718096",
    "DeepSeek": "#0f766e",
    "Moonshot": "#b7791f",
    "Zhipu": "#9f1239",
    "MAPLE": "#c05621",
    "Other": "#4a5568",
}

MAPLE_WITH_META_VARIANTS = {
    "maple_1b_tplus_eplus": "MAPLE 1B",
    "maple_3b_tplus_eplus": "MAPLE 3B",
}

MAPLE_NO_META_VARIANTS = {
    "maple_1b_tminus_eminus": "MAPLE 1B",
    "maple_3b_tminus_eminus": "MAPLE 3B",
}


def _canonical_label(label: str) -> str:
    if label.startswith("MAPLE 1B"):
        return "MAPLE 1B"
    if label.startswith("MAPLE 3B"):
        return "MAPLE 3B"
    return label


def _family_for_row(label: str, model: str) -> str:
    text = f"{label} {model}".lower()
    if "maple" in text:
        return "MAPLE"
    if str(label).startswith("GPT") or str(model).startswith("gpt"):
        return "OpenAI"
    if "claude" in text:
        return "Anthropic"
    if "gemini" in text or "gemma" in text:
        return "Google"
    if "llama" in text:
        return "Meta"
    if "qwen" in text:
        return "Qwen"
    if "deepseek" in text:
        return "DeepSeek"
    if "kimi" in text:
        return "Moonshot"
    if "glm" in text:
        return "Zhipu"
    return "Other"


def _load_country_centroid() -> tuple[float, float]:
    human_df = load_country_map(DATA_DIR)
    return float(human_df["RC1_final"].mean()), float(human_df["RC2_final"].mean())


def _load_with_meta_points() -> pd.DataFrame:
    frame = pd.read_csv(WITH_META_POINTS_CSV)[["label", "model", "RC1", "RC2"]].copy()
    frame = frame.loc[~frame["label"].astype(str).str.startswith("MAPLE ")].reset_index(drop=True)
    frame["canonical_label"] = frame["label"].map(_canonical_label)
    return frame


def _load_no_meta_points() -> pd.DataFrame:
    vendor = pd.read_csv(NO_META_POINTS_CSV)[["label", "model", "RC1", "RC2"]].copy()
    vendor = vendor.loc[~vendor["label"].astype(str).str.startswith("MAPLE ")].reset_index(drop=True)

    projection_model = derive_projection_model(DATA_DIR)
    paper = load_paper_general_points(DATA_DIR, projection_model)[["label", "model", "RC1", "RC2"]].copy()

    combined = pd.concat([paper, vendor], ignore_index=True)
    combined["canonical_label"] = combined["label"].map(_canonical_label)
    combined = combined.drop_duplicates(subset=["canonical_label"], keep="last").reset_index(drop=True)
    return combined


def _load_maple_variant_points(variant_map: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_csv(MAPLE_POINTS_CSV)
    frame = frame.loc[frame["variant"].isin(variant_map)].copy()
    grouped = (
        frame.groupby("variant", as_index=False)[["RC1", "RC2"]]
        .mean()
        .assign(label=lambda df: df["variant"].map(variant_map), model="")
    )
    grouped["canonical_label"] = grouped["label"]
    return grouped[["label", "model", "RC1", "RC2", "canonical_label"]]


def _build_delta_table() -> pd.DataFrame:
    centroid_rc1, centroid_rc2 = _load_country_centroid()

    with_meta = pd.concat(
        [_load_with_meta_points(), _load_maple_variant_points(MAPLE_WITH_META_VARIANTS)],
        ignore_index=True,
    )
    no_meta = pd.concat(
        [_load_no_meta_points(), _load_maple_variant_points(MAPLE_NO_META_VARIANTS)],
        ignore_index=True,
    )

    with_meta["distance_with_meta"] = (
        (with_meta["RC1"] - centroid_rc1) ** 2 + (with_meta["RC2"] - centroid_rc2) ** 2
    ) ** 0.5
    no_meta["distance_no_meta"] = (
        (no_meta["RC1"] - centroid_rc1) ** 2 + (no_meta["RC2"] - centroid_rc2) ** 2
    ) ** 0.5

    merged = with_meta.merge(
        no_meta,
        on="canonical_label",
        how="inner",
        suffixes=("_with_meta", "_no_meta"),
    )
    merged["label"] = merged["canonical_label"]
    merged["family"] = [
        _family_for_row(str(label), str(model))
        for label, model in zip(merged["label"], merged["model_with_meta"].fillna(""), strict=False)
    ]
    merged["delta_distance"] = merged["distance_no_meta"] - merged["distance_with_meta"]
    merged = merged.sort_values(["distance_with_meta", "label"], ascending=[True, True]).reset_index(drop=True)
    merged["rank"] = merged.index + 1
    return merged[
        [
            "rank",
            "label",
            "family",
            "distance_with_meta",
            "distance_no_meta",
            "delta_distance",
            "RC1_with_meta",
            "RC2_with_meta",
            "RC1_no_meta",
            "RC2_no_meta",
        ]
    ]


def _draw_delta_plot(delta_df: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    delta_df.to_csv(DELTA_CSV, index=False)

    fig, ax = plt.subplots(figsize=(11.6, 8.4))
    y_positions = list(range(len(delta_df)))
    colors = [COLOR_BY_FAMILY.get(family, COLOR_BY_FAMILY["Other"]) for family in delta_df["family"]]

    ax.barh(y_positions, delta_df["delta_distance"], color=colors, height=0.72, alpha=0.95)
    ax.axvline(0.0, color="#1a202c", linewidth=1.4, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(delta_df["label"], fontsize=10)
    ax.invert_yaxis()

    min_delta = float(delta_df["delta_distance"].min())
    max_delta = float(delta_df["delta_distance"].max())
    left_pad = max(0.25, abs(min_delta) * 0.08)
    right_pad = max(0.35, abs(max_delta) * 0.20)
    ax.set_xlim(min_delta - left_pad, max_delta + right_pad)
    ax.set_xlabel("Delta Distance From Country Centroid (no meta - with meta)", fontsize=12, fontweight="bold")

    ax.text(
        0.0,
        1.01,
        "Positive values mean the metadata-conditioned point moved closer to the country centroid.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#4a5568",
    )

    label_pad = max(0.05, (abs(min_delta) + abs(max_delta) + left_pad + right_pad) * 0.015)
    for y_position, (_, row) in zip(y_positions, delta_df.iterrows(), strict=False):
        value = float(row["delta_distance"])
        if value < 0:
            x_position = value - label_pad
            ha = "right"
        else:
            x_position = value + label_pad
            ha = "left"
        ax.text(
            x_position,
            y_position,
            f"{value:+.3f}",
            va="center",
            ha=ha,
            fontsize=9,
            family="monospace",
            color="#1a202c",
        )

    present_families = list(dict.fromkeys(delta_df["family"].tolist()))
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOR_BY_FAMILY[family], markeredgecolor="none", markersize=7, label=family)
        for family in present_families
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        ncol=1,
        fontsize=9,
        columnspacing=1.1,
        handletextpad=0.4,
    )

    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#4a5568")
    ax.spines["bottom"].set_color("#4a5568")

    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(PDF_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(LATEX_PDF_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    delta_df = _build_delta_table()
    _draw_delta_plot(delta_df)
    print(DELTA_CSV)
    print(PNG_PATH)
    print(PDF_PATH)
    print(LATEX_PDF_PATH)


if __name__ == "__main__":
    main()
