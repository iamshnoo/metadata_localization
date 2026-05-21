#!/usr/bin/env python3
from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
from matplotlib.lines import Line2D


RAW_ROOT = Path(
    "/path/to/metacul/results/downstream_localnewsqa_pretrained_figure9_frozen_full_multiseed"
)
OUT_DIR = Path("/path/to/metacul/slides")

FILES = {
    ("1B", "T+/I+"): RAW_ROOT
    / "1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_1b_tplus_eplus_seed41_c0.jsonl",
    ("1B", "T+/I-"): RAW_ROOT
    / "1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_1b_tplus_eminus_seed41_c0.jsonl",
    ("1B", "T-/I+"): RAW_ROOT
    / "1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_1b_tminus_eplus_seed41_c0.jsonl",
    ("1B", "T-/I-"): RAW_ROOT
    / "1b_codeg_labels_question_final/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_1b_tminus_eminus_seed41_c0.jsonl",
    ("3B", "T+/I+"): RAW_ROOT
    / "3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_3b_tplus_eplus_seed41_c0.jsonl",
    ("3B", "T+/I-"): RAW_ROOT
    / "3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_3b_tplus_eminus_seed41_c0.jsonl",
    ("3B", "T-/I+"): RAW_ROOT
    / "3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_with_metadata_custom_frozenfig9_3b_tminus_eplus_seed41_c0.jsonl",
    ("3B", "T-/I-"): RAW_ROOT
    / "3b_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos/seed_41/localnewsqa_eval_target_without_metadata_custom_frozenfig9_3b_tminus_eminus_seed41_c0.jsonl",
}


def summarize_jsonl(path: Path) -> dict:
    total = correct = 0
    explicit_total = explicit_correct = 0
    ambiguous_total = ambiguous_correct = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("processed_answer") is None:
                continue
            total += 1
            ok = int(bool(row.get("is_correct")))
            correct += ok
            split_type = str(row.get("split_type", "")).strip().lower()
            if split_type == "ambiguous":
                ambiguous_total += 1
                ambiguous_correct += ok
            else:
                explicit_total += 1
                explicit_correct += ok
    if total == 0:
        raise ValueError(f"No scored rows found in {path}")
    return {
        "Overall": correct / total,
        "Explicit": explicit_correct / explicit_total if explicit_total else 0.0,
        "Ambiguous": ambiguous_correct / ambiguous_total if ambiguous_total else 0.0,
        "rows": total,
    }


def load_df() -> pd.DataFrame:
    rows = []
    for (family, series), path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        stats = summarize_jsonl(path)
        for split in ("Overall", "Explicit", "Ambiguous"):
            rows.append(
                {
                    "family": family,
                    "series": f"{family} {series}",
                    "split": split,
                    "accuracy": stats[split],
                }
            )
    return pd.DataFrame(rows)


def _series_row(df, family, series, split):
    row = df[(df["family"] == family) & (df["series"] == series) & (df["split"] == split)]
    if row.empty:
        return None
    return row.iloc[0]


def build_overlay(df, visible_series, out_stem):
    split_order = ["Explicit", "Ambiguous", "Overall"]
    y_positions = {"Explicit": 2, "Ambiguous": 1, "Overall": 0}
    all_series_order = ["T-/I-", "T-/I+", "T+/I-", "T+/I+"]
    series_offsets = {
        "T+/I+": 0.18,
        "T+/I-": 0.06,
        "T-/I+": -0.06,
        "T-/I-": -0.18,
    }
    family_titles = ["1B", "3B"]

    colors = {
        "T+/I+": "#2f9e44",
        "T+/I-": "#78c679",
        "T-/I+": "#2b8cbe",
        "T-/I-": "#8a919c",
        "band": "#ebf6ec",
    }
    marker_style = {
        "T+/I+": dict(marker="o", s=210, facecolor=colors["T+/I+"], edgecolor="black", linewidth=1.8),
        "T+/I-": dict(marker="^", s=210, facecolor=colors["T+/I-"], edgecolor="black", linewidth=1.8),
        "T-/I+": dict(marker="D", s=165, facecolor=colors["T-/I+"], edgecolor="black", linewidth=1.65),
        "T-/I-": dict(marker="s", s=165, facecolor=colors["T-/I-"], edgecolor="black", linewidth=1.65),
    }

    x_vals = (df["accuracy"] * 100.0).tolist()
    x_min = max(0.0, min(x_vals) - 1.5) if x_vals else 0.0
    x_max = min(100.0, max(x_vals) + 1.5) if x_vals else 100.0

    tick_fs = 16
    label_fs = 19
    title_fs = 21
    legend_fs = 13
    row_label_fs = 18
    bbox_props = dict(
        facecolor="lightgrey",
        edgecolor="black",
        linewidth=1.2,
        alpha=0.84,
        boxstyle="round,pad=0.3",
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.6, 8.2),
        sharex=True,
        gridspec_kw={"hspace": 0.40},
    )

    for ax, family in zip(axes, family_titles):
        ax.axhspan(0.5, 1.5, color=colors["band"], zorder=0)

        for split in split_order:
            base_y = y_positions[split]
            row_points = []
            for short_series in all_series_order:
                if short_series not in visible_series:
                    continue
                full_series = f"{family} {short_series}"
                row = _series_row(df, family, full_series, split)
                if row is None:
                    continue
                x_val = float(row["accuracy"]) * 100.0
                y_val = base_y + series_offsets[short_series]
                row_points.append((x_val, y_val, short_series))

            row_points.sort(key=lambda item: item[0])
            if len(row_points) >= 2:
                connector = ax.plot(
                    [point[0] for point in row_points],
                    [point[1] for point in row_points],
                    color="#d6d9df",
                    linewidth=2.2,
                    alpha=0.85,
                    solid_capstyle="round",
                    zorder=1,
                )[0]
                connector.set_path_effects(
                    [pe.Stroke(linewidth=3.2, foreground="white", alpha=0.72), pe.Normal()]
                )

            for x_val, y_val, short_series in row_points:
                ax.scatter([x_val], [y_val], zorder=4, **marker_style[short_series])

        ax.set_yticks([y_positions[s] for s in split_order])
        ax.set_yticklabels(split_order, fontsize=row_label_fs)
        ax.set_title(
            family,
            fontsize=title_fs,
            fontweight="bold",
            pad=8,
            y=0.88,
            bbox=bbox_props,
        )
        ax.grid(axis="x", linestyle="-", linewidth=1.05, alpha=0.38, color="#c7ced8")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.35, 2.65)
        ax.tick_params(axis="x", labelsize=tick_fs)
        ax.tick_params(axis="y", length=0, pad=10)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color("black")

    axes[-1].set_xlabel("Target accuracy (%)", fontsize=label_fs)

    legend_label_map = {
        "T-/I-": ("s", colors["T-/I-"], "(T-, I-)"),
        "T-/I+": ("D", colors["T-/I+"], "(T-, I+)"),
        "T+/I-": ("^", colors["T+/I-"], "(T+, I-)"),
        "T+/I+": ("o", colors["T+/I+"], "(T+, I+)"),
    }
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=legend_label_map[short_series][0],
            color="none",
            markerfacecolor=legend_label_map[short_series][1],
            markeredgecolor="black",
            markeredgewidth=1.5 if short_series.startswith("T+") else 1.4,
            markersize=10.5 if short_series.startswith("T+") else 9.5,
            label=legend_label_map[short_series][2],
        )
        for short_series in all_series_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=True,
        fancybox=True,
        framealpha=0.94,
        edgecolor="black",
        fontsize=legend_fs,
        bbox_to_anchor=(0.5, 0.99),
        handletextpad=0.5,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.08, top=0.90, hspace=0.42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{out_stem}.png"
    pdf_path = OUT_DIR / f"{out_stem}.pdf"
    fig.savefig(png_path, dpi=220, pad_inches=0.0)
    fig.savefig(pdf_path, dpi=600, pad_inches=0.0)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


def main():
    df = load_df()
    build_overlay(df, {"T-/I-"}, "figure9_seed41_raw_overlay_step1_tminus_iminus")
    build_overlay(df, {"T-/I-", "T-/I+"}, "figure9_seed41_raw_overlay_step2_add_tminus_iplus")
    build_overlay(df, {"T-/I-", "T-/I+", "T+/I-"}, "figure9_seed41_raw_overlay_step3_add_tplus_iminus")
    build_overlay(df, {"T-/I-", "T-/I+", "T+/I-", "T+/I+"}, "figure9_seed41_raw_overlay_step4_add_tplus_iplus")


if __name__ == "__main__":
    main()
