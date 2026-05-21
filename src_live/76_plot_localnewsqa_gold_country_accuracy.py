#!/usr/bin/env python3
import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path("/path/to/metacul")
DEFAULT_SFT_ROOT = ROOT / "results/localnewsqa_gold_20260516/sft_target"
DEFAULT_OUT_DIR = ROOT / "results/localnewsqa_gold_20260516/plots"
DEFAULT_LATEX_DIR = ROOT / "latex/figs/appendix"

CONTINENT_ORDER = ["Africa", "America", "Asia", "Europe"]
CONTINENT_BG = {
    "Africa": "#f8e8ea",
    "America": "#e8f1fb",
    "Asia": "#eef7e8",
    "Europe": "#f7f0e6",
}
VARIANT_COLORS = {
    "(T-, I-)": "#d95f02",
    "(T+, I+)": "#1b9e77",
}
SPECS = {
    "1B": {
        "root_name": "1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos",
        "variants": [
            {
                "label": "(T-, I-)",
                "run_tag": "sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tminus_eminus",
                "eval_meta": "without_metadata",
            },
            {
                "label": "(T+, I+)",
                "run_tag": "sftgold_1b_chat_nameplain_country_final_qanswer_nolabel_qmask025_bos_tplus_eplus",
                "eval_meta": "with_metadata",
            },
        ],
        "latex_name": "14_country_qa_accuracy_1b.pdf",
        "plot_name": "country_qa_accuracy_1b.pdf",
    },
    "3B": {
        "root_name": "3b_best3b_chat_name_grounded_final_answer_colon_qanswer_nolabel_qmask_bos",
        "variants": [
            {
                "label": "(T-, I-)",
                "run_tag": "sftgold_3b_best3b_chat_tminus_eminus",
                "eval_meta": "without_metadata",
            },
            {
                "label": "(T+, I+)",
                "run_tag": "sftgold_3b_best3b_chat_tplus_eplus",
                "eval_meta": "with_metadata",
            },
        ],
        "latex_name": "15_country_qa_accuracy_3b.pdf",
        "plot_name": "country_qa_accuracy_3b.pdf",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LocalNewsQA gold country-level SFT accuracy.")
    parser.add_argument("--sft-root", type=Path, default=DEFAULT_SFT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--latex-dir", type=Path, default=DEFAULT_LATEX_DIR)
    parser.add_argument("--seeds", default="41,42,43,44,45")
    parser.add_argument("--write-latex", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def parse_seeds(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def sample_std(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) <= 1:
        return 0.0
    avg = sum(vals) / len(vals)
    return math.sqrt(sum((v - avg) ** 2 for v in vals) / (len(vals) - 1))


def normalize_continent(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:].lower()


def country_name(row: Dict[str, object]) -> str:
    return str(row.get("country") or row.get("target_country") or "").strip()


def find_run(seed_dir: Path, eval_meta: str, run_tag: str, seed: int) -> Optional[Path]:
    pattern = f"localnewsqa_eval_target_{eval_meta}_custom*{run_tag}_seed{seed}_c0*.jsonl"
    matches = sorted(seed_dir.glob(pattern))
    return matches[0] if matches else None


def load_country_seed(path: Path, size: str, variant: str, seed: int) -> List[Dict[str, object]]:
    counts: Dict[tuple, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "is_correct" not in row:
                continue
            country = country_name(row)
            continent = normalize_continent(row.get("continent"))
            if not country or continent not in CONTINENT_ORDER:
                continue
            key = (continent, country)
            stats = counts.setdefault(
                key,
                {"correct": 0, "total": 0, "explicit": 0, "ambiguous": 0},
            )
            stats["correct"] = int(stats["correct"]) + int(bool(row["is_correct"]))
            stats["total"] = int(stats["total"]) + 1
            split = str(row.get("split_type", "")).strip().lower()
            if split == "ambiguous":
                stats["ambiguous"] = int(stats["ambiguous"]) + 1
            else:
                stats["explicit"] = int(stats["explicit"]) + 1
    records = []
    for (continent, country), stats in counts.items():
        total = int(stats["total"])
        if total <= 0:
            continue
        records.append(
            {
                "size": size,
                "variant": variant,
                "seed": seed,
                "continent": continent,
                "country": country,
                "accuracy": int(stats["correct"]) / total,
                "correct": int(stats["correct"]),
                "total": total,
                "explicit": int(stats["explicit"]),
                "ambiguous": int(stats["ambiguous"]),
                "path": str(path),
            }
        )
    return records


def collect(args: argparse.Namespace, seeds: List[int]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    missing: List[str] = []
    for size, spec in SPECS.items():
        for variant in spec["variants"]:
            for seed in seeds:
                seed_dir = args.sft_root / spec["root_name"] / f"seed_{seed}"
                path = find_run(seed_dir, variant["eval_meta"], variant["run_tag"], seed)
                if path is None:
                    missing.append(f"{size} {variant['label']} seed={seed}")
                    continue
                records.extend(load_country_seed(path, size, variant["label"], seed))
    if missing:
        message = "Missing country-plot inputs: " + ", ".join(missing)
        if args.strict:
            raise FileNotFoundError(message)
        print("[warn] " + message)
    return pd.DataFrame(records)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["size", "variant", "continent", "country"]
    for keys, sub in raw.groupby(group_cols):
        size, variant, continent, country = keys
        values = sub["accuracy"].astype(float).tolist()
        rows.append(
            {
                "size": size,
                "variant": variant,
                "continent": continent,
                "country": country,
                "accuracy": mean(values),
                "accuracy_std": sample_std(values),
                "seed_count": len(values),
                "mean_total_per_seed": mean(sub["total"].astype(float).tolist()),
            }
        )
    out = pd.DataFrame(rows)
    out["continent_order"] = out["continent"].map({c: i for i, c in enumerate(CONTINENT_ORDER)})
    out.sort_values(["size", "continent_order", "country", "variant"], inplace=True)
    out.drop(columns=["continent_order"], inplace=True)
    return out


def render_size(df: pd.DataFrame, size: str, out_dir: Path) -> Path:
    sub = df[df["size"] == size].copy()
    if sub.empty:
        raise ValueError(f"No rows for size={size}")

    order: List[str] = []
    group_specs = []
    positions: Dict[str, float] = {}
    cursor = 0.0
    continent_gap = 1.25
    for continent in CONTINENT_ORDER:
        continent_df = sub[sub["continent"] == continent].copy()
        if continent_df.empty:
            continue
        plus = continent_df[continent_df["variant"] == "(T+, I+)"]
        countries = plus.sort_values("accuracy", ascending=False)["country"].tolist()
        if not countries:
            countries = sorted(continent_df["country"].unique().tolist())
        x = np.arange(len(countries), dtype=float) + cursor
        for country, pos in zip(countries, x):
            positions[country] = float(pos)
            order.append(country)
        group_specs.append(
            {
                "continent": continent,
                "start": float(x[0]),
                "end": float(x[-1]),
                "center": float((x[0] + x[-1]) / 2),
                "width": 0.70 if continent != "Asia" else 0.58,
            }
        )
        cursor = float(x[-1]) + continent_gap

    fig, ax = plt.subplots(figsize=(18.5, 6.4), facecolor="white")
    ax.set_facecolor("white")
    for spec in group_specs:
        ax.axvspan(
            spec["start"] - spec["width"] * 0.75,
            spec["end"] + spec["width"] * 0.75,
            color=CONTINENT_BG.get(spec["continent"], "#f5f5f5"),
            alpha=0.30,
            linewidth=0,
            zorder=0.1,
        )

    offsets = {"(T-, I-)": -0.15, "(T+, I+)": 0.15}
    for country in order:
        country_rows = sub[sub["country"] == country].set_index("variant")
        if "(T-, I-)" in country_rows.index and "(T+, I+)" in country_rows.index:
            y0 = float(country_rows.loc["(T-, I-)", "accuracy"])
            y1 = float(country_rows.loc["(T+, I+)", "accuracy"])
            ax.plot(
                [positions[country] + offsets["(T-, I-)"], positions[country] + offsets["(T+, I+)"]],
                [y0, y1],
                color="#777777",
                linewidth=1.8,
                alpha=0.75,
                zorder=2.0,
            )
    for variant in ["(T-, I-)", "(T+, I+)"]:
        variant_rows = sub[sub["variant"] == variant].copy()
        variant_rows["x"] = variant_rows["country"].map(positions) + offsets[variant]
        ax.scatter(
            variant_rows["x"],
            variant_rows["accuracy"],
            s=86,
            color=VARIANT_COLORS[variant],
            edgecolors="#4f4f4f",
            linewidths=0.8,
            zorder=3.0,
            label=variant,
        )

    y_min = max(0.0, float(sub["accuracy"].min()) - 0.035)
    y_max = min(1.0, float(sub["accuracy"].max()) + 0.065)
    if y_max - y_min < 0.12:
        pad = (0.12 - (y_max - y_min)) / 2
        y_min = max(0.0, y_min - pad)
        y_max = min(1.0, y_max + pad)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(min(positions.values()) - 0.9, max(positions.values()) + 0.9)
    ax.set_xticks([positions[c] for c in order])
    ax.set_xticklabels(order, rotation=38, ha="right", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.23)
    ax.set_axisbelow(True)
    ax.set_ylabel("Accuracy", fontsize=15)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)
    for spec in group_specs:
        ax.text(
            spec["center"],
            y_max - 0.035,
            spec["continent"],
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
            bbox=dict(facecolor="white", edgecolor="#9a9a9a", alpha=0.88, boxstyle="round,pad=0.28"),
            zorder=4,
        )
    handles = [
        Line2D([], [], marker="o", linestyle="None", markersize=8.5, color=VARIANT_COLORS["(T-, I-)"], label="(T-, I-)"),
        Line2D([], [], marker="o", linestyle="None", markersize=8.5, color=VARIANT_COLORS["(T+, I+)"], label="(T+, I+)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True, fontsize=12, bbox_to_anchor=(0.5, 0.90))
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, bottom=0.30, left=0.055, right=0.995)
    out_path = out_dir / SPECS[size]["plot_name"]
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = collect(args, seeds)
    if raw.empty:
        raise SystemExit("No country-level rows found.")
    raw.to_csv(args.out_dir / "country_qa_accuracy_seed_rows.csv", index=False)
    summary = aggregate(raw)
    summary.to_csv(args.out_dir / "country_qa_accuracy_summary.csv", index=False)

    generated: Dict[str, Path] = {}
    for size in SPECS:
        generated[size] = render_size(summary, size, args.out_dir)
        print(f"[ok] wrote {generated[size]}")

    if args.write_latex:
        args.latex_dir.mkdir(parents=True, exist_ok=True)
        for size, path in generated.items():
            dst = args.latex_dir / SPECS[size]["latex_name"]
            shutil.copy2(path, dst)
            print(f"[ok] copied {path} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
