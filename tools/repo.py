#!/usr/bin/env python3
"""Small Python entrypoint index for the repository."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


COMPONENTS = [
    (
        "culture_map",
        "Installable WVS/cultural projection package.",
        "python -m pip install -e culture_map && culture-map --help",
    ),
    (
        "LocalNewsQA Core",
        "Dataset generation, validation, audit, and release pipeline.",
        "python tools/repo.py localnewsqa-pipeline",
    ),
    (
        "LocalNewsQA gold data",
        "Tracked 18,700 row final gold benchmark and HF parquet export.",
        "qa_data/localnewsqa_core/final_gold_20260516/",
    ),
    (
        "Nanotron pretraining",
        "MAPLE training recipes for the iamshnoo/nanotron fork.",
        "python tools/repo.py pretraining-recipes",
    ),
    (
        "SFT",
        "Supervised fine-tuning and LoRA merge scripts.",
        "python tools/repo.py sft",
    ),
    (
        "Evaluations",
        "Pretraining, SFT, LocalNewsQA, external, WVS, and plot pipelines.",
        "python tools/repo.py evals",
    ),
]


LOCALNEWSQA_GROUPS = [
    (
        "Generation",
        [
            "qa_data/localnewsqa_core/01_build_generation_requests.py",
            "qa_data/localnewsqa_core/02_extract_generation_candidates.py",
            "qa_data/localnewsqa_core/02b_prune_generation_candidates.py",
            "qa_data/localnewsqa_core/05_finalize_core_dataset.py",
        ],
    ),
    (
        "Human validation and evidence collection",
        [
            "qa_data/localnewsqa_core/06_build_human_validation_sample.py",
            "qa_data/localnewsqa_core/07_web_validate_ambiguous_sample.py",
            "qa_data/localnewsqa_core/09_build_human_validation_dashboard.py",
            "qa_data/localnewsqa_core/11_fill_validation_evidence_urls.py",
            "qa_data/localnewsqa_core/13_certify_validation_sources.py",
        ],
    ),
    (
        "Automated audit and semantic verification",
        [
            "qa_data/localnewsqa_core/24_audit_full_dataset_quality.py",
            "qa_data/localnewsqa_core/25_estimate_semantic_quality.py",
            "qa_data/localnewsqa_core/26_flag_weak_locale_ambiguous.py",
            "qa_data/localnewsqa_core/27_llm_verify_core_quality.py",
            "qa_data/localnewsqa_core/46_final_gold_quality_audit.py",
            "qa_data/localnewsqa_core/47_reviewer_risk_audit.py",
        ],
    ),
    (
        "Strict gold release",
        [
            "qa_data/localnewsqa_core/61_build_explicit_strict_defensible_balanced.py",
            "qa_data/localnewsqa_core/64_build_explicit_strict_defensible_1000_curated.py",
            "qa_data/localnewsqa_core/67_build_gold_eval_dataset.py",
            "qa_data/localnewsqa_core/67_validate_explicit_strict_defensible_1000_final.py",
            "qa_data/localnewsqa_core/70_audit_ambiguous_human_weak_locale_pattern.py",
        ],
    ),
]


SFT_GROUPS = [
    (
        "Training",
        [
            "src/step4_sft/4b_sft/12_sft.py",
            "src/step4_sft/4b_sft/15_sft.py",
            "src/step4_sft/4b_sft/scripts/run_sft_3b_with_metadata.sbatch",
            "src/step4_sft/4b_sft/scripts/run_sft_3b_without_metadata.sbatch",
        ],
    ),
    (
        "LoRA merge",
        [
            "src/step4_sft/4b_sft/13_merge_lora.py",
            "src/step4_sft/4b_sft/16_merge_lora.py",
            "src/step4_sft/4b_sft/scripts/merge_sft_3b.sbatch",
            "src/step4_sft/4b_sft/scripts/merge_sft_3b_best.sbatch",
        ],
    ),
    (
        "SFT evaluation",
        [
            "src/step4_sft/4c_sft_eval/14_sft_eval.py",
            "src/step4_sft/4c_sft_eval/18_sft_eval_grid.py",
            "src/step4_sft/4c_sft_eval/24_sft_paired_significance.py",
            "src/step4_sft/4c_sft_eval/25_sft_eval_external.py",
        ],
    ),
]


EVAL_GROUPS = [
    (
        "Pretraining evaluation",
        [
            "src/step3_pretraining/3b_pretrain_eval/12_perplexity_eval.py",
            "src/step3_pretraining/3b_pretrain_eval/18_perplexity_eval.py",
            "src/step3_pretraining/3b_pretrain_eval/29_merge_perplexity_csvs.py",
        ],
    ),
    (
        "LocalNewsQA and external benchmarks",
        [
            "slurm_live/pretrained_localnewsqa_eval_single.slurm",
            "slurm_live/pretrained_external_eval_single.slurm",
            "slurm_live/localnewsqa_eval_single_combo.slurm",
            "slurm_live/submit_external_localnewsqa_contrast_multiseed.sh",
        ],
    ),
    (
        "Cultural projection",
        [
            "culture_map/README.md",
            "slurm_live/submit_culture_map_wvs_maple.sh",
            "slurm_live/culture_map_wvs_country_eval_single.slurm",
        ],
    ),
    (
        "Analysis and plots",
        [
            "src/step5_plots/20_significance_tests.py",
            "src/step5_plots/67_plot_localnewsqa_accuracy_switch_composite.py",
            "src/step5_plots/73_plot_localnewsqa_gold_accuracy_switch.py",
            "results/analysis/localnewsqa_significance/summary.json",
        ],
    ),
]


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def print_table(rows: Iterable[tuple[str, str, str]]) -> None:
    names = [name for name, _, _ in rows]
    width = max(len(name) for name in names)
    for name, purpose, command in rows:
        print(f"{name:<{width}}  {purpose}")
        print(f"{'':<{width}}  {command}")


def print_groups(groups: Iterable[tuple[str, list[str]]]) -> None:
    for title, paths in groups:
        print(f"\n{title}")
        print("-" * len(title))
        for path in paths:
            marker = "" if (REPO_ROOT / path).exists() else " (missing)"
            print(f"{path}{marker}")


def pretraining_recipes() -> None:
    roots = [
        REPO_ROOT / "src/step3_pretraining/3a_pretrain/continents",
        REPO_ROOT / "src/step3_pretraining/3a_pretrain",
    ]
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path in seen:
                continue
            if path.suffix in {".md", ".sh", ".py"} and path.name != "__init__.py":
                seen.add(path)
                print(rel(path))


def clean_pycache() -> None:
    count = 0
    for path in REPO_ROOT.rglob("__pycache__"):
        shutil.rmtree(path)
        count += 1
    for path in REPO_ROOT.rglob("*.pyc"):
        path.unlink()
        count += 1
    print(f"removed {count} Python cache paths")


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover reusable repository components.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("components", help="List the major reusable components.")
    subparsers.add_parser("pretraining-recipes", help="List Nanotron/MAPLE recipe files.")
    subparsers.add_parser("localnewsqa-pipeline", help="List LocalNewsQA generation and audit entrypoints.")
    subparsers.add_parser("sft", help="List SFT training, merge, and evaluation entrypoints.")
    subparsers.add_parser("evals", help="List evaluation and analysis entrypoints.")
    subparsers.add_parser("clean-pycache", help="Remove Python cache files from the checkout.")

    args = parser.parse_args()
    if args.command == "components":
        print_table(COMPONENTS)
    elif args.command == "pretraining-recipes":
        pretraining_recipes()
    elif args.command == "localnewsqa-pipeline":
        print_groups(LOCALNEWSQA_GROUPS)
    elif args.command == "sft":
        print_groups(SFT_GROUPS)
    elif args.command == "evals":
        print_groups(EVAL_GROUPS)
    elif args.command == "clean-pycache":
        clean_pycache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
