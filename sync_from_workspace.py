#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO = Path(os.environ.get('METADATA_LOCALIZATION_REPO', Path(__file__).resolve().parent))
WORKSPACE = Path(os.environ.get('METACUL_WORKSPACE', REPO.parent / 'metacul'))

SRC_MAP = {
    'src/0_data_now.py': 'src/step0_dataset/0_data_now.py',
    'src/utils/stats_now.py': 'src/step0_dataset/stats_now.py',
    'src/scripts/0_now_data.slurm': 'src/step0_dataset/scripts/0_now_data.slurm',
    'src/scripts/0_now_years_splits.txt': 'src/step0_dataset/scripts/0_now_years_splits.txt',
    'src/1_lda.py': 'src/step1_lda_analysis/1_lda.py',
    'src/scripts/1_lda.slurm': 'src/step1_lda_analysis/scripts/1_lda.slurm',
    'src/scripts/1_lda_splits.txt': 'src/step1_lda_analysis/scripts/1_lda_splits.txt',
    'src/config.py': 'src/step2_process_data/config.py',
    'src/2_generate_document_metadata.py': 'src/step2_process_data/2a_metadata_processing/2_generate_document_metadata.py',
    'src/3_identify_dominant_themes.py': 'src/step2_process_data/2a_metadata_processing/3_identify_dominant_themes.py',
    'src/4_update_metadata_with_themes.py': 'src/step2_process_data/2a_metadata_processing/4_update_metadata_with_themes.py',
    'src/5_create_meta_index.py': 'src/step2_process_data/2a_metadata_processing/5_create_meta_index.py',
    'src/scripts/2_generate_document_metadata.slurm': 'src/step2_process_data/2a_metadata_processing/scripts/2_generate_document_metadata.slurm',
    'src/scripts/2_doc_metadata_splits.txt': 'src/step2_process_data/2a_metadata_processing/scripts/2_doc_metadata_splits.txt',
    'src/6_data_splitter_ids_only.py': 'src/step2_process_data/2b_hf_dataset_creation/6_data_splitter_ids_only.py',
    'src/7_create_hf_datasets_streaming.py': 'src/step2_process_data/2b_hf_dataset_creation/7_create_hf_datasets_streaming.py',
    'src/8_create_meco_datasets.py': 'src/step2_process_data/2b_hf_dataset_creation/8_create_meco_datasets.py',
    'src/9_create_combined_datasets.py': 'src/step2_process_data/2b_hf_dataset_creation/9_create_combined_datasets.py',
    'src/9_create_combined_metadata_ablations.py': 'src/step2_process_data/2b_hf_dataset_creation/9_create_combined_metadata_ablations.py',
    'src/10_create_leave_one_out_datasets.py': 'src/step2_process_data/2b_hf_dataset_creation/10_create_leave_one_out_datasets.py',
    'src/utils/check_train_data.py': 'src/step2_process_data/2b_hf_dataset_creation/check_train_data.py',
    'src/utils/data_stats.py': 'src/step2_process_data/2b_hf_dataset_creation/data_stats.py',
    'src/utils/upload_datasets.py': 'src/step2_process_data/2b_hf_dataset_creation/upload_datasets.py',
    'src/scripts/7_create_hf_datasets.slurm': 'src/step2_process_data/2b_hf_dataset_creation/scripts/7_create_hf_datasets.slurm',
    'src/scripts/7_hf_dataset_splits.txt': 'src/step2_process_data/2b_hf_dataset_creation/scripts/7_hf_dataset_splits.txt',
    'src/scripts/8_meco_data.slurm': 'src/step2_process_data/2b_hf_dataset_creation/scripts/8_meco_data.slurm',
    'src/scripts/8_meco_splits.txt': 'src/step2_process_data/2b_hf_dataset_creation/scripts/8_meco_splits.txt',
    'src/12_sft.py': 'src/step4_sft/4b_sft/12_sft.py',
    'src/13_merge_lora.py': 'src/step4_sft/4b_sft/13_merge_lora.py',
    'src/chat_template.jinja': 'src/step4_sft/4b_sft/chat_template.jinja',
    'src/14_sft_eval.py': 'src/step4_sft/4c_sft_eval/14_sft_eval.py',
    'src/15_sft_eval_grid.py': 'src/step4_sft/4c_sft_eval/15_sft_eval_grid.py',
    'src/16_sft_analyse.py': 'src/step4_sft/4c_sft_eval/16_sft_analyse.py',
    'src/17_eval_list.py': 'src/step3_pretraining/3b_pretrain_eval/17_eval_list.py',
    'src/18_perplexity_eval.py': 'src/step3_pretraining/3b_pretrain_eval/18_perplexity_eval.py',
    'src/22_noamerica_cluster.py': 'src/step3_pretraining/3b_pretrain_eval/22_noamerica_cluster.py',
    'src/23_sft_adversarial_url_analysis.py': 'src/step4_sft/4c_sft_eval/23_sft_adversarial_url_analysis.py',
    'src/24_sft_paired_significance.py': 'src/step4_sft/4c_sft_eval/24_sft_paired_significance.py',
    'src/25_sft_eval_external.py': 'src/step4_sft/4c_sft_eval/25_sft_eval_external.py',
    'src/27_build_3b_ppl_eval_list.py': 'src/step3_pretraining/3b_pretrain_eval/27_build_3b_ppl_eval_list.py',
    'src/28_build_country_continent_only_eval_list.py': 'src/step3_pretraining/3b_pretrain_eval/28_build_country_continent_only_eval_list.py',
    'src/29_merge_perplexity_csvs.py': 'src/step3_pretraining/3b_pretrain_eval/29_merge_perplexity_csvs.py',
    'src/19_perplexity_plot.py': 'src/step5_plots/19_perplexity_plot.py',
    'src/20_significance_tests.py': 'src/step5_plots/20_significance_tests.py',
    'src/21_url_signal_analysis.py': 'src/step5_plots/21_url_signal_analysis.py',
    'src/26_geomlama_metadata_hypothesis_summary.py': 'src/step5_plots/26_geomlama_metadata_hypothesis_summary.py',
    'src/67_plot_localnewsqa_accuracy_switch_composite.py': 'src/step5_plots/67_plot_localnewsqa_accuracy_switch_composite.py',
    'src/73_plot_localnewsqa_gold_accuracy_switch.py': 'src/step5_plots/73_plot_localnewsqa_gold_accuracy_switch.py',
    'src/utils/estimate_token_lengths.py': 'src/step3_pretraining/3b_pretrain_eval/estimate_token_lengths.py',
    'src/scripts/11_combined_sft.slurm': 'src/step4_sft/4b_sft/scripts/11_combined_sft.slurm',
    'src/scripts/12_downstream_grid_eval.slurm': 'src/step4_sft/4c_sft_eval/scripts/12_downstream_grid_eval.slurm',
    'src/scripts/25_sft_eval_external.slurm': 'src/step4_sft/4c_sft_eval/scripts/25_sft_eval_external.slurm',
}

SLURM_MAP = {
    'slurm/convert_intermediate_3b_to_hf.sbatch': 'src/step3_pretraining/3b_pretrain_eval/scripts/convert_intermediate_3b_to_hf.sbatch',
    'slurm/run_ppl_3b_requested.sbatch': 'src/step3_pretraining/3b_pretrain_eval/scripts/run_ppl_3b_requested.sbatch',
    'slurm/run_ppl_country_continent_only.sbatch': 'src/step3_pretraining/3b_pretrain_eval/scripts/run_ppl_country_continent_only.sbatch',
    'slurm/convert_intermediate_1b_metadata_family.sbatch': 'src/step3_pretraining/3b_pretrain_eval/scripts/convert_intermediate_1b_metadata_family.sbatch',
    'slurm/run_qwen_arch_ppl_10k.sbatch': 'src/step3_pretraining/3b_pretrain_eval/scripts/run_qwen_arch_ppl_10k.sbatch',
    'slurm/upload_models_to_hf.sbatch': 'src/step3_pretraining/3b_pretrain_eval/scripts/upload_models_to_hf.sbatch',
    'slurm/run_sft_3b_with_metadata.sbatch': 'src/step4_sft/4b_sft/scripts/run_sft_3b_with_metadata.sbatch',
    'slurm/run_sft_3b_without_metadata.sbatch': 'src/step4_sft/4b_sft/scripts/run_sft_3b_without_metadata.sbatch',
    'slurm/merge_sft_3b.sbatch': 'src/step4_sft/4b_sft/scripts/merge_sft_3b.sbatch',
    'slurm/run_sft_eval_grid_3b.sbatch': 'src/step4_sft/4c_sft_eval/scripts/run_sft_eval_grid_3b.sbatch',
}

QA_FILES = ['qa_gen.py', 'build_hf_dataset.py', 'developer.md', 'user.md']

LIVE_SRC_SUFFIXES = {'.jinja', '.py', '.slurm'}
LIVE_SRC_SCRIPT_SUFFIXES = {'.jinja', '.py', '.sbatch', '.sh', '.slurm', '.txt'}
LIVE_SLURM_SUFFIXES = {'.py', '.sbatch', '.sh', '.slurm'}
RESULT_DIRS = [
    'results/plots',
    'results/significance',
    'results/url_analysis',
    'results/mechanistic',
    'results/appendix_model_gain_tables_20260505',
    'results/appendix_model_gain_tables_20260505_check',
    'results/appendix_model_gain_tables_20260505_probe',
]
RESULT_FILTERED_DIRS = [
    'results/analysis',
    'results/localnewsqa_gold_20260516',
    'results/final_benchmark_matrix',
    'results/external_benchmarks_maple_tuned_seed41',
    'results/external_paper_table_tokenizerfix_20260504',
    'results/external_scale3b_tinker_20260505',
]
RESULT_FILTERED_PREFIXES = (
    'results/culture_map_wvs',
    'results/external_protocol',
)
RESULT_SUMMARY_SUFFIXES = {'.csv', '.json', '.md', '.tex'}


def copy_file(src_rel: str, dst_rel: str) -> None:
    src = WORKSPACE / src_rel
    dst = REPO / dst_rel
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def replace_tree(src_rel: str, dst_rel: str) -> None:
    src = WORKSPACE / src_rel
    dst = REPO / dst_rel
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_flat_files(src_rel: str, dst_rel: str, suffixes: set[str]) -> None:
    src_dir = WORKSPACE / src_rel
    dst_dir = REPO / dst_rel
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        if src.is_file() and src.suffix in suffixes:
            shutil.copy2(src, dst_dir / src.name)


def copy_tree_if_present(src_rel: str, dst_rel: str) -> None:
    src = WORKSPACE / src_rel
    if not src.exists():
        return
    replace_tree(src_rel, dst_rel)


def copy_tree_filtered(src_rel: str, dst_rel: str, suffixes: set[str]) -> None:
    src_root = WORKSPACE / src_rel
    dst_root = REPO / dst_rel
    if not src_root.exists():
        return
    for src in src_root.rglob('*'):
        if src.is_file() and src.suffix in suffixes:
            dst = dst_root / src.relative_to(src_root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def copy_result_artifacts() -> None:
    for rel in RESULT_DIRS:
        copy_tree_if_present(rel, rel)
    for rel in RESULT_FILTERED_DIRS:
        copy_tree_filtered(rel, rel, RESULT_SUMMARY_SUFFIXES)
    results_root = WORKSPACE / 'results'
    if results_root.exists():
        for src in results_root.iterdir():
            rel = f'results/{src.name}'
            if src.is_dir() and rel.startswith(RESULT_FILTERED_PREFIXES):
                copy_tree_filtered(rel, rel, RESULT_SUMMARY_SUFFIXES)


shutil.copy2(WORKSPACE / 'README.md', REPO / 'README.md')
replace_tree('qa_data', 'qa_data')
copy_result_artifacts()
for src_rel, dst_rel in SRC_MAP.items():
    copy_file(src_rel, dst_rel)
for src_rel, dst_rel in SLURM_MAP.items():
    copy_file(src_rel, dst_rel)
for name in QA_FILES:
    copy_file(f'qa_data/{name}', f'src/step4_sft/4a_qa_data_generation/{name}')
copy_flat_files('src', 'src_live', LIVE_SRC_SUFFIXES)
copy_flat_files('src/scripts', 'src_live/scripts', LIVE_SRC_SCRIPT_SUFFIXES)
copy_flat_files('slurm', 'slurm_live', LIVE_SLURM_SUFFIXES)

for candidate in (REPO / 'src/scripts').iterdir():
    if candidate.is_file() and candidate.name.startswith('.' + 'env'):
        candidate.unlink()

print('sync complete')
