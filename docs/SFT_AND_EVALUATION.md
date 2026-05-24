# SFT And Evaluation

This guide collects the supervised fine-tuning and evaluation components that
sit around the pretraining and dataset pipelines.

## SFT Components

The SFT stage lives under:

```text
src/step4_sft/4a_qa_data_generation/
src/step4_sft/4b_sft/
src/step4_sft/4c_sft_eval/
```

Use the Python component index:

```bash
python tools/repo.py sft
```

Main pieces:

| Component | Files | Role |
| --- | --- | --- |
| QA data generation | `4a_qa_data_generation/` | Legacy QA generation and dataset build scripts mirrored from `qa_data/`. |
| SFT training | `4b_sft/12_sft.py`, `4b_sft/15_sft.py`, `4b_sft/scripts/run_sft_*.sbatch` | Train adapters for with-metadata and without-metadata model variants. |
| LoRA merge | `4b_sft/13_merge_lora.py`, `4b_sft/16_merge_lora.py`, `4b_sft/scripts/merge_sft_*.sbatch` | Merge trained adapters back into model checkpoints for evaluation. |
| SFT evaluation | `4c_sft_eval/14_sft_eval.py`, `4c_sft_eval/18_sft_eval_grid.py`, `4c_sft_eval/24_sft_paired_significance.py`, `4c_sft_eval/25_sft_eval_external.py` | Run LocalNewsQA, adversarial, paired-significance, and external benchmark evaluations. |

The `.sbatch` and `.slurm` files are cluster run manifests. They document exact
submission arguments from the paper runs, but paths, accounts, partitions, and
cache locations may need editing on another cluster.

## Evaluation Families

List the main evaluation entrypoints:

```bash
python tools/repo.py evals
```

The repository covers several evaluation surfaces:

| Evaluation | Entry points | Outputs |
| --- | --- | --- |
| Pretraining perplexity | `src/step3_pretraining/3b_pretrain_eval/` | Perplexity CSVs, merged summaries, and pretraining plots. |
| LocalNewsQA pretrained models | `slurm_live/pretrained_localnewsqa_eval_*.slurm`, `slurm_live/localnewsqa_eval_*.slurm` | Localized QA accuracy and locale-switch analyses. |
| LocalNewsQA SFT models | `src/step4_sft/4c_sft_eval/`, `slurm_live/submit_figure9_*.sh` | SFT accuracy, contrast, multiseed, and ablation summaries. |
| External benchmarks | `src/step4_sft/4c_sft_eval/25_sft_eval_external.py`, `slurm_live/pretrained_external_eval_*.slurm`, `slurm_live/submit_external_*.sh` | GeoMLAMA, GlobalOpinionQA, GlobalMMLU, GOQA, CROQ-related, and related benchmark summaries. |
| Cultural projection | `culture_map/`, `slurm_live/submit_culture_map_wvs_maple.sh`, `slurm_live/culture_map_wvs_country_eval_single.slurm` | WVS-style country projections, distances, and map overlays. |
| Open-ended geographic probes | `slurm_live/openended_geo_eval_single.slurm`, `slurm_live/submit_openended_*.sh` | Open-ended locale and geography response analyses. |
| Paper plots and significance | `src/step5_plots/`, `results/analysis/` | Figure inputs, bootstrap/significance files, and paper-ready plot artifacts. |

## Result Artifacts

The `results/` directory tracks summary-heavy artifacts rather than full raw
logs. Useful starting points:

```text
results/analysis/localnewsqa_significance/
results/analysis/localnewsqa_locale_switch/
results/analysis/localnewsqa_sft_locale_switch/
results/appendix_model_gain_tables_20260505/
results/culture_map_wvs_3b_improve/
```

See `docs/RESULTS.md` for the broader inventory.

## Relationship To Pretraining

Pretraining itself uses the project Nanotron fork documented in
`docs/PRETRAINING.md`. The SFT and evaluation scripts consume the converted
pretrained checkpoints and compare with-metadata, without-metadata,
leave-one-continent, metadata-field-ablation, and SFT-adapted variants.
