# Nanotron Pretraining Guide

The pretraining component uses the project Nanotron fork:

- https://github.com/iamshnoo/nanotron

Nanotron is the training engine. This repository stores the MAPLE-specific
training recipes, dataset names, metadata ablations, checkpoint conversion
helpers, and evaluation scripts.

## What Lives Where

List the tracked recipe files with:

```bash
python tools/repo.py pretraining-recipes
```

| Path | Purpose |
| --- | --- |
| `src/step3_pretraining/3a_pretrain/continents/` | Markdown command manifests for Nanotron training runs. |
| `src/step3_pretraining/3a_pretrain/convert_to_hf.py` | Convert a Nanotron checkpoint into Hugging Face model format. |
| `src/step3_pretraining/3a_pretrain/convert_weights.py` | Weight/key mapping utilities used by conversion. |
| `src/step3_pretraining/3a_pretrain/converter.md` | Example conversion command. |
| `src/step3_pretraining/3b_pretrain_eval/` | Perplexity, clustering, model-card sync, and checkpoint-evaluation scripts. |
| `src/step3_pretraining/3b_pretrain_eval/scripts/` | SLURM wrappers for conversion, perplexity, upload, and sync jobs. |
| `slurm_live/` | Live cluster launchers used for the paper runs. |

## External Dependency

Clone and install the Nanotron fork next to this repository or wherever your
cluster expects it:

```bash
git clone https://github.com/iamshnoo/nanotron.git
cd nanotron
python -m pip install -e .
```

The recipe manifests call Nanotron's `slurm_launcher.py`. Run them from the
Nanotron checkout, or adjust the command to point at your local launcher path.

## Required Inputs

The public Git repository does not include the large training corpora or
checkpoints. The recipes assume datasets built by `src/step2_process_data/` are
available under paths like:

```text
/scratch/$HF_USER$/pretrain/datasets/<dataset-name>/train
/scratch/$HF_USER$/pretrain/datasets/<dataset-name>/validation
```

The recipes also assume scratch output roots:

```text
/scratch/$HF_USER$/pretrain/logs/configs
/scratch/$HF_USER$/pretrain/logs/slurm_logs
/scratch/$HF_USER$/pretrain/logs/slurm_scripts
/scratch/$HF_USER$/pretrain/logs/checkpoints
```

Set these before adapting a command:

```bash
export HF_USER=<cluster-user-or-scratch-namespace>
export GPU_GROUP=<slurm-qos-or-account>
```

## Training Recipe Matrix

The manifests are grouped by model size and condition.

```text
src/step3_pretraining/3a_pretrain/continents/
├── 500m/
│   ├── with_meta/       # Africa, America, Asia, Europe, combined
│   └── without_meta/    # Africa, America, Asia, Europe, combined
└── 1b/
    ├── with_meta/       # Africa, America, Asia, Europe, combined
    ├── without_meta/    # Africa, America, Asia, Europe, combined
    └── ablations/       # leave-one-continent and metadata-field ablations
```

Representative full metadata run:

```bash
sed -n '1,120p' src/step3_pretraining/3a_pretrain/continents/1b/with_meta/combined_with.md
```

Representative metadata ablation:

```bash
sed -n '1,120p' src/step3_pretraining/3a_pretrain/continents/1b/ablations/url_only.md
```

Each manifest is a copy-pastable `python slurm_launcher.py ...` command with:

- model scale (`--model 500m` or `--model 1b`);
- data/pipeline/tensor parallel settings (`--dp`, `--tp`, `--pp`);
- batch schedule (`--mbs`, `--acc`, `--seq`, `--steps`);
- tokenizer and vocabulary size;
- dataset and validation dataset paths;
- checkpoint, config, log, and generated SLURM script output paths.

## Run Pattern

From the Nanotron checkout:

```bash
cd /path/to/nanotron
# paste or adapt a command from one of the recipe markdown files
python slurm_launcher.py \
  --run combined_with_metadata_1b \
  --model 1b \
  --dataset /scratch/$HF_USER$/pretrain/datasets/combined-with-metadata/train \
  --validation-dataset /scratch/$HF_USER$/pretrain/datasets/combined-with-metadata/validation \
  --checkpoints-path /scratch/$HF_USER$/pretrain/logs/checkpoints \
  --configs-path /scratch/$HF_USER$/pretrain/logs/configs \
  --slurm-logs-path /scratch/$HF_USER$/pretrain/logs/slurm_logs \
  --slurm-scripts-dir /scratch/$HF_USER$/pretrain/logs/slurm_scripts \
  --auto-resume
```

Prefer the complete manifests over this shortened example, because the manifests
include the exact scheduler, optimization, and batch settings used in the runs.

## Convert Checkpoints To Hugging Face Format

After Nanotron writes a checkpoint, convert it with:

```bash
cd src/step3_pretraining/3a_pretrain
python -m torch.distributed.run \
  --nproc_per_node=1 convert_to_hf.py \
  --checkpoint_path=/scratch/$HF_USER$/pretrain/logs/checkpoints/<run-name>/<step> \
  --save_path=/scratch/$HF_USER$/metacul/models/<run-name>
```

The converter supports the project Llama/Qwen Nanotron checkpoint layouts used
by the submission runs. See `converter.md` for a concrete command.

## Evaluate Pretrained Checkpoints

Evaluation scripts live in `src/step3_pretraining/3b_pretrain_eval/`.

Common entrypoints:

| Script | Purpose |
| --- | --- |
| `17_eval_list.py`, `27_build_3b_ppl_eval_list.py`, `28_build_country_continent_only_eval_list.py` | Build evaluation manifests. |
| `18_perplexity_eval.py` | Run perplexity evaluation for converted checkpoints. |
| `29_merge_perplexity_csvs.py` | Merge perplexity CSV outputs. |
| `22_noamerica_cluster.py` | Run leave-one-region clustering analysis. |
| `scripts/convert_intermediate_*.sbatch` | Convert intermediate checkpoints on the cluster. |
| `scripts/run_ppl_*.sbatch` | Launch perplexity evaluation jobs. |
| `scripts/upload_models_to_hf.sbatch` | Upload converted model artifacts when configured. |

Tracked outputs from these scripts are under `results/perplexity*`,
`results/noamerica_cluster/`, and related `results/plots/` folders.

## Reusability Boundary

Reusable and public:

- Nanotron run manifests;
- model/dataset condition names;
- checkpoint conversion utilities;
- evaluation and plotting scripts;
- summary result artifacts.

External by design:

- raw NOW corpus data;
- tokenized training datasets;
- Nanotron checkpoints;
- cluster logs and generated configs;
- Hugging Face and Weights & Biases credentials.
