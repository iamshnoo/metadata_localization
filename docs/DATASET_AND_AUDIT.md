# Dataset Generation And Audit

This repository has two QA data surfaces:

- `qa_data/localnewsqa_core/`: the LocalNewsQA Core generation, validation,
  automated audit, repair, and release pipeline.
- `qa_data/`: the older `qa_metacul` benchmark kept for compatibility with
  earlier downstream runs.

For new evaluation work, start with LocalNewsQA Core.

## Released Gold Data

Tracked release files:

```text
qa_data/localnewsqa_core/final_gold_20260516/localnewsqa_gold_explicit17000_ambiguous1700.jsonl
qa_data/localnewsqa_core/final_gold_20260516/localnewsqa_gold_explicit17000_ambiguous1700_summary.json
qa_data/hf_dataset_localnewsqa_gold_20260516/
```

The release contains 18,700 rows: 17,000 explicit rows and 1,700 ambiguous
rows. Both splits are balanced across 17 countries. The parquet directory is a
Hugging Face-style dataset export of the same release.

Load the JSONL:

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="qa_data/localnewsqa_core/final_gold_20260516/localnewsqa_gold_explicit17000_ambiguous1700.jsonl",
    split="train",
)
```

Load the parquet export:

```python
from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files="qa_data/hf_dataset_localnewsqa_gold_20260516/data/train-00000-of-00001.parquet",
    split="train",
)
```

## Component Entry Points

Use the Python index to see the main reusable scripts:

```bash
python tools/repo.py localnewsqa-pipeline
```

Each script can be inspected or run directly, and the portable scripts expose
their arguments through `--help`.

## Pipeline Stages

| Stage | Files | Role |
| --- | --- | --- |
| Request generation | `01_build_generation_requests.py`, `config.py`, `prompts/` | Build Batch API request JSONL for explicit and ambiguous candidate generation. |
| Candidate extraction | `02_extract_generation_candidates.py`, `02b_prune_generation_candidates.py`, `05_finalize_core_dataset.py` | Convert provider outputs into normalized candidates, remove malformed or leaky rows, and prepare core pools. |
| Human validation | `06_*.py` through `23_*.py`, `human_validation_guidelines.md` | Build reviewer samples, dashboards, evidence fields, source certification, and replacement pools. |
| Automated audit | `24_audit_full_dataset_quality.py`, `25_estimate_semantic_quality.py`, `26_flag_weak_locale_ambiguous.py`, `27_llm_verify_core_quality.py` | Apply heuristic, semantic, weak-locale, and LLM-verifier checks. |
| Web evidence audit | `32_*.py` through `40_*.py` | Audit URLs and evidence, fetch missing support, and repair weak ambiguous rows. |
| Strict gold construction | `41_*.py` through `70_*.py` | Build semantic-gold ambiguous rows, strict explicit rows, reviewer-risk reports, and final release data. |

Late numbered scripts preserve the exact curation and repair manifests used in
the submission. Some expect the `runs/` workspace produced by earlier steps.
They remain valuable because they expose the actual quality gates and manual
repair logic, while large intermediate artifacts stay out of Git.

## Automated Quality Gates

The audit stack covers:

- schema and formatting validity;
- duplicate IDs and duplicate normalized questions;
- country and answer leakage checks;
- weak locale cues and target-only anchoring risks;
- semantic support and evidence quality;
- target/contrast relation support for ambiguous rows;
- final reviewer-risk and human weak-locale pattern audits.

The final summary at
`qa_data/localnewsqa_core/final_gold_20260516/localnewsqa_gold_explicit17000_ambiguous1700_summary.json`
records balanced country counts, duplicate checks, and split-level validity.

## Artifact Policy

Tracked:

- generation/audit source code;
- prompts and validation guidelines;
- final JSONL release;
- final release summary;
- compact Hugging Face-style parquet export.

Excluded:

- `qa_data/localnewsqa_core/runs/`;
- raw Batch API outputs;
- web fetch caches;
- reviewer dashboard artifacts;
- multi-GB intermediate audit pools.
