# LocalNewsQA Core

`LocalNewsQA Core` is the reusable dataset generation, validation, audit, and
release pipeline for the final LocalNewsQA benchmark used in the submission.
It is separate from the older `qa_metacul` benchmark preserved one directory
up.

## Quick Use

List the main pipeline entrypoints:

```bash
python tools/repo.py localnewsqa-pipeline
```

Inspect the portable generation and audit CLIs:

```bash
python qa_data/localnewsqa_core/01_build_generation_requests.py --help
python qa_data/localnewsqa_core/24_audit_full_dataset_quality.py --help
python qa_data/localnewsqa_core/27_llm_verify_core_quality.py --help
python qa_data/localnewsqa_core/46_final_gold_quality_audit.py --help
```

Load the released gold set directly:

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="qa_data/localnewsqa_core/final_gold_20260516/localnewsqa_gold_explicit17000_ambiguous1700.jsonl",
    split="train",
)
```

The same release is also tracked as a Hugging Face-style parquet export in
`qa_data/hf_dataset_localnewsqa_gold_20260516/`.

## Release Artifact

The final tracked release is:

```text
final_gold_20260516/localnewsqa_gold_explicit17000_ambiguous1700.jsonl
final_gold_20260516/localnewsqa_gold_explicit17000_ambiguous1700_summary.json
```

Release size:

- 18,700 total rows.
- 17,000 explicit rows, balanced at 1,000 rows per country.
- 1,700 ambiguous rows, balanced at 100 rows per country.
- 17 countries across America, Asia, Africa, and Europe.

The explicit split names the locale in the question. The ambiguous split omits
the locale and keeps target/contrast metadata for locale-dependent answers.
Rows include options, correct answers, country/continent/topic metadata,
evidence fields, validation fields, and release provenance.

## Pipeline

| Stage | Scripts | Purpose |
| --- | --- | --- |
| Candidate generation | `01_build_generation_requests.py`, `02_extract_generation_candidates.py`, `02b_prune_generation_candidates.py`, `05_finalize_core_dataset.py` | Build Batch API request JSONL, extract model candidates, prune malformed rows, and assemble core candidates. |
| Human validation | `06_build_human_validation_sample.py` through `23_manual_curate_full_tail.py` | Build reviewer samples, dashboards, evidence URL fills, source certification, and curated replacement pools. |
| Automated quality audit | `24_audit_full_dataset_quality.py`, `25_estimate_semantic_quality.py`, `26_flag_weak_locale_ambiguous.py`, `27_llm_verify_core_quality.py` | Run heuristic, semantic, weak-locale, and LLM-based checks before final curation. |
| Web and semantic repair | `31_build_no_api_splits.py` through `45_build_relation_strict_gold_ambiguous.py` | Build no-API splits, audit web evidence, repair ambiguous rows, and enforce relation support. |
| Strict release | `46_final_gold_quality_audit.py` through `70_audit_ambiguous_human_weak_locale_pattern.py` | Run final audits, explicit strict balancing, manual repair manifests, final validation, and release build. |

The numbered files intentionally preserve the experiment order. Some late
repair scripts are historical manifests that expect the `runs/` workspace
created by earlier stages. They are tracked for reproducibility and
auditability; use the `--help` output on scripts that expose arguments when
rerunning in a new workspace.

## Prompts And Configuration

Generation behavior is controlled by:

- `config.py` for country lists, topics, quotas, prompt paths, and split names.
- `prompts/developer_explicit.md` for locale-explicit questions.
- `prompts/developer_ambiguous.md` for locale-ambiguous questions.
- `human_validation_guidelines.md` for the validation rubric.

The generation scripts create request files and do not submit paid API jobs
automatically.

## Large Artifacts

The repository tracks source code and the small final release artifacts. It
does not track `runs/`, raw Batch API outputs, evidence-fetch caches, web audit
fetch logs, reviewer dashboards, or intermediate CSV/JSONL pools. Those files
were several GB in the source workspace and are ignored by Git.
