# QA Data

This directory contains the tracked QA benchmark artifacts for the metadata
localization repository. New work should start with LocalNewsQA Core; the older
`qa_metacul` files remain available for compatibility with earlier runs.

## Contents

| Path | Purpose |
| --- | --- |
| `localnewsqa_core/` | LocalNewsQA Core generation, human validation, automated audit, repair, and final release pipeline. |
| `localnewsqa_core/final_gold_20260516/` | Final LocalNewsQA gold JSONL release and summary. |
| `hf_dataset_localnewsqa_gold_20260516/` | Hugging Face-style parquet export of the final LocalNewsQA gold release. |
| `README_qa_metacul.md` | Dataset-card style documentation for the legacy `qa_metacul` benchmark. |
| `africa/`, `america/`, `asia/`, `europe/` | Continent-level source JSON files. |
| `hf_dataset.jsonl` | Normalized JSONL export built from the source files. |
| `hf_dataset/` | Saved Hugging Face dataset directory. |
| `build_hf_dataset.py` | Builder that validates, normalizes, and exports the dataset. |
| `qa_gen.py` | Generation helper used to create benchmark questions. |
| `developer.md`, `user.md` | Prompt templates used by the generation helper. |

## Build

List LocalNewsQA Core pipeline entrypoints:

```bash
python tools/repo.py localnewsqa-pipeline
```

Load the final LocalNewsQA gold release:

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="qa_data/localnewsqa_core/final_gold_20260516/localnewsqa_gold_explicit17000_ambiguous1700.jsonl",
    split="train",
)
```

Build or inspect the legacy `qa_metacul` export:

```bash
python qa_data/build_hf_dataset.py --help
```

The same files are mirrored under
`src/step4_sft/4a_qa_data_generation/` so the step-structured pipeline has a
self-contained QA generation stage.

## Notes

The LocalNewsQA Core final release is tracked in this repository. Multi-GB
intermediate generation, validation, and web-audit workspaces live under
`qa_data/localnewsqa_core/runs/` in the source workspace and are ignored here.
