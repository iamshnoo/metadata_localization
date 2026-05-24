# QA Data

This directory contains the tracked QA benchmark artifacts for the metadata
localization repository.

## Contents

| Path | Purpose |
| --- | --- |
| `README_qa_metacul.md` | Dataset-card style documentation for the legacy `qa_metacul` benchmark. |
| `africa/`, `america/`, `asia/`, `europe/` | Continent-level source JSON files. |
| `hf_dataset.jsonl` | Normalized JSONL export built from the source files. |
| `hf_dataset/` | Saved Hugging Face dataset directory. |
| `build_hf_dataset.py` | Builder that validates, normalizes, and exports the dataset. |
| `qa_gen.py` | Generation helper used to create benchmark questions. |
| `developer.md`, `user.md` | Prompt templates used by the generation helper. |

## Build

```bash
python qa_data/build_hf_dataset.py --help
```

The same files are mirrored under
`src/step4_sft/4a_qa_data_generation/` so the step-structured pipeline has a
self-contained QA generation stage.

## Notes

`qa_metacul` is preserved for compatibility with earlier downstream evaluation
runs. Newer LocalNewsQA and gold-set summaries live under `results/`; their
large private generation workspaces are not tracked in this repository.
