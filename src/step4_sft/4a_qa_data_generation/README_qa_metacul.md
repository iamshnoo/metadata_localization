---
pretty_name: QA MetaCul
license: mit
task_categories:
- question-answering
- text-generation
language:
- en
size_categories:
- n<1K
tags:
- metadata-localization
- evaluation
- multiple-choice
- geography
---

# qa_metacul

## Summary

`qa_metacul` is an 800-question multiple-choice benchmark used to evaluate metadata-conditioned language models in the Metadata Conditioned LLMs project.

The benchmark tests whether a model can answer culturally and geographically grounded factual questions for different parts of the world, and whether metadata-aware models correctly adapt their answers when continent- or country-level context changes.

Paper: https://arxiv.org/abs/2601.15236

Project repository: https://github.com/YOUR_HF_USERNAME/metadata_localization

## How The Dataset Was Created

The dataset was assembled from JSON files in the project workspace under `qa_data/{continent}/{generator}.json`, then normalized and uploaded using [`build_hf_dataset.py`](/path/to/metacul/qa_data/build_hf_dataset.py).

Creation process:

1. For each continent, we prompted LLMs to generate fact-based, culturally relevant multiple-choice questions.
2. Each continent used fixed country quotas so the questions were distributed across specific countries within that region.
3. The generators were instructed to produce questions with exactly 4 options, 1 correct answer, and 3 distractors.
4. The resulting JSON files were normalized into a single Hugging Face dataset with consistent fields.
5. The final dataset was uploaded to this repo as the benchmark used for SFT and evaluation.

Generator sources included:

- `chatgpt`
- `gpt-5-2-pro`

The generation prompt required the questions to be:

- factual
- culturally grounded
- multiple-choice with one unambiguous correct answer
- balanced across predefined country lists within each continent

## Geographic Coverage

The benchmark contains 800 questions total, constructed as 100 questions for each continent-generator pair.

Continents:

- Africa
- America
- Asia
- Europe

Country allocation instructions used during generation:

- America: 34 USA, 33 Canada, 33 Jamaica
- Asia: 15 India, 15 Pakistan, 15 Bangladesh, 15 Sri Lanka, 15 Hong Kong, 15 Malaysia, 10 Philippines
- Africa: 20 Nigeria, 20 South Africa, 20 Kenya, 20 Ghana, 20 Tanzania
- Europe: 50 United Kingdom, 50 Ireland

Because two generator variants were used for each continent, this yields 8 source files and 800 total rows.

## Data Fields

Each row contains:

- `question`: multiple-choice question text
- `options`: list of 4 answer options
- `correct_answer`: correct option string
- `distractors`: list of the 3 incorrect options
- `country`: country targeted by the question
- `continent`: continent label
- `generated_by`: source generator label

## Intended Use

This dataset is intended for:

- evaluating metadata-conditioned language models
- comparing metadata-aware and metadata-agnostic SFT models
- testing whether model answers shift appropriately under different geographic metadata
- downstream benchmarking in the Metadata Conditioned LLMs paper

It is not intended as a general-purpose QA benchmark outside this evaluation setting.

## Relationship To Training Data

This benchmark is an evaluation artifact for models trained in the project.

The project's model training data comes from the English News on the Web (NOW) corpus:
- https://www.english-corpora.org/now/

This dataset does not contain raw NOW corpus text. It contains synthetic benchmark questions created for evaluation of the resulting models.

## Dataset Build Script

The dataset card corresponds to the build pipeline in:
- [`build_hf_dataset.py`](/path/to/metacul/qa_data/build_hf_dataset.py)

That script:

- reads continent-level JSON files
- validates required keys
- adds `continent` and `generated_by`
- writes a normalized JSONL export
- saves a Hugging Face dataset to disk
- optionally pushes the dataset to `YOUR_HF_USERNAME/qa_metacul`

## Citation

If you use this dataset, please cite the paper:

```bibtex
@misc{metadata_conditioned_llms_2026,
  title={Metadata Conditioned LLMs},
  author={A. Author and collaborators},
  year={2026},
  eprint={2601.15236},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
