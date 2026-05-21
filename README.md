# MeCo: Metacultural Training Pipeline

This repository contains the complete pipeline for training language models on metacultural data with various splits and configurations.

## 📁 Project Structure

```
metacul/
├── src/                          # Source code
├── train_configs/                # Generated training configurations
├── data/                         # Raw data (ignored)
├── training_data/                # Processed datasets (ignored)
├── logs/                         # Training logs and checkpoints (ignored)
├── quick_llama/                  # Quick LLaMA library (ignored)
├── MeCo/                         # MeCo dataset repository (ignored)
└── scripts/                      # SLURM job scripts
```

## 🚀 Execution Order

### Phase 1: Data Preparation

#### 1. **Raw Data Processing** (`0_data_now.py`)
- **Purpose**: Download and preprocess raw NOW corpus data
- **SLURM**: `scripts/now_data.slurm`
- **Output**: Cleaned text data in `data/`

#### 2. **LDA Topic Modeling** (`1_lda.py`)
- **Purpose**: Generate topic models for document clustering
- **SLURM**: `scripts/lda.slurm`
- **Input**: Raw text data
- **Output**: LDA models and topic assignments

#### 3. **Document Metadata Generation** (`2_generate_document_metadata.py`)
- **Purpose**: Extract metadata (dates, sources, topics) for each document
- **SLURM**: `scripts/generate_document_metadata.slurm`
- **Input**: Raw data + LDA topics
- **Output**: Document metadata in `document_metadata/`

#### 4. **Theme Identification** (`3_identify_dominant_themes.py`)
- **Purpose**: Identify dominant themes across different time periods and regions
- **Input**: Document metadata
- **Output**: Theme classifications in `themes/`

#### 5. **Metadata Enhancement** (`4_update_metadata_with_themes.py`)
- **Purpose**: Enhance document metadata with identified themes
- **Input**: Document metadata + themes
- **Output**: Enhanced metadata

#### 6. **Meta Index Creation** (`5_create_meta_index.py`)
- **Purpose**: Create searchable index of all documents with metadata
- **Input**: Enhanced metadata
- **Output**: Meta index for efficient querying

### Phase 2: Dataset Creation

#### 7. **Data Splitting** (`6_data_splitter_ids_only.py`)
- **Purpose**: Split data into different configurations (continents, concepts, etc.)
- **Input**: Meta index
- **Output**: Split configurations

#### 8. **HuggingFace Dataset Creation** (`7_create_hf_datasets_streaming.py`)
- **Purpose**: Convert splits into HuggingFace dataset format
- **SLURM**: `scripts/create_hf_datasets.slurm`
- **Input**: Split configurations
- **Output**: HF datasets in `training_data/hf_datasets/`

#### 9. **MECO Dataset Creation** (`8_create_meco_datasets.py`)
- **Purpose**: Create final training datasets with/without metadata
- **SLURM**: `scripts/meco_data.slurm`
- **Input**: HF datasets
- **Output**: MECO datasets in `training_data/meco_datasets/`

### Phase 3: Training

#### 10. **Training Configuration Generation** (`generate_train_configs.py`)
- **Purpose**: Generate all training configuration files
- **Input**: `scripts/meco_splits.txt`
- **Output**: Training configs in `train_configs/`

#### 11. **Model Training** (`9_train_meco.py`)
- **Purpose**: Train language models on MECO datasets
- **SLURM**: `scripts/train_meco.slurm`
- **Input**: MECO datasets + training configs
- **Output**: Trained models in `logs/`

## 🎯 Training Modes

The pipeline supports 4 training modes for each data split:

1. **Pretraining**: Train from scratch on MECO data
2. **Continued Pretraining**: Continue training from HuggingFace LLaMA model
3. **Pretrain + Instruct**: Pretraining → Instruction tuning
4. **Continued Pretrain + Instruct**: Continued pretraining → Instruction tuning

## 📊 Data Splits

Available in `scripts/meco_splits.txt`:

- **Continents**: africa, asia, europe, america
- **Novel Concepts**: pivot_2012, pivot_2015, pivot_2018, pivot_2021
- **Concept Change**: development_&_society, global_politics, identity_&_gender, innovation_&_markets, urban_governance

Each split has **with_metadata** and **without_metadata** variants.

## 🚀 Quick Start

### 1. Generate Training Configs
```bash
cd src
python generate_train_configs.py
```

### 2. Run Training
```bash
# Single GPU
sbatch scripts/train_meco.slurm train_configs/continents/africa/with_metadata/pretraining.yaml

# Multi-GPU (4 GPUs)
sbatch scripts/train_meco.slurm train_configs/continents/africa/with_metadata/pretraining.yaml 4
```

### 3. Monitor Training
```bash
# Check logs
tail -f /scratch/amukher6/logs/culture/out/meco_train_*.out.txt

# Check completion log
tail -f /scratch/amukher6/metacul/logs/meco_training_completion.log
```

## 📋 Configuration Files

Training configurations are automatically generated in:
```
train_configs/
├── continents/africa/with_metadata/
│   ├── pretraining.yaml
│   ├── continued_pretraining.yaml
│   ├── pretrain_instruct.yaml
│   └── continued_pretrain_instruct.yaml
└── ... (all other splits)
```

Each config file is self-sufficient and contains all parameters needed for training.

## 🔧 Key Features

- **Scalable**: Supports 1-8 GPU training
- **Reproducible**: All configs version controlled
- **Comprehensive**: Covers all training paradigms
- **Efficient**: Uses Quick LLaMA for fast training
- **Monitored**: WandB integration for experiment tracking

## 📝 Notes

- All data directories are git-ignored to keep repository clean
- SLURM scripts are configured for the cluster environment
- Training checkpoints are saved every 1000 steps
- Logs are automatically organized by experiment type
