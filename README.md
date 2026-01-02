Anonymous ARR submission

# Environment Setup

## Create venv and install dependencies

```bash
echo "export HF_HOME=/scratch/$USER$/cache/hf_cache" >> ~/.bashrc
cat ~/.bashrc

ml load gnu12/12.3.0
ml load python/3.11.7-cx
ml load cuda/12.4.0
ml load git
python -m venv ~/nanotron-env
source ~/nanotron-env/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
cd /scratch/$USER$/pretrain
mkdir models
mkdir datasets
mkdir logs
mkdir logs/data_processing
mkdir logs/slurm_logs
mkdir logs/checkpoints
mkdir logs/configs
mkdir logs/slurm_scripts
mkdir src
git clone https://github.com/huggingface/nanotron.git
cd nanotron
pip install -e .
pip install datasets==3.6.0 transformers numba wandb ninja triton datatrove==0.3.0
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
rm flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install psutil
pip install pybind11
hf auth login
wandb login
pip install trl bitsandbytes peft liger-kernel rich
echo "export VIRTUAL_ENV=/home/$USER$/nanotron-env" >> ~/.bashrc
echo "export VIRTUAL_ENV" >> ~/.bashrc
echo "export PATH=$HOME/.local/bin:$PATH" >> ~/.bashrc

git clone https://github.com/mimno/Mallet.git
cd Mallet
ml load ant
ant
ant jar
cd ..
pip install little_mallet_wrapper
```

# Dataset (Step 1)

```0_data_now.py``` is used to unzip purchased data from
https://www.english-corpora.org/now/ and convert it into text files.

We also did some initial LDA analysis of the data (```1_lda.py```)

## Data Preprocessing (Step 2)

Step 2 contains all the files relevant to extracting metadata and correctly
formatting everything into a huggingface compatible dataset for each experiment
used in the paper. Note that we initially experimented with many different
variations of metadata and dataset splits, not all of which are used in the
final paper. (e.g., year metadata, time based splits, LDA topics as part of
metadata, etc)

Since the data is purchased from a third party, we are forbidden by license
agreements from sharing the processed datasets. However, all code used for
processing is included here for reproducibility. Running the code would require
purchasing the data from https://www.english-corpora.org/now/ first. Similar data can be obtained from other sources like CommonCrawl for example.

# Training (Step 3, Step 4)

All trained models and intermediate checkpoints will be made publicly available on
Huggingface after anonymity period is over. All code for reproducing training
setup is included here, including model parameterization, training
hyperparameters, etc.

## Step a: Data tokenization using datatrove

Example file is given in ```data_slurm.h```
move the file to pretrain/nanotron/data_slurm.h
modify dataset paths as needed for different datasets

## Step b: Launch training job

Commands used for all training jobs are given in the ```continents``` folder for
reproducibility. This step uses the ```slurm_launcher.py``` file in nanotron to
submit jobs to SLURM.

## Step c: Convert weights from nanotron format to hf format

This step uses the files, ```__init__.py```, ```convert_to_hf.py``` and
```convert_weights.py```. Example command for converting weights is given in
```converter.md```.

## Step d: Perplexity evaluation for pretrained LMs

```11_eval_list.py``` creates all the combinations of models and test sets that
are then evaluated using ```12_eval_perplexity.py```.
Results of all evaluations are available in the ``` results``` folder in a
single CSV file ```perplexity_eval.csv```.

## Step e: Supervised Fine-tuning

```15_sft.py``` is the relevant file for performing sft using LoRA. Parameters
used are adapted from https://huggingface.co/docs/trl/lora_without_regret. To
create chat based models from base models, we use the same chat template as used
by Llama-3.2-1b-Instruct model, the jinja file for which is included as
```chat_template.jinja```.

```16_merge_lora.py``` is used to merge the LoRA weights with the base model.

## Step f: Downstream eval of SFT models

```13_qa_gen.py``` is an example API call used for creating our downstream
dataset. Complete prompts for user and assistant are available in the included
```scripts``` folder.

```14_build_hf_dataset.py``` is used to combine all of the created task data
into a single HF dataset.

```17_sft_eval.py``` is used to evaluate the SFT models on the downstream tasks.
```18_sft_eval_grid.py``` is used to create a grid of different URLs and models
so that the evals can be run in parallel using SLURM.
```19_sft_analyse.py``` combines all of the cleaned eval results into a single
CSV file in the results folder called ```qa_metacul_eval.csv```.

# Plots (Step 5)

```20_perplexity_plot.py``` contains all relevant plotting code for all figures
in the paper.
