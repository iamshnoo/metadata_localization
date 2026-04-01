#!/bin/bash

#SBATCH --job-name=sft-eval
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40GB
#SBATCH --time=0-23:59:58
#SBATCH --output=/scratch/amukher6/logs/culture/out/sft_eval.%A_%a.out.txt
#SBATCH --error=/scratch/amukher6/logs/culture/err/sft_eval.%A_%a.err.txt

python 15_sft_eval_grid.py
