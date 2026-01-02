#!/bin/bash

#SBATCH --job-name=sft-without
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=$GPU_GROUP$
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40GB
#SBATCH --time=0-23:59:58
#SBATCH --output=/scratch/$USER$/logs/culture/out/sft_without_n.%A_%a.out.txt
#SBATCH --error=/scratch/$USER$/logs/culture/err/sft_without_n.%A_%a.err.txt

python sft.py
