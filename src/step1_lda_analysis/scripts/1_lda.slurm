#!/bin/bash

#SBATCH --job-name=meta-lda
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40GB
#SBATCH --time=0-23:59:58
#SBATCH --array=1-270%10
#SBATCH --output=/path/to/logs/culture/out/mallet_lda_n.%A_%a.out.txt
#SBATCH --error=/path/to/logs/culture/err/mallet_lda_n.%A_%a.err.txt


# for some reason, java/ant only works on GPU cluster. 
module load ant
which java
export MALLET_MEMORY=8g

readarray -t ARGS < 1_lda_splits.txt
ARGUMENTS=${ARGS[$SLURM_ARRAY_TASK_ID - 1]}

echo $ARGUMENTS
python lda.py $ARGUMENTS
