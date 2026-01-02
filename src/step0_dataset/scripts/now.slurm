#!/bin/bash

#SBATCH --job-name=now-data
#SBATCH --partition=normal
#SBATCH --mem=40GB
#SBATCH --time=0-12:00:00
#SBATCH --array=1-15%15
#SBATCH --output=/scratch/$USER$/logs/culture/out/now_data_n.%A_%a.out.txt
#SBATCH --error=/scratch/$USER$/logs/culture/err/now_data_n.%A_%a.err.txt

readarray -t ARGS < now_years_splits.txt
ARGUMENTS=${ARGS[$SLURM_ARRAY_TASK_ID - 1]}

echo $ARGUMENTS
python data_now.py $ARGUMENTS
