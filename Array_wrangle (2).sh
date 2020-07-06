#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=standard
#SBATCH --account=behnaz_lab
#SBATCH --time=1-20:00:00


module load anaconda/2019.10-py3.7

python array_wrangle.py $SLURM_ARRAY_TASK_ID

