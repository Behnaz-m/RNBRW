#!/bin/bash 

#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=standard
#SBATCH --account=behnaz_lab
#SBATCH --time=2-20:00:00


module load anaconda/2019.10-py3.7

python walkhole.py $SLURM_ARRAY_TASK_ID
