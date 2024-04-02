#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --clusters=alex
#SBATCH --partition=a100
##SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --output %x-%j.out
#SBATCH --error %x-%j.out
#SBATCH --time=00:10:00
#SBATCH --job-name=varnet

# Export environment from this script to srun
unset SLURM_EXPORT_ENV             

# Activate environment and load modules
module load python
conda activate varnet


srun --unbuffered bash "$@"