#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output %x-%j.out
#SBATCH --error %x-%j.out
#SBATCH --time=12:00:00
#SBATCH --job-name=varnet

# Export environment from this script to srun
unset SLURM_EXPORT_ENV             

# Activate environment and load modules
module load python
conda activate varnet


srun --unbuffered bash ./scripts/train_radial_varnet.sh
