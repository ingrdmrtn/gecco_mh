#!/bin/bash -l
## SBATCH --partition=ailab
#SBATCH -J gecco_study2
#SBATCH -N 1
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out
## SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -t 12:00:00
#SBATCH -c 32

cd ~/gecco-1/
module purge
module load anaconda3/2025.12
module load proxy/default
conda activate gecco

# Study 2 group run (no transdiagnostic factors)
python scripts/two_step_psychiatry_group.py --config two_step_study2_group.yaml
