#!/bin/bash -l
## SBATCH --partition=ailab
#SBATCH -J gecco
#SBATCH -N 1
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out
## SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH -c 32
#SBATCH --mail-user=akshay.jagadish@princeton.edu

cd ~/gecco-1/
module purge
module load anaconda3/2025.12 
module load proxy/default
conda activate gecco

python scripts/two_step_individual_function.py --config two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting.yaml
# python scripts/two_step_individual_function.py --config two_step_psychiatry_individual_ocd_function_gemini-3-pro_ocd.yaml
# python scripts/two_step_individual_function.py --config two_step_psychiatry_individual_function_gemini-3-pro_ocd.yaml