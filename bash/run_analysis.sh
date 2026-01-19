#!/bin/bash -l
## SBATCH --partition=ailab
#SBATCH -J gecco
#SBATCH -N 1
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out
## SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -t 12:00:00
#SBATCH -c 32
#SBATCH --mail-user=akshay.jagadish@princeton.edu

cd ~/gecco-1/
module purge
module load anaconda3/2025.12 
module load proxy/default
conda activate gecco

# ppc generation for group models
python analysis/two_step_task/ppc_group_oci.py --config two_step_psychiatry_group_ocd_maxsetting.yaml
python analysis/two_step_task/ppc_group_oci.py --config two_step_psychiatry_group_metadata_ocd_maxsetting.yaml

# ppc generation for individual models
python analysis/two_step_task/ppc_individual_oci.py --config two_step_psychiatry_individual_ocd_function_gemini-3-pro_ocd_maxsetting.yaml
python analysis/two_step_task/ppc_individual_oci.py --config two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting.yaml

# group bmc analysis for group models
python analysis/two_step_task/group_bmc_analysis_oci.py --config two_step_psychiatry_group_ocd_maxsetting.yaml
python analysis/two_step_task/group_bmc_analysis_oci.py --config two_step_psychiatry_group_metadata_ocd_maxsetting.yaml

# group bmc analysis for individual models
python analysis/two_step_task/group_bmc_analysis_oci.py --config two_step_psychiatry_individual_ocd_function_gemini-3-pro_ocd_maxsetting.yaml
python analysis/two_step_task/group_bmc_analysis_oci.py --config two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting.yaml

# plot bic comparison
python analysis/two_step_task/plot_mean_bics_between_models.py 
