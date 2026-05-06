#!/bin/bash -l
#SBATCH -J gecco-test-eval
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH --output=logs/gecco-test-eval-%j.out
#SBATCH --error=logs/gecco-test-eval-%j.err

# Post-processing script: runs test evaluation after all distributed clients complete.
#
# Usage (normally called by launch_distributed.py):
#   sbatch --dependency=afterok:$CLIENT_JOB_ID bash/run_test_evaluation.sh two_step_factors_distributed.yaml results/two_step_factors

CONFIG=${1:-"two_step_factors_distributed.yaml"}
RESULTS_DIR=${2:-"results/two_step_factors"}
CONDA_ENV=${3:-""}

mkdir -p logs

# Change to the directory where sbatch was submitted (repo root)
cd "${SLURM_SUBMIT_DIR:-.}"

# Keep joblib/loky temp files off generic /tmp and within this run directory.
export GECCO_TMPDIR="${GECCO_TMPDIR:-${PWD}/tmp}"
export TMPDIR="${TMPDIR:-$GECCO_TMPDIR}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-${GECCO_TMPDIR}/joblib}"
export LOKY_TEMP_FOLDER="${LOKY_TEMP_FOLDER:-$JOBLIB_TEMP_FOLDER}"
mkdir -p "$JOBLIB_TEMP_FOLDER"
echo "[test-eval] Temp dir: $TMPDIR"
echo "[test-eval] Joblib/loky temp dir: $JOBLIB_TEMP_FOLDER"

# Activate conda environment if specified
if [ -n "$CONDA_ENV" ]; then
    echo "[test-eval] Activating conda env: $CONDA_ENV"
    conda activate "$CONDA_ENV"
fi

echo "[test-eval] Starting test evaluation"
echo "[test-eval] Config: $CONFIG"
echo "[test-eval] Results dir: $RESULTS_DIR"
echo "[test-eval] Python: $(which python)"

# Run the test evaluation script
python scripts/run_test_evaluation.py \
    --config "$CONFIG" \
    --results-dir "$RESULTS_DIR" \
    --write-store

echo "[test-eval] Complete"
