#!/bin/bash -l
#SBATCH -J gecco-pipeline-allocation
#SBATCH -N 1
# --cpus-per-task is set dynamically by `gecco run distributed-batch --pipeline-allocation`
#SBATCH -t 8:00:00
#SBATCH --output=logs/gecco-pipeline-allocation-%j.out
#SBATCH --error=logs/gecco-pipeline-allocation-%j.err
# NOTE: launcher overrides SBATCH log paths with config-mirrored nested directories.

# Per-pipeline allocation wrapper.
# Submitted by `gecco run distributed-batch --pipeline-allocation` to run
# one full config/replicate pipeline inside a single SLURM job allocation.
#
# Usage (via launcher; nested log directories are provided by sbatch overrides):
#   sbatch bash/run_pipeline_allocation.sh <config> <profiles_csv> <vllm_url> <conda_env> <results_dir>

CONFIG=${1:-""}
PROFILES_CSV=${2:-""}
VLLM_URL_ARG=${3:-""}
CONDA_ENV=${4:-""}
RESULTS_DIR=${5:-""}

# Change to the directory where sbatch was submitted (repo root)
cd "${SLURM_SUBMIT_DIR:-.}"

mkdir -p logs

# Keep joblib/loky temp files off generic /tmp and within this run directory.
export GECCO_TMPDIR="${GECCO_TMPDIR:-${PWD}/tmp}"
export TMPDIR="${TMPDIR:-$GECCO_TMPDIR}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-${GECCO_TMPDIR}/joblib}"
export LOKY_TEMP_FOLDER="${LOKY_TEMP_FOLDER:-$JOBLIB_TEMP_FOLDER}"
mkdir -p "$JOBLIB_TEMP_FOLDER"
echo "[pipeline-allocation] Temp dir: $TMPDIR"
echo "[pipeline-allocation] Joblib/loky temp dir: $JOBLIB_TEMP_FOLDER"

# Resolve environment manager: conda when CONDA_ENV is set, otherwise uv
if [ -n "$CONDA_ENV" ]; then
    echo "[pipeline-allocation] Activating conda env: $CONDA_ENV"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    PYTHON_CMD="python"
    ENV_MANAGER="conda"
else
    echo "[pipeline-allocation] Using uv environment"
    PYTHON_CMD="uv run python"
    ENV_MANAGER="uv"
fi

echo "[pipeline-allocation] Config: $CONFIG"
echo "[pipeline-allocation] Profiles: $PROFILES_CSV"
echo "[pipeline-allocation] Results dir: $RESULTS_DIR"
echo "[pipeline-allocation] Env manager: $ENV_MANAGER"
if ! PYTHON_EXECUTABLE=$($PYTHON_CMD -c "import sys; print(sys.executable)"); then
    echo "[pipeline-allocation] ERROR: Failed to run Python via $ENV_MANAGER"
    exit 1
fi
echo "[pipeline-allocation] Python: $PYTHON_EXECUTABLE"

# Build results-dir argument
RESULTS_DIR_ARG=()
if [ -n "$RESULTS_DIR" ]; then
    RESULTS_DIR_ARG=(--results-dir "$RESULTS_DIR")
fi

# Build vLLM URL argument
VLLM_ARG=()
if [ -n "$VLLM_URL_ARG" ]; then
    VLLM_ARG=(--vllm-url "$VLLM_URL_ARG")
fi

# Build profiles-csv argument
PROFILES_ARG=()
if [ -n "$PROFILES_CSV" ]; then
    PROFILES_ARG=(--profiles-csv "$PROFILES_CSV")
fi

# Run the internal pipeline allocation runner
$PYTHON_CMD -m gecco internal pipeline-allocation \
    --config "$CONFIG" \
    "${PROFILES_ARG[@]}" \
    "${VLLM_ARG[@]}" \
    "${RESULTS_DIR_ARG[@]}"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "[pipeline-allocation] ERROR: Pipeline allocation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
echo "[pipeline-allocation] Pipeline allocation complete"
