#!/bin/bash -l
#SBATCH -J gecco-cmg-evaluator
#SBATCH -N 1
# --cpus-per-task is set dynamically by `gecco run distributed`
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH --output=logs/gecco-cmg-evaluator-%A_%a.out
#SBATCH --error=logs/gecco-cmg-evaluator-%A_%a.err

# CMG evaluator client for distributed GeCCo runs.
#
# Usage (via launcher; nested log directories are provided by sbatch overrides):
#   python -m gecco run distributed --config <yaml>  # with centralized_model_generation.enabled: true
#
# Manual usage (flat logs/ fallback only):
#   sbatch --array=0-1 bash/run_cmg_evaluator.sh two_step_factors_cmg.yaml "http://gpu-node:8000/v1" "my_env"

CONFIG=${1:-"two_step_factors_cmg.yaml"}
VLLM_URL_ARG=${2:-""}
CONDA_ENV=${3:-""}
RESULTS_DIR=${4:-""}

# Change to the directory where sbatch was submitted (repo root)
cd "${SLURM_SUBMIT_DIR:-.}"

mkdir -p logs

# Keep joblib/loky temp files off generic /tmp and within this run directory.
export GECCO_TMPDIR="${GECCO_TMPDIR:-${PWD}/tmp}"
export TMPDIR="${TMPDIR:-$GECCO_TMPDIR}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-${GECCO_TMPDIR}/joblib}"
export LOKY_TEMP_FOLDER="${LOKY_TEMP_FOLDER:-$JOBLIB_TEMP_FOLDER}"
mkdir -p "$JOBLIB_TEMP_FOLDER"
echo "[CMG evaluator] Temp dir: $TMPDIR"
echo "[CMG evaluator] Joblib/loky temp dir: $JOBLIB_TEMP_FOLDER"

# Resolve environment manager: conda when CONDA_ENV is set, otherwise uv
if [ -n "$CONDA_ENV" ]; then
    echo "[CMG evaluator] Activating conda env: $CONDA_ENV"
    conda activate "$CONDA_ENV"
    PYTHON_CMD="python"
    ENV_MANAGER="conda"
else
    echo "[CMG evaluator] Using uv environment"
    PYTHON_CMD="uv run python"
    ENV_MANAGER="uv"
fi

echo "[CMG evaluator] Client $SLURM_ARRAY_TASK_ID starting"
echo "[CMG evaluator] Config: $CONFIG"
if [ -n "$RESULTS_DIR" ]; then
    echo "[CMG evaluator] Results dir: $RESULTS_DIR"
fi
echo "[CMG evaluator] Env manager: $ENV_MANAGER"
if ! PYTHON_EXECUTABLE=$($PYTHON_CMD -c "import sys; print(sys.executable)"); then
    echo "[CMG evaluator] ERROR: Failed to run Python via $ENV_MANAGER"
    exit 1
fi
echo "[CMG evaluator] Python: $PYTHON_EXECUTABLE"

# Detect provider from config to skip vLLM setup for API-based providers
if ! PROVIDER=$($PYTHON_CMD -m gecco.cli.slurm_preflight --config "$CONFIG"); then
    echo "[CMG evaluator] ERROR: Failed to detect provider from config via $ENV_MANAGER"
    exit 1
fi
echo "[CMG evaluator] Provider: $PROVIDER"

VLLM_ARG=""
RESULTS_DIR_ARG=()
if [ -n "$RESULTS_DIR" ]; then
    RESULTS_DIR_ARG=(--results-dir "$RESULTS_DIR")
fi
if [ "$PROVIDER" = "vllm" ]; then
    # Resolve vLLM server URL: explicit arg > .vllm_env > environment
    if [ -n "$VLLM_URL_ARG" ]; then
        export VLLM_BASE_URL="$VLLM_URL_ARG"
        echo "[CMG evaluator] vLLM server (from arg): $VLLM_BASE_URL"
    elif [ -f "$HOME/.vllm_env" ]; then
        source "$HOME/.vllm_env"
        echo "[CMG evaluator] vLLM server (from .vllm_env): $VLLM_BASE_URL"
    elif [ -n "$VLLM_BASE_URL" ]; then
        echo "[CMG evaluator] vLLM server (from env): $VLLM_BASE_URL"
    else
        echo "[CMG evaluator] WARNING: No vLLM URL found. Set VLLM_BASE_URL, pass as 2nd arg, or create \$HOME/.vllm_env"
    fi

    # Wait for vLLM server to be ready (retry for up to 5 minutes)
    MAX_RETRIES=30
    RETRY_INTERVAL=10
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -s "${VLLM_BASE_URL}/models" > /dev/null 2>&1; then
            echo "[CMG evaluator] vLLM server is ready"
            break
        fi
        if [ $i -eq $MAX_RETRIES ]; then
            echo "[CMG evaluator] ERROR: vLLM server not reachable after ${MAX_RETRIES} retries"
            exit 1
        fi
        echo "[CMG evaluator] Waiting for vLLM server (attempt $i/$MAX_RETRIES)..."
        sleep $RETRY_INTERVAL
    done

    # Build vLLM URL argument
    if [ -n "$VLLM_BASE_URL" ]; then
        VLLM_ARG="--vllm-url $VLLM_BASE_URL"
    fi
else
    echo "[CMG evaluator] Using API provider '$PROVIDER' — skipping vLLM server check"
fi

# Run the CMG evaluator client
echo "[CMG evaluator] Starting CMG evaluator client..."
$PYTHON_CMD -m gecco internal distributed-client \
    --config "$CONFIG" \
    --client-id "$SLURM_ARRAY_TASK_ID" \
    $VLLM_ARG \
    "${RESULTS_DIR_ARG[@]}"
