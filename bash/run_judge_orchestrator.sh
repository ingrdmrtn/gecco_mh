#!/bin/bash -l
#SBATCH -J gecco-orchestrator
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 8:00:00
#SBATCH --output=logs/gecco-orchestrator-%j.out
#SBATCH --error=logs/gecco-orchestrator-%j.err

# Centralized judge orchestrator for distributed GeCCo runs.
#
# Usage (via launcher; nested log directories are provided by sbatch overrides):
#   python -m gecco internal judge-orchestrate --config <yaml>
#
# Manual usage (flat logs/ fallback only):
#   sbatch bash/run_judge_orchestrator.sh two_step_factors.yaml "http://gpu-node:8000/v1" "4" "my_env"

CONFIG=${1:-"two_step_factors.yaml"}
VLLM_URL_ARG=${2:-""}
N_CLIENTS=${3:-""}
CONDA_ENV=${4:-""}

# Change to the directory where sbatch was submitted (repo root)
cd "${SLURM_SUBMIT_DIR:-.}"

mkdir -p logs

# Keep joblib/loky temp files off generic /tmp and within this run directory.
export GECCO_TMPDIR="${GECCO_TMPDIR:-${PWD}/tmp}"
export TMPDIR="${TMPDIR:-$GECCO_TMPDIR}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-${GECCO_TMPDIR}/joblib}"
export LOKY_TEMP_FOLDER="${LOKY_TEMP_FOLDER:-$JOBLIB_TEMP_FOLDER}"
mkdir -p "$JOBLIB_TEMP_FOLDER"
echo "[Orchestrator] Temp dir: $TMPDIR"
echo "[Orchestrator] Joblib/loky temp dir: $JOBLIB_TEMP_FOLDER"

# Resolve environment manager: conda when CONDA_ENV is set, otherwise uv
if [ -n "$CONDA_ENV" ]; then
    echo "[Orchestrator] Activating conda env: $CONDA_ENV"
    conda activate "$CONDA_ENV"
    PYTHON_CMD="python"
    ENV_MANAGER="conda"
else
    echo "[Orchestrator] Using uv environment"
    PYTHON_CMD="uv run python"
    ENV_MANAGER="uv"
fi

echo "[Orchestrator] Config: $CONFIG"
echo "[Orchestrator] Env manager: $ENV_MANAGER"
if ! PYTHON_EXECUTABLE=$($PYTHON_CMD -c "import sys; print(sys.executable)"); then
    echo "[Orchestrator] ERROR: Failed to run Python via $ENV_MANAGER"
    exit 1
fi
echo "[Orchestrator] Python: $PYTHON_EXECUTABLE"

# Detect provider from config to skip vLLM setup for API-based providers
if ! PROVIDER=$($PYTHON_CMD -m gecco.cli.slurm_preflight --config "$CONFIG"); then
    echo "[Orchestrator] ERROR: Failed to detect provider from config via $ENV_MANAGER"
    exit 1
fi
echo "[Orchestrator] Provider: $PROVIDER"

VLLM_ARG=""
if [ "$PROVIDER" = "vllm" ]; then
    # Resolve vLLM server URL: explicit arg > .vllm_env > environment
    if [ -n "$VLLM_URL_ARG" ]; then
        export VLLM_BASE_URL="$VLLM_URL_ARG"
        echo "[Orchestrator] vLLM server (from arg): $VLLM_BASE_URL"
    elif [ -f "$HOME/.vllm_env" ]; then
        source "$HOME/.vllm_env"
        echo "[Orchestrator] vLLM server (from .vllm_env): $VLLM_BASE_URL"
    elif [ -n "$VLLM_BASE_URL" ]; then
        echo "[Orchestrator] vLLM server (from env): $VLLM_BASE_URL"
    fi

    # Wait for vLLM server to be ready (retry for up to 5 minutes)
    MAX_RETRIES=30
    RETRY_INTERVAL=10
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -s "${VLLM_BASE_URL}/models" > /dev/null 2>&1; then
            echo "[Orchestrator] vLLM server is ready"
            break
        fi
        if [ $i -eq $MAX_RETRIES ]; then
            echo "[Orchestrator] ERROR: vLLM server not reachable after ${MAX_RETRIES} retries"
            exit 1
        fi
        echo "[Orchestrator] Waiting for vLLM server (attempt $i/$MAX_RETRIES)..."
        sleep $RETRY_INTERVAL
    done

    # Build vLLM URL argument
    if [ -n "$VLLM_BASE_URL" ]; then
        VLLM_ARG="--vllm-url $VLLM_BASE_URL"
    fi
else
    echo "[Orchestrator] Using API provider '$PROVIDER' — skipping vLLM server check"
fi

# Build n-clients argument
N_CLIENTS_ARG=""
if [ -n "$N_CLIENTS" ]; then
    N_CLIENTS_ARG="--n-clients $N_CLIENTS"
fi

# Run the orchestrator
echo "[Orchestrator] Starting centralized judge orchestrator..."
$PYTHON_CMD -m gecco internal judge-orchestrate \
    --config "$CONFIG" \
    $VLLM_ARG \
    $N_CLIENTS_ARG
