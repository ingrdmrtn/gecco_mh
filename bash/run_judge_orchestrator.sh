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
# Usage (via launcher):
#   python scripts/launch_distributed.py --config <yaml> --launch-orchestrator
#
# Manual usage:
#   sbatch bash/run_judge_orchestrator.sh two_step_factors.yaml "http://gpu-node:8000/v1" "4" "my_env"

CONFIG=${1:-"two_step_factors.yaml"}
VLLM_URL_ARG=${2:-""}
N_CLIENTS=${3:-""}
CONDA_ENV=${4:-""}

mkdir -p logs

# Change to the directory where sbatch was submitted (repo root)
cd "${SLURM_SUBMIT_DIR:-.}"

# Keep joblib/loky temp files off generic /tmp and within this run directory.
export GECCO_TMPDIR="${GECCO_TMPDIR:-${PWD}/tmp}"
export TMPDIR="${TMPDIR:-$GECCO_TMPDIR}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-${GECCO_TMPDIR}/joblib}"
export LOKY_TEMP_FOLDER="${LOKY_TEMP_FOLDER:-$JOBLIB_TEMP_FOLDER}"
mkdir -p "$JOBLIB_TEMP_FOLDER"
echo "[Orchestrator] Temp dir: $TMPDIR"
echo "[Orchestrator] Joblib/loky temp dir: $JOBLIB_TEMP_FOLDER"

# Activate conda environment if specified
if [ -n "$CONDA_ENV" ]; then
    echo "[Orchestrator] Activating conda env: $CONDA_ENV"
    conda activate "$CONDA_ENV"
fi

echo "[Orchestrator] Config: $CONFIG"
echo "[Orchestrator] Python: $(which python)"

# Detect provider from config to skip vLLM setup for API-based providers
PROVIDER=$(python -c "
import yaml, sys
with open('config/$CONFIG' if '/' not in '$CONFIG' else '$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('llm', {}).get('provider', 'vllm'))
" 2>/dev/null || echo "vllm")
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
python scripts/run_judge_orchestrator.py \
    --config "$CONFIG" \
    $VLLM_ARG \
    $N_CLIENTS_ARG
