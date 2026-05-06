#!/bin/bash -l
#SBATCH -J gecco-cmg-generator
#SBATCH -N 1
# --cpus-per-task is set dynamically by launch_cmg_distributed.py
#SBATCH -t 8:00:00
#SBATCH --output=logs/gecco-cmg-generator-%j.out
#SBATCH --error=logs/gecco-cmg-generator-%j.err

# CMG generator client for distributed GeCCo runs.
#
# Usage (via launcher):
#   python scripts/launch_cmg_distributed.py --config <yaml>
#
# Manual usage:
#   sbatch bash/run_cmg_generator.sh two_step_factors_cmg.yaml "generator" "http://gpu-node:8000/v1" "my_env"

CONFIG=${1:-"two_step_factors_cmg.yaml"}
PROFILE=${2:-"generator"}
VLLM_URL_ARG=${3:-""}
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
echo "[Generator] Temp dir: $TMPDIR"
echo "[Generator] Joblib/loky temp dir: $JOBLIB_TEMP_FOLDER"

# Activate conda environment if specified
if [ -n "$CONDA_ENV" ]; then
    echo "[Generator] Activating conda env: $CONDA_ENV"
    conda activate "$CONDA_ENV"
fi

echo "[Generator] CMG generator starting (profile: $PROFILE)"
echo "[Generator] Config: $CONFIG"
echo "[Generator] Python: $(which python)"

# Detect provider from config to skip vLLM setup for API-based providers
PROVIDER=$(python -c "
import yaml, sys
with open('config/$CONFIG' if '/' not in '$CONFIG' else '$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('llm', {}).get('provider', 'vllm'))
" 2>/dev/null || echo "vllm")
echo "[Generator] Provider: $PROVIDER"

VLLM_ARG=""
if [ "$PROVIDER" = "vllm" ]; then
    # Resolve vLLM server URL: explicit arg > .vllm_env > environment
    if [ -n "$VLLM_URL_ARG" ]; then
        export VLLM_BASE_URL="$VLLM_URL_ARG"
        echo "[Generator] vLLM server (from arg): $VLLM_BASE_URL"
    elif [ -f "$HOME/.vllm_env" ]; then
        source "$HOME/.vllm_env"
        echo "[Generator] vLLM server (from .vllm_env): $VLLM_BASE_URL"
    elif [ -n "$VLLM_BASE_URL" ]; then
        echo "[Generator] vLLM server (from env): $VLLM_BASE_URL"
    else
        echo "[Generator] WARNING: No vLLM URL found. Set VLLM_BASE_URL, pass as 3rd arg, or create \$HOME/.vllm_env"
    fi

    # Wait for vLLM server to be ready (retry for up to 5 minutes)
    MAX_RETRIES=30
    RETRY_INTERVAL=10
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -s "${VLLM_BASE_URL}/models" > /dev/null 2>&1; then
            echo "[Generator] vLLM server is ready"
            break
        fi
        if [ $i -eq $MAX_RETRIES ]; then
            echo "[Generator] ERROR: vLLM server not reachable after ${MAX_RETRIES} retries"
            exit 1
        fi
        echo "[Generator] Waiting for vLLM server (attempt $i/$MAX_RETRIES)..."
        sleep $RETRY_INTERVAL
    done

    # Build vLLM URL argument
    if [ -n "$VLLM_BASE_URL" ]; then
        VLLM_ARG="--vllm-url $VLLM_BASE_URL"
    fi
else
    echo "[Generator] Using API provider '$PROVIDER' — skipping vLLM server check"
fi

# Run the CMG generator client
echo "[Generator] Starting CMG generator client..."
python scripts/run_gecco_distributed.py \
    --config "$CONFIG" \
    --client-profile "$PROFILE" \
    $VLLM_ARG
