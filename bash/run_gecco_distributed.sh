#!/bin/bash -l
#SBATCH -J gecco-client
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH --output=logs/gecco-client-%A_%a.out
#SBATCH --error=logs/gecco-client-%A_%a.err
# NOTE: --array is set dynamically by launch_distributed.py (or override with sbatch --array=...)

# Usage (preferred — reads profiles from config automatically):
#   python scripts/launch_distributed.py --config two_step_factors_distributed.yaml
#
# Manual usage:
#   sbatch --array=0-4 --dependency=afterok:$VLLM_JOB \
#       bash/run_gecco_distributed.sh two_step_factors_distributed.yaml "exploit,explore,diverse,minimal,hybrid" "http://gpu-node:8000/v1"

CONFIG=${1:-"two_step_factors_distributed.yaml"}
PROFILES_CSV=${2:-""}
VLLM_URL_ARG=${3:-""}
CONDA_ENV=${4:-""}

# Resolve profile for this array task from the comma-separated list
if [ -n "$PROFILES_CSV" ]; then
    IFS=',' read -ra PROFILES <<< "$PROFILES_CSV"
    PROFILE=${PROFILES[$SLURM_ARRAY_TASK_ID]}
else
    PROFILE=""
fi

# Build profile argument
if [ -n "$PROFILE" ]; then
    PROFILE_ARG="--client-profile $PROFILE"
else
    PROFILE_ARG=""
fi

mkdir -p logs

# Activate conda environment if specified
if [ -n "$CONDA_ENV" ]; then
    echo "[GeCCo] Activating conda env: $CONDA_ENV"
    conda activate "$CONDA_ENV"
fi

echo "[GeCCo] Client $SLURM_ARRAY_TASK_ID starting (profile: ${PROFILE:-default})"
echo "[GeCCo] Config: $CONFIG"
echo "[GeCCo] Python: $(which python)"

# Detect provider from config to skip vLLM setup for API-based providers
PROVIDER=$(python -c "
import yaml, sys
with open('config/$CONFIG' if '/' not in '$CONFIG' else '$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('llm', {}).get('provider', 'vllm'))
" 2>/dev/null || echo "vllm")
echo "[GeCCo] Provider: $PROVIDER"

VLLM_ARG=""
if [ "$PROVIDER" = "vllm" ]; then
    # Resolve vLLM server URL: explicit arg > .vllm_env > environment
    if [ -n "$VLLM_URL_ARG" ]; then
        export VLLM_BASE_URL="$VLLM_URL_ARG"
        echo "[GeCCo] vLLM server (from arg): $VLLM_BASE_URL"
    elif [ -f "$HOME/.vllm_env" ]; then
        source "$HOME/.vllm_env"
        echo "[GeCCo] vLLM server (from .vllm_env): $VLLM_BASE_URL"
    elif [ -n "$VLLM_BASE_URL" ]; then
        echo "[GeCCo] vLLM server (from env): $VLLM_BASE_URL"
    else
        echo "[GeCCo] WARNING: No vLLM URL found. Set VLLM_BASE_URL, pass as 3rd arg, or create \$HOME/.vllm_env"
    fi

    # Wait for vLLM server to be ready (retry for up to 5 minutes)
    MAX_RETRIES=30
    RETRY_INTERVAL=10
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -s "${VLLM_BASE_URL}/models" > /dev/null 2>&1; then
            echo "[GeCCo] vLLM server is ready"
            break
        fi
        if [ $i -eq $MAX_RETRIES ]; then
            echo "[GeCCo] ERROR: vLLM server not reachable after ${MAX_RETRIES} retries"
            exit 1
        fi
        echo "[GeCCo] Waiting for vLLM server (attempt $i/$MAX_RETRIES)..."
        sleep $RETRY_INTERVAL
    done

    # Build vLLM URL argument
    if [ -n "$VLLM_BASE_URL" ]; then
        VLLM_ARG="--vllm-url $VLLM_BASE_URL"
    fi
else
    echo "[GeCCo] Using API provider '$PROVIDER' — skipping vLLM server check"
fi

# Run the distributed client
python scripts/run_gecco_distributed.py \
    --config "$CONFIG" \
    --client-id "$SLURM_ARRAY_TASK_ID" \
    $PROFILE_ARG \
    $VLLM_ARG
