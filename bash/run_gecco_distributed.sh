#!/bin/bash -l
#SBATCH -J gecco-client
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/gecco-client-%A_%a.out
#SBATCH --error=logs/gecco-client-%A_%a.err
# Add dependency on vLLM server job:
#   sbatch --dependency=afterok:<VLLM_JOB_ID> bash/run_gecco_distributed.sh

# Usage:
#   # 1. Launch vLLM server
#   VLLM_JOB=$(sbatch --parsable bash/launch_vllm_server.sh)
#
#   # 2. Launch distributed clients (waits for vLLM)
#   sbatch --dependency=afterok:$VLLM_JOB bash/run_gecco_distributed.sh [CONFIG]
#
# Environment variables:
#   CONFIG  — config YAML filename (default: two_step_factors.yaml)

CONFIG=${1:-"two_step_factors_distributed.yaml"}

# Map array task IDs to client profiles
PROFILES=("exploit" "explore" "diverse" "minimal" "hybrid")
PROFILE=${PROFILES[$SLURM_ARRAY_TASK_ID]}

# Fall back to default if no matching profile
if [ -z "$PROFILE" ]; then
    PROFILE=""
    PROFILE_ARG=""
else
    PROFILE_ARG="--client-profile $PROFILE"
fi

mkdir -p logs

echo "[GeCCo] Client $SLURM_ARRAY_TASK_ID starting (profile: ${PROFILE:-default})"
echo "[GeCCo] Config: $CONFIG"

# Source vLLM connection info
if [ -f "$HOME/.vllm_env" ]; then
    source "$HOME/.vllm_env"
    echo "[GeCCo] vLLM server: $VLLM_BASE_URL"
else
    echo "[GeCCo] WARNING: $HOME/.vllm_env not found. vLLM may not be ready."
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

# Run the distributed client
python scripts/run_gecco_distributed.py \
    --config "$CONFIG" \
    --client-id "$SLURM_ARRAY_TASK_ID" \
    $PROFILE_ARG
