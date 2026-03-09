#!/bin/bash -l
#SBATCH -J vllm-server
#SBATCH -N 1
#SBATCH --gres=gpu
#SBATCH --mem=100G
#SBATCH -t 4:00:00
#SBATCH --output=logs/vllm-server-%j.out
#SBATCH --error=logs/vllm-server-%j.err
#SBATCH --constraints="a100|h200|b200"

# Usage: sbatch bash/launch_vllm_server.sh [MODEL] [PORT] [TP_SIZE]
#
# This launches a vLLM server on a GPU node. Once running, the GeCCo script
# can connect to it by setting VLLM_BASE_URL.
#
# The server writes its connection info to $HOME/.vllm_env. Source this file
# in your GeCCo job script:
#
#   source $HOME/.vllm_env
#   python scripts/two_step_psychiatry_group.py --config two_step_vllm_example.yaml

MODEL=${1:-"Qwen/Qwen2.5-14B-Instruct"}
PORT=${2:-8000}
TP_SIZE=${3:-1}

export HF_HOME=/scratch/prj/bcn_neudec/huggingface

mkdir -p logs

# Write connection info for the GeCCo job
HOSTNAME=$(hostname)
echo "export VLLM_BASE_URL=http://${HOSTNAME}:${PORT}/v1" > "$HOME/.vllm_env"
echo "[vLLM] Server will be available at: http://${HOSTNAME}:${PORT}/v1"
echo "[vLLM] Connection info written to: \$HOME/.vllm_env"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --trust-remote-code
