#!/bin/bash
#SBATCH --job-name=final_pipe_2gpu
#SBATCH --output=final_pipeline/logs/%j.out
#SBATCH --partition=gpu_a100
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=128G

# 2-GPU variant of run.sh for models that need device_map sharding (QwQ-32B, Llama-70B).
# Usage:
#   sbatch final_pipeline/run_2gpu.sh final_pipeline/config_qwq_32b_pilot.yaml runs/qwq_32b_pilot
#   sbatch final_pipeline/run_2gpu.sh <config.yaml> <output_dir> [--stages ...]

set -e

CONFIG="${1:?Usage: sbatch run_2gpu.sh <config.yaml> <output_dir> [--stages ...]}"
OUTPUT_DIR="${2:?Usage: sbatch run_2gpu.sh <config.yaml> <output_dir> [--stages ...]}"
shift 2
EXTRA_ARGS="$@"

export HF_HOME=/scratch-shared/lfletcher/huggingface
export HF_HUB_CACHE=/scratch-shared/lfletcher/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /home/lfletcher/miniconda3/etc/profile.d/conda.sh
conda activate nnsight_experiments

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
mkdir -p final_pipeline/logs

echo "============================================"
echo "Final Pipeline (2 GPU)"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "============================================"

python final_pipeline/run_full_pipeline.py \
    --config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}

echo ""
echo "Done: $(date)"
