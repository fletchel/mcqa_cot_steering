#!/bin/bash
#SBATCH --job-name=final_pipe
#SBATCH --output=final_pipeline/logs/%j.out
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem=64G

# Usage:
#   sbatch final_pipeline/run.sh final_pipeline/default_config.yaml runs/qwen_mixed
#   sbatch final_pipeline/run.sh configs/llama.yaml runs/llama_mixed

set -e

CONFIG="${1:-final_pipeline/default_config.yaml}"
OUTPUT_DIR="${2:-runs/default}"

export HF_HOME=/home/lfletcher/scratch
export HF_HUB_CACHE=/home/lfletcher/scratch

source /home/lfletcher/miniconda3/etc/profile.d/conda.sh
conda activate nnsight_experiments

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
mkdir -p final_pipeline/logs

echo "============================================"
echo "Final Pipeline"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "============================================"

python final_pipeline/run_full_pipeline.py \
    --config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "Done: $(date)"
