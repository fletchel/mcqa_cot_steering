#!/bin/bash
#SBATCH --job-name=cot_sample
#SBATCH --output=final_pipeline/logs/%j_cot_sample.out
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=64G

# Usage: sbatch final_pipeline/run_cot_sample.sh <cot_sample.py args...>
# e.g.:  sbatch final_pipeline/run_cot_sample.sh --data_run runs/qwen_big \
#            --source_dataset hellaswag --num_datapoints 200 --run_baseline

set -e

export HF_HOME=/home/lfletcher/scratch
export HF_HUB_CACHE=/home/lfletcher/scratch

source /home/lfletcher/miniconda3/etc/profile.d/conda.sh
conda activate nnsight_experiments

# sbatch runs a spooled copy of this script, so BASH_SOURCE cannot locate the
# repo under SLURM; use the submission directory (submit from the repo root).
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"
mkdir -p final_pipeline/logs

python final_pipeline/cot_sample.py "$@"

echo ""
echo "Done: $(date)"
