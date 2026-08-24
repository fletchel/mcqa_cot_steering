#!/bin/bash
# Submit the final pipeline with maximum parallelism.
# Each independent stage gets its own GPU job.
#
# Usage:
#   bash final_pipeline/submit_parallel.sh final_pipeline/default_config.yaml runs/qwen_mixed
#   bash final_pipeline/submit_parallel.sh configs/llama.yaml runs/llama_mixed [--partition gpu_h100]

set -e

CONFIG="${1:?Usage: $0 <config.yaml> <output_dir> [--partition PARTITION] [--mem MEM]}"
OUTPUT_DIR="${2:?Usage: $0 <config.yaml> <output_dir>}"
PARTITION="${3:---partition gpu_a100}"
MEM="${4:---mem 64G}"

# Parse optional named args
while [[ $# -gt 2 ]]; do
    case "$3" in
        --partition) PARTITION="--partition $4"; shift 2;;
        --mem) MEM="--mem $4"; shift 2;;
        *) shift;;
    esac
done

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMON="--ntasks=1 --cpus-per-task=4 ${PARTITION} ${MEM}"
WRAPPER="source /home/lfletcher/miniconda3/etc/profile.d/conda.sh && conda activate nnsight_experiments && export HF_HOME=/home/lfletcher/scratch && export HF_HUB_CACHE=/home/lfletcher/scratch && cd ${EXP_DIR}"
PY="python final_pipeline/run_full_pipeline.py --config ${CONFIG} --output_dir ${OUTPUT_DIR}"

mkdir -p "${EXP_DIR}/final_pipeline/logs"

echo "============================================"
echo "Final Pipeline (parallel submission)"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# --- Stage 1: Filter (no dependencies) ---
JID_FILTER=$(sbatch --parsable \
    --job-name=fp_filter \
    --output="${EXP_DIR}/final_pipeline/logs/%j_filter.out" \
    --gpus=1 ${COMMON} --time=01:30:00 \
    --wrap="${WRAPPER} && ${PY} --stages filter")
echo "1. Filter: ${JID_FILTER}"

# --- Stage 2a: Direction (depends on filter) ---
JID_DIRECTION=$(sbatch --parsable \
    --dependency=afterok:${JID_FILTER} \
    --job-name=fp_direction \
    --output="${EXP_DIR}/final_pipeline/logs/%j_direction.out" \
    --gpus=1 ${COMMON} --time=02:00:00 \
    --wrap="${WRAPPER} && ${PY} --stages direction")
echo "2a. Direction: ${JID_DIRECTION} (after ${JID_FILTER})"

# --- Stage 2b: Patching (depends on filter, parallel with direction) ---
JID_PATCHING=$(sbatch --parsable \
    --dependency=afterok:${JID_FILTER} \
    --job-name=fp_patching \
    --output="${EXP_DIR}/final_pipeline/logs/%j_patching.out" \
    --gpus=1 ${COMMON} --time=02:00:00 \
    --wrap="${WRAPPER} && ${PY} --stages patching")
echo "2b. Patching: ${JID_PATCHING} (after ${JID_FILTER})"

# --- Stage 3a: Test projections + BC heatmaps (depends on direction) ---
JID_TEST_PROJ=$(sbatch --parsable \
    --dependency=afterok:${JID_DIRECTION} \
    --job-name=fp_test_proj \
    --output="${EXP_DIR}/final_pipeline/logs/%j_test_proj.out" \
    --gpus=1 ${COMMON} --time=01:00:00 \
    --wrap="${WRAPPER} && ${PY} --stages test_projections")
echo "3a. Test projections: ${JID_TEST_PROJ} (after ${JID_DIRECTION})"

# --- Stage 3b: Layer sweeps on dev (depends on direction) ---
JID_SWEEPS=$(sbatch --parsable \
    --dependency=afterok:${JID_DIRECTION} \
    --job-name=fp_sweeps \
    --output="${EXP_DIR}/final_pipeline/logs/%j_sweeps.out" \
    --gpus=1 ${COMMON} --time=02:00:00 \
    --wrap="${WRAPPER} && ${PY} --stages layer_sweeps")
echo "3b. Layer sweeps: ${JID_SWEEPS} (after ${JID_DIRECTION})"

# --- Stage 4a: Logit steering on test (depends on layer sweeps) ---
JID_LS=$(sbatch --parsable \
    --dependency=afterok:${JID_SWEEPS} \
    --job-name=fp_logit_steer \
    --output="${EXP_DIR}/final_pipeline/logs/%j_logit_steer.out" \
    --gpus=1 ${COMMON} --time=01:00:00 \
    --wrap="${WRAPPER} && ${PY} --stages logit_steering")
echo "4a. Logit steering: ${JID_LS} (after ${JID_SWEEPS})"

# --- Stage 4b: CoT steering on test (depends on layer sweeps) ---
JID_COT=$(sbatch --parsable \
    --dependency=afterok:${JID_SWEEPS} \
    --job-name=fp_cot_steer \
    --output="${EXP_DIR}/final_pipeline/logs/%j_cot_steer.out" \
    --gpus=1 ${COMMON} --time=04:00:00 \
    --wrap="${WRAPPER} && ${PY} --stages cot_steering")
echo "4b. CoT steering: ${JID_COT} (after ${JID_SWEEPS})"

echo ""
echo "============================================"
echo "Dependency graph:"
echo "  ${JID_FILTER} (filter)"
echo "    ├── ${JID_DIRECTION} (direction)"
echo "    │   ├── ${JID_TEST_PROJ} (test projections)"
echo "    │   └── ${JID_SWEEPS} (layer sweeps)"
echo "    │       ├── ${JID_LS} (logit steering)"
echo "    │       └── ${JID_COT} (CoT steering)"
echo "    └── ${JID_PATCHING} (patching)"
echo "============================================"
echo ""
echo "Monitor: sacct -j ${JID_FILTER},${JID_DIRECTION},${JID_PATCHING},${JID_TEST_PROJ},${JID_SWEEPS},${JID_LS},${JID_COT} --format=JobID,JobName%15,State,Elapsed"
