#!/bin/bash

#-- SLURM Job Directives --#
#SBATCH --nodes=1                   # Request a single node
#SBATCH --ntasks-per-node=4         # Request 4 CPU cores
#SBATCH --time=2:30:00              # Set a 2.5-hour time limit
#SBATCH --partition=a100_normal_q   # Specify the GPU partition
#SBATCH --account=tml_2026          # Your class-specific account
#SBATCH --gres=gpu:1                # Request 1 GPU

DATA=$1

if command -v module &> /dev/null; then
    module load Miniconda3
    module load CUDA/12.6.0
fi

# Hardcode fallback to evalenv
ENV_NAME="${CONDA_ENV:-evalenv}"
source activate "$ENV_NAME"

# Direct absolute binary to prevent subshell path resolution failure
PYTHON_BIN="/home/seanstafford/.conda/envs/evalenv/bin/python"

# eval.py loads prompts/data relative to the working directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

if [ -z "$DATA" ] || [ ! -f "data/$DATA/test.jsonl" ]; then
    echo "Usage: bash eval_single.sh <dataset>"
    echo "Available datasets: $(ls data | tr '\n' ' ')"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.88}
mkdir -p "$OUTPUT_DIR"

# Leaderboard setup: temperature 0.6, top-p 0.95, 8 samples, 32768 max tokens
$PYTHON_BIN eval.py \
--model_name_or_path "$MODEL" \
--data_name "$DATA" \
--prompt_type "qwen-instruct" \
--temperature 0.6 \
--top_p 0.95 \
--start_idx 0 \
--end_idx -1 \
--n_sampling 8 \
--k 1 \
--split "test" \
--max_tokens 32768 \
--seed 0 \
--surround_with_messages \
--output_dir "$OUTPUT_DIR" \
--completions_save_dir "$OUTPUT_DIR/completions" \
--gpu_memory_utilization "$GPU_MEM_UTIL" \
2>&1 | tee "$OUTPUT_DIR/log_${DATA}.txt"