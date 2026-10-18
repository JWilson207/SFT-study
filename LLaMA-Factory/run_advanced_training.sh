#!/bin/bash
#SBATCH --job-name=sft-adv15k-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --partition=h200_normal_q
#SBATCH --account=tml_2026
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out

module load Miniconda3
module load CUDA/12.6.0

source /apps/common/software/Miniconda3/25.11.1-1/etc/profile.d/conda.sh
conda activate sft

cd "$SLURM_SUBMIT_DIR"

# Required for DeepSpeed + LLaMA-Factory execution
export FORCE_TORCHRUN=1
export WANDB_PROJECT=sft-reasoning

# Redirect Triton cache from NFS home to local compute node scratch
export TRITON_CACHE_DIR="/tmp/${USER}_triton_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

# Verify GPU device availability
nvidia-smi

OUTPUT_DIR="saves/qwen25_3b_instruct/advanced15k"
EXTRA_ARGS=""

if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR"/checkpoint-* 2>/dev/null)" ]; then
    LATEST_CKPT=$(ls -td "$OUTPUT_DIR"/checkpoint-* | head -1)
    echo "=== Prior checkpoint found: $LATEST_CKPT. Resuming training ==="
    EXTRA_ARGS="--resume_from_checkpoint $LATEST_CKPT"
else
    echo "=== Starting fresh training run ==="
fi

llamafactory-cli train sft_advanced15k.yaml $EXTRA_ARGS