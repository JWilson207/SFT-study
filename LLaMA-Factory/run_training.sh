#!/bin/bash
#SBATCH --job-name=sft-rand-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --partition=h200_normal_q
#SBATCH --account=tml_2026
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out

module load Miniconda3 CUDA/12.6.0

source /apps/common/software/Miniconda3/25.11.1-1/etc/profile.d/conda.sh
conda activate sft

cd "$SLURM_SUBMIT_DIR"

export FORCE_TORCHRUN=1
export WANDB_PROJECT=sft-reasoning
export TRITON_CACHE_DIR="/tmp/${USER}_triton_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

nvidia-smi

llamafactory-cli train sft_random15k.yaml