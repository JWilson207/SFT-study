#!/bin/bash
#SBATCH --job-name=sft-rand-b200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --partition=b200_normal_q
#SBATCH --account=tml_2026
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out

module purge
module load Miniconda3
module load CUDA/12.6.0

export CUDA_HOME="${CUDA_HOME:-$EBROOTCUDA}"
export PATH="$CUDA_HOME/bin:$PATH"

ENV_PATH="/home/seanstafford/.conda/envs/sft_b200"
export PATH="$ENV_PATH/bin:$PATH"
export CONDA_PREFIX="$ENV_PATH"

source /apps/common/software/Miniconda3/25.11.1-1/etc/profile.d/conda.sh
conda activate sft_b200

cd "$SLURM_SUBMIT_DIR"

export DS_BUILD_OPS=0
export FORCE_TORCHRUN=1
export MASTER_PORT=29505
export WANDB_PROJECT=sft-reasoning

export TRITON_CACHE_DIR="/tmp/${USER}_triton_rand_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

"$ENV_PATH/bin/llamafactory-cli" train sft_random15k.yaml