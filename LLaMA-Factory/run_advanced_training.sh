#!/bin/bash
#SBATCH --job-name=sft-adv15k-b200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:30:00
#SBATCH --partition=b200_normal_q
#SBATCH --account=tml_2026
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out

module purge
module load Miniconda3
module load CUDA/12.6.0

# Export CUDA_HOME so DeepSpeed's op builder finds nvcc
export CUDA_HOME="${CUDA_HOME:-$EBROOTCUDA}"
export PATH="$CUDA_HOME/bin:$PATH"

# Force environment path directly to sft_b200
ENV_PATH="/home/seanstafford/.conda/envs/sft_b200"
export PATH="$ENV_PATH/bin:$PATH"
export CONDA_PREFIX="$ENV_PATH"

source /apps/common/software/Miniconda3/25.11.1-1/etc/profile.d/conda.sh
conda activate sft_b200

cd "$SLURM_SUBMIT_DIR"

# Prevent DeepSpeed from trying to build optional C++ ops at runtime
export DS_BUILD_OPS=0
export FORCE_TORCHRUN=1
export WANDB_PROJECT=sft-reasoning

# Redirect Triton cache to compute node local scratch
export TRITON_CACHE_DIR="/tmp/${USER}_triton_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

echo "Using Python: $(which python)"
echo "CUDA_HOME is: $CUDA_HOME"
nvidia-smi

"$ENV_PATH/bin/llamafactory-cli" train sft_advanced15k.yaml


# #!/bin/bash
# #SBATCH --job-name=sft-adv15k-b200
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --time=6:30:00
# #SBATCH --partition=b200_normal_q
# #SBATCH --account=tml_2026
# #SBATCH --gres=gpu:1
# #SBATCH --output=%x-%j.out

# module load Miniconda3
# module load CUDA/12.6.0

# source /apps/common/software/Miniconda3/25.11.1-1/etc/profile.d/conda.sh
# conda activate sft_b200

# cd "$SLURM_SUBMIT_DIR"

# # Required for DeepSpeed + LLaMA-Factory execution
# export FORCE_TORCHRUN=1
# export WANDB_PROJECT=sft-reasoning

# # Redirect Triton cache from NFS home to local compute node scratch
# export TRITON_CACHE_DIR="/tmp/${USER}_triton_${SLURM_JOB_ID}"
# mkdir -p "$TRITON_CACHE_DIR"

# # Verify GPU device availability
# nvidia-smi

# OUTPUT_DIR="saves/qwen25_3b_instruct/advanced15k"
# EXTRA_ARGS=""

# llamafactory-cli train sft_advanced15k.yaml