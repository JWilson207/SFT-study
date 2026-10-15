#!/bin/bash
# Submit from inside LLaMA-Factory/:   cd LLaMA-Factory && sbatch run_training.sh
# Fixes vs README: real time limit, enough CPUs for data preprocessing, correct env name and YAML path.

#SBATCH --job-name=sft-random15k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16          # README asked for 2; preprocessing uses 16 workers
#SBATCH --time=48:00:00             # README had 1:00:00, far too short. Check the max: sinfo -p h200_normal_q -o "%l"
#SBATCH --partition=h200_normal_q
#SBATCH --account=tml_2026          # your class account
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out

module load Miniconda3
module load CUDA/12.6.0
source activate sft                 # training env from the runbook (README said 'your_env_name')

cd "$SLURM_SUBMIT_DIR"              # LLaMA-Factory/, so 'data/' and 'examples/deepspeed/' resolve

export WANDB_PROJECT=sft-reasoning
export FORCE_TORCHRUN=1

nvidia-smi
llamafactory-cli train sft_random15k.yaml
