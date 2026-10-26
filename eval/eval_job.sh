#!/bin/bash
#SBATCH --account=tml_2026
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out
# Usage (from eval/): sbatch --job-name=<name> eval_job.sh <model> <output name> <benchmark ...>
# Same eval.py, flags and settings as eval_single.sh (README section 2); whole GPU for one benchmark.
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
GPU_MEM_UTIL=0.90 bash run_evals.sh "$@"
