#!/bin/bash
#SBATCH --job-name=eval-qwen3b-remaining
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=06:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --account=tml_2026
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out

module load Miniconda3
module load CUDA/12.6.0
source activate evalenv

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

export CONDA_ENV="evalenv"
export GPU_MEM_UTIL="0.88"

# Only run the three remaining benchmarks
for DATA in math minerva olympiadbench; do
    echo "=== Starting evaluation on ${DATA} at $(date) ==="
    bash eval_single.sh "$DATA"
done
