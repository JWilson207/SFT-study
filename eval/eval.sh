#!/bin/bash
#SBATCH --job-name=eval-qwen3b-base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:30:00
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

for DATA in $(ls data); do
    echo "=== Starting evaluation on ${DATA} ==="
    bash eval_single.sh "$DATA"
done




# #!/bin/bash


# #-- SLURM Job Directives --#
# #SBATCH --nodes=1                   # Request a single node
# #SBATCH --ntasks-per-node=2         # Request 2 CPU cores
# #SBATCH --time=2:30:00             # Set a 12-hour time limit
# #SBATCH --partition=h200_normal_q   # Specify the GPU partition: h200_normal_q, a100_normal_q on Tinkercliffs | a30_normal_q on Falcon
# #SBATCH --account=tml_2026          # Your class-specific account
# #SBATCH --gres=gpu:1                # Request 1 GPU

# # Evaluate a model on all datasets under ./data (one after another).
# # Usage: sbatch eval.sh
# #        MODEL=/path/to/your/model sbatch eval.sh
# # To evaluate a single dataset, use eval_single.sh instead.

# # eval_single.sh loads prompts/data relative to the working directory
# cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# for DATA in $(ls data); do
#     bash eval_single.sh "$DATA"
# done
