#!/bin/bash
# Evaluate one problem range of one benchmark on ARC (sbatch --wrap or interactive).
# Same flags as the course's eval_single.sh; only --start_idx/--end_idx differ. scores.py merges ranges.
# Usage (from eval/): bash run_chunk.sh <model path> <output name> <benchmark> <start> <end>
#   end = -1 means "to the last problem". A claim folder stops two jobs (any cluster) running the same range.
set -u
[ $# -eq 5 ] || { sed -n '2,6p' "$0"; exit 1; }
MODEL=$1; NAME=$2; B=$3; S=$4; E=$5
PY=${PY:-$HOME/.conda/envs/evalenv/bin/python}
export PYTHONNOUSERSITE=1 PATH="$(dirname "$PY"):$PATH" CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$(dirname "$0")"
[ -f "data/$B/test.jsonl" ] || { echo "unknown benchmark $B"; exit 1; }
OUT=./outputs/$NAME; mkdir -p "$OUT"
TAG=${B}_s${S}_e${E}; LOG=$OUT/log_$TAG.txt
grep -q "^Acc:" "$LOG" 2>/dev/null && { echo "skip $TAG (already scored)"; exit 0; }
mkdir "$OUT/.running_$TAG" 2>/dev/null || { echo "skip $TAG (another job is running it; if not, rmdir $OUT/.running_$TAG)"; exit 0; }
trap 'rmdir "$OUT/.running_$TAG" 2>/dev/null' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import sys, transformers, vllm; print('python', sys.version.split()[0], '| transformers', transformers.__version__, '| vllm', vllm.__version__)" || exit 1
echo "=== $(date '+%m-%d %H:%M') $NAME / $B problems [$S,$E) ==="
"$PY" eval.py --model_name_or_path "$MODEL" --data_name "$B" --prompt_type qwen-instruct \
  --temperature 0.6 --top_p 0.95 --start_idx "$S" --end_idx "$E" --n_sampling 8 --k 1 \
  --split test --max_tokens 32768 --seed 0 --surround_with_messages \
  --output_dir "$OUT" --completions_save_dir "$OUT/completions" \
  --gpu_memory_utilization "${GPU_MEM_UTIL:-0.90}" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
