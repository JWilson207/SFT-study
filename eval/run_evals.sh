#!/bin/bash
# Usage (from eval/): bash run_evals.sh <model> <output name> [benchmark ...]   Same flags as eval_single.sh.
set -u
if [ $# -lt 2 ]; then echo "usage: bash run_evals.sh <model> <output name> [benchmarks...]"; exit 1; fi
MODEL=$1; NAME=$2; shift 2
BENCHES="${*:-amc cn_math_2024 kaoyan minerva gpqa olympiadbench aime math}"
PY=${PY:-$HOME/.conda/envs/evalenv/bin/python}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
export PYTHONNOUSERSITE=1
export PATH="$(dirname "$PY"):$PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$(dirname "$0")"
OUT="./outputs/$NAME"; mkdir -p "$OUT"
"$PY" -c "import sys, transformers, vllm; print('python', sys.version.split()[0], '| transformers', transformers.__version__, '| vllm', vllm.__version__)" || exit 1
for b in $BENCHES; do
  [ -f "data/$b/test.jsonl" ] || { echo "unknown benchmark '$b'"; continue; }
  grep -q "^Acc:" "$OUT/log_$b.txt" 2>/dev/null && { echo "skip $b (already scored)"; continue; }
  mkdir "$OUT/.running_$b" 2>/dev/null || { echo "skip $b (another job is running it; if not, rmdir $OUT/.running_$b)"; continue; }
  echo "=== $(date '+%H:%M') $NAME / $b ==="
  "$PY" eval.py --model_name_or_path "$MODEL" --data_name "$b" --prompt_type qwen-instruct \
    --temperature 0.6 --top_p 0.95 --start_idx 0 --end_idx -1 --n_sampling 8 --k 1 \
    --split test --max_tokens 32768 --seed 0 --surround_with_messages \
    --output_dir "$OUT" --completions_save_dir "$OUT/completions" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" 2>&1 | tee "$OUT/log_$b.txt"
  rc=${PIPESTATUS[0]}; rmdir "$OUT/.running_$b" 2>/dev/null
  [ "$rc" -eq 0 ] || { echo "eval.py failed on $b; stopping"; exit 1; }
done
