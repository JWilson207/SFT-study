# SFT Data Selection for Reasoning: Qwen2.5-3B-Instruct on AceReason-1.1

Course project (Reasoning SFT). We fine-tune **Qwen2.5-3B-Instruct** on 15K examples chosen from the
[AceReason-1.1-SFT 100K subset](https://huggingface.co/datasets/redsgnaoh/acereason11_100k) and compare a
**curated selection (advanced15k)** with a **random selection (random15k)** and the base model.

The full write-up is in **Needs Input Jai**.

## Results (pass@8, course evaluation settings)

| Model | CN-Math 24 | Kaoyan | AMC 23 | Minerva | OlympiadBench | GPQA-D | **AVG (6)** | AIME 24 | MATH |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct (base) | 0.267 | 0.497 | **0.775** | **0.478** | 0.483 | **0.758** | 0.543 | 0.133 | 0.840 |
| **advanced15k (ours)** | **0.400** | **0.678** | 0.750 | 0.426 | **0.526** | 0.667 | **0.575** | **0.233** | **0.864** |

AVG (6) is the mean of the six leaderboard benchmarks available in the course repo (the leaderboard's
seventh, AIME 2025, is not in `eval/data`). Pass@1 results and the analysis are in the report.
Full table: [`eval/outputs/scores_table.txt`](eval/outputs/scores_table.txt).

## Artifacts

| | Link |
|---|---|
| Fine-tuned model (advanced15k, submitted) | `https://huggingface.co/<hf-user>/qwen25-3b-advanced15k` |
| Fine-tuned model (random15k) | `https://huggingface.co/<hf-user>/qwen25-3b-random15k` |
| Dataset: advanced15k (15,000 examples) | `https://huggingface.co/datasets/<hf-user>/acereason-advanced15k` |
| Dataset: random15k (15,000 examples) | `https://huggingface.co/datasets/<hf-user>/acereason-random15k` |

## Repository layout

| Path | What it is |
|---|---|
| `make_random_subset.py` | Random 15K baseline (seed 42); writes LLaMA-Factory JSON and registers it |
| `make_advanced_subset.py` | 4-stage curation pipeline for advanced15k (see report §3) |
| `LLaMA-Factory/sft_random15k.yaml`, `sft_advanced15k.yaml` | Training configs (identical hyperparameters) |
| `LLaMA-Factory/run_training.sh`, `run_advanced_training.sh` | SLURM training jobs (B200) |
| `LLaMA-Factory/data/dataset_info.json` | Dataset registration, with the `system` column mapped |
| `eval/` | Course evaluation code (`eval.py`, unchanged) plus our runners |
| `eval/run_evals.sh` | Runs `eval.py` on a list of benchmarks with the exact flags of `eval_single.sh` |
| `eval/run_chunk.sh` | Same, for a problem range (`--start_idx/--end_idx`); used to split OlympiadBench |
| `eval/scores.py` | Builds the score table from the result files, merging chunks |
| `eval/outputs/base/`, `eval/outputs/advanced15k/` | Logs and per-problem results (`*.jsonl`) for every benchmark |

## Reproduce

**Environments** (two conda envs, Python 3.10, to avoid version conflicts between LLaMA-Factory and vLLM):
- training `sft`: `torch==2.7.1`, LLaMA-Factory (bundled, v0.9.4.dev0), `deepspeed==0.16.8`, `transformers==4.55.0`
- evaluation `evalenv`: `vllm==0.10.0`, `transformers==4.55.0`, `datasets==3.6.0`, `huggingface_hub<1.0`,
  `word2number`, `antlr4-python3-runtime==4.9.3`, `latex2sympy2==1.9.1 --no-deps`
- B200 nodes need the CUDA 12.8 builds (`--extra-index-url https://download.pytorch.org/whl/cu128`).

**1. Data**
```bash
python make_random_subset.py          # -> LLaMA-Factory/data/random15k.json
python make_advanced_subset.py        # -> LLaMA-Factory/data/advanced15k.json
```

**2. Training** (from `LLaMA-Factory/`). The configs stop at `max_steps: 3750` (2 epochs) of a 3-epoch cosine
schedule; delete the `resume_from_checkpoint` line for a fresh run.
```bash
sbatch run_training.sh                # random15k
sbatch run_advanced_training.sh       # advanced15k
```

**3. Evaluation** (from `eval/`; settings from the course README: temperature 0.6, top-p 0.95, 8 samples,
32,768 max tokens, `qwen-instruct` prompt)
```bash
bash run_evals.sh <model path> <name> amc cn_math_2024 kaoyan minerva gpqa aime math
# OlympiadBench in 27 ranges of 25 problems (one SLURM array):
W="S=\$((SLURM_ARRAY_TASK_ID*25)); E=\$((S+25)); [ \$E -gt 675 ] && E=675; bash $PWD/run_chunk.sh <model path> <name> olympiadbench \$S \$E"
sbatch --array=0-26 --gres=gpu:1 --time=4:00:00 --wrap="$W"
python scores.py
```
