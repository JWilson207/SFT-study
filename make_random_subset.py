#!/usr/bin/env python
"""Build the random 15K SFT baseline (README sections 1.5 + 3) for LLaMA-Factory.

What it does
  1. Downloads redsgnaoh/acereason11_100k (data/acereason11_100k.json) unless --input is given.
  2. Samples N rows uniformly at random with a fixed seed (reproducible).
  3. Writes them in LLaMA-Factory alpaca format: instruction / input / output / system.
  4. Registers the dataset in LLaMA-Factory/data/dataset_info.json WITH the "system"
     column mapped. (The README's entry leaves it out, so LLaMA-Factory would silently
     train with Qwen's default system prompt instead of the one the eval uses.)
  5. Writes <out>_meta.jsonl (row index, category, source) for the report.

Run from the repo root (the folder that contains LLaMA-Factory/ and eval/):
  python make_random_subset.py
"""
import argparse
import json
import os
import random
import statistics
from collections import Counter

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def load_rows(path):
    """Load a JSON array or a JSON Lines file."""
    with open(path, encoding="utf-8") as f:
        head = f.read(4096).lstrip()
        f.seek(0)
        if head.startswith("["):
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def register(dataset_info_path, name, file_name):
    with open(dataset_info_path, encoding="utf-8") as f:
        info = json.load(f)
    info[name] = {
        "file_name": file_name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system",
        },
    }
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="local acereason11_100k.json; downloaded if omitted")
    p.add_argument("--download_dir", default="raw_data")
    p.add_argument("--n", type=int, default=15000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="random15k", help="dataset name used in the training YAML")
    p.add_argument("--lf_data_dir", default="LLaMA-Factory/data")
    args = p.parse_args()

    path = args.input
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="redsgnaoh/acereason11_100k",
            repo_type="dataset",
            filename="data/acereason11_100k.json",
            local_dir=args.download_dir,
        )
    print(f"loading {path} ...")
    rows = load_rows(path)
    print(f"loaded {len(rows):,} rows")
    if len(rows) < args.n:
        raise SystemExit(f"only {len(rows)} rows, cannot sample {args.n}")

    rng = random.Random(args.seed)
    picked = sorted(rng.sample(range(len(rows)), args.n))

    out_rows, meta = [], []
    for i in picked:
        r = rows[i]
        if not r.get("instruction") or not r.get("output"):
            raise SystemExit(f"row {i} is missing instruction/output: keys={list(r)}")
        out_rows.append(
            {
                "instruction": r["instruction"],
                "input": r.get("input") or "",
                "output": r["output"],
                "system": r.get("system") or SYSTEM_PROMPT,
            }
        )
        extra = r.get("extra_info") or {}
        meta.append({"row": i, "category": extra.get("category"), "source": extra.get("source"), "index": extra.get("index")})

    os.makedirs(args.lf_data_dir, exist_ok=True)
    file_name = f"{args.name}.json"
    out_path = os.path.join(args.lf_data_dir, file_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False)
    meta_path = os.path.join(args.lf_data_dir, f"{args.name}_meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    register(os.path.join(args.lf_data_dir, "dataset_info.json"), args.name, file_name)

    # quick report for the write-up
    systems = Counter(r["system"] for r in out_rows)
    lens = [len(r["output"]) for r in out_rows]
    print(f"\nwrote {len(out_rows):,} rows -> {out_path}")
    print(f"wrote meta -> {meta_path}")
    print(f"registered '{args.name}' in {args.lf_data_dir}/dataset_info.json (system column mapped)")
    print(f"system prompts used: {dict(systems)}")
    print("category:", dict(Counter(m["category"] for m in meta).most_common()))
    print("source:  ", dict(Counter(m["source"] for m in meta).most_common()))
    print(f"output chars: median {statistics.median(lens):,.0f}, max {max(lens):,}")


if __name__ == "__main__":
    main()
