#!/usr/bin/env python
"""Build the curated 15K SFT dataset (advanced15k) for LLaMA-Factory using a 4-stage pipeline.

What it does:
  1. Downloads redsgnaoh/acereason11_100k (data/acereason11_100k.json) unless --input is given.
  2. Stage 1 (Format & Boundary Constraints): Enforces non-empty \boxed{...} answers,
     minimum reasoning depth (>= 200 words and >= 800 chars), and context ceiling
     (<= 10,000 words and <= 45,000 chars) to prevent cutoff_len truncation.
  3. Stage 2 (Domain Alignment): Filters out programming/coding problems via metadata
     (category/source) and content signatures (code fences and language syntax tokens).
  4. Stage 3 (Cognitive Quality Scoring): Scores candidates on verification (+2),
     backtracking/reflection (+2), and structured decomposition (+1), discarding score == 0.
  5. Stage 4 (Stratified Semantic Clustering): Embeds instructions using
     all-MiniLM-L6-v2, clusters into K semantic groups via KMeans, and selects top-scoring
     traces per cluster (with global greedy top-up if any cluster is under-populated).
  6. Writes LLaMA-Factory alpaca format JSON, <name>_meta.jsonl, and registers the
     dataset in LLaMA-Factory/data/dataset_info.json with the "system" column mapped.

Run from the repo root (the folder that contains LLaMA-Factory/ and eval/):
  python make_advanced_subset.py
"""
import argparse
import json
import os
import re
import statistics
from collections import Counter, defaultdict

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# ---------------------------------------------------------------------------
# Stage 1 & 2 Patterns
# ---------------------------------------------------------------------------
# Ensures at least one \boxed{...} with non-whitespace content inside
NON_EMPTY_BOXED_RE = re.compile(r"\\boxed\{\s*[^}\s][^}]*\}")

CODE_META_KEYWORDS = (
    "code",
    "coding",
    "programming",
    "leetcode",
    "taco",
    "codeforces",
    "livecodebench",
    "humaneval",
    "mbpp",
    "apps",
)

CODE_CONTENT_RE = re.compile(
    r"(?:```(?:python|cpp|c\+\+|java|javascript|js|c|bash|sh|rust|go)\b"
    r"|\bdef\s+[a-zA-Z_]\w*\s*\("
    r"|\bclass\s+Solution\s*[:(]"
    r"|\bimport\s+sys\b"
    r"|#include\s*<"
    r"|\bstd::"
    r"|\bpublic\s+static\s+void\s+main\b)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Stage 3 Cognitive Scoring Patterns (case-insensitive)
# Note: Prefix stems (verif\w*, substitut\w*, re-evaluat\w*) and colon patterns
# are matched cleanly so trailing punctuation or word inflections don't fail \b.
# ---------------------------------------------------------------------------
VERIFICATION_RE = re.compile(
    r"\b(?:let's check|plugging\s+[^\n.]{0,50}?back|verif\w*|substitut\w*|double[- ]check\w*)\b",
    re.IGNORECASE,
)

BACKTRACKING_RE = re.compile(
    r"\b(?:wait\b|alternatively\b|let me rethink\b|however\b|re-evaluat\w*\b|contradiction\b)",
    re.IGNORECASE,
)

DECOMPOSITION_RE = re.compile(
    r"(?:\bcase\s+\d+\s*:|\bstep\s+\d+\s*:|\bfirst,\s+we\b|\bgoal\s*:)",
    re.IGNORECASE,
)


def load_rows(path):
    """Load a JSON array or a JSON Lines file."""
    with open(path, encoding="utf-8") as f:
        head = f.read(4096).lstrip()
        f.seek(0)
        if head.startswith("["):
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def register(dataset_info_path, name, file_name):
    """Register the dataset in LLaMA-Factory/data/dataset_info.json with system mapped."""
    info = {}
    if os.path.exists(dataset_info_path):
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


def passes_stage1(instruction, output):
    """Stage 1: Format & Boundary Hard Constraints."""
    # 1. Answer Verification: must have \boxed{...} with content
    if "\\boxed{" not in output or not NON_EMPTY_BOXED_RE.search(output):
        return False, "missing_boxed"

    # 2. Triviality Floor: >= 200 words and >= 800 characters
    if len(output) < 800 or len(output.split()) < 200:
        return False, "too_short"

    # 3. Context Ceiling: <= 10,000 words and <= 45,000 characters combined
    total_chars = len(instruction) + len(output)
    if total_chars > 45000:
        return False, "too_long"
    total_words = len(instruction.split()) + len(output.split())
    if total_words > 10000:
        return False, "too_long"

    return True, "ok"


def passes_stage2(instruction, output, extra_info):
    """Stage 2: Target Domain Alignment (Filter Out Code)."""
    # 1. Metadata Category / Source Check
    cat = str(extra_info.get("category") or "").lower()
    src = str(extra_info.get("source") or "").lower()
    if any(k in cat or k in src for k in CODE_META_KEYWORDS):
        return False, "code_metadata"

    # 2. Content Code Filtering
    if CODE_CONTENT_RE.search(instruction) or CODE_CONTENT_RE.search(output):
        return False, "code_content"

    return True, "ok"


def compute_cognitive_score(output):
    """Stage 3: Cognitive Quality Scoring."""
    num_verif = len(VERIFICATION_RE.findall(output))
    num_backtrack = len(BACKTRACKING_RE.findall(output))
    num_subgoals = len(DECOMPOSITION_RE.findall(output))
    score = (2 * num_verif) + (2 * num_backtrack) + (1 * num_subgoals)
    return score, num_verif, num_backtrack, num_subgoals


def stratify_and_select(candidates, n_select, n_clusters, seed, batch_size):
    """Stage 4: Stratified Semantic Clustering & Breadth Maximization."""
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"\n[Stage 4] Encoding {len(candidates):,} instructions with {model_name} on {device.upper()} ...")

    embedder = SentenceTransformer(model_name, device=device)
    instructions = [c["instruction"] for c in candidates]
    embeddings = embedder.encode(
        instructions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    effective_clusters = min(n_clusters, len(candidates))
    print(f"[Stage 4] Running KMeans (n_clusters={effective_clusters}, seed={seed}) ...")
    kmeans = KMeans(n_clusters=effective_clusters, random_state=seed, n_init="auto")
    cluster_ids = kmeans.fit_predict(embeddings)

    clusters = defaultdict(list)
    for idx, cid in enumerate(cluster_ids):
        candidates[idx]["cluster"] = int(cid)
        clusters[int(cid)].append(candidates[idx])

    target_per_cluster = n_select // effective_clusters
    selected = []
    selected_row_ids = set()
    deficit_clusters = 0

    # Sort within each cluster descending by (quality_score, output_len, -row_id)
    for cid in range(effective_clusters):
        members = clusters[cid]
        members.sort(key=lambda c: (c["quality_score"], c["output_len"], -c["row"]), reverse=True)
        take = members[:target_per_cluster]
        if len(take) < target_per_cluster:
            deficit_clusters += 1
        for item in take:
            selected.append(item)
            selected_row_ids.add(item["row"])

    # Fill any deficit (from small clusters or integer division remainder) greedily across all candidates
    remaining_needed = n_select - len(selected)
    if remaining_needed > 0:
        print(
            f"[Stage 4] Harvested {len(selected):,} via uniform cluster quota "
            f"({deficit_clusters} clusters under quota). Filling {remaining_needed:,} globally ..."
        )
        all_sorted = sorted(
            candidates,
            key=lambda c: (c["quality_score"], c["output_len"], -c["row"]),
            reverse=True,
        )
        for item in all_sorted:
            if item["row"] not in selected_row_ids:
                selected.append(item)
                selected_row_ids.add(item["row"])
                if len(selected) == n_select:
                    break

    # Sort final selection by original row index for deterministic ordering matching make_random_subset.py
    selected.sort(key=lambda c: c["row"])
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="local acereason11_100k.json; downloaded if omitted")
    p.add_argument("--download_dir", default="raw_data")
    p.add_argument("--n", type=int, default=15000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="advanced15k", help="dataset name used in the training YAML")
    p.add_argument("--lf_data_dir", default="LLaMA-Factory/data")
    p.add_argument("--clusters", type=int, default=50, help="number of semantic clusters for stratification")
    p.add_argument("--batch_size", type=int, default=256, help="batch size for embedding inference")
    args = p.parse_args()

    path = args.input
    if path is None:
        local_default = os.path.join(args.download_dir, "data", "acereason11_100k.json")
        if os.path.exists(local_default):
            path = local_default
        else:
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

    # Track curation funnel stats
    stats = Counter()
    candidates = []
    zero_score_backfill = []

    for i, r in enumerate(rows):
        instruction = r.get("instruction") or ""
        output = r.get("output") or ""
        if not instruction or not output:
            stats["drop_missing_fields"] += 1
            continue

        # Stage 1: Format & Boundary Hard Constraints
        ok1, reason1 = passes_stage1(instruction, output)
        if not ok1:
            stats[f"stage1_{reason1}"] += 1
            continue
        stats["pass_stage1"] += 1

        # Stage 2: Target Domain Alignment (Filter Out Code)
        extra = r.get("extra_info") or {}
        ok2, reason2 = passes_stage2(instruction, output, extra)
        if not ok2:
            stats[f"stage2_{reason2}"] += 1
            continue
        stats["pass_stage2"] += 1

        # Stage 3: Cognitive Quality Scoring
        q_score, n_verif, n_back, n_sub = compute_cognitive_score(output)
        item = {
            "row": i,
            "instruction": instruction,
            "input": r.get("input") or "",
            "output": output,
            "system": r.get("system") or SYSTEM_PROMPT,
            "extra_info": extra,
            "quality_score": q_score,
            "verifications": n_verif,
            "backtrackings": n_back,
            "subgoals": n_sub,
            "output_len": len(output),
        }

        if q_score == 0:
            stats["stage3_zero_score"] += 1
            zero_score_backfill.append(item)
            continue

        stats["pass_stage3"] += 1
        candidates.append(item)

    print("\n--- Curation Funnel Summary ---")
    print(f"Total raw rows:              {len(rows):,}")
    print(f"Passed Stage 1 (Boundaries): {stats['pass_stage1']:,} "
          f"(dropped: no_boxed={stats['stage1_missing_boxed']:,}, "
          f"short={stats['stage1_too_short']:,}, long={stats['stage1_too_long']:,})")
    print(f"Passed Stage 2 (No Code):    {stats['pass_stage2']:,} "
          f"(dropped: meta={stats['stage2_code_metadata']:,}, content={stats['stage2_code_content']:,})")
    print(f"Passed Stage 3 (Score > 0):  {stats['pass_stage3']:,} "
          f"(dropped: score_zero={stats['stage3_zero_score']:,})")

    if len(candidates) < args.n:
        needed = args.n - len(candidates)
        print(
            f"WARNING: Only {len(candidates):,} candidates had quality_score > 0. "
            f"Topping up {needed:,} longest valid math traces from Stage 2 pool."
        )
        zero_score_backfill.sort(key=lambda c: (c["output_len"], -c["row"]), reverse=True)
        candidates.extend(zero_score_backfill[:needed])

    if len(candidates) < args.n:
        raise SystemExit(f"Only {len(candidates):,} candidates survived filtering, cannot select {args.n:,}")

    # Stage 4: Stratified Semantic Clustering
    selected = stratify_and_select(
        candidates=candidates,
        n_select=args.n,
        n_clusters=args.clusters,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    out_rows, meta = [], []
    for c in selected:
        out_rows.append(
            {
                "instruction": c["instruction"],
                "input": c["input"],
                "output": c["output"],
                "system": c["system"],
            }
        )
        extra = c["extra_info"]
        meta.append(
            {
                "row": c["row"],
                "category": extra.get("category"),
                "source": extra.get("source"),
                "index": extra.get("index"),
                "cluster": c["cluster"],
                "quality_score": c["quality_score"],
                "verifications": c["verifications"],
                "backtrackings": c["backtrackings"],
                "subgoals": c["subgoals"],
            }
        )

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

    # Quick report for the write-up
    systems = Counter(r["system"] for r in out_rows)
    lens = [len(r["output"]) for r in out_rows]
    scores = [m["quality_score"] for m in meta]
    cluster_counts = Counter(m["cluster"] for m in meta)

    print(f"\nwrote {len(out_rows):,} rows -> {out_path}")
    print(f"wrote meta -> {meta_path}")
    print(f"registered '{args.name}' in {args.lf_data_dir}/dataset_info.json (system column mapped)")
    print(f"system prompts used: {dict(systems)}")
    print("category:", dict(Counter(m["category"] for m in meta).most_common()))
    print("source:  ", dict(Counter(m["source"] for m in meta).most_common()))
    print(f"output chars:  median {statistics.median(lens):,.0f}, mean {statistics.mean(lens):,.0f}, max {max(lens):,}")
    print(f"quality score: median {statistics.median(scores):.1f}, mean {statistics.mean(scores):.2f}, max {max(scores)}")
    print(
        f"clusters populated: {len(cluster_counts)}/{args.clusters} "
        f"(min={min(cluster_counts.values())}, median={int(statistics.median(cluster_counts.values()))}, "
        f"max={max(cluster_counts.values())})"
    )


if __name__ == "__main__":
    main()