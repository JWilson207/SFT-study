"""Score table from eval result files (outputs/<run>/**/<benchmark>/*.jsonl).

Merges every result file of a benchmark (full runs, or chunks run with --start_idx/--end_idx on
different machines) and counts each problem once, identified by its position in data/<bench>/test.jsonl
(start index from the file name + row number). pass@8 = Acc in the course logs.
Usage (from eval/): python scores.py
"""
import glob
import json
import os
import re

LB6 = ["cn_math_2024", "kaoyan", "amc", "minerva", "olympiadbench", "gpqa"]
ALL = LB6 + ["aime", "math"]
SIZE = {b: sum(1 for l in open(f"data/{b}/test.jsonl", encoding="utf-8") if l.strip()) for b in ALL}

rows = {}
for f in glob.glob("outputs/*/**/*.jsonl", recursive=True):
    parts = f.split(os.sep)
    run, bench = parts[1], parts[-2]
    if bench not in ALL:
        continue
    m = re.search(r"_s(\d+)_e(-?\d+)\.jsonl$", f)
    start = int(m.group(1)) if m else 0
    seen = rows.setdefault(run, {}).setdefault(bench, {})
    i = 0
    for line in open(f, encoding="utf-8"):
        if line.strip():
            seen.setdefault(start + i, json.loads(line))   # problem index -> result, counted once
            i += 1

res = {}
for run, benches in rows.items():
    for bench, items in benches.items():
        rs = list(items.values())
        p8 = sum(r["is_correct"] for r in rs) / len(rs)
        p1 = sum(sum(r["answers_correctness"]) / len(r["answers_correctness"]) for r in rs) / len(rs)
        res.setdefault(run, {})[bench] = (p8, p1, len(rs))

for metric, i in [("pass@8 (Acc, leaderboard metric)", 0), ("Pass@1", 1)]:
    print(f"\n{metric}")
    print("run".ljust(14) + "".join(b[:12].rjust(13) for b in ALL) + "   AVG_LB6")
    for run in sorted(res):
        cells = []
        for b in ALL:
            c = res[run].get(b)
            if c is None:
                cells.append("-")
            else:
                cells.append(f"{c[i]:.3f}" + ("" if c[2] == SIZE[b] else f"*{c[2]}"))
        full = [res[run][b][i] for b in LB6 if b in res[run] and res[run][b][2] == SIZE[b]]
        avg = f"{sum(full)/6:.3f}" if len(full) == 6 else f"({len(full)}/6 complete)"
        print(run.ljust(14) + "".join(c.rjust(13) for c in cells) + "   " + avg)
print("\nproblems per benchmark:", SIZE)
print("*N = only N of the benchmark's problems are scored so far (partial).")
