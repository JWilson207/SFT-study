#!/usr/bin/env python
"""Summarize eval logs into one table.

Usage (from eval/):  python collect_scores.py outputs/base outputs/random15k

Reads <dir>/log_<benchmark>.txt for each benchmark and prints Acc (pass@8, what the
leaderboard uses) and Pass@1. AVG_LB6 = mean pass@8 over the six leaderboard benchmarks
that exist in eval/data. The leaderboard's seventh, AIME25, is not in the repo (it ships
AIME24), so AIME24 is shown separately and left out of the average.
"""
import os
import re
import sys

LB6 = ["cn_math_2024", "kaoyan", "amc", "minerva", "olympiadbench", "gpqa", "aime"]
EXTRA = ["math"]


def read_log(path):
    if not os.path.exists(path):
        return None, None
    text = open(path, encoding="utf-8", errors="ignore").read()
    acc = re.findall(r"^Acc: ([0-9.]+)", text, re.M)
    p1 = re.findall(r"^Pass@1: .*= ([0-9.]+)", text, re.M)
    return (float(acc[-1]) if acc else None), (float(p1[-1]) if p1 else None)


def main(dirs):
    cols = LB6 + EXTRA
    headers = ["run"] + cols + ["AVG_LB6"]
    
    col_w = 14
    run_w = 16

    for metric_name, idx in [("pass@8 (Acc, leaderboard metric)", 0), ("Pass@1", 1)]:
        print(f"\n{metric_name}")
        header_str = f"{'run':<{run_w}}" + "".join([f"{c[:col_w-2]:>{col_w}}" for c in cols]) + f"{'AVG_LB6':>{col_w}}"
        print(header_str)

        for d in dirs:
            vals = {b: read_log(os.path.join(d, f"log_{b}.txt"))[idx] for b in cols}
            lb = [vals[b] for b in LB6 if vals[b] is not None]
            
            cells = [f"{os.path.basename(os.path.normpath(d)):<{run_w}}"]
            for b in cols:
                v = vals[b]
                cells.append(f"{'-':>{col_w}}" if v is None else f"{v:>{col_w}.3f}")
            
            if len(lb) == len(LB6):
                avg_str = f"{sum(lb) / len(LB6):>{col_w}.3f}"
            else:
                avg_str = f"{f'({len(lb)}/{len(LB6)} complete)':>{col_w}}"
            cells.append(avg_str)
            
            print("".join(cells))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
