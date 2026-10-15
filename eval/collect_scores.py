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

LB6 = ["cn_math_2024", "kaoyan", "amc", "minerva", "olympiadbench", "gpqa"]
EXTRA = ["aime", "math"]


def read_log(path):
    if not os.path.exists(path):
        return None, None
    text = open(path, encoding="utf-8", errors="ignore").read()
    acc = re.findall(r"^Acc: ([0-9.]+)", text, re.M)
    p1 = re.findall(r"^Pass@1: .*= ([0-9.]+)", text, re.M)
    return (float(acc[-1]) if acc else None), (float(p1[-1]) if p1 else None)


def main(dirs):
    cols = LB6 + EXTRA
    header = ["run"] + cols + ["AVG_LB6"]
    print("pass@8 (Acc)")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for d in dirs:
        vals = {b: read_log(os.path.join(d, f"log_{b}.txt"))[0] for b in cols}
        lb = [vals[b] for b in LB6 if vals[b] is not None]
        avg = sum(lb) / len(LB6) if len(lb) == len(LB6) else None
        cells = [os.path.basename(os.path.normpath(d))]
        cells += ["-" if vals[b] is None else f"{vals[b]:.3f}" for b in cols]
        cells.append("-" if avg is None else f"{avg:.3f}")
        print("| " + " | ".join(cells) + " |")
    print("\nPass@1 (mean over 8 samples)")
    print("| " + " | ".join(["run"] + cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for d in dirs:
        vals = {b: read_log(os.path.join(d, f"log_{b}.txt"))[1] for b in cols}
        cells = [os.path.basename(os.path.normpath(d))]
        cells += ["-" if vals[b] is None else f"{vals[b]:.3f}" for b in cols]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
