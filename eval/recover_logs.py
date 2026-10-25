import os
import glob
import json
import argparse

BENCHMARKS = [
    "cn_math_2024", "kaoyan", "amc", "minerva", 
    "olympiadbench", "gpqa", "aime", "math"
]

def recover_logs_for_dir(output_dir):
    print(f"Scanning {output_dir}...")
    
    for bench in BENCHMARKS:
        log_file = os.path.join(output_dir, f"log_{bench}.txt")
        
        # Check if the existing log already has valid Acc and Pass@1
        has_valid_scores = False
        if os.path.exists(log_file):
            content = open(log_file, errors="ignore").read()
            if "Acc:" in content and "Pass@1:" in content:
                has_valid_scores = True

        if has_valid_scores:
            print(f"[{bench}] Log file already contains scores, skipping.")
            continue

        # Look for the generated jsonl file in subdirectories
        pattern = os.path.join(output_dir, "**", bench, "*.jsonl")
        jsonl_files = glob.glob(pattern, recursive=True)
        # Exclude temporary files
        jsonl_files = [f for f in jsonl_files if not f.endswith(".tmp")]

        if not jsonl_files:
            print(f"[{bench}] No .jsonl output found. This benchmark likely didn't finish.")
            continue

        target_jsonl = jsonl_files[0]
        
        # Read items and compute scores
        total_problems = 0
        pass_at_8_cnt = 0
        pass_at_1_sum = 0.0

        with open(target_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                total_problems += 1
                
                # Check pass@8 (any sample correct)
                if data.get("is_correct", False):
                    pass_at_8_cnt += 1
                
                # Check pass@1 (mean accuracy across samples)
                corr_list = data.get("answers_correctness", [])
                if corr_list:
                    pass_at_1_sum += sum(1 for c in corr_list if c) / len(corr_list)

        if total_problems == 0:
            print(f"[{bench}] {target_jsonl} was empty.")
            continue

        acc = pass_at_8_cnt / total_problems
        p1 = pass_at_1_sum / total_problems

        # Append or rewrite log in the exact format eval.py produces
        log_content = (
            f"\n--- Recovered from {os.path.basename(target_jsonl)} ---\n"
            f"correct cnt / total cnt: {pass_at_8_cnt}/{total_problems}\n"
            f"Acc: {acc:.4f}\n"
            f"Pass@1: {pass_at_1_sum:.3f}/{total_problems} = {p1:.4f}\n"
        )

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_content)

        print(f"[{bench}] Successfully regenerated {log_file} (Acc: {acc:.4f}, Pass@1: {p1:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="Directories containing logs (e.g. outputs/base outputs/random15k)")
    args = parser.parse_args()

    for d in args.dirs:
        recover_logs_for_dir(d)