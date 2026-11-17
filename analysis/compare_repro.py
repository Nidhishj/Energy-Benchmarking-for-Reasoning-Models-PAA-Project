"""Compare a B2b repro run with a reference run of the same problems: energy per output token per (problem, mode).

Used as the first gate of a GPU session (eval2_rtxpro5000_plan.md, step 1): every later number on the card is
only comparable to the Eval 1 baselines if this matches.

Usage: python analysis/compare_repro.py results/runs/B2b_repro/<new> results/runs/B2b_repro/<reference> [--tol 5]
Exit code 1 if any (problem, mode) differs by more than --tol percent.
"""

import argparse
import json
import sys
from pathlib import Path


def per_token(run_dir: str) -> dict:
    out = {}
    for line in open(Path(run_dir) / "measurements.jsonl"):
        r = json.loads(line)
        if r.get("method", "nvml") != "nvml" or not r.get("output_len"):
            continue
        out.setdefault((r["problem_id"], r["mode"]), []).append((r["E_end_mJ"] - r["E_start_mJ"]) / r["output_len"])
    return {k: sum(v) / len(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("new")
    p.add_argument("ref")
    p.add_argument("--tol", type=float, default=5.0, help="max difference in percent")
    a = p.parse_args()
    new, ref = per_token(a.new), per_token(a.ref)
    keys = sorted(set(new) & set(ref))
    if not keys:
        sys.exit("No (problem, mode) in common")
    worst = 0.0
    print(f"| Problem | Mode | Reference mJ/token | New mJ/token | Difference |\n|---|---|---|---|---|")
    for k in keys:
        d = 100 * (new[k] - ref[k]) / ref[k]
        worst = max(worst, abs(d))
        print(f"| {k[0]} | {k[1]} | {ref[k]:.1f} | {new[k]:.1f} | {d:+.1f}% |")
    ok = worst <= a.tol
    print(f"\nLargest difference {worst:.1f}% (limit {a.tol:.0f}%): {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
