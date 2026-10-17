"""B2a analysis: MRPC accuracy/F1, dataset-pass energy per example (NVML vs Zeus), CLEAR per-example energy.

Usage: python analysis/analyze_b2a.py results/runs/B2a/<run_id>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.common import RESULTS, load_run, write_md  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    a = p.parse_args()
    run, session, rows = load_run(Path(a.run_dir))
    idle_W = session["idle"]["mean_W"]

    acc = rows[rows.kind == "accuracy"].iloc[0]
    dp = rows[rows.kind == "dataset_pass"].copy()
    dp["E_mJ"] = (dp.E_end_mJ - dp.E_start_mJ) / dp.n_examples
    dp["E_idle_sub_mJ"] = dp.E_mJ - idle_W * dp.dur_s * 1000 / dp.n_examples
    md = [f"# B2a: BERT-base on MRPC, run `{run['run_id']}`", "",
          f"- GPU: {session['gpu_name']}, driver {session['driver']}, idle {idle_W:.1f} W",
          f"- Model: {run['model']['id']} ({run['model']['dtype']}, {run['model']['attn_impl']})",
          f"- Git: {run['git_commit']} (dirty={run['git_dirty']}), status: {run['status']}", "",
          f"**Accuracy {100 * acc.accuracy:.2f}%, F1 {100 * acc.f1:.2f}%** on {int(acc.n)} validation pairs.", "",
          "## Dataset pass (408 pairs, batch 1, one window per trial)", "",
          "| Quantity | mean | std | std/mean |", "|---|---|---|---|"]

    def line(name, s, unit="mJ"):
        md.append(f"| {name} | {s.mean():.3f} {unit} | {s.std(ddof=0):.3f} | {100 * s.std(ddof=0) / s.mean():.1f}% |")

    line("Energy per example (NVML)", dp.E_mJ)
    line("Energy per example, idle subtracted", dp.E_idle_sub_mJ)
    if "zeus_mJ" in dp and dp.zeus_mJ.notna().all():
        dp["zeus_per_ex"] = dp.zeus_mJ / dp.n_examples
        line("Energy per example (Zeus)", dp.zeus_per_ex)
        gap = 100 * (dp.zeus_mJ / (dp.E_end_mJ - dp.E_start_mJ) - 1)
        md.append(f"| Zeus vs NVML gap | {gap.mean():+.2f}% | {gap.std(ddof=0):.2f} | must be within ±1% |")
    line("Time per example", dp.dur_s / dp.n_examples * 1000, "ms")

    pe = rows[rows.kind == "per_example"].copy()
    if len(pe):
        pe["E_mJ"] = (pe.E_end_mJ - pe.E_start_mJ) / pe.N
        g = pe.groupby(["example_idx", "seq_len"]).E_mJ.agg(["mean", lambda s: s.std(ddof=0), "size"]).reset_index()
        g.columns = ["example_idx", "seq_len", "mean", "std", "n"]
        md += ["", "## Per example (CLEAR, N calls per trial)", "",
               "| Example | Tokens | mean mJ | std | std/mean | T |", "|---|---|---|---|---|---|"]
        for r in g.sort_values("seq_len").itertuples():
            md.append(f"| {r.example_idx} | {r.seq_len} | {r.mean:.3f} | {r.std:.3f} | "
                      f"{100 * r.std / r.mean:.1f}% | {r.n} |")
        md += ["", f"Dataset-pass mean per example: {dp.E_mJ.mean():.3f} mJ vs mean of CLEAR per-example values: "
               f"{g['mean'].mean():.3f} mJ (different length mixes; the comparison checks scale, not equality)."]
    write_md(RESULTS / "tables" / f"B2a_{run['run_id']}.md", md)


if __name__ == "__main__":
    main()
