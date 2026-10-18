"""B2b analysis: Qwen3 on MATH500, thinking on vs off (baselines.md section 3).

Per mode: accuracy, token counts, energy per query/token/correct answer, token-cap rate, trial-to-trial spread,
NVML-vs-Zeus agreement, and the paired on/off energy ratio. Several run dirs (e.g. a run continued with --resume in a
second slot, or a second GPU) can be given; rows are grouped by GPU.

Usage: python analysis/analyze_b2b.py results/runs/B2b/<run_id> [more run dirs]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.common import RESULTS, load_runs, write_md  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--out", default="B2b_summary")
    a = p.parse_args()
    rows = load_runs(a.run_dirs)
    rows["E_J"] = (rows.E_end_mJ - rows.E_start_mJ) / 1000
    rows["E_idle_sub_J"] = rows.E_J - rows.idle_W * rows.dur_s
    rows["mJ_per_tok"] = 1000 * rows.E_J / rows.output_len.clip(lower=1)

    md = ["# B2b: Qwen3 on MATH500, thinking on vs off", "", f"Runs: {', '.join(Path(d).name for d in a.run_dirs)}", ""]
    for gpu, g in rows.groupby("gpu"):
        md += [f"## {gpu}", "",
               "| Mode | Rows | Problems | Accuracy | Output tok | Thinking tok | Answer tok | J/query | "
               "J/query idle-sub | mJ/token | J/correct | Token cap hit | Trial spread J/query | Trial spread mJ/token | "
               "Zeus vs NVML |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        per_mode = {}
        for mode in ["off", "on"]:
            m = g[g["mode"] == mode]
            if not len(m):
                continue
            n_correct = int(m.correct.sum())
            spread = m.groupby("problem_id").E_J.agg(lambda s: s.std(ddof=0) / s.mean() if len(s) > 1 else None)
            spread = spread.dropna()
            spread_tok = m.groupby("problem_id").mJ_per_tok.agg(
                lambda s: s.std(ddof=0) / s.mean() if len(s) > 1 else None).dropna()
            zeus = ""
            if "zeus_mJ" in m and m.zeus_mJ.notna().any():
                gap = 100 * (m.zeus_mJ / 1000 / m.E_J - 1)
                zeus = f"{gap.mean():+.2f}% (max {gap.abs().max():.2f}%)"
            md.append(
                f"| {mode} | {len(m)} | {m.problem_id.nunique()} | {100 * m.correct.mean():.1f}% | "
                f"{m.output_len.mean():.0f} | {m.thinking_tokens.mean():.0f} | {m.answer_tokens.mean():.0f} | "
                f"{m.E_J.mean():.1f} | {m.E_idle_sub_J.mean():.1f} | {m.mJ_per_tok.mean():.1f} | "
                f"{m.E_J.sum() / n_correct if n_correct else float('nan'):.1f} | {100 * m.hit_token_cap.mean():.1f}% | "
                f"{'' if spread.empty else f'{100 * spread.mean():.1f}% (max {100 * spread.max():.1f}%)'} | "
                f"{'' if spread_tok.empty else f'{100 * spread_tok.mean():.1f}% (max {100 * spread_tok.max():.1f}%)'} | "
                f"{zeus} |")
            per_mode[mode] = m.groupby("problem_id").E_J.mean()
        if {"on", "off"} <= set(per_mode):
            ratio = (per_mode["on"] / per_mode["off"]).dropna()
            md += ["", f"**On/off energy ratio** (per problem, mean over trials, {len(ratio)} problems): "
                   f"median {ratio.median():.1f}x, range {ratio.min():.1f}x to {ratio.max():.1f}x, "
                   f"ratio of means {per_mode['on'].mean() / per_mode['off'].mean():.1f}x. "
                   "Published reference: Jin et al. 3.7-113x, Oviedo et al. ~13x."]
        by_level = g.groupby(["mode", "level"]).agg(acc=("correct", "mean"), E_J=("E_J", "mean"),
                                                     tok=("output_len", "mean")).reset_index()
        md += ["", "| Mode | Level | Accuracy | J/query | Output tok |", "|---|---|---|---|---|"]
        for r in by_level.itertuples():
            md.append(f"| {r.mode} | {r.level} | {100 * r.acc:.0f}% | {r.E_J:.1f} | {r.tok:.0f} |")
        md.append("")
    md += ["Trial spread = std/mean across trials of the same problem. Sampled answers differ in length between trials,",
           "so J/query spread mostly reflects length; the 5% check (baselines.md) applies to mJ/token.",
           "Token-capped runs are included in the averages; their J/query understates the true cost."]
    write_md(RESULTS / "tables" / f"{a.out}.md", md)


if __name__ == "__main__":
    main()
