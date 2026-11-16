"""Experiment C analysis: accuracy vs energy when thinking is cut short (expc/run_expc.py).

Budget curves are built on the same set of traces for every budget, so points are comparable:
  - percentage budgets (p25, p50, p75, full) exist for every trace;
  - an absolute budget k (t512 ...) is only cut where k < natural_len. For a trace that ends thinking before k the
    cap never binds, so its value at k is its "full" row. Each absolute point is therefore "thinking capped at k".
Thinking off (optional, a B2b run with --modes off from the same session) is matched on (problem_id, trial) to the
C traces, so it covers the same problems with the same weights.

Per budget: accuracy (95% Wilson range), mean thinking/answer tokens, J/query (raw and idle-subtracted), J per correct
answer, energy saved vs full thinking, and answer flips vs full (right->wrong, wrong->right on the same trace).

Usage: python analysis/analyze_expc.py results/runs/C/<run_id> [--off results/runs/B2b_expc_off/<run_id>]
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.common import INK, INK_2, RESULTS, SERIES, load_run, style, write_md  # noqa: E402

KEY = ["problem_id", "trial"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def budget_table(rows, pct: list[int], abs_: list[int]) -> dict:
    """{label: DataFrame with one row per trace}; absolute budgets fall back to the trace's full row."""
    full = rows[rows.label == "full"].set_index(KEY)
    out = {}
    for p in pct:
        out[f"p{p}"] = rows[rows.label == f"p{p}"].set_index(KEY)
    for k in abs_:
        cut = rows[rows.label == f"t{k}"].set_index(KEY)
        out[f"t{k}"] = cut.combine_first(full.loc[~full.index.isin(cut.index)]).loc[full.index]
    out["full"] = full
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--off", help="B2b run dir with thinking-off answers from the same session")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run, session, rows = load_run(Path(a.run_dir))
    cfg = run["config"]
    rows["ans_dur_s"] = rows.answer_t_end_s - rows.answer_t_start_s
    rows["E_J"] = rows.total_E_mJ / 1000
    rows["E_idle_sub_J"] = rows.E_J - rows.idle_W * (rows.think_t_s + rows.ans_dur_s)
    rows["think_tok"] = rows.budget_tokens

    # Only traces that have every percentage budget and full (a run stopped mid-trace leaves partial traces).
    need = {f"p{p}" for p in cfg["pct"]} | {"full"}
    have = rows.groupby(KEY).label.agg(set)
    complete = have[have.apply(lambda s: need <= s)].index
    rows = rows.set_index(KEY).loc[complete].reset_index()
    n_tr = len(complete)

    tabs = budget_table(rows, cfg["pct"], cfg["abs"])
    full = tabs["full"]

    def summarize(label, t, think_tok=None, ans_tok=None):
        n, k = len(t), int(t.correct.sum())
        lo, hi = wilson(k, n)
        E = t.E_J.mean()
        r = {"label": label, "n": n, "acc": k / n, "lo": lo, "hi": hi, "E": E, "E_idle": t.E_idle_sub_J.mean(),
             "J_per_correct": t.E_J.sum() / k if k else float("nan"),
             "think_tok": t.think_tok.mean() if think_tok is None else think_tok,
             "ans_tok": t.answer_tokens.mean() if ans_tok is None else ans_tok,
             "saved": 1 - E / full.E_J.mean()}
        if "label" in t and label != "full":
            j = t.join(full[["correct"]], rsuffix="_full")
            r["flip_bad"] = int((j.correct_full & ~j.correct).sum())
            r["flip_good"] = int((~j.correct_full & j.correct).sum())
            r["cut"] = int((t.label != "full").sum())
            r["ans_cap"] = int(t.answer_hit_cap.sum())
            r["reopen"] = int(t.reopened_think.sum())
        return r

    summ = [summarize(lab, tabs[lab]) for lab in [f"p{p}" for p in cfg["pct"]] + ["full"] + [f"t{k}" for k in cfg["abs"]]]

    off = None
    if a.off:
        _, off_sess, off_rows = load_run(Path(a.off))
        off_rows = off_rows[off_rows["mode"] == "off"].set_index(KEY)
        off_rows = off_rows.loc[off_rows.index.intersection(complete)]
        off_rows["E_J"] = (off_rows.E_end_mJ - off_rows.E_start_mJ) / 1000
        off_rows["E_idle_sub_J"] = off_rows.E_J - off_rows.idle_W * off_rows.dur_s
        off_rows["think_tok"] = 0
        off = summarize("off", off_rows, think_tok=0, ans_tok=off_rows.output_len.mean())
        same = off_sess["gpu_uuid"] == session["gpu_uuid"] and \
            off_sess["versions"].get("torch") == session["versions"].get("torch")

    name = a.out or f"C_{run['run_id']}"
    md = [f"# Experiment C: cutting thinking short (Qwen3-8B, MATH-500)", "",
          f"Run `{run['run_id']}` (status {run['status']}), source traces `{cfg['source_run']}`, "
          f"GPU {session['gpu_name']}, idle {session['idle']['mean_W']:.1f} W. "
          f"Forced end of thinking: `{cfg['end_text']!r}`; answer cap {cfg['max_answer']} tokens.",
          f"Traces: {n_tr} complete ({rows.problem_id.nunique()} problems); "
          f"{len(cfg['skipped_capped'])} source traces skipped (hit the token cap, no natural end).", ""]
    if off:
        md += [f"Thinking off: `{Path(a.off).name}`, matched on (problem, trial): {off['n']} rows"
               + f", session `{off_sess['session_id']}`"
               + ("" if same else " **(different physical GPU or torch version: energy not comparable)**") + ".", ""]
    md += ["| Budget | Traces cut | Thinking tok | Answer tok | Accuracy (95% range) | J/query | J/query idle-sub | "
           "J/correct | Energy saved vs full | Right→wrong | Wrong→right | Answer hit cap | Reopened thinking |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ([off] if off else []) + summ:
        md.append(f"| {r['label']} | {r.get('cut', '')} | {r['think_tok']:.0f} | {r['ans_tok']:.0f} | "
                  f"{100 * r['acc']:.1f}% ({100 * r['lo']:.0f}–{100 * r['hi']:.0f}) | {r['E']:.0f} | {r['E_idle']:.0f} | "
                  f"{r['J_per_correct']:.0f} | {100 * r['saved']:+.0f}% | {r.get('flip_bad', '')} | "
                  f"{r.get('flip_good', '')} | {r.get('ans_cap', '')} | {r.get('reopen', '')} |")

    md += ["", "## By MATH level (percentage budgets): accuracy / J per correct answer", "",
           "| Level | Traces | " + " | ".join(r["label"] for r in summ[:len(cfg["pct"]) + 1]) + (" | off |" if off else " |"),
           "|---|---|" + "---|" * (len(cfg["pct"]) + 1 + (1 if off else 0))]
    for lvl in sorted(rows.level.unique()):
        cells = []
        for lab in [f"p{p}" for p in cfg["pct"]] + ["full"]:
            t = tabs[lab][tabs[lab].level == lvl]
            k = int(t.correct.sum())
            cells.append(f"{100 * t.correct.mean():.0f}% / {t.E_J.sum() / k:.0f} J" if k else f"0% / –")
        if off:
            t = off_rows[off_rows.level == lvl]
            k = int(t.correct.sum())
            cells.append(f"{100 * t.correct.mean():.0f}% / {t.E_J.sum() / k:.0f} J" if k else "0% / –")
        md.append(f"| {lvl} | {len(tabs['full'][tabs['full'].level == lvl])} | " + " | ".join(cells) + " |")

    md += ["", "Notes:",
           "- Thinking energy comes from a teacher-forced replay of the saved tokens (one forward per token); in a "
           "test it ran ~2% faster than generate() at the same power, so thinking energy is ~2% low.",
           "- Absolute budgets: traces whose thinking ends before k are not cut (column 'Traces cut'); they use their "
           "full row.",
           "- J/correct = total energy of all queries at that budget ÷ number of correct answers.",
           "- Skipped token-capped traces are the longest ones, mostly on hard problems; accuracy here is for "
           "problems the model finishes thinking on."]
    write_md(RESULTS / "tables" / f"{name}.md", md)

    # Figure: accuracy vs J/query (one curve per budget kind), and J per correct answer per budget.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    pct = [r for r in summ if r["label"].startswith("p")] + [summ[len(cfg["pct"])]]
    absb = [r for r in summ if r["label"].startswith("t")] + [summ[len(cfg["pct"])]]
    for pts, color, name_ in [(pct, SERIES[0], "% of natural length"), (absb, SERIES[1], "token cap")]:
        pts = sorted(pts, key=lambda r: r["E"])
        x = [r["E"] / 1000 for r in pts]
        y = [100 * r["acc"] for r in pts]
        ax1.plot(x, y, "-o", color=color, lw=2, ms=8, mec="white", mew=2, label=name_)
        for r in pts:
            ax1.vlines(r["E"] / 1000, 100 * r["lo"], 100 * r["hi"], color=color, lw=1, alpha=0.5)
            if r["label"] != "full":
                ax1.annotate(r["label"], (r["E"] / 1000, 100 * r["acc"]), textcoords="offset points",
                             xytext=(0, -16 if color == SERIES[1] else 9), ha="center", fontsize=8, color=INK_2)
    full_r = summ[len(cfg["pct"])]
    ax1.annotate("full", (full_r["E"] / 1000, 100 * full_r["acc"]), textcoords="offset points", xytext=(6, 6),
                 fontsize=8, color=INK)
    if off:
        ax1.plot([off["E"] / 1000], [100 * off["acc"]], "s", color=SERIES[2], ms=9, mec="white", mew=2,
                 label="thinking off")
        ax1.vlines(off["E"] / 1000, 100 * off["lo"], 100 * off["hi"], color=SERIES[2], lw=1, alpha=0.5)
    style(ax1, "Energy per query (kJ)", "Accuracy (%)")
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    ax1.set_title("Accuracy vs energy (bars: 95% range)", fontsize=10, color=INK, loc="left")

    bars = ([off] if off else []) + [r for r in summ if r["label"].startswith("p")] + [full_r] + \
        [r for r in summ if r["label"].startswith("t")]
    colors = [SERIES[2] if r["label"] == "off" else SERIES[1] if r["label"].startswith("t") else SERIES[0]
              for r in bars]
    ax2.bar(range(len(bars)), [r["J_per_correct"] / 1000 for r in bars], color=colors, width=0.7)
    ax2.set_xticks(range(len(bars)), [r["label"] for r in bars], fontsize=8)
    style(ax2, "Budget", "Energy per correct answer (kJ)")
    ax2.set_title("Energy per correct answer", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    figdir = RESULTS / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{name}.png", dpi=150)
    print(f"Wrote {figdir / f'{name}.png'}")


if __name__ == "__main__":
    main()
