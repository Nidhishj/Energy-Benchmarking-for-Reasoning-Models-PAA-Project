"""Fixed thinking caps analysis (expc/run_caps.py): accuracy vs energy when thinking is limited to k tokens.

Each copy is a pair (B2b source run, Caps run) from the same GPU. Energy is never mixed across GPUs: energy columns
are per copy; accuracy is also pooled over the copies.

For a trace and cap k:
  - k < thinking length: the Caps row (thinking cut at k, answer forced; energy = E(k) + answer energy);
  - otherwise the cap never binds: the source run's own uncapped row.
"full" = the source run uncapped (thinking up to its token limit); "full+forced" = the same, but traces whose thinking
never ended use the answer forced at the limit (Caps label "limit"). "off" = thinking off from the source run.
Only traces whose Caps jobs are all done are used, and "off" is restricted to the same (problem, trial) pairs.

Difficulty groups: for AIME the problem number within its paper (1-5, 6-10, 11-15; AIME problems get harder in
order); otherwise the MATH level.

Usage: python analysis/analyze_caps.py --copy <B2b run> <Caps run> [--copy <B2b run> <Caps run> ...] --out NAME
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.analyze_expc import wilson  # noqa: E402
from analysis.common import INK, INK_2, RESULTS, SERIES, load_run, style, write_md  # noqa: E402


def group_of(pid: str, level) -> str:
    m = re.match(r"aime\d{4}/I+-(\d+)$", pid)
    if m:
        n = int(m.group(1))
        return ["1-5 (easier)", "6-10 (medium)", "11-15 (hardest)"][(n - 1) // 5]
    return f"L{level}"


def load_copy(src_dir: str, caps_dir: str) -> tuple[pd.DataFrame, dict]:
    """One row per (trace, budget) with columns budget, correct, E_J, think_tok, ans_tok, ans_cap."""
    _, s_sess, src = load_run(Path(src_dir))
    caps_run, c_sess, caps = load_run(Path(caps_dir))
    if s_sess["gpu_uuid"] != c_sess["gpu_uuid"]:
        sys.exit(f"{caps_dir}: Caps run and source run are on different GPUs")
    ks = caps_run["config"]["caps"]
    with_limit = caps_run["config"]["limit_answer"]
    src["E_J"] = (src.E_end_mJ - src.E_start_mJ) / 1000
    on = src[src["mode"] == "on"].set_index(["problem_id", "trial"])
    off = src[src["mode"] == "off"].set_index(["problem_id", "trial"])
    caps["E_J"] = caps.total_E_mJ / 1000
    cidx = caps.set_index(["problem_id", "trial", "label"])

    out = []
    for (pid, trial), r in on.iterrows():
        ended = r.natural_len is not None and r.natural_len == r.natural_len
        think_len = int(r.natural_len) if ended else int(r.output_len)
        need = [f"cap{k}" for k in ks if k < think_len] + ([] if ended or not with_limit else ["limit"])
        if not all((pid, trial, lab) in cidx.index for lab in need):
            continue  # Caps run has not reached this trace yet
        base = {"problem_id": pid, "trial": trial, "group": group_of(pid, r.level), "gpu": c_sess["gpu_name"],
                "copy_id": Path(caps_dir).name}
        unc = {"correct": bool(r.correct), "E_J": r.E_J, "think_tok": r.thinking_tokens, "ans_tok": r.answer_tokens,
               "ans_cap": False}
        for k in ks:
            if k < think_len:
                c = cidx.loc[(pid, trial, f"cap{k}")]
                out.append({**base, "budget": f"≤{k // 1024}k", "correct": bool(c.correct), "E_J": c.E_J,
                            "think_tok": k, "ans_tok": c.answer_tokens, "ans_cap": bool(c.answer_hit_cap)})
            else:
                out.append({**base, "budget": f"≤{k // 1024}k", **unc})
        out.append({**base, "budget": "full", **unc})
        if not ended and with_limit:
            c = cidx.loc[(pid, trial, "limit")]
            out.append({**base, "budget": "full+forced", "correct": bool(c.correct), "E_J": c.E_J,
                        "think_tok": think_len, "ans_tok": c.answer_tokens, "ans_cap": bool(c.answer_hit_cap)})
        else:
            out.append({**base, "budget": "full+forced", **unc})
        o = off.loc[(pid, trial)]
        out.append({**base, "budget": "off", "correct": bool(o.correct), "E_J": o.E_J, "think_tok": 0,
                    "ans_tok": o.output_len, "ans_cap": False})
    info = {"name": Path(caps_dir).name, "gpu": c_sess["gpu_name"], "uuid": c_sess["gpu_uuid"][-8:],
            "status": caps_run["status"], "n_on": len(on), "ks": ks,
            "gpu_tag": (re.search(r"_(gpu\d+)_", Path(caps_dir).name) or re.search("(.*)", "GPU")).group(1).upper()}
    return pd.DataFrame(out), info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy", nargs=2, action="append", required=True, metavar=("B2B_RUN", "CAPS_RUN"))
    ap.add_argument("--out", default="Caps_summary")
    ap.add_argument("--title", default="Fixed thinking caps (Qwen3-8B)")
    a = ap.parse_args()

    parts, infos = zip(*[load_copy(s, c) for s, c in a.copy])
    df = pd.concat(parts, ignore_index=True)
    ks = infos[0]["ks"]
    order = ["off"] + [f"≤{k // 1024}k" for k in ks] + ["full", "full+forced"]

    md = [f"# {a.title}", ""]
    for i, inf in enumerate(infos):
        n = df[(df.copy_id == inf["name"]) & (df.budget == "full")].shape[0]
        md.append(f"- Copy {i + 1}: `{inf['name']}` ({inf['status']}), GPU {inf['gpu']} …{inf['uuid']}: "
                  f"{n} of {inf['n_on']} traces done")
    md += ["", "## Accuracy and energy per thinking budget", "",
           "| Budget | Accuracy, all copies (95% range) | "
           + " | ".join(f"Copy {i + 1}: acc / J per query / J per correct" for i in range(len(infos)))
           + " | Mean thinking tok | Mean answer tok | Answer hit cap |",
           "|---|---|" + "---|" * len(infos) + "---|---|---|"]
    summ = {}
    for b in order:
        t = df[df.budget == b]
        k, n = int(t.correct.sum()), len(t)
        lo, hi = wilson(k, n)
        cells = []
        for inf in infos:
            u = t[t.copy_id == inf["name"]]
            kc = int(u.correct.sum())
            summ[(inf["name"], b)] = (u.E_J.mean(), u.correct.mean(), wilson(kc, len(u)))
            cells.append(f"{100 * u.correct.mean():.0f}% / {u.E_J.mean() / 1000:.1f} kJ / "
                         + (f"{u.E_J.sum() / kc / 1000:.1f} kJ" if kc else "–"))
        md.append(f"| {b} | {k}/{n} = {100 * k / n:.0f}% ({100 * lo:.0f}–{100 * hi:.0f}) | " + " | ".join(cells)
                  + f" | {t.think_tok.mean():.0f} | {t.ans_tok.mean():.0f} | {int(t.ans_cap.sum())} |")

    groups = sorted(df.group.unique(), key=lambda g: int(re.match(r"L?(\d+)", g).group(1)))
    md += ["", "## Accuracy by difficulty group (all copies)", "",
           "| Group | Traces | " + " | ".join(order) + " |", "|---|---|" + "---|" * len(order)]
    for g in groups:
        t = df[df.group == g]
        cells = [f"{100 * t[t.budget == b].correct.mean():.0f}%" for b in order]
        md.append(f"| {g} | {len(t[t.budget == 'full'])} | " + " | ".join(cells) + " |")
    md += ["", "## Energy per query by difficulty group (kJ, per copy)", "",
           "| Group | Copy | " + " | ".join(order) + " |", "|---|---|" + "---|" * len(order)]
    for g in groups:
        for i, inf in enumerate(infos):
            t = df[(df.group == g) & (df.copy_id == inf["name"])]
            md.append(f"| {g} | {i + 1} | " + " | ".join(f"{t[t.budget == b].E_J.mean() / 1000:.1f}" for b in order)
                      + " |")

    md += ["", "Notes:",
           "- ≤k: thinking stopped after k tokens, then `</think>` forced and the answer sampled (max 4,096 tokens). "
           "Traces that finish thinking before k are not cut and keep their own answer.",
           "- Thinking energy at a cap is the energy counter read at token k during the uncapped run itself (same "
           "GPU), plus the energy of the forced answer.",
           "- full+forced: runs whose thinking never ended get an answer forced at the token limit instead of "
           "no answer.",
           "- Energy is only compared within a copy (each copy ran on one GPU; GPUs differ by ~8–10%)."]
    write_md(RESULTS / "tables" / f"{a.out}.md", md)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, inf in enumerate(infos):
        pts = [summ[(inf["name"], b)] for b in order if b != "full+forced"]
        x = [p[0] / 1000 for p in pts]
        ax1.plot(x[1:], [100 * p[1] for p in pts[1:]], "-o", color=SERIES[i % 3], lw=2, ms=7, mec="white", mew=2,
                 label=f"copy {i + 1} ({inf['gpu_tag']})")
        ax1.plot(x[:1], [100 * pts[0][1]], "s", color=SERIES[i % 3], ms=8, mec="white", mew=2)
        for xi, p, b in zip(x, pts, [b for b in order if b != "full+forced"]):
            ax1.vlines(xi, 100 * p[2][0], 100 * p[2][1], color=SERIES[i % 3], lw=1, alpha=0.4)
            if i == 0:
                ax1.annotate(b, (xi, 100 * p[1]), textcoords="offset points", xytext=(0, 9), ha="center",
                             fontsize=8, color=INK_2)
    style(ax1, "Energy per query (kJ)", "Accuracy (%)")
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    ax1.set_title("Accuracy vs energy (square: thinking off)", fontsize=10, color=INK, loc="left")

    width = 0.8 / len(groups)
    for j, g in enumerate(groups):
        t = df[df.group == g]
        acc = [100 * t[t.budget == b].correct.mean() for b in order[:-1]]
        ax2.bar([i + (j - (len(groups) - 1) / 2) * width for i in range(len(acc))], acc, width,
                color=SERIES[j % 3], label=g)
    ax2.set_xticks(range(len(order) - 1), order[:-1], fontsize=8)
    style(ax2, "Thinking budget", "Accuracy (%)")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    ax2.set_title("Accuracy by difficulty (all copies)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / f"{a.out}.png", dpi=150)
    print(f"Wrote {RESULTS / 'figures' / f'{a.out}.png'}")


if __name__ == "__main__":
    main()
