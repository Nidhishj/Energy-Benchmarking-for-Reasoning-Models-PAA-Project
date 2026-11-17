"""Can we tell when to stop thinking? Offline test of stopping rules on the fixed-cap runs (expc/run_caps.py).

For every trace we already have forced answers ("probes") after 2k, 4k, 8k, 16k thinking tokens, plus the uncapped
answer. Each rule picks where to stop; we score its accuracy and energy with the measured numbers:
  - fixed cap k: stop at k (the cap curve, for reference);
  - agree-twice: probe at 2k, 4k, 8k, 16k; stop at the first probe whose answer equals the previous probe's answer
    and use that answer; if no two agree, think to the natural end. Cost = thinking energy up to the stop + the
    energy of every probe answer generated on the way (probes are real extra generations);
  - agree-twice (free probes): the same, without the probe answer cost (if probing were nearly free);
  - oracle: stop at the first probe that is correct (needs the gold answer, so it is only an upper bound on savings).
Energy within one copy only (one GPU each); accuracy pooled.

Usage: python analysis/stop_rules.py --copy <B2b run> <Caps run> [--copy ...] --out NAME
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.analyze_caps import group_of  # noqa: E402
from analysis.common import RESULTS, load_run, write_md  # noqa: E402


def traces(src_dir, caps_dir):
    _, s_sess, src = load_run(Path(src_dir))
    run, c_sess, caps = load_run(Path(caps_dir))
    assert s_sess["gpu_uuid"] == c_sess["gpu_uuid"]
    ks = run["config"]["caps"]
    cidx = {(r.problem_id, r.trial, r.label): r for r in caps.itertuples()}
    for r in src[src["mode"] == "on"].itertuples():
        ended = r.natural_len == r.natural_len and r.natural_len is not None
        think_len = int(r.natural_len) if ended else int(r.output_len)
        probes = []
        for k in ks:
            if k < think_len:
                c = cidx[(r.problem_id, r.trial, f"cap{k}")]
                probes.append({"k": k, "pred": c.pred, "correct": bool(c.correct), "think_E": c.think_E_mJ / 1e3,
                               "ans_E": c.answer_E_mJ / 1e3})
        full = {"pred": r.pred, "correct": bool(r.correct), "E": (r.E_end_mJ - r.E_start_mJ) / 1e3}
        yield {"pid": r.problem_id, "trial": r.trial, "group": group_of(r.problem_id, r.level), "probes": probes,
               "full": full, "think_len": think_len}


def agree_twice(t, pay_probes=True):
    paid = 0.0
    prev = None
    for p in t["probes"]:
        paid += p["ans_E"]
        if p["pred"] is not None and p["pred"] == prev:
            return p["correct"], p["think_E"] + (paid if pay_probes else p["ans_E"]), p["k"]
        prev = p["pred"]
    return t["full"]["correct"], t["full"]["E"] + (paid if pay_probes else 0.0), None


def oracle(t):
    for p in t["probes"]:
        if p["correct"]:
            return True, p["think_E"] + p["ans_E"], p["k"]
    return t["full"]["correct"], t["full"]["E"], None


def fixed(t, k):
    for p in t["probes"]:
        if p["k"] == k:
            return p["correct"], p["think_E"] + p["ans_E"], k
    return t["full"]["correct"], t["full"]["E"], None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy", nargs=2, action="append", required=True)
    ap.add_argument("--out", default="stop_rules")
    a = ap.parse_args()
    copies = [list(traces(s, c)) for s, c in a.copy]
    rules = [("full thinking", lambda t: (t["full"]["correct"], t["full"]["E"], None))]
    rules += [(f"fixed cap {k // 1024}k", (lambda k: lambda t: fixed(t, k))(k)) for k in [4096, 8192, 16384]]
    rules += [("agree-twice (probe cost paid)", agree_twice),
              ("agree-twice (probes free)", lambda t: agree_twice(t, False)),
              ("oracle: first correct probe", oracle)]

    md = ["# Stopping rules on AIME (from the fixed-cap runs)", "",
          "| Rule | Accuracy (all copies) | " + " | ".join(f"Copy {i + 1}: kJ/query" for i in range(len(copies)))
          + " | " + " | ".join(f"Copy {i + 1}: saved vs full" for i in range(len(copies))) + " | Stopped early |",
          "|---|---|" + "---|" * (2 * len(copies)) + "---|"]
    base = [sum(t["full"]["E"] for t in c) / len(c) for c in copies]
    for name, f in rules:
        res = [[f(t) for t in c] for c in copies]
        flat = [r for c in res for r in c]
        acc = sum(r[0] for r in flat) / len(flat)
        E = [sum(r[1] for r in c) / len(c) for c in res]
        early = sum(r[2] is not None for r in flat)
        md.append(f"| {name} | {100 * acc:.0f}% ({sum(r[0] for r in flat)}/{len(flat)}) | "
                  + " | ".join(f"{e / 1e3:.1f}" for e in E) + " | "
                  + " | ".join(f"{100 * (1 - e / b):+.0f}%" for e, b in zip(E, base)) + f" | {early} |")

    md += ["", "## agree-twice by difficulty (probe cost paid)", "", "| Group | Full: acc | Rule: acc | Stopped at 4k / 8k / 16k / not early |",
           "|---|---|---|---|"]
    flat_t = [t for c in copies for t in c]
    for g in sorted({t["group"] for t in flat_t}, key=lambda s: int(s.split("-")[0].lstrip("L"))):
        ts = [t for t in flat_t if t["group"] == g]
        rs = [agree_twice(t) for t in ts]
        stops = [sum(r[2] == k for r in rs) for k in [4096, 8192, 16384]] + [sum(r[2] is None for r in rs)]
        md.append(f"| {g} | {100 * sum(t['full']['correct'] for t in ts) / len(ts):.0f}% | "
                  f"{100 * sum(r[0] for r in rs) / len(rs):.0f}% | {' / '.join(map(str, stops))} |")
    md += ["", "Notes:", "- Probes are the forced answers of `run_caps.py` (answer up to 4,096 tokens each).",
           "- 'Stopped early' = the rule stopped before the model finished thinking on its own.",
           "- Oracle uses the gold answer; no real rule can do better than it with these probe points."]
    write_md(RESULTS / "tables" / f"{a.out}.md", md)


if __name__ == "__main__":
    main()
