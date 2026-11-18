"""Thinking on vs off at the component level (two expa/run_decode_components.py runs from the same GPU).

1. Equal-context table: at each context measured in both runs, the attention share of one layer
   (attention / whole layer), the attention and MLP shares of a whole decode step (n_layers x module / full step),
   and J per token, thinking on vs off.
2. Per-answer split: every saved answer of the given B2b runs is priced token by token with the per-token
   component energies of its own mode (linear interpolation between measured contexts, held flat outside them),
   so a long answer pays the higher attention cost of its late tokens. Reports the share of the answer's
   generation energy in attention / MLP / norms / LM head + embedding / other, thinking on vs off, and correct vs
   wrong. Prefill (reading the prompt) is not included: only generated tokens.

Usage: python analysis/compare_on_off.py --on results/runs/A_decode/<run> --off results/runs/A_decode_off/<run> \
           --answers results/runs/B2b_aime/<run> [...] --out NAME
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.common import INK, INK_2, RESULTS, SERIES, load_run, style, write_md  # noqa: E402

PARTS = ["attention", "MLP", "norms", "LM head + embed", "other"]


def curves(run_dir):
    run, sess, rows = load_run(Path(run_dir))
    rows["mJ"] = (rows.E_end_mJ - rows.E_start_mJ) / rows.N
    m = rows.groupby(["context", "component"]).mJ.mean().unstack()
    nl = run["config"]["n_layers"]
    c = {"ctx": m.index.to_numpy(dtype=float), "full": m.full_step.to_numpy(), "layer": m.layer.to_numpy(),
         "attention_1": m.attention.to_numpy(), "mlp_1": m.mlp.to_numpy()}
    c["attention"] = nl * m.attention.to_numpy()
    c["MLP"] = nl * m.mlp.to_numpy()
    c["norms"] = nl * (m.input_norm + m.post_norm).to_numpy() + m.final_norm.to_numpy()
    c["LM head + embed"] = (m.lm_head + m.embed).to_numpy()
    c["other"] = c["full"] - sum(c[p] for p in PARTS[:-1])
    return run, sess, c


def price(c, start: int, n: int) -> dict:
    """Energy (mJ) of generating n tokens whose contexts are start, start+1, ..., start+n-1."""
    t = np.arange(start, start + n, dtype=float)
    return {p: float(np.interp(t, c["ctx"], c[p]).sum()) for p in PARTS + ["full"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--on", required=True)
    ap.add_argument("--off", required=True)
    ap.add_argument("--answers", nargs="+", required=True, help="B2b run dirs with traces.jsonl + measurements")
    ap.add_argument("--out", default="A_on_vs_off")
    a = ap.parse_args()

    on_run, on_sess, on = curves(a.on)
    off_run, off_sess, off = curves(a.off)
    same_gpu = on_sess["gpu_uuid"] == off_sess["gpu_uuid"]
    cur = {"on": on, "off": off}

    md = ["# Thinking on vs off, component level (Qwen3-8B)", "",
          f"- Thinking on: `{on_run['run_id']}`, trace `{on_run['config']['problem_id']}`",
          f"- Thinking off: `{off_run['run_id']}`, trace `{off_run['config']['problem_id']}`",
          f"- Same physical GPU: {'yes' if same_gpu else '**no: J/token not comparable, shares only**'} "
          f"({on_sess['gpu_name']} …{on_sess['gpu_uuid'][-8:]}); layer {on_run['config']['layer']}, "
          f"T = {on_run['config']['T']}", "",
          "## 1. Same context, on vs off", "",
          "| Context | Attention % of one layer (on / off) | Attention % of a decode step (on / off) | "
          "MLP % of a decode step (on / off) | J per token (on / off) |", "|---|---|---|---|---|"]
    shared = sorted(set(on["ctx"]) & set(off["ctx"]))
    for L in shared:
        i, j = list(on["ctx"]).index(L), list(off["ctx"]).index(L)
        md.append(f"| {int(L)} | {100 * on['attention_1'][i] / on['layer'][i]:.0f}% / "
                  f"{100 * off['attention_1'][j] / off['layer'][j]:.0f}% | "
                  f"{100 * on['attention'][i] / on['full'][i]:.0f}% / {100 * off['attention'][j] / off['full'][j]:.0f}% | "
                  f"{100 * on['MLP'][i] / on['full'][i]:.0f}% / {100 * off['MLP'][j] / off['full'][j]:.0f}% | "
                  f"{on['full'][i] / 1000:.2f} / {off['full'][j] / 1000:.2f} |")
    only_on = [int(L) for L in on["ctx"] if L not in shared]
    if only_on:
        md.append(f"\nThinking on also measured at {only_on} (thinking-off answers are never that long).")

    # 2. per-answer split
    ans = []
    for d in a.answers:
        meas = {(r["problem_id"], r["mode"], r["trial"]): r
                for r in map(json.loads, open(Path(d) / "measurements.jsonl"))}
        for line in open(Path(d) / "traces.jsonl"):
            t = json.loads(line)
            r = meas[(t["problem_id"], t["mode"], t["trial"])]
            e = price(cur[t["mode"]], len(t["prompt_ids"]), len(t["output_ids"]))
            ans.append({"mode": t["mode"], "correct": bool(r["correct"]), "n": len(t["output_ids"]), **e,
                        "measured": (r["E_end_mJ"] - r["E_start_mJ"])})

    def split_row(label, xs):
        tot = sum(x["full"] for x in xs)
        mean_share = {p: np.mean([x[p] / x["full"] for x in xs]) for p in PARTS}
        return (f"| {label} | {len(xs)} | {np.median([x['n'] for x in xs]):.0f} | "
                + " | ".join(f"{100 * sum(x[p] for x in xs) / tot:.0f}%" for p in PARTS)
                + f" | {100 * mean_share['attention']:.0f}% | {tot / len(xs) / 1e6:.1f} | "
                f"{100 * tot / sum(x['measured'] for x in xs):.0f}% |"), {p: sum(x[p] for x in xs) / tot for p in PARTS}

    md += ["", "## 2. Where each answer's generation energy goes (every saved answer, priced token by token)", "",
           "| Answers | n | Median tokens | " + " | ".join(PARTS) + " | Attention, mean per answer | "
           "Predicted kJ/answer | Predicted ÷ measured |", "|---|---|---|" + "---|" * len(PARTS) + "---|---|---|"]
    bars = []
    for label, sel in [("thinking off", lambda x: x["mode"] == "off"), ("thinking on", lambda x: x["mode"] == "on"),
                       ("on, correct", lambda x: x["mode"] == "on" and x["correct"]),
                       ("on, wrong", lambda x: x["mode"] == "on" and not x["correct"]),
                       ("off, correct", lambda x: x["mode"] == "off" and x["correct"]),
                       ("off, wrong", lambda x: x["mode"] == "off" and not x["correct"])]:
        xs = [x for x in ans if sel(x)]
        if xs:
            row, sh = split_row(label, xs)
            md.append(row)
            bars.append((label, sh))
    md += ["", "Notes:",
           "- Shares in section 2 are energy-weighted (sum over answers); 'mean per answer' averages each answer's own "
           "share.",
           "- Predicted ÷ measured: priced generation energy vs the energy measured for the whole generate() call "
           "(which also includes prefill, sampling and generate() bookkeeping), so it is below 100%.",
           "- Thinking-off per-token costs beyond the longest measured off context are held at that value."]
    write_md(RESULTS / "tables" / f"{a.out}.md", md)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, c, col in [("on", on, SERIES[0]), ("off", off, SERIES[1])]:
        ax1.plot(c["ctx"], 100 * c["attention"] / c["full"], "-o", color=col, lw=2, ms=7, mec="white", mew=2,
                 label=f"thinking {m}: attention")
        ax1.plot(c["ctx"], 100 * c["MLP"] / c["full"], "--s", color=col, lw=1.5, ms=6, mec="white", mew=1.5,
                 label=f"thinking {m}: MLP")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(on["ctx"], [f"{int(L) // 1024}k" if L >= 1024 else str(int(L)) for L in on["ctx"]])
    style(ax1, "Context length (tokens so far)", "Share of one decode step (%)")
    ax1.legend(frameon=False, fontsize=8, loc="center left")
    ax1.set_title("Same context: on vs off", fontsize=10, color=INK, loc="left")

    cols = [SERIES[0], SERIES[1], INK_2, SERIES[2], "#c9c7c1"]
    left = [0.0] * len(bars)
    for p, col in zip(PARTS, cols):
        vals = [100 * sh[p] for _, sh in bars]
        ax2.barh(range(len(bars)), vals, left=left, color=col, label=p, height=0.6)
        for k, (v_, l_) in enumerate(zip(vals, left)):
            if v_ >= 8:
                ax2.text(l_ + v_ / 2, k, f"{v_:.0f}%", ha="center", va="center", fontsize=8, color="white")
        left = [l_ + v_ for l_, v_ in zip(left, vals)]
    ax2.set_yticks(range(len(bars)), [b[0] for b in bars], fontsize=9)
    ax2.invert_yaxis()
    style(ax2, "Share of the answer's generation energy (%)", "")
    ax2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=5)
    ax2.set_title("Whole answers: share of generation energy", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / f"{a.out}.png", dpi=150)
    print(f"Wrote {RESULTS / 'figures' / f'{a.out}.png'}")


if __name__ == "__main__":
    main()
