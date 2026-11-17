"""Experiment A analysis: component energy of one decode step vs context length (expa/run_decode_components.py).

Per context and component: mean energy per call (raw and idle-subtracted), std/mean over trials, time per call, power.
Whole-model shares: attention = n_layers x attention, MLP = n_layers x MLP, norms = n_layers x (2 norms) + final norm,
LM head, embedding; "other" = full step - sum of those.
CLEAR checks: block %Capture = (input_norm + attention + post_norm + mlp) / layer;
model %Capture = (n_layers x layer + embed + final_norm + lm_head) / full_step.

Usage: python analysis/analyze_a_decode.py results/runs/A_decode/<run_id> [--out NAME]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.common import INK, INK_2, RESULTS, SERIES, load_run, style, write_md  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    run, sess, rows = load_run(Path(a.run_dir))
    cfg = run["config"]
    nl = cfg["n_layers"]
    rows["mJ"] = (rows.E_end_mJ - rows.E_start_mJ) / rows.N
    rows["s_call"] = rows.dur_s / rows.N
    rows["mJ_idle"] = rows.mJ - rows.idle_W * rows.s_call * 1000
    g = rows.groupby(["context", "component"])
    s = g.agg(mJ=("mJ", "mean"), sd=("mJ", "std"), mJ_idle=("mJ_idle", "mean"), us=("s_call", "mean"),
              T=("mJ", "size")).reset_index()
    s["us"] *= 1e6
    s["W"] = s.mJ / s.us * 1000
    v = {(r.context, r.component): r for r in s.itertuples()}
    ctxs = sorted(s.context.unique())

    name = a.out or f"A_decode_{run['run_id']}"
    md = [f"# Experiment A: component energy of one decoding step vs context length (Qwen3-8B)", "",
          f"Run `{run['run_id']}` ({run['status']}), GPU {sess['gpu_name']} (`…{sess['gpu_uuid'][-8:]}`), idle "
          f"{sess['idle']['mean_W']:.1f} W. Trace `{cfg['problem_id']}` (thinking on), layer {cfg['layer']} of {nl}, "
          f"{cfg['dtype']}, {cfg['attn_impl']}, T = {cfg['T']}, ~{cfg['target_s']} s per trial.", "",
          "## Energy per call (mJ, raw; std/mean over trials) and time per call", "",
          "| Context | " + " | ".join(cfg["components"]) + " |", "|---|" + "---|" * len(cfg["components"])]
    for L in ctxs:
        md.append(f"| {L} | " + " | ".join(
            f"{v[(L, c)].mJ:.1f} ({100 * v[(L, c)].sd / v[(L, c)].mJ:.0f}%), {v[(L, c)].us:.0f} µs"
            for c in cfg["components"]) + " |")

    md += ["", "## Where one new token's energy goes (whole model)", "",
           "| Context | Full step (J/token) | ms/token | Attention (all layers) | MLP | Norms | LM head | Other | "
           "Block %Capture | Model %Capture |", "|---|---|---|---|---|---|---|---|---|---|"]
    shares = []
    for L in ctxs:
        full = v[(L, "full_step")].mJ
        att, mlp = nl * v[(L, "attention")].mJ, nl * v[(L, "mlp")].mJ
        norms = nl * (v[(L, "input_norm")].mJ + v[(L, "post_norm")].mJ) + v[(L, "final_norm")].mJ
        head = v[(L, "lm_head")].mJ + v[(L, "embed")].mJ
        other = full - att - mlp - norms - head
        blk = (v[(L, "input_norm")].mJ + v[(L, "attention")].mJ + v[(L, "post_norm")].mJ + v[(L, "mlp")].mJ) \
            / v[(L, "layer")].mJ
        mdl = (nl * v[(L, "layer")].mJ + v[(L, "embed")].mJ + v[(L, "final_norm")].mJ + v[(L, "lm_head")].mJ) / full
        shares.append((L, full, att, mlp, norms, head, other))
        md.append(f"| {L} | {full / 1000:.2f} | {v[(L, 'full_step')].us / 1000:.1f} | {100 * att / full:.0f}% | "
                  f"{100 * mlp / full:.0f}% | {100 * norms / full:.0f}% | {100 * head / full:.0f}% | "
                  f"{100 * other / full:+.0f}% | {100 * blk:.1f}% | {100 * mdl:.1f}% |")
    md += ["", "Notes:",
           "- One call = one module for one new token with L tokens already in the KV cache. Attention includes "
           "q/k/v/o projections, q/k norms, RoPE, the DynamicCache concatenation and SDPA.",
           "- Shares multiply layer-" + str(cfg["layer"]) + " values by the number of layers (all layers have the "
           "same shapes; Qwen3-8B uses full attention in every layer).",
           "- 'Other' = full step minus the parts (glue: residual adds, mask/rotary set-up, Python overhead); "
           "negative means the parts sum to more than the whole.",
           "- Full step = model forward for one token with logits; it excludes sampling and generate() bookkeeping."]
    write_md(RESULTS / "tables" / f"{name}.md", md)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for c, col, lab in [("attention", SERIES[0], "attention"), ("mlp", SERIES[1], "MLP"),
                        ("layer", SERIES[2], "whole layer")]:
        ax1.plot(ctxs, [v[(L, c)].mJ for L in ctxs], "-o", color=col, lw=2, ms=7, mec="white", mew=2, label=lab)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ctxs, [f"{L // 1024}k" if L >= 1024 else str(L) for L in ctxs])
    style(ax1, "Context length (tokens in KV cache)", "Energy per new token, one layer (mJ)")
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    ax1.set_title("One layer: attention overtakes the MLP", fontsize=10, color=INK, loc="left")

    bottom = [0.0] * len(shares)
    for idx, col, lab in [(2, SERIES[0], "attention"), (3, SERIES[1], "MLP"), (5, SERIES[2], "LM head + embed"),
                          (4, INK_2, "norms")]:
        vals = [r[idx] / 1000 for r in shares]
        ax2.bar(range(len(shares)), vals, bottom=bottom, color=col, width=0.7, label=lab)
        bottom = [b + x for b, x in zip(bottom, vals)]
    ax2.plot(range(len(shares)), [r[1] / 1000 for r in shares], "_", color=INK, ms=22, mew=2, label="full step")
    ax2.set_xticks(range(len(shares)), [f"{L // 1024}k" if L >= 1024 else str(L) for L in ctxs])
    style(ax2, "Context length", "Energy per new token, whole model (J)")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    ax2.set_title("Whole model: where each token's energy goes", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / f"{name}.png", dpi=150)
    print(f"Wrote {RESULTS / 'figures' / f'{name}.png'}")


if __name__ == "__main__":
    main()
