"""B3 analysis: energy per token vs output tokens (extended CLEAR Fig. 1; baselines.md section 4).

Inputs: B3-short run dirs (forced k = 1..30, CLEAR method) and B2b run dirs (checkpoints E(k) inside real traces).
Outputs (results/figures, results/tables):
  B3-1  E(k)/k vs k for k <= 30: one panel per mode, one line per prompt (shortest/middle/longest), GPU by line style.
  B3-2  E(k)/k vs k on a log x-axis up to the longest trace, combining B3-short (k <= 30) and B3-long (k >= 64).
  B3-3  Delta E per token between consecutive checkpoints vs k, thinking-on traces (does cost per token grow?).
  Table B3: E(1) per prompt, plateau E(k)/k, k where E(k)/k is within 5% of the plateau, per GPU and mode.

Plateau = median energy per token between checkpoints over k in [256, 1024] in the B2b traces (decode cost per
token once prefill is amortised). Only checkpoints k >= 64 are used from B2b; below that one checkpoint interval is
shorter than the counter's update period, and B3-short covers that range.

Usage: python analysis/analyze_b3.py --short results/runs/B3short/<id> --long results/runs/B2b/<id> [...]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.common import INK, RESULTS, SERIES, load_runs, style, write_md  # noqa: E402

MODES = {"off": "Thinking off", "on": "Thinking on"}
ROLES = ["shortest", "middle", "longest"]
LINESTYLES = ["-", "--", ":"]
MIN_LONG_K = 64


def short_table(dirs):
    s = load_runs(dirs)
    s["E_mJ"] = (s.E_end_mJ - s.E_start_mJ) / s.N
    g = s.groupby(["gpu", "mode", "role", "problem_id", "prompt_len", "k"]).E_mJ.agg(
        mean="mean", std=lambda x: x.std(ddof=0)).reset_index()
    g["per_tok"] = g["mean"] / g.k
    return g


def long_points(dirs):
    rows = load_runs(dirs)
    pts = []
    for r in rows.itertuples():
        prev = None
        for c in r.checkpoints:
            if c["k"] < MIN_LONG_K:
                continue
            pts.append({"gpu": r.gpu, "mode": r.mode, "problem_id": r.problem_id, "trial": r.trial, "k": c["k"],
                        "E_mJ": c["E_mJ"], "per_tok": c["E_mJ"] / c["k"],
                        "dE_per_tok": None if prev is None else (c["E_mJ"] - prev["E_mJ"]) / (c["k"] - prev["k"])})
            prev = c
    return pd.DataFrame(pts)


def fig_short(g, out):
    gpus = sorted(g.gpu.unique())
    modes = [m for m in MODES if m in set(g["mode"])]
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 3.8), sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        for gi, gpu in enumerate(gpus):
            for ri, role in enumerate(ROLES):
                d = g[(g.gpu == gpu) & (g["mode"] == mode) & (g.role == role)].sort_values("k")
                if not len(d):
                    continue
                label = f"{role} prompt ({int(d.prompt_len.iloc[0])} tok)" + (f", {gpu}" if len(gpus) > 1 else "")
                ax.plot(d.k, d.per_tok / 1000, LINESTYLES[gi], color=SERIES[ri], lw=2, marker="o", ms=4,
                        label=label)
        style(ax, "Output tokens k", "Energy per token, E(k)/k (J)" if mode == modes[0] else "")
        ax.set_title(MODES[mode], color=INK, fontsize=11, loc="left")
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.suptitle("B3-1: energy per token over the first 30 output tokens (CLEAR Fig. 1 range)", color=INK,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_long(g_short, pts, out):
    gpus = sorted(set(g_short.gpu) | set(pts.gpu if len(pts) else []))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for gi, gpu in enumerate(gpus):
        for mi, mode in enumerate(MODES):
            s = g_short[(g_short.gpu == gpu) & (g_short["mode"] == mode)].groupby("k").per_tok.median()
            p = pts[(pts.gpu == gpu) & (pts["mode"] == mode)] if len(pts) else pts
            if not len(s) and not len(p):
                continue
            lab = MODES[mode] + (f", {gpu}" if len(gpus) > 1 else "")
            if len(p):
                q = p.groupby("k").per_tok.agg(med="median", lo=lambda x: x.quantile(0.25),
                                               hi=lambda x: x.quantile(0.75), n="size")
                q = q[q.n >= 3]
                ax.fill_between(q.index, q.lo / 1000, q.hi / 1000, color=SERIES[mi], alpha=0.15, lw=0)
                ax.plot(q.index, q.med / 1000, LINESTYLES[gi], color=SERIES[mi], lw=2, label=lab + " (traces)")
            if len(s):
                ax.plot(s.index, s / 1000, LINESTYLES[gi], color=SERIES[mi], lw=2, marker="o", ms=4,
                        label=lab + " (forced k)")
    ax.set_xscale("log")
    style(ax, "Output tokens k (log scale)", "Energy per token, E(k)/k (J)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("B3-2: energy per token from 1 token to full traces (median, band = IQR across traces)",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_marginal(pts, out):
    p = pts[(pts["mode"] == "on") & pts.dE_per_tok.notna()] if len(pts) else pts
    if not len(p):
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for gi, gpu in enumerate(sorted(p.gpu.unique())):
        q = p[p.gpu == gpu].groupby("k").dE_per_tok.agg(med="median", lo=lambda x: x.quantile(0.25),
                                                        hi=lambda x: x.quantile(0.75), n="size")
        q = q[q.n >= 3]
        ax.fill_between(q.index, q.lo / 1000, q.hi / 1000, color=SERIES[gi], alpha=0.15, lw=0)
        ax.plot(q.index, q.med / 1000, "-", color=SERIES[gi], lw=2, marker="o", ms=4, label=gpu)
    style(ax, "Output tokens k (end of checkpoint window)", "Energy per token in window, ΔE/Δk (J)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("B3-3: per-token decode energy vs position, thinking on (median, band = IQR)", color=INK,
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return True


def table(g_short, pts):
    md = ["| GPU | Mode | E(1) shortest / middle / longest (J) | Plateau (J/token) | k within 5% of plateau |",
          "|---|---|---|---|---|"]
    for gpu in sorted(set(g_short.gpu) | set(pts.gpu if len(pts) else [])):
        for mode in MODES:
            s = g_short[(g_short.gpu == gpu) & (g_short["mode"] == mode)]
            e1 = " / ".join(f"{s[(s.role == r) & (s.k == 1)]['mean'].mean() / 1000:.2f}" for r in ROLES)
            p = pts[(pts.gpu == gpu) & (pts["mode"] == mode)] if len(pts) else pts
            plateau, k5 = float("nan"), "n/a"
            if len(p):
                w = p[(p.k >= 256) & (p.k <= 1024)].dE_per_tok.dropna()
                if len(w):
                    plateau = w.median()
                    curve = pd.concat([s.groupby("k").per_tok.median(), p.groupby("k").per_tok.median()])
                    within = curve[np.abs(curve / plateau - 1) <= 0.05]
                    k5 = str(int(within.index.min())) if len(within) else "> max k"
            md.append(f"| {gpu} | {mode} | {e1} | {plateau / 1000:.3f} | {k5} |")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", nargs="+", default=[], help="B3short run dirs")
    ap.add_argument("--long", nargs="+", default=[], help="B2b run dirs (checkpoints)")
    ap.add_argument("--out", default="B3")
    a = ap.parse_args()
    figs = RESULTS / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    g_short = short_table(a.short) if a.short else pd.DataFrame(
        columns=["gpu", "mode", "role", "problem_id", "prompt_len", "k", "mean", "std", "per_tok"])
    pts = long_points(a.long) if a.long else pd.DataFrame()

    md = ["# B3: energy per token vs output tokens", "",
          f"Short runs: {', '.join(Path(d).name for d in a.short) or 'none'}",
          f"Long runs (B2b checkpoints): {', '.join(Path(d).name for d in a.long) or 'none'}", ""]
    if len(g_short):
        fig_short(g_short, figs / f"{a.out}-1_short.png")
        md += [f"![B3-1](../figures/{a.out}-1_short.png)", ""]
        md += ["| GPU | Mode | Prompt | k | E(k) mJ | std | E(k)/k mJ |", "|---|---|---|---|---|---|---|"]
        for r in g_short.sort_values(["gpu", "mode", "role", "k"]).itertuples():
            md.append(f"| {r.gpu} | {r.mode} | {r.role} | {r.k} | {r.mean:.1f} | {r.std:.1f} | {r.per_tok:.1f} |")
        md.append("")
    if len(g_short) or len(pts):
        fig_long(g_short, pts, figs / f"{a.out}-2_long.png")
        md += [f"![B3-2](../figures/{a.out}-2_long.png)", ""]
    if fig_marginal(pts, figs / f"{a.out}-3_marginal.png"):
        md += [f"![B3-3](../figures/{a.out}-3_marginal.png)", ""]
    md += ["## Table B3", ""] + table(g_short, pts)
    write_md(RESULTS / "tables" / f"{a.out}.md", md)


if __name__ == "__main__":
    main()
