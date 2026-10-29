"""Tables for the Laya runs (indic_models/run_laya.py), computed only from the raw rows in measurements.jsonl.

Usage:
  python indic_models/analyze_laya.py results/runs/Laya_english_ours/<run> results/runs/Laya_english_clear/<run>
  -> results/tables/Laya_<checkpoint>.md (one section per method, then the two methods side by side)

Checks (energy per call, NVML counter differences only; W = energy / time is a diagnostic):
  Block (layer i)    = (attn_norm + attn + mlp_norm + mlp) / whole layer i
  Model, CLEAR       = (n_layers x layer 5 + embedding + final norm + decision head) / full model   (CLEAR's rule)
  Model, by type     = (n_global x layer 6 + n_sliding x layer 5 + ...) / full model
  Model, parts       = (n_global x parts of layer 6 + n_sliding x parts of layer 5 + ...) / full model
  Model, cycled      = (all layers in sequence, each with its own input + ...) / full model
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PARTS = ["attn_norm", "attn", "mlp_norm", "mlp"]


def load(run_dir):
    run = json.loads((Path(run_dir) / "run.json").read_text())
    vals, secs = defaultdict(list), defaultdict(list)
    states = defaultdict(list)
    with open(Path(run_dir) / "measurements.jsonl") as f:
        for line in f:
            r = json.loads(line)
            k = (r["seq_len"], r["component"])
            vals[k].append((r["E_end_mJ"] - r["E_start_mJ"]) / r["N"])
            secs[k].append((r["t_end_s"] - r["t_start_s"]) / r["N"])
            states[k].append((r["sm_clock_MHz"], r["temp_C"], bool(r.get("clock_reasons", 0) & 0x4)))  # 0x4: SW power cap
    stats = {k: {"mean": np.mean(v), "std": np.std(v), "n": len(v), "us": 1e6 * np.mean(secs[k]),
                 "W": np.sum(v) / np.sum(secs[k]) / 1000 if np.sum(secs[k]) else float("nan"),
                 "MHz": np.median([s[0] for s in states[k]]), "cap": sum(s[2] for s in states[k])}
             for k, v in vals.items()}
    return run, stats


def checks(run, st, L):
    m = run["model"]
    types = m["layer_types"]
    n_g, n_s, n = types.count("global"), types.count("sliding"), m["n_layers"]
    g = lambda c: st[(L, c)]["mean"] if (L, c) in st else float("nan")  # noqa: E731
    parts = {i: sum(g(f"L{i}.{p}") for p in PARTS) for i in (5, 6)}
    rest = g("embedding") + g("final_norm") + g("decision_head")
    full = g("full_model")
    out = {f"Block, layer {i} ({types[i]})": 100 * parts[i] / g(f"L{i}.block") for i in (5, 6)}
    out |= {
        f"Model, CLEAR rule ({n} x layer 5)": 100 * (n * g("L5.block") + rest) / full,
        f"Model, by type ({n_g} x layer 6 + {n_s} x layer 5)": 100 * (n_g * g("L6.block") + n_s * g("L5.block") + rest) / full,
        "Model, parts of layers 5/6": 100 * (n_g * parts[6] + n_s * parts[5] + rest) / full,
        "Model, all layers in sequence": 100 * (g("encoder_cycled") + rest) / full,
    }
    for i in (5, 6):
        out[f"Attention share of layer {i} parts"] = 100 * g(f"L{i}.attn") / parts[i]
    out["Decision head share of model"] = 100 * g("decision_head") / full
    return out


def section(run, st):
    lens = sorted({k[0] for k in st})
    comps = list(dict.fromkeys(k[1] for k in sorted(st, key=lambda k: k[0])))
    c = run["config"]
    lines = [f"## {c['method']}: `{run['run_id']}` (status {run['status']})", "",
             f"warm-up {c['warmup_calls']}, sync before read {c['sync_before_read']}, N {c['N_component']}/{c['N_model']}, "
             f"T {c['T']}, {c['dtype']}. mJ per call: mean (std/mean); µs per call; W = energy/time.", "",
             "| Component | " + " | ".join(f"{L} tok: mJ | µs | W" for L in lens) + " |",
             "|---|" + "---|---|---|" * len(lens)]
    for comp in comps:
        cells = []
        for L in lens:
            s = st.get((L, comp))
            cells.append("- | - | -" if s is None else
                         f"{s['mean']:.3f} ({100 * s['std'] / s['mean']:.1f}%) | {s['us']:.0f} | {s['W']:.0f}")
        lines.append(f"| {comp} | " + " | ".join(cells) + " |")
    lines += ["", "| Check (%) | " + " | ".join(f"{L} tok" for L in lens) + " |", "|---|" + "---|" * len(lens)]
    ck = {L: checks(run, st, L) for L in lens}
    for name in ck[lens[0]]:
        lines.append(f"| {name} | " + " | ".join(f"{ck[L][name]:.1f}" for L in lens) + " |")
    lines += ["", "Power cap (trials with the 300 W software cap active) and median SM clock, largest parts:", ""]
    for comp in ("L5.mlp", "L6.attn", "L6.block", "full_model"):
        lines.append(f"- {comp}: " + ", ".join(
            f"{L}: {st[(L, comp)]['cap']}/{st[(L, comp)]['n']} at {st[(L, comp)]['MHz']:.0f} MHz"
            for L in lens if (L, comp) in st))
    return lines, ck


def main():
    runs = [load(d) for d in sys.argv[1:]]
    if not runs:
        sys.exit(__doc__)
    ckpt = runs[0][0]["config"]["checkpoint"]
    m = runs[0][0]["model"]
    out = [f"# Laya ({m['id']}), component energy with CLEAR's method and ours", "",
           f"{m['encoder']} encoder, {m['n_layers']} layers ({m['layer_types'].count('global')} global, "
           f"{m['layer_types'].count('sliding')} sliding-window {m['sliding_window']}), hidden {m['hidden_size']}, "
           f"{m['n_params_M']} M params, revision {m['revision'][:7]}, laya {m['laya_version']}. Batch 1. "
           "Measured layers: 5 (sliding) and 6 (global).", ""]
    all_ck = {}
    for run, st in runs:
        lines, ck = section(run, st)
        out += lines + [""]
        all_ck[run["config"]["method"]] = ck
    if len(all_ck) > 1:
        lens = sorted(set.intersection(*(set(v) for v in all_ck.values())))
        out += ["## Methods side by side (%)", "", "| Check | " + " | ".join(
            f"{L}: {meth}" for L in lens for meth in all_ck) + " |", "|---|" + "---|" * (len(lens) * len(all_ck))]
        for name in next(iter(next(iter(all_ck.values())).values())):
            out.append(f"| {name} | " + " | ".join(f"{all_ck[meth][L][name]:.1f}" for L in lens for meth in all_ck) + " |")
    path = REPO / "results" / "tables" / f"Laya_{ckpt}.md"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
