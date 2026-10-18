"""B1 analysis (stage 3, validation): per-component mean/std, %Capture, and comparison with CLEAR Tables 8 and 12.

Everything is computed from the raw rows in measurements.jsonl, so new energy definitions only need a change here.

Usage:
  python b1/analyze_b1.py results/runs/B1/<run_id>       # writes results/tables/B1_<run_id>.{md,csv}
  python b1/analyze_b1.py --selftest                     # checks the %Capture formulas on CLEAR's Table 8
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# CLEAR (arXiv:2510.02810) BERT-base, FP16, RTX 6000 Ada. Values in mJ: (mean, std).
CLEAR_TABLE8 = {  # input length not stated in the paper; the released code uses seq_len = 8
    "attention": (13.676, 0.370), "ffn_intermediate": (2.765, 0.192), "ffn_output": (4.944, 0.343),
    "norm_all": (6.216, 0.845), "block": (29.802, 0.607), "embedding": (5.918, 0.307),
    "cls_head": (23.666, 0.979), "full_model": (400.449, 4.805),
}
CLEAR_TABLE8_DERIVED = {"captured_block": 27.601, "pct_capture_block": 92.616,
                        "captured_model": 387.205, "pct_capture_model": 96.693}
CLEAR_TABLE12 = {  # BERT-Base rows; no Norm row is reported here
    32: {"attention": (15.337, 0.398), "ffn_intermediate": (3.472, 0.393), "ffn_output": (5.210, 0.374),
         "block": (32.585, 0.487), "embedding": (6.938, 0.304), "cls_head": (24.876, 0.906),
         "full_model": (450.756, 4.650)},
    64: {"attention": (17.988, 0.466), "ffn_intermediate": (3.695, 0.186), "ffn_output": (5.947, 0.361),
         "block": (34.980, 0.871), "embedding": (8.004, 0.401), "cls_head": (31.174, 0.798),
         "full_model": (465.372, 2.665)},
    96: {"attention": (18.949, 0.237), "ffn_intermediate": (4.628, 0.494), "ffn_output": (7.450, 0.290),
         "block": (39.130, 0.938), "embedding": (7.812, 0.378), "cls_head": (31.497, 0.615),
         "full_model": (523.566, 8.469)},
    128: {"attention": (19.518, 0.358), "ffn_intermediate": (5.626, 0.653), "ffn_output": (8.774, 0.272),
          "block": (41.482, 1.039), "embedding": (7.553, 0.046), "cls_head": (39.772, 0.897),
          "full_model": (557.446, 6.968)},
}
ORDER = ["embedding", "attention", "attention_self", "ffn_intermediate", "ffn_output", "norm_all", "block",
         "layers_cycled",
         "cls_head", "full_model"]


def derived(m: dict, n_layers: int = 12) -> dict:
    """%Capture as defined in baselines.md section 2 (CLEAR section 4.2), from per-component means."""
    out = {}
    if all(k in m for k in ["attention", "ffn_intermediate", "ffn_output", "block"]):
        parts = m["attention"] + m["ffn_intermediate"] + m["ffn_output"]
        if "norm_all" in m:  # CLEAR's definition; LayerNorms are also inside attention and ffn_output
            out["captured_block"] = parts + m["norm_all"]
            out["pct_capture_block"] = 100 * out["captured_block"] / m["block"]
        out["captured_block_no_double_norm"] = parts
        out["pct_capture_block_no_double_norm"] = 100 * parts / m["block"]
    if all(k in m for k in ["block", "embedding", "cls_head", "full_model"]):
        out["captured_model"] = n_layers * m["block"] + m["embedding"] + m["cls_head"]
        out["pct_capture_model"] = 100 * out["captured_model"] / m["full_model"]
    if all(k in m for k in ["layers_cycled", "embedding", "cls_head", "full_model"]):
        # Our definition: the 12 layers replayed in sequence instead of 12 x one layer (cache residency, slot 2).
        out["captured_model_cycled"] = m["layers_cycled"] + m["embedding"] + m["cls_head"]
        out["pct_capture_model_cycled"] = 100 * out["captured_model_cycled"] / m["full_model"]
        if "block" in m:
            out["layers_cycled_over_12x_block"] = m["layers_cycled"] / (n_layers * m["block"])
    return out


def selftest():
    d = derived({k: v[0] for k, v in CLEAR_TABLE8.items()})
    ok = True
    for k, ref in CLEAR_TABLE8_DERIVED.items():
        good = abs(d[k] - ref) < 0.01
        ok &= good
        print(f"{k:22s} ours {d[k]:9.3f}  CLEAR {ref:9.3f}  {'OK' if good else 'MISMATCH'}")
    for L, t in CLEAR_TABLE12.items():
        print(f"Table 12, {L:3d} tokens: %Capture(Model) = {derived({k: v[0] for k, v in t.items()})['pct_capture_model']:.3f}")
    sys.exit(0 if ok else 1)


def load(run_dir: Path):
    run = json.loads((run_dir / "run.json").read_text())
    rows = pd.read_json(run_dir / "measurements.jsonl", lines=True)
    session = json.loads((REPO / "results" / "sessions" / f"{run['session_id']}.json").read_text())
    idle_W = session["idle"]["mean_W"]
    rows["E_mJ"] = (rows.E_end_mJ - rows.E_start_mJ) / rows.N
    rows["dur_s"] = rows.t_end_s - rows.t_start_s
    rows["E_idle_sub_mJ"] = rows.E_mJ - idle_W * rows.dur_s * 1000 / rows.N
    rows["us_per_call"] = rows.dur_s / rows.N * 1e6
    return run, session, rows


def analyze(run_dir: Path):
    run, session, rows = load(run_dir)
    n_layers = run["model"].get("n_layers", 12)
    g = rows.groupby(["seq_len", "component"])
    stats = g.agg(mean_mJ=("E_mJ", "mean"), std_mJ=("E_mJ", lambda s: s.std(ddof=0)), n=("E_mJ", "size"),
                  idle_sub_mean_mJ=("E_idle_sub_mJ", "mean"), us_per_call=("us_per_call", "mean"),
                  temp_max_C=("temp_C", "max"), sm_clock_min_MHz=("sm_clock_MHz", "min")).reset_index()
    stats["cv_pct"] = 100 * stats.std_mJ / stats.mean_mJ
    stats["pass_cv"] = (stats.mean_mJ <= 5) | (stats.cv_pct <= 9.5)

    def clear_ref(r):
        t = CLEAR_TABLE12.get(r.seq_len) if r.seq_len != 8 else CLEAR_TABLE8
        return (t or {}).get(r.component, (None, None))[0]
    stats["clear_mJ"] = stats.apply(clear_ref, axis=1)
    stats["ours_vs_clear_pct"] = 100 * (stats.mean_mJ / stats.clear_mJ - 1)

    out_dir = REPO / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"B1_{run['run_id']}"
    stats.to_csv(out_dir / f"{stem}.csv", index=False)

    md = [f"# B1: CLEAR on BERT-base, run `{run['run_id']}`", "",
          f"- GPU: {session['gpu_name']} (nvml {session['nvml_index']}), driver {session['driver']}, "
          f"idle {session['idle']['mean_W']:.1f} W",
          f"- Config: {json.dumps(run['config'])}", f"- Git: {run['git_commit']} (dirty={run['git_dirty']}), "
          f"status: {run['status']}",
          "- Energies in mJ per call (CLEAR method, idle not subtracted). `vs CLEAR` = ours / CLEAR - 1.",
          "- Reference: seq_len 8 vs CLEAR Table 8 (length not stated in the paper); 32-128 vs Table 12.", ""]
    for L, s in stats.groupby("seq_len"):
        s = s.set_index("component").reindex([c for c in ORDER if c in set(s.component)])
        md += [f"## {L} tokens", "", "| Component | Ours mean | std | std/mean | CLEAR | vs CLEAR | us/call |",
               "|---|---|---|---|---|---|---|"]
        for c, r in s.iterrows():
            ref = "" if pd.isna(r.clear_mJ) else f"{r.clear_mJ:.3f}"
            diff = "" if pd.isna(r.clear_mJ) else f"{r.ours_vs_clear_pct:+.1f}%"
            flag = "" if r.pass_cv else " ⚠"
            md.append(f"| {c} | {r.mean_mJ:.3f} | {r.std_mJ:.3f} | {r.cv_pct:.1f}%{flag} | {ref} | {diff} | "
                      f"{r.us_per_call:.1f} |")
        d = derived(s.mean_mJ.to_dict(), n_layers)
        ref_t = CLEAR_TABLE8 if L == 8 else CLEAR_TABLE12.get(L, {})
        d_ref = derived({k: v[0] for k, v in ref_t.items()}, n_layers) if ref_t else {}
        md += ["", "| Validation | Ours | CLEAR |", "|---|---|---|"]
        for k, v in d.items():
            rv = d_ref.get(k)
            md.append(f"| {k} | {v:.3f} | {'' if rv is None else f'{rv:.3f}'} |")
        md.append("")
    (out_dir / f"{stem}.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nWrote {out_dir / stem}.md and .csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif a.run_dir:
        analyze(Path(a.run_dir))
    else:
        p.error("give a run_dir or --selftest")
