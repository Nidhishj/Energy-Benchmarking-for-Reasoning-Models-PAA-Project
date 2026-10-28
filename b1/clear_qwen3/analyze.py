"""Analyse a b1/clear_qwen3/run.sh run: CLEAR's method on Qwen3-8B, laid out like CLEAR Table 1.

  Captured(Block) = attention + MLP + input_layernorm + post_attention_layernorm;  %Capture(Block) = / block
  Captured(Model) = n_layers x block + embedding + LM head + final norm;           %Capture(Model) = / full model

Usage:
  python b1/clear_qwen3/analyze.py results/runs/B1clearQwen3/<run_id>     # table -> results/tables/B1clearQwen3_<run_id>.md
  python b1/clear_qwen3/analyze.py --one <run dir>/L32                      # one-line summary (used by run.sh)
"""

import argparse
import json
import statistics as st
from pathlib import Path

N_LAYERS = 36  # Qwen3-8B
ROWS = [("Attention Block", "attention"), ("MLP", "mlp"), ("Norm. (All)", "norm_all"), ("Captured (Block)", "captured_block"),
        ("Block", "block"), ("% Capture (Block)", "pct_block"), ("Embedding Layer", "embeddings"), ("LM Head", "lm_head"),
        ("Final Layer Norm.", "final_norm"), ("Captured (Model)", "captured_model"), ("Model", "full_model"),
        ("% Capture (Model)", "pct_model")]


def raw(w: Path) -> dict:
    out = {}
    for name in ("block", "parts", "full"):
        for k, v in json.loads((w / f"{name}.json").read_text()).items():
            out["block" if k.startswith("layer_") else k] = v
    return out


def load_one(w: Path, key: str = "average_mJ") -> dict:
    """Per component: mean and std (mJ per call), mean time per call (µs) and average power (W).
    key = "average_with_tail_mJ" (Run A) uses the energy with the queued tail CLEAR's loop misses added back."""
    out = {}
    for k, v in raw(w).items():
        t = st.mean(v["trial_seconds"]) / v["repeat_count"]
        out[k] = {"mean": v[key], "std": v["std_dev_mJ"], "us": t * 1e6, "W": v[key] / 1e3 / t}
    n1, n2 = out["input_layernorm"], out["post_attention_layernorm"]
    out["norm_all"] = {"mean": n1["mean"] + n2["mean"], "std": (n1["std"] ** 2 + n2["std"] ** 2) ** 0.5,
                       "us": n1["us"] + n2["us"], "W": None}
    m = {k: v["mean"] for k, v in out.items()}
    out["captured_block"] = {"mean": m["attention"] + m["mlp"] + m["norm_all"]}
    out["pct_block"] = {"mean": 100 * out["captured_block"]["mean"] / m["block"]}
    out["captured_model"] = {"mean": N_LAYERS * m["block"] + m["embeddings"] + m["lm_head"] + m["final_norm"]}
    out["pct_model"] = {"mean": 100 * out["captured_model"]["mean"] / m["full_model"]}
    return out


def table(by_len: dict, lens: list) -> list:
    md = ["| Component | " + " | ".join(f"{L} tokens: Avg. | Std. Dev." for L in lens) + " |",
          "|---|" + "---|---|" * len(lens)]
    for label, k in ROWS:
        cells = []
        for L in lens:
            v = by_len[L][k]
            fmt = "{:.1f}%" if k.startswith("pct") else "{:.3f}"
            cells += [fmt.format(v["mean"]), f"{v['std']:.3f}" if "std" in v else "-"]
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    return md


PARTS = ["attention", "mlp", "input_layernorm", "post_attention_layernorm", "block", "embeddings", "lm_head",
         "final_norm", "full_model"]


def tail_section(dirs: dict, lens: list) -> list:
    """Run A: the same table with the lost tail added back, and what CLEAR's loop misses per component."""
    md = ["", "## With the queued GPU work CLEAR's loop misses added back", ""]
    md += table({L: load_one(dirs[L], "average_with_tail_mJ") for L in lens}, lens)
    md += ["", "## What CLEAR's loop misses, per component",
           "missed % (counter) = lost energy / (counted + lost), from the energy counter, which updates in steps, so short "
           "tails read as 0 or one whole step; missed % (calls) = calls lost / N, from GPU timing (no steps); "
           "calls lost = lost GPU time / GPU time per call, per trial; "
           "tail in pause = trials whose queued work finished inside the 0.5 s pause (rest spilled into the next trial); "
           "cap = trials with the 300 W power limit active at E_end; MHz = median SM clock at E_end.", "",
           "| Component | " + " | ".join(f"{L}: missed % (counter) | missed % (calls) | calls lost | tail in pause | cap | MHz" for L in lens) + " |",
           "|---|" + "---|---|---|---|---|---|" * len(lens)]
    for k in PARTS:
        cells = []
        for L in lens:
            v = raw(dirs[L])[k]
            tr = v["trials"]
            counted = v["average_mJ"] * v["repeat_count"] * len(tr)
            cells += [f"{100 * v['lost_mJ_total'] / (counted + v['lost_mJ_total']):.2f}",
                      f"{100 * v['calls_lost_per_trial'] / v['repeat_count']:.2f}",
                      f"{v['calls_lost_per_trial']:.0f}",
                      f"{sum(t['tail_done_within_pause'] for t in tr)}/{len(tr)}",
                      f"{sum(t['power_cap'] for t in tr)}/{len(tr)}",
                      f"{st.median(t['sm_MHz'] for t in tr):.0f}"]
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    return md


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--one", action="store_true", help="run_dir is one L<len> dir; print a one-line summary")
    a = p.parse_args()

    if a.one:
        d = load_one(Path(a.run_dir))
        line = (f"block {d['block']['mean']:.1f} mJ, full {d['full_model']['mean']:.1f} mJ, "
                f"%Capture(Block) {d['pct_block']['mean']:.1f}%, %Capture(Model) {d['pct_model']['mean']:.1f}%")
        if "average_with_tail_mJ" in raw(Path(a.run_dir))["block"]:
            t = load_one(Path(a.run_dir), "average_with_tail_mJ")
            line += f"; with tail: Block {t['pct_block']['mean']:.1f}%, Model {t['pct_model']['mean']:.1f}%"
        print(line)
        return

    run_dir = Path(a.run_dir).resolve()
    run = json.loads((run_dir / "run.json").read_text())
    dirs = {int(w.name[1:]): w for w in run_dir.glob("L*") if (w / "full.json").exists()}
    lens = sorted(dirs)
    by_len = {L: load_one(dirs[L]) for L in lens}
    tail = run.get("tail_tracking", False)

    md = [f"# CLEAR's method on Qwen3-8B (FP16): `{run['run_id']}`", "",
          f"GPU {run['gpu_name']} (nvml {run['nvml_index']}), driver {run['driver']}, idle {run['idle']['mean_W']:.1f} W; "
          f"torch {run['versions'].get('torch')}, transformers {run['versions'].get('transformers')}; status {run['status']}.",
          "Method: CLEAR's BERT scripts (`clear_repo/`) adapted to a decoder (`b1/clear_qwen3/`); their measurement loop "
          "unchanged (no warm-up, no sync, 0.5 s pause, N = 10,000 parts / 1,000 model, T = "
          f"{run['trials']}). Layer {run['layer']}. Layout as CLEAR Table 1. Energy in mJ per call.", "",
          "## CLEAR's numbers (their loop, no sync)" if tail else ""]
    md += table(by_len, lens)
    if tail:
        md += tail_section(dirs, lens)

    md += ["", "## Time per call and average power", "",
           "| Component | " + " | ".join(f"{L} tokens: µs | W" for L in lens) + " |", "|---|" + "---|---|" * len(lens)]
    for k in ["attention", "mlp", "input_layernorm", "post_attention_layernorm", "block", "embeddings", "lm_head",
              "final_norm", "full_model"]:
        md.append(f"| {k} | " + " | ".join(f"{by_len[L][k]['us']:.0f} | {by_len[L][k]['W']:.0f}" for L in lens) + " |")

    out = run_dir.parents[2] / "tables" / f"{run.get('experiment', 'B1clearQwen3')}_{run['run_id']}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
