"""Analyse a b1/clear_original/run.sh run: the CLEAR authors' scripts, run back to back in one session.

%Capture as in the paper (section 4.2; Table 12 rows, recomputed exactly from their numbers):
  Captured(Model) = n_layers x block + embedding + CLS/LM head;  %Capture(Model) = Captured(Model) / full model
  %Capture(Block) = (attention + FFN-intermediate + FFN-output) / block   (Table 12 reports no norm row)
Compared with the paper (CLEAR Tables 8 and 12, from b1/analyze_b1.py) and with our b1/run_b1.py run in the same
session (the control), whose %Capture is computed the same way plus with the 12 layers replayed in sequence.

Usage:
  python b1/clear_original/analyze.py results/runs/B1clear/<run_id>     # table -> results/tables/B1clear_<run_id>.md
  python b1/clear_original/analyze.py --one <run dir>/scripts/r1/L32     # one-line summary (used by run.sh)
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from b1.analyze_b1 import CLEAR_TABLE8, CLEAR_TABLE12  # noqa: E402

N_LAYERS = 12
# their JSON keys -> our component names
KEYS = {"embeddings": "embedding", "encoder.attention": "attention", "encoder.intermediate": "ffn_intermediate",
        "encoder.output": "ffn_output", "cls": "cls_head"}


def capture(c: dict) -> dict:
    model = N_LAYERS * c["block"] + c["embedding"] + c["cls_head"]
    out = {"captured_model": model, "pct_model": 100 * model / c["full_model"],
           "pct_block": 100 * (c["attention"] + c["ffn_intermediate"] + c["ffn_output"]) / c["block"]}
    if "layers_cycled" in c:
        out["pct_model_cycled"] = 100 * (c["layers_cycled"] + c["embedding"] + c["cls_head"]) / c["full_model"]
    return out


def load_one(w: Path) -> dict:
    """Mean mJ per call and trial spread (std/mean, %) of each component in one work dir."""
    raw = {}
    for name in ("block", "parts", "full"):
        for k, v in json.loads((w / f"{name}.json").read_text()).items():
            raw["block" if k.startswith("layer_") else KEYS.get(k, k)] = v
    mean = {k: v["average_mJ"] for k, v in raw.items()}
    spread = {k: 100 * v["std_dev_mJ"] / v["average_mJ"] for k, v in raw.items()}
    return {"mean": mean, "spread": spread}


def load_control(run_dir: Path) -> dict:
    """Mean mJ per call per (seq_len, component) from our b1/run_b1.py measurements."""
    vals = {}
    for line in open(run_dir / "measurements.jsonl"):
        r = json.loads(line)
        vals.setdefault(r["seq_len"], {}).setdefault(r["component"], []).append(
            (r["E_end_mJ"] - r["E_start_mJ"]) / r["N"])
    return {L: {c: st.mean(v) for c, v in comps.items()} for L, comps in vals.items()}


def rel(path: Path) -> str:
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def pct(x) -> str:
    return "-" if x is None else f"{x:.1f}%"


def paper(L: int) -> dict | None:
    t = CLEAR_TABLE8 if L == 8 else CLEAR_TABLE12.get(L)
    return {k: v[0] for k, v in t.items()} if t else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--one", action="store_true", help="run_dir is one work dir; print a one-line summary")
    a = p.parse_args()

    if a.one:
        d = load_one(Path(a.run_dir))
        m, c = d["mean"], capture(d["mean"])
        print(f"block {m['block']:.3f} mJ, full {m['full_model']:.3f} mJ, %Capture(Model) {c['pct_model']:.1f}%, "
              f"%Capture(Block) {c['pct_block']:.1f}%")
        return

    run_dir = Path(a.run_dir).resolve()
    run = json.loads((run_dir / "run.json").read_text())
    rounds = sorted((run_dir / "scripts").glob("r*"), key=lambda q: int(q.name[1:]))
    by_len: dict[int, list[dict]] = {}
    for r in rounds:
        for w in sorted(r.glob("L*"), key=lambda q: int(q.name[1:])):
            if (w / "full.json").exists():
                by_len.setdefault(int(w.name[1:]), []).append({"round": int(r.name[1:]), **load_one(w)})

    round_ids = sorted({r["round"] for rs in by_len.values() for r in rs})
    ctrl_file = run_dir / "control_run_dir"
    ctrl_dir = Path(ctrl_file.read_text().strip()) if ctrl_file.exists() else None
    ctrl = load_control(ctrl_dir) if ctrl_dir and (ctrl_dir / "measurements.jsonl").exists() else {}

    md = [f"# B1 with the CLEAR authors' scripts, one session: `{run['run_id']}`", "",
          f"GPU {run['gpu_name']} (nvml {run['nvml_index']}), driver {run['driver']}, idle {run['idle']['mean_W']:.1f} W; "
          f"torch {run['versions'].get('torch')}, transformers {run['versions'].get('transformers')}; status {run['status']}.",
          f"Scripts: `clear_repo/` (sha256 prefixes {run['clear_scripts_sha256_prefix']}), patched only to run "
          f"(diffs in `scripts/r*/L*/*.diff`). Control (our `b1/run_b1.py`, same session): "
          f"`{rel(ctrl_dir) if ctrl_dir else 'none'}`.", "",
          "## %Capture(Model) = 12 x block + embedding + head, / full model", "",
          "| Tokens | " + " | ".join(f"CLEAR code, round {i}" for i in round_ids) +
          " | CLEAR code, mean | CLEAR paper | Ours (control), 12 x block | Ours (control), 12 layers in sequence |",
          "|---|" + "---|" * (len(round_ids) + 4)]
    for L, rs in sorted(by_len.items()):
        by_round = {r["round"]: capture(r["mean"])["pct_model"] for r in rs}  # a stopped run can miss some rounds
        ref = paper(L)
        c = capture(ctrl[L]) if L in ctrl else {}
        cells = [pct(capture(ref)["pct_model"] if ref else None), pct(c.get("pct_model")),
                 pct(c.get("pct_model_cycled"))]
        md.append(f"| {L} | " + " | ".join(pct(by_round.get(i)) for i in round_ids) +
                  f" | **{pct(st.mean(by_round.values()))}** | " + " | ".join(cells) + " |")

    comps = ["embedding", "attention", "ffn_intermediate", "ffn_output", "block", "cls_head", "full_model"]
    for L, rs in sorted(by_len.items()):
        ref = paper(L) or {}
        md += ["", f"## {L} tokens: mJ per call (mean of rounds; spread across rounds; trial spread in the worst round)", "",
               "| Component | CLEAR code | across rounds | trial std/mean | CLEAR paper | Ours (control) |",
               "|---|---|---|---|---|---|"]
        for k in comps:
            v = [r["mean"][k] for r in rs]
            across = 100 * st.pstdev(v) / st.mean(v) if len(v) > 1 else 0.0
            worst = max(r["spread"][k] for r in rs)
            md.append(f"| {k} | {st.mean(v):.3f} | {across:.1f}% | {worst:.1f}% | "
                      f"{ref.get(k, float('nan')):.3f} | {ctrl.get(L, {}).get(k, float('nan')):.3f} |")
        ref_blk = capture(ref)["pct_block"] if ref else None
        ctrl_blk = capture(ctrl[L])["pct_block"] if L in ctrl else None
        md.append(f"| %Capture(Block), attn + FFN int. + FFN out | "
                  f"{pct(st.mean(capture(r['mean'])['pct_block'] for r in rs))} | | | {pct(ref_blk)} | {pct(ctrl_blk)} |")

    out = run_dir.parents[2] / "tables" / f"B1clear_{run['run_id']}.md"  # <results>/runs/B1clear/<id> -> <results>/tables
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
