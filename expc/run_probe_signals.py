"""Stopping signals every --every thinking tokens of saved AIME traces (Phase 3, energy-aware stopping rules).

For each thinking-on trace of the source B2b runs, at thinking positions k = Δ, 2Δ, ... before its end of thinking:
  1. Prefill the trace up to prompt + k tokens (chunked, unmeasured, cache kept between probes).
  2. Probe (measured, synced reads): append `--probe-text` (forced end of thinking + "The final answer is \\boxed{")
     in one forward pass, then greedy-decode the answer until the box closes or --max-answer tokens.
     Recorded: entropy of the next token right after `</think>` (EAT), entropy and top-1 probability of the first
     answer token, every answer token's top-1 probability (DEER's geometric mean is computed offline), the answer and
     its grade, the probe's energy and time, and the final-layer hidden state (hidden/<trace>.pt, for a probe).
  3. Crop the probe tokens from the cache and go on.
The signals depend on the trace, not the GPU, so a run can be finished on another card. Stopping rules are then
simulated offline (analysis/stop_rules*.py) with thinking energy priced by the Exp A cost curve.

Usage:
  python expc/run_probe_signals.py --source results/runs/B2b_aime/<copy 1> results/runs/B2b_aime/<copy 2> --stop-at 11:45
"""

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import torch
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from energy import GpuBusy, Recorder, get_meter  # noqa: E402
from expc.run_expc import load_source  # noqa: E402
from llm import qwen  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", nargs="+", required=True, help="B2b run dirs with saved thinking-on traces")
    p.add_argument("--every", type=int, default=1024, help="probe spacing in thinking tokens")
    p.add_argument("--probe-text", default="\n</think>\n\nThe final answer is \\boxed{")
    p.add_argument("--max-answer", type=int, default=24)
    p.add_argument("--limit", type=int, help="first N traces only (testing)")
    p.add_argument("--stop-at", help="HH:MM: finish the current trace and stop")
    p.add_argument("--model", default="models/Qwen--Qwen3-8B")
    p.add_argument("--model-id", default="Qwen/Qwen3-8B")
    p.add_argument("--dtype", choices=qwen.DTYPES, default="bf16")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--prefill-chunk", type=int, default=4096)
    p.add_argument("--idle-s", type=float, default=10.0)
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--tag", default="")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def entropy(logp: torch.Tensor) -> float:
    return float(-(logp.exp() * logp).sum())


@torch.inference_mode()
def probe(model, tok, cache, probe_ids: torch.Tensor, think_pos: int, max_answer: int) -> dict:
    """Forced answer from the current cache (which is left with the probe tokens appended; the caller crops)."""
    out = model(input_ids=probe_ids, past_key_values=cache, use_cache=True, logits_to_keep=probe_ids.shape[1],
                output_hidden_states=True)
    lp = out.logits[0].float().log_softmax(-1)
    rec = {"H_eat": entropy(lp[think_pos]), "H_first": entropy(lp[-1]), "p_first": float(lp[-1].max().exp())}
    hidden = out.hidden_states[-1][0, -1].to(torch.float16).cpu()
    toks, probs, cur = [], [], lp[-1]
    closed = False
    for _ in range(max_answer):
        p, t = cur.max(-1)
        toks.append(int(t))
        probs.append(float(p.exp()))
        text = tok.decode(toks, skip_special_tokens=True)
        if text.count("}") > text.count("{"):
            closed = True
            break
        if int(t) in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")):
            break
        o = model(input_ids=t.view(1, 1), past_key_values=cache, use_cache=True, logits_to_keep=1)
        cur = o.logits[0, -1].float().log_softmax(-1)
    rec.update({"answer_ids": toks, "answer_probs": probs, "answer_tokens": len(toks), "closed": closed,
                "answer_text": tok.decode(toks, skip_special_tokens=True)})
    return rec, hidden


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    stop_ts = None
    if a.stop_at:
        h, m = map(int, a.stop_at.split(":"))
        stop_ts = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0).timestamp()

    traces = []
    for s in a.source:
        rid = json.loads((Path(s) / "run.json").read_text())["run_id"]
        traces += [{**t, "source_run": rid} for t in load_source(Path(s))]
    traces.sort(key=lambda t: (t["problem_id"], t["source_run"]))  # both copies of a problem together
    if a.limit:
        traces = traces[:a.limit]

    meter = get_meter(False)
    others = meter.other_processes()
    if others and not a.allow_busy:
        sys.exit(f"GPU in use by other processes {others}; pick a free GPU or --allow-busy")
    meter.strict = not a.allow_busy
    tok, model = qwen.load_model(a.model, a.dtype, a.attn, a.device)
    probe_ids = tok(a.probe_text, add_special_tokens=False).input_ids
    think_end_id = tok.convert_tokens_to_ids("</think>")
    if think_end_id not in probe_ids:
        sys.exit(f"--probe-text {a.probe_text!r} does not contain </think>")
    think_pos = probe_ids.index(think_end_id)
    print(f"GPU: {meter.name}; {len(traces)} traces; probe every {a.every} tokens; probe = {len(probe_ids)} tokens",
          flush=True)
    idle = meter.measure_idle(a.idle_s)

    config = {"sources": [t for t in a.source], "every": a.every, "probe_text": a.probe_text,
              "probe_ids": probe_ids, "max_answer": a.max_answer, "decoding": "greedy", "dtype": a.dtype,
              "attn_impl": a.attn, "limit": a.limit}
    experiment = "Probe_signals" + (f"_{a.tag}" if a.tag else "")
    rec = Recorder(experiment, config, qwen.model_info(model, a.model_id, a.model, a.dtype), meter, idle,
                   resume_dir=a.resume)
    done = rec.done_keys(["source_run", "problem_id", "trial", "k"])
    (rec.dir / "hidden").mkdir(exist_ok=True)
    print(f"Run dir: {rec.dir}", flush=True)
    dev_probe = torch.tensor([probe_ids], device=a.device)

    status = "failed"
    try:
        for n, tr in enumerate(traces):
            if stop_ts and time.time() > stop_ts:
                status = "stopped"
                print("Stop time reached", flush=True)
                break
            P = len(tr["prompt_ids"])
            ended = tr["natural_len"] is not None
            think_len = tr["natural_len"] if ended else len(tr["output_ids"])
            ks = [k for k in range(a.every, think_len, a.every)
                  if (tr["source_run"], tr["problem_id"], tr["trial"], k) not in done]
            if not ks:
                continue
            ids = torch.tensor([tr["prompt_ids"] + tr["output_ids"]], device=a.device)
            cache, pos, hid = DynamicCache(), 0, {}
            t_tr = time.time()
            for k in ks:
                L = P + k
                while pos < L:
                    m = min(a.prefill_chunk, L - pos)
                    model(input_ids=ids[:, pos:pos + m], past_key_values=cache, use_cache=True, logits_to_keep=1)
                    pos += m
                e0, t0 = meter.read()
                r, hidden = probe(model, tok, cache, dev_probe, think_pos, a.max_answer)
                e1, t1 = meter.read()
                cache.crop(L)
                hid[k] = hidden
                g = qwen.grade("\\boxed{" + r["answer_text"], tr["gold"])
                rec.write({"experiment": experiment, "source_run": tr["source_run"], "problem_id": tr["problem_id"],
                           "trial": tr["trial"], "k": k, "L": L, "prompt_len": P, "think_len": think_len,
                           "thinking_ended": ended, "orig_correct": tr["orig_correct"], "gold": tr["gold"],
                           **r, **g, "E_start_mJ": e0, "E_end_mJ": e1, "t_start_s": t0, "t_end_s": t1})
            key = f"{tr['source_run'][-8:]}_{tr['problem_id'].replace('/', '_')}_t{tr['trial']}"
            hpath = rec.dir / "hidden" / f"{key}.pt"
            if hpath.exists():  # resumed mid-trace: keep the earlier probes' states
                hid = {**torch.load(hpath), **hid}
            torch.save(hid, hpath)
            del cache
            print(f"  [{n + 1}/{len(traces)}] {tr['problem_id']} t{tr['trial']}: {len(ks)} probes, "
                  f"think {think_len} tok, {time.time() - t_tr:.0f} s", flush=True)
        else:
            status = "complete"
    except GpuBusy:
        status = "gpu_busy"
        raise
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        rec.finish(status)
        print(f"Run {status}: {rec.dir}", flush=True)


if __name__ == "__main__":
    main()
