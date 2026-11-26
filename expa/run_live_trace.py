"""Live decode of a saved trace, energy every `--window` tokens (Phase 3: live vs replayed step; KV-cache ablation).

Teacher-forced decoding: the saved answer's tokens are fed one forward pass per token, each with the logits of that
token, exactly the work of a decode step in generate() minus sampling. Same trace and GPU as Exp A, so the energy per
token of window [L, L + W) can be compared directly with Exp A's replayed full step at context L.

  1. Prefill the prompt (chunked, unmeasured).
  2. For each window of W tokens: synced counter read, W one-token forward passes, synced counter read.

--cache inplace uses llm/inplace_cache.py (no torch.cat copy of the cache each step) instead of DynamicCache, so
the same run measures what the KV-cache change saves in real decoding. --resume rebuilds the cache (unmeasured) up
to the first missing window and goes on.

Usage:
  python expa/run_live_trace.py --source results/runs/B2b_aime/<run> --problem aime2025/I-11 --trial 0 --cache dynamic
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from energy import GpuBusy, Recorder, get_meter  # noqa: E402
from expa.run_decode_components import pick_trace  # noqa: E402
from llm import qwen  # noqa: E402
from llm.inplace_cache import InplaceCache  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="B2b run dir with saved traces")
    p.add_argument("--problem", required=True)
    p.add_argument("--trial", type=int)
    p.add_argument("--mode", choices=["on", "off"], default="on")
    p.add_argument("--cache", choices=["dynamic", "inplace"], default="dynamic")
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--max-len", type=int, help="stop at this many tokens in the cache (default: whole trace)")
    p.add_argument("--model", default="models/Qwen--Qwen3-8B")
    p.add_argument("--model-id", default="Qwen/Qwen3-8B")
    p.add_argument("--dtype", choices=qwen.DTYPES, default="bf16")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--prefill-chunk", type=int, default=8192)
    p.add_argument("--idle-s", type=float, default=30.0)
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--tag", default="")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.inference_mode()
def main():
    a = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    src = Path(a.source)
    tr = pick_trace(src, a.problem, a.trial, a.mode)
    ids = tr["prompt_ids"] + tr["output_ids"]
    P = len(tr["prompt_ids"])
    end = min(len(ids), a.max_len or len(ids))

    meter = get_meter(False)
    others = meter.other_processes()
    if others and not a.allow_busy:
        sys.exit(f"GPU in use by other processes {others}; pick a free GPU or --allow-busy")
    meter.strict = not a.allow_busy
    print(f"GPU: {meter.name}; trace {tr['problem_id']} t{tr['trial']}: prompt {P} + {len(tr['output_ids'])} "
          f"tokens; live decode to {end}, cache {a.cache}", flush=True)

    tok, model = qwen.load_model(a.model, a.dtype, a.attn, a.device)
    print(f"Measuring idle power for {a.idle_s:.0f} s ...", flush=True)
    idle = meter.measure_idle(a.idle_s)
    print(f"  idle: {idle['mean_W']:.1f} W", flush=True)

    config = {"source_run": json.loads((src / "run.json").read_text())["run_id"], "problem_id": tr["problem_id"],
              "trace_mode": tr["mode"], "trace_trial": tr["trial"], "prompt_len": P, "end": end,
              "window": a.window, "cache": a.cache, "dtype": a.dtype, "attn_impl": a.attn, "batch_size": 1,
              "method": "teacher-forced decode, one forward per token, logits_to_keep=1; synced reads per window"}
    experiment = "Live_trace" + (f"_{a.tag}" if a.tag else "")
    rec = Recorder(experiment, config, qwen.model_info(model, a.model_id, a.model, a.dtype), meter, idle,
                   resume_dir=a.resume)
    done = {k[0] for k in rec.done_keys(["window_start"])}
    print(f"Run dir: {rec.dir}", flush=True)

    starts = list(range(P, end - a.window + 1, a.window))
    todo = [s for s in starts if s not in done]
    status = "failed"
    try:
        if todo:
            dev_ids = torch.tensor([ids[:end]], device=a.device)
            cache = InplaceCache(model.config.num_hidden_layers, end + 16) if a.cache == "inplace" else DynamicCache()
            first = todo[0]
            for i in range(0, first, a.prefill_chunk):  # prompt (+ already measured windows), unmeasured
                model(input_ids=dev_ids[:, i:min(first, i + a.prefill_chunk)], past_key_values=cache, use_cache=True,
                      logits_to_keep=1)
            for i in range(first, first + 32):  # warm-up: 32 decode steps, unmeasured, then cut back
                model(input_ids=dev_ids[:, i:i + 1], past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache.crop(first)
            meter.sync()
            assert cache.get_seq_length() == first
            t_run = time.time()
            for s in starts:
                if s < first:
                    continue
                e0, t0 = meter.read()
                for i in range(s, s + a.window):
                    model(input_ids=dev_ids[:, i:i + 1], past_key_values=cache, use_cache=True, logits_to_keep=1)
                e1, t1 = meter.read()
                if s in done:
                    continue
                rec.write({"experiment": experiment, "method": "live_window", "window_start": s,
                           "window_end": s + a.window, "n_tokens": a.window, "E_start_mJ": e0, "E_end_mJ": e1,
                           "t_start_s": t0, "t_end_s": t1, **meter.state()})
                if (s - P) % (8 * a.window) == 0:
                    print(f"  L={s:6d}: {(e1 - e0) / a.window / 1e3:6.3f} J/token, "
                          f"{1e3 * (t1 - t0) / a.window:6.2f} ms/token  ({time.time() - t_run:5.0f} s)", flush=True)
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
