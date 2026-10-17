"""B3-short: E(k) for k = 1..30 forced output tokens, CLEAR's Fig. 1 range (baselines.md section 4).

One generate() of a few tokens takes well under the energy counter's update period, so each (problem, mode, k) is
measured with CLEAR's amplification: the same generate() call is repeated N times between two counter reads, N
chosen so one trial lasts >= --target-s seconds, and repeated T times. Exactly k tokens are forced with
min_new_tokens = max_new_tokens = k. E(k) includes the prefill, as in CLEAR Fig. 1.

The three problems are subset["b3_short"] (shortest, middle and longest prompt of the 30).

Usage:
  python b3/run_b3_short.py --stop-at 19:45
  python b3/run_b3_short.py --ks 1 5 30 --T 2 --tag test
"""

import argparse
import datetime as dt
import math
import sys
import time
from pathlib import Path

import torch
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from energy import Recorder, clear_measure, get_meter  # noqa: E402
from llm import qwen  # noqa: E402

KS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/Qwen--Qwen3-8B")
    p.add_argument("--model-id", default="Qwen/Qwen3-8B")
    p.add_argument("--dtype", choices=qwen.DTYPES, default="bf16")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--modes", nargs="+", choices=["off", "on"], default=["off", "on"])
    p.add_argument("--ks", type=int, nargs="+", default=KS)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--target-s", type=float, default=1.0, help="minimum duration of one trial (sets N)")
    p.add_argument("--pause-s", type=float, default=0.5)
    p.add_argument("--stop-at", help="local time HH:MM; stop before starting a config that would not finish")
    p.add_argument("--idle-s", type=float, default=30.0)
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--tag", default="")
    p.add_argument("--fake-meter", action="store_true", help="dry run without an NVIDIA GPU (numbers meaningless)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    a = parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available (use --fake-meter --device cpu for a dry run)")
    stop_ts = None
    if a.stop_at:
        h, m = map(int, a.stop_at.split(":"))
        stop_ts = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0).timestamp()

    meter = get_meter(a.fake_meter)
    others = meter.other_processes()
    if others and not a.allow_busy:
        sys.exit(f"GPU in use by other processes {others}; pick a free GPU or --allow-busy")
    print(f"GPU: {meter.name}", flush=True)

    subset = qwen.load_subset()
    by_id = {p["problem_id"]: p for p in subset["problems"]}
    picks = [(by_id[b["problem_id"]], b["role"]) for b in subset["b3_short"]]
    tok, model = qwen.load_model(a.model, a.dtype, a.attn, a.device)

    print(f"Measuring idle power for {a.idle_s:.0f} s ...", flush=True)
    idle = meter.measure_idle(a.idle_s)
    print(f"  idle: {idle['mean_W']:.1f} W", flush=True)

    config = {"problems": [p["problem_id"] for p, _ in picks], "modes": a.modes, "ks": a.ks, "T": a.T,
              "target_s": a.target_s, "pause_s": a.pause_s, "decoding": {m: qwen.DECODING[m] for m in a.modes},
              "instruction": qwen.MATH_INSTRUCTION, "dtype": a.dtype, "attn_impl": a.attn, "batch_size": 1,
              "forced_length": "min_new_tokens = max_new_tokens = k", "N": "calibrated per config, >= target_s"}
    experiment = "B3short" + (f"_{a.tag}" if a.tag else "")
    rec = Recorder(experiment, config, qwen.model_info(model, a.model_id, a.model, a.dtype), meter, idle,
                   resume_dir=a.resume)
    done = rec.done_keys(["problem_id", "mode", "k", "trial"])
    print(f"Run dir: {rec.dir}", flush=True)

    status = "failed"
    try:
        for prob, role in picks:
            for mode in a.modes:
                ids = qwen.prompt_ids(tok, prob["problem"], mode, a.device)
                for k in a.ks:
                    todo = [t for t in range(a.T) if (prob["problem_id"], mode, k, t) not in done]
                    if not todo:
                        continue
                    kw = {**qwen.generation_kwargs(tok, mode, k), "min_new_tokens": k}

                    def fn():
                        return model.generate(ids, **kw)

                    # Warm up and time one call, then pick N so a trial lasts >= target_s.
                    set_seed(qwen.seed_for(prob["problem_id"], mode, k))
                    with torch.inference_mode():
                        n_out = fn().shape[1] - ids.shape[1]
                        if n_out != k:
                            sys.exit(f"forced length failed: asked for {k} tokens, got {n_out}")
                        meter.sync()
                        t0 = time.time()
                        for _ in range(3):
                            fn()
                        meter.sync()
                    per_call = (time.time() - t0) / 3
                    N = max(1, math.ceil(a.target_s / per_call))
                    est = len(todo) * (N * per_call + a.pause_s)
                    if stop_ts is not None and time.time() + 1.2 * est > stop_ts:
                        print("Stopping: next config would pass --stop-at", flush=True)
                        status = "stopped"
                        raise StopIteration
                    vals = []
                    for i, row in enumerate(clear_measure(fn, meter, N=N, T=len(todo), pause_s=a.pause_s,
                                                          warmup_calls=0)):
                        row["trial"] = todo[i]
                        rec.write({"experiment": experiment, "method": "clear", "problem_id": prob["problem_id"],
                                   "role": role, "level": prob["level"], "mode": mode, "k": k,
                                   "prompt_len": ids.shape[1], "per_call_s_calibration": per_call, **row})
                        vals.append((row["E_end_mJ"] - row["E_start_mJ"]) / N)
                    v = torch.tensor(vals, dtype=torch.float64)
                    print(f"{role:8s} {mode:3s} k={k:3d} N={N:4d} E(k) {v.mean():9.1f} mJ "
                          f"({100 * v.std(unbiased=False) / v.mean():4.1f}%)  E(k)/k {v.mean() / k:8.1f} mJ/tok",
                          flush=True)
        status = "complete"
    except StopIteration:
        pass
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        rec.finish(status)
        print(f"Run {status}: {rec.dir}", flush=True)


if __name__ == "__main__":
    main()
