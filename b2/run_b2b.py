"""B2b + B3-long: Qwen3 on MATH500, thinking on vs off (baselines.md sections 3 and 4).

Each generation is one measurement:
  - whole-query energy, read two ways at once: raw NVML counter (method "nvml") and a Zeus window (zeus_mJ);
  - E(k) at output-token checkpoints during the same generate() call (B3-long);
  - accuracy (math-verify), output/thinking/answer token counts, natural length and token-cap flag;
  - the full trace (prompt and output token IDs, text) in traces.jsonl next to measurements.jsonl, so
    Experiment C can truncate exactly the traces that were measured.

Order is trial-major: every problem gets trial 0 before any problem gets trial 1. If a slot ends early, the run
still covers all problems once; `--resume` continues where it stopped.

Usage:
  python b2/run_b2b.py --modes on --T-on 1 --limit 3 --tag speedcheck      # speed check (baselines.md 5)
  python b2/run_b2b.py --modes off on --stop-at 19:45                       # full B2b + B3-long
  python b2/run_b2b.py --resume results/runs/B2b/<run_id> --stop-at 19:45   # continue an interrupted run
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import StoppingCriteriaList, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from energy import Recorder, get_meter  # noqa: E402
from energy.checkpoint import EnergyCheckpoints, checkpoint_grid  # noqa: E402
from energy.window import Window  # noqa: E402
from llm import qwen  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/Qwen--Qwen3-8B")
    p.add_argument("--model-id", default="Qwen/Qwen3-8B", help="recorded in run.json")
    p.add_argument("--dtype", choices=qwen.DTYPES, default="bf16")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--modes", nargs="+", choices=["off", "on"], default=["off", "on"])
    p.add_argument("--T-on", type=int, default=2, help="trials per problem, thinking on")
    p.add_argument("--T-off", type=int, default=3, help="trials per problem, thinking off")
    p.add_argument("--max-new-on", type=int, default=qwen.MAX_NEW_TOKENS["on"])
    p.add_argument("--max-new-off", type=int, default=qwen.MAX_NEW_TOKENS["off"])
    p.add_argument("--limit", type=int, help="only the first n problems of the subset (speed check, dry runs)")
    p.add_argument("--problems", nargs="+", help="only these problem_ids")
    p.add_argument("--no-zeus", action="store_true", help="skip the Zeus cross-check")
    p.add_argument("--no-checkpoints", action="store_true", help="no E(k) reads (to measure their overhead)")
    p.add_argument("--stop-at", help="local time HH:MM; no generation is started that would not finish before it")
    p.add_argument("--idle-s", type=float, default=30.0)
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--tag", default="")
    p.add_argument("--fake-meter", action="store_true", help="dry run without an NVIDIA GPU (numbers meaningless)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def deadline(stop_at: str | None) -> float | None:
    if not stop_at:
        return None
    h, m = map(int, stop_at.split(":"))
    now = dt.datetime.now()
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if t <= now:
        sys.exit(f"--stop-at {stop_at} is already past")
    return t.timestamp()


def main():
    a = parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available (use --fake-meter --device cpu for a dry run)")
    stop_ts = deadline(a.stop_at)

    meter = get_meter(a.fake_meter)
    others = meter.other_processes()
    if others and not a.allow_busy:
        sys.exit(f"GPU in use by other processes {others}; pick a free GPU or --allow-busy")
    print(f"GPU: {meter.name}", flush=True)

    subset = qwen.load_subset()
    problems = subset["problems"]
    if a.problems:
        problems = [p for p in problems if p["problem_id"] in set(a.problems)]
    if a.limit:
        problems = problems[:a.limit]

    tok, model = qwen.load_model(a.model, a.dtype, a.attn, a.device)
    max_new = {"on": a.max_new_on, "off": a.max_new_off}
    T = {"on": a.T_on, "off": a.T_off}

    window = Window(meter, use_zeus=not (a.no_zeus or a.fake_meter))
    print(f"Measuring idle power for {a.idle_s:.0f} s ...", flush=True)
    idle = meter.measure_idle(a.idle_s)
    print(f"  idle: {idle['mean_W']:.1f} W", flush=True)

    config = {
        "dataset": subset["source"], "subset_selection": subset["selection"],
        "problem_ids": [p["problem_id"] for p in problems], "modes": a.modes, "T": T, "max_new_tokens": max_new,
        "decoding": {m: qwen.DECODING[m] for m in a.modes}, "instruction": qwen.MATH_INSTRUCTION,
        "dtype": a.dtype, "attn_impl": a.attn, "batch_size": 1, "zeus": window.zeus is not None,
        "checkpoints": None if a.no_checkpoints else "1,2,4,...,512 then every 512",
        "grader": "math-verify", "order": "mode, then trial, then problem",
    }
    experiment = "B2b" + (f"_{a.tag}" if a.tag else "")
    rec = Recorder(experiment, config, qwen.model_info(model, a.model_id, a.model, a.dtype), meter, idle,
                   resume_dir=a.resume)
    done = rec.done_keys(["problem_id", "mode", "trial"])
    traces = open(rec.dir / "traces.jsonl", "a")
    print(f"Run dir: {rec.dir}", flush=True)

    # Untimed warm-up per mode: CUDA kernels, cuBLAS handles, allocator.
    with torch.inference_mode():
        for mode in a.modes:
            ids = qwen.prompt_ids(tok, problems[0]["problem"], mode, a.device)
            model.generate(ids, **qwen.generation_kwargs(tok, mode, 32))
    meter.sync()

    durations = {"on": [], "off": []}
    status = "failed"
    try:
        for mode in a.modes:
            for trial in range(T[mode]):
                for prob in problems:
                    key = (prob["problem_id"], mode, trial)
                    if key in done:
                        continue
                    if stop_ts is not None:
                        guess = max(durations[mode]) if durations[mode] else (120.0 if mode == "on" else 30.0)
                        if time.time() + 1.2 * guess > stop_ts:
                            print(f"Stopping: next {mode} generation (~{guess:.0f} s) would pass --stop-at", flush=True)
                            status = "stopped"
                            raise StopIteration
                    seed = qwen.seed_for(prob["problem_id"], mode, trial)
                    set_seed(seed)
                    ids = qwen.prompt_ids(tok, prob["problem"], mode, a.device)
                    prompt_len = ids.shape[1]
                    ks = [] if a.no_checkpoints else checkpoint_grid(max_new[mode])
                    ckpt = EnergyCheckpoints(meter, prompt_len, ks)
                    others = meter.other_processes()
                    with torch.inference_mode(), window.measure() as w:
                        out = model.generate(ids, **qwen.generation_kwargs(tok, mode, max_new[mode]),
                                             stopping_criteria=StoppingCriteriaList([ckpt]))
                    out_ids = out[0, prompt_len:].tolist()
                    split = qwen.split_output(tok, out_ids, mode, max_new[mode])
                    g = qwen.grade(split["answer_text"], prob["answer"])
                    dur = w["t_end_s"] - w["t_start_s"]
                    durations[mode].append(dur)
                    row = {
                        "experiment": experiment, "method": "nvml", "problem_id": prob["problem_id"],
                        "level": prob["level"], "mode": mode, "trial": trial, "seed": seed,
                        "prompt_len": prompt_len, **{k: v for k, v in split.items() if k != "answer_text"},
                        **g, "gold": prob["answer"], **w,
                        "checkpoints": [{"k": c["k"], "E_mJ": c["E_mJ"] - w["E_start_mJ"],
                                         "t_s": c["t_s"] - w["t_start_s"]} for c in ckpt.rows],
                        **meter.state(), "other_processes": others,
                    }
                    rec.write(row)
                    traces.write(json.dumps({"problem_id": prob["problem_id"], "mode": mode, "trial": trial,
                                             "seed": seed, "prompt_ids": ids[0].tolist(), "output_ids": out_ids,
                                             "think_end_pos": split["think_end_pos"],
                                             "text": tok.decode(out_ids, skip_special_tokens=False)}) + "\n")
                    traces.flush()
                    os.fsync(traces.fileno())
                    E_J = (w["E_end_mJ"] - w["E_start_mJ"]) / 1000
                    zeus = f" zeus {w['zeus_mJ'] / 1000:8.1f} J" if "zeus_mJ" in w else ""
                    print(f"{mode:3s} t{trial} {prob['problem_id'][:38]:38s} L{prob['level']} "
                          f"{split['output_len']:5d} tok {dur:6.1f} s {E_J:8.1f} J{zeus} "
                          f"{1000 * E_J / max(split['output_len'], 1):7.1f} mJ/tok "
                          f"{'OK ' if g['correct'] else 'BAD'}{' CAP' if split['hit_token_cap'] else ''}",
                          flush=True)
        status = "complete"
    except StopIteration:
        pass
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        traces.close()
        rec.finish(status)
        print(f"Run {status}: {rec.dir}", flush=True)


if __name__ == "__main__":
    main()
