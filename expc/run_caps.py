"""Fixed thinking caps on saved thinking-on traces: "think at most k tokens, then answer" (Experiment C, live budgets).

Unlike expc/run_expc.py (which replays the thinking to measure it), this script takes the thinking energy from the
source run itself: b2/run_b2b.py reads the energy counter during generation at k = 1, 2, 4, ..., 512, then every 512
tokens (B3-long checkpoints). E(k) from that run is exactly the energy of a capped run up to its k-th thinking token
(prefill included), because sampling with the same seed gives the same first k tokens whether or not a cap is set.
So the source run must come from the same GPU and session setup as this one (checked: same GPU UUID).

For each thinking-on trace and each cap k below its thinking length:
  1. Rebuild the KV cache for prompt + first k-1 thinking tokens in a few large forward passes. Unmeasured: in a real
     capped run this cache already exists when the cap is reached.
  2. Feed token k plus the forced end of thinking (`--end-text`) and sample the answer with model.generate(), inside
     a measurement window (raw NVML + Zeus).
  3. Total energy at cap k = E(k) from the source run + answer energy.
Caps at or above the thinking length are not cut: the result is the source run's own answer (the analysis uses it).

Traces whose thinking never ended (hit the token limit without </think>) also get label "limit": the answer forced
at the end of all their thinking, to see whether forcing an answer rescues them.

Usage:
  CUDA_VISIBLE_DEVICES=1 python expc/run_caps.py --source results/runs/B2b_aime/<run_id>
  CUDA_VISIBLE_DEVICES=1 python expc/run_caps.py --source ... --caps 2048 4096 --limit 1 --tag test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import DynamicCache, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from energy import GpuBusy, Recorder, get_meter  # noqa: E402
from energy.record import RESULTS  # noqa: E402
from energy.window import Window  # noqa: E402
from llm import qwen  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="b2/run_b2b.py run dir (thinking on, with checkpoints)")
    p.add_argument("--model", default="models/Qwen--Qwen3-8B")
    p.add_argument("--model-id", default="Qwen/Qwen3-8B")
    p.add_argument("--dtype", choices=qwen.DTYPES, default="bf16")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--caps", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    p.add_argument("--no-limit", action="store_true", help="skip the forced answer at the end of unfinished traces")
    p.add_argument("--end-text", default="\n</think>\n\n")
    p.add_argument("--max-answer", type=int, default=4096, help="max new tokens after the forced end")
    p.add_argument("--prefill-chunk", type=int, default=8192)
    p.add_argument("--trials", type=int, nargs="+")
    p.add_argument("--problems", nargs="+")
    p.add_argument("--limit", type=int, help="only the first n traces")
    p.add_argument("--no-zeus", action="store_true")
    p.add_argument("--idle-s", type=float, default=30.0)
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--allow-other-gpu", action="store_true",
                   help="source run from another GPU (tests only: E(k) and answer energy then do not add up)")
    p.add_argument("--resume")
    p.add_argument("--tag", default="")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_source(src: Path) -> list[dict]:
    meas = {}
    for line in open(src / "measurements.jsonl"):
        r = json.loads(line)
        meas[(r["problem_id"], r["mode"], r["trial"])] = r
    out = []
    for line in open(src / "traces.jsonl"):
        t = json.loads(line)
        if t["mode"] != "on":
            continue
        m = meas[(t["problem_id"], "on", t["trial"])]
        out.append({**t, "row": m})
    return out


@torch.inference_mode()
def build_cache(model, ids: list[int], chunk: int, device: str) -> DynamicCache:
    cache = DynamicCache()
    for i in range(0, len(ids), chunk):
        model(input_ids=torch.tensor([ids[i:i + chunk]], device=device), past_key_values=cache, use_cache=True,
              logits_to_keep=1)
    return cache


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    src = Path(a.source)
    src_run = json.loads((src / "run.json").read_text())
    if src_run["model"]["id"] != a.model_id:
        sys.exit(f"Source traces come from {src_run['model']['id']}, not {a.model_id}")
    sess_file = RESULTS / "sessions" / f"{src_run['session_id']}.json"
    src_sess = json.loads(sess_file.read_text()) if sess_file.exists() else None

    meter = get_meter(False)
    others = meter.other_processes()
    if others and not a.allow_busy:
        sys.exit(f"GPU in use by other processes {others}; pick a free GPU or --allow-busy")
    meter.strict = not a.allow_busy
    same_gpu = src_sess is not None and src_sess["gpu_uuid"] == meter.info()["gpu_uuid"]
    if not same_gpu and not a.allow_other_gpu:
        sys.exit("Source run is from another GPU (or its session file is missing): thinking energy E(k) would come "
                 "from a different card than the answer energy. Use the same GPU, or --allow-other-gpu for tests.")
    print(f"GPU: {meter.name}; source on the same GPU: {same_gpu}", flush=True)

    traces = load_source(src)
    if a.trials is not None:
        traces = [t for t in traces if t["trial"] in set(a.trials)]
    if a.problems:
        traces = [t for t in traces if t["problem_id"] in set(a.problems)]
    if a.limit:
        traces = traces[:a.limit]
    print(f"{len(traces)} thinking-on traces", flush=True)

    tok, model = qwen.load_model(a.model, a.dtype, a.attn, a.device)
    end_ids = tok(a.end_text, add_special_tokens=False).input_ids
    think_end_id = tok.convert_tokens_to_ids("</think>")
    if think_end_id not in end_ids:
        sys.exit(f"--end-text {a.end_text!r} does not contain the </think> token")
    eos_ids = tok.eos_token_id if isinstance(tok.eos_token_id, list) else [tok.eos_token_id]

    window = Window(meter, use_zeus=not a.no_zeus)
    print(f"Measuring idle power for {a.idle_s:.0f} s ...", flush=True)
    idle = meter.measure_idle(a.idle_s)
    print(f"  idle: {idle['mean_W']:.1f} W", flush=True)

    config = {
        "source_run": src_run["run_id"], "source_dir": str(src), "source_same_gpu": same_gpu,
        "source_max_new_on": src_run["config"]["max_new_tokens"]["on"], "subset_file": src_run["config"].get("subset_file"),
        "caps": a.caps, "limit_answer": not a.no_limit, "end_text": a.end_text, "end_ids": end_ids,
        "max_answer": a.max_answer, "decoding": qwen.DECODING["on"], "dtype": a.dtype, "attn_impl": a.attn,
        "batch_size": 1, "zeus": window.zeus is not None, "grader": "math-verify",
        "think_energy": "E(k) checkpoint of the source generation (same GPU); cache rebuilt unmeasured",
    }
    if a.trials is not None:
        config["trials"] = a.trials
    experiment = "Caps" + (f"_{a.tag}" if a.tag else "")
    rec = Recorder(experiment, config, qwen.model_info(model, a.model_id, a.model, a.dtype), meter, idle,
                   resume_dir=a.resume)
    done = rec.done_keys(["problem_id", "trial", "label"])
    answers = open(rec.dir / "answers.jsonl", "a")
    print(f"Run dir: {rec.dir}", flush=True)

    with torch.inference_mode():  # untimed warm-up
        t0 = traces[0]
        c = build_cache(model, t0["prompt_ids"] + t0["output_ids"][:7], a.prefill_chunk, a.device)
        ctx = torch.tensor([t0["prompt_ids"] + t0["output_ids"][:8] + end_ids], device=a.device)
        model.generate(ctx, attention_mask=torch.ones_like(ctx), past_key_values=c,
                       **qwen.generation_kwargs(tok, "on", 16))
        del c
    meter.sync()

    status = "failed"
    try:
        for tr in traces:
            pid, trial, src_row = tr["problem_id"], tr["trial"], tr["row"]
            out_ids, prompt = tr["output_ids"], tr["prompt_ids"]
            ended = src_row["natural_len"] is not None
            think_len = src_row["natural_len"] if ended else len(out_ids)
            ckpt = {c["k"]: c["E_mJ"] for c in src_row["checkpoints"]}
            jobs = [(k, f"cap{k}") for k in a.caps if k < think_len]
            if not ended and not a.no_limit:
                jobs.append((think_len, "limit"))
            for k, label in jobs:
                if (pid, trial, label) in done:
                    continue
                if k not in ckpt:
                    print(f"  skip {pid} t{trial} {label}: no checkpoint at k={k}", flush=True)
                    continue
                cache = build_cache(model, prompt + out_ids[:k - 1], a.prefill_chunk, a.device)
                ctx_list = prompt + out_ids[:k] + end_ids
                ctx = torch.tensor([ctx_list], device=a.device)
                seed = qwen.seed_for(pid, f"cap:{k}", trial)
                set_seed(seed)
                others = meter.other_processes()
                with torch.inference_mode(), window.measure() as w:
                    gen = model.generate(ctx, attention_mask=torch.ones_like(ctx), past_key_values=cache,
                                         **qwen.generation_kwargs(tok, "on", a.max_answer))
                del cache
                ans_ids = gen[0, len(ctx_list):].tolist()
                ans_cap = len(ans_ids) >= a.max_answer and not (ans_ids and ans_ids[-1] in eos_ids)
                ans_text = tok.decode(ans_ids, skip_special_tokens=True)
                g = qwen.grade(ans_text, src_row["gold"])
                E_ans = w["E_end_mJ"] - w["E_start_mJ"]
                rec.write({
                    "experiment": experiment, "problem_id": pid, "level": src_row["level"], "trial": trial,
                    "label": label, "cap": k, "think_len": think_len, "thinking_ended": ended,
                    "prompt_len": len(prompt), "seed": seed,
                    "think_E_mJ": ckpt[k], "answer_E_mJ": E_ans, **{f"answer_{x}": v for x, v in w.items()},
                    "total_E_mJ": ckpt[k] + E_ans, "answer_tokens": len(ans_ids), "answer_hit_cap": ans_cap,
                    "reopened_think": think_end_id in ans_ids,
                    **g, "gold": src_row["gold"], "uncapped_correct": src_row["correct"],
                    **meter.state(), "other_processes": others,
                })
                answers.write(json.dumps({"problem_id": pid, "trial": trial, "label": label, "cap": k, "seed": seed,
                                          "answer_ids": ans_ids,
                                          "text": tok.decode(ans_ids, skip_special_tokens=False)}) + "\n")
                answers.flush()
                os.fsync(answers.fileno())
                print(f"t{trial} {pid[:28]:28s} L{src_row['level']} think {think_len:5d}{'' if ended else '+'} "
                      f"{label:9s} think {ckpt[k] / 1000:7.1f} J ans {len(ans_ids):5d} tok {E_ans / 1000:6.1f} J "
                      f"{'OK ' if g['correct'] else 'BAD'}{' CAP' if ans_cap else ''} "
                      f"(uncapped {'OK' if src_row['correct'] else 'BAD'})", flush=True)
        status = "complete"
    except GpuBusy:
        status = "gpu_busy"
        raise
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        answers.close()
        rec.finish(status)
        print(f"Run {status}: {rec.dir}", flush=True)


if __name__ == "__main__":
    main()
