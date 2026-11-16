"""Experiment C: cut thinking short (force `</think>`) at budgets inside saved B2b traces; energy and accuracy.

For each saved thinking-on trace whose thinking ended on its own (natural_len known; traces that hit the token cap
are skipped, as planned in todo.md), with budgets b sorted by size:

  1. Replay the trace's saved thinking tokens one forward pass per token (teacher forcing). This is the same work
     as decoding them, but gives back exactly the trace measured in B2b. The counter is read (with sync) at each
     budget point, so E_think(b) = prefill + b thinking tokens, measured once per trace for all budgets.
  2. At each b: copy the KV cache, append the forced end of thinking (`--end-text`, default "\n</think>\n\n")
     after the first b thinking tokens, and sample the answer with model.generate() from the copy, in its own
     measurement window (raw NVML + Zeus). The replay then continues from the uncopied cache. Copying the cache
     and the gaps between windows are in no window.
  3. Budget "full": the trace's own `</think>` is kept and only the answer is regenerated (control at 100%).

  Energy of one budget-forced query = E_think(b) + E_answer(b). Both are raw counter differences; idle power is in
  the session file, not subtracted (as in B2b).

Budgets: pct  25/50/75% of natural_len (labels p25, p50, p75)
         abs  512/1024/2048/4096 tokens, only where b < natural_len (labels t512, ...)
         full natural_len, the trace's own end of thinking
Budgets that round to the same b share one answer; one row is written per label (field `labels` lists all).

Thinking-off answers are not part of this script: run `b2/run_b2b.py --modes off` in the same session.

Usage:
  CUDA_VISIBLE_DEVICES=1 python expc/run_expc.py --source results/runs/B2b/20261019-122926_B2b_gpu0_feea8f05
  CUDA_VISIBLE_DEVICES=1 python expc/run_expc.py --source ... --limit 1 --tag test       # one trace, all budgets
  CUDA_VISIBLE_DEVICES=1 python expc/run_expc.py --source ... --resume results/runs/C/<run_id>
"""

import argparse
import copy
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import DynamicCache, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from energy import GpuBusy, Recorder, get_meter  # noqa: E402
from energy.window import Window  # noqa: E402
from llm import qwen  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="B2b run dir with traces.jsonl and measurements.jsonl")
    p.add_argument("--model", default="models/Qwen--Qwen3-8B")
    p.add_argument("--model-id", default="Qwen/Qwen3-8B", help="recorded in run.json")
    p.add_argument("--dtype", choices=qwen.DTYPES, default="bf16")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--pct", type=int, nargs="*", default=[25, 50, 75], help="budgets in %% of natural_len")
    p.add_argument("--abs", type=int, nargs="*", default=[512, 1024, 2048, 4096], help="budgets in tokens")
    p.add_argument("--end-text", default="\n</think>\n\n", help="appended after the first b thinking tokens")
    p.add_argument("--max-answer", type=int, default=2048, help="max new tokens after the forced end")
    p.add_argument("--trials", type=int, nargs="+", help="only these trial indices of the source")
    p.add_argument("--problems", nargs="+", help="only these problem_ids")
    p.add_argument("--limit", type=int, help="only the first n eligible traces")
    p.add_argument("--no-zeus", action="store_true")
    p.add_argument("--stop-at", help="local time HH:MM; no trace is started that would not finish before it")
    p.add_argument("--idle-s", type=float, default=30.0)
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--tag", default="")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_source(src: Path) -> list[dict]:
    """Thinking-on traces of a B2b run joined with their measurement rows, in the source's order."""
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
        out.append({**t, "natural_len": m["natural_len"], "level": m["level"], "gold": m["gold"],
                    "orig_correct": m["correct"], "orig_output_len": m["output_len"]})
    return out


def budgets_for(natural_len: int, pcts: list[int], abs_: list[int]) -> list[tuple[int, list[str]]]:
    by_b: dict[int, list[str]] = {}
    for p in pcts:
        b = max(1, round(p / 100 * natural_len))
        if b < natural_len:
            by_b.setdefault(b, []).append(f"p{p}")
    for k in abs_:
        if k < natural_len:
            by_b.setdefault(k, []).append(f"t{k}")
    by_b.setdefault(natural_len, []).append("full")
    return sorted(by_b.items())


def deadline(stop_at: str | None) -> float | None:
    if not stop_at:
        return None
    h, m = map(int, stop_at.split(":"))
    t = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    if t.timestamp() <= time.time():
        sys.exit(f"--stop-at {stop_at} is already past")
    return t.timestamp()


@torch.inference_mode()
def feed(model, ids: torch.Tensor, cache: DynamicCache):
    """One forward pass that extends the cache; only the last position's logits are computed, as in generate()."""
    model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    stop_ts = deadline(a.stop_at)
    src = Path(a.source)

    meter = get_meter(False)
    others = meter.other_processes()
    if others and not a.allow_busy:
        sys.exit(f"GPU in use by other processes {others}; pick a free GPU or --allow-busy")
    meter.strict = not a.allow_busy
    print(f"GPU: {meter.name}", flush=True)

    traces = load_source(src)
    n_all = len(traces)
    if a.trials is not None:
        traces = [t for t in traces if t["trial"] in set(a.trials)]
    if a.problems:
        traces = [t for t in traces if t["problem_id"] in set(a.problems)]
    capped = [t for t in traces if t["natural_len"] is None]
    traces = [t for t in traces if t["natural_len"] is not None]
    if a.limit:
        traces = traces[:a.limit]
    print(f"{n_all} thinking-on traces in source; {len(capped)} skipped (token cap, no natural end); "
          f"{len(traces)} to run", flush=True)

    tok, model = qwen.load_model(a.model, a.dtype, a.attn, a.device)
    end_ids = tok(a.end_text, add_special_tokens=False).input_ids
    think_end_id = tok.convert_tokens_to_ids("</think>")
    if think_end_id not in end_ids:
        sys.exit(f"--end-text {a.end_text!r} does not contain the </think> token")

    window = Window(meter, use_zeus=not a.no_zeus)
    print(f"Measuring idle power for {a.idle_s:.0f} s ...", flush=True)
    idle = meter.measure_idle(a.idle_s)
    print(f"  idle: {idle['mean_W']:.1f} W", flush=True)

    src_run = json.loads((src / "run.json").read_text())
    config = {
        "source_run": src_run["run_id"], "source_dir": str(src), "source_model": src_run["model"]["id"],
        "pct": a.pct, "abs": a.abs, "end_text": a.end_text, "end_ids": end_ids, "max_answer": a.max_answer,
        "decoding": qwen.DECODING["on"], "dtype": a.dtype, "attn_impl": a.attn, "batch_size": 1,
        "zeus": window.zeus is not None, "grader": "math-verify",
        "think_energy": "teacher-forced replay of saved tokens, one forward per token, counter read at each b",
        "skipped_capped": [[t["problem_id"], t["trial"]] for t in capped],
    }
    if a.trials is not None:
        config["trials"] = a.trials
    if src_run["model"]["id"] != a.model_id:
        sys.exit(f"Source traces come from {src_run['model']['id']}, not {a.model_id}")
    experiment = "C" + (f"_{a.tag}" if a.tag else "")
    rec = Recorder(experiment, config, qwen.model_info(model, a.model_id, a.model, a.dtype), meter, idle,
                   resume_dir=a.resume)
    done = rec.done_keys(["problem_id", "trial", "label"])
    answers = open(rec.dir / "answers.jsonl", "a")
    print(f"Run dir: {rec.dir}", flush=True)

    # Untimed warm-up: kernels, cuBLAS handles, allocator, cache copy.
    with torch.inference_mode():
        ids = torch.tensor([traces[0]["prompt_ids"]], device=a.device)
        c = DynamicCache()
        feed(model, ids, c)
        feed(model, torch.tensor([[traces[0]["output_ids"][0]]], device=a.device), c)
        c2 = copy.deepcopy(c)
        ctx = torch.tensor([traces[0]["prompt_ids"] + traces[0]["output_ids"][:1] + end_ids], device=a.device)
        model.generate(ctx, attention_mask=torch.ones_like(ctx), past_key_values=c2,
                       **qwen.generation_kwargs(tok, "on", 16))
        del c, c2
    meter.sync()

    durations: list[float] = []
    status = "failed"
    try:
        for tr in traces:
            pid, trial, nat = tr["problem_id"], tr["trial"], tr["natural_len"]
            out_ids, prompt = tr["output_ids"], tr["prompt_ids"]
            assert out_ids[nat] == think_end_id, f"{pid} t{trial}: no </think> at natural_len {nat}"
            budgets = [(b, labels) for b, labels in budgets_for(nat, a.pct, a.abs)
                       if any((pid, trial, lab) not in done for lab in labels)]
            if not budgets:
                continue
            if stop_ts is not None:
                guess = max(durations) if durations else 300.0
                if time.time() + 1.2 * guess > stop_ts:
                    print(f"Stopping: next trace (~{guess:.0f} s) would pass --stop-at", flush=True)
                    status = "stopped"
                    raise StopIteration
            t_trace = time.time()
            others = meter.other_processes()
            cache = DynamicCache()
            think_E = 0
            think_t = 0.0
            pos = 0  # thinking tokens in the cache
            e_seg, t_seg = meter.read()
            feed(model, torch.tensor([prompt], device=a.device), cache)  # prefill, as generate() does first
            for b, labels in budgets:
                # The cache holds prompt + first b-1 thinking tokens; token b-1 goes in with the end text (or, for
                # "full", with nothing: the trace's own </think> is token nat). This matches real generation, where
                # the b-th sampled token is fed together with whatever follows it.
                while pos < b - 1:
                    feed(model, torch.tensor([[out_ids[pos]]], device=a.device), cache)
                    pos += 1
                e, t = meter.read()
                think_E += e - e_seg
                think_t += t - t_seg
                if labels == ["full"]:
                    ctx_list = prompt + out_ids[:nat + 1]
                else:
                    ctx_list = prompt + out_ids[:b] + end_ids
                ctx = torch.tensor([ctx_list], device=a.device)
                c = copy.deepcopy(cache)
                seed = qwen.seed_for(pid, f"C:{b}", trial)
                set_seed(seed)
                with torch.inference_mode(), window.measure() as w:
                    gen = model.generate(ctx, attention_mask=torch.ones_like(ctx), past_key_values=c,
                                         **qwen.generation_kwargs(tok, "on", a.max_answer))
                del c
                ans_ids = gen[0, len(ctx_list):].tolist()
                eos_ids = tok.eos_token_id if isinstance(tok.eos_token_id, list) else [tok.eos_token_id]
                ans_cap = len(ans_ids) >= a.max_answer and not (ans_ids and ans_ids[-1] in eos_ids)
                ans_text = tok.decode(ans_ids, skip_special_tokens=True)
                g = qwen.grade(ans_text, tr["gold"])
                E_ans = w["E_end_mJ"] - w["E_start_mJ"]
                for lab in labels:
                    if (pid, trial, lab) in done:
                        continue
                    rec.write({
                        "experiment": experiment, "problem_id": pid, "level": tr["level"], "trial": trial,
                        "label": lab, "labels": labels, "budget_tokens": b, "natural_len": nat,
                        "budget_frac": b / nat, "prompt_len": len(prompt), "seed": seed,
                        "think_E_mJ": think_E, "think_t_s": think_t,
                        "answer_E_mJ": E_ans, **{f"answer_{k}": v for k, v in w.items()},
                        "total_E_mJ": think_E + E_ans,
                        "answer_tokens": len(ans_ids), "answer_hit_cap": ans_cap,
                        "reopened_think": think_end_id in ans_ids or "<think>" in tok.decode(ans_ids),
                        **g, "gold": tr["gold"], "orig_correct": tr["orig_correct"],
                        **meter.state(), "other_processes": others,
                    })
                answers.write(json.dumps({"problem_id": pid, "trial": trial, "labels": labels, "budget_tokens": b,
                                          "seed": seed, "answer_ids": ans_ids,
                                          "text": tok.decode(ans_ids, skip_special_tokens=False)}) + "\n")
                answers.flush()
                os.fsync(answers.fileno())
                print(f"t{trial} {pid[:30]:30s} L{tr['level']} nat {nat:5d} b {b:5d} {','.join(labels):12s} "
                      f"think {think_E / 1000:7.1f} J ans {len(ans_ids):5d} tok {E_ans / 1000:6.1f} J "
                      f"{'OK ' if g['correct'] else 'BAD'}{' CAP' if ans_cap else ''}", flush=True)
                e_seg, t_seg = meter.read()  # the replay continues from here
            del cache
            durations.append(time.time() - t_trace)
        status = "complete"
    except StopIteration:
        pass
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
