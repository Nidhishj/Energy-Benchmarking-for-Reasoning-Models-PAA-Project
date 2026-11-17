"""Experiment A: CLEAR component energy of ONE decoding step (one new thinking token) at growing context lengths.

For each context length L (tokens already in the KV cache):
  1. Rebuild the real KV cache for the first L tokens of a saved thinking trace (prompt + thinking), unmeasured.
  2. Run the real decode step for token L+1 once with forward hooks, to capture the exact inputs of each module of
     layer `--layer` (hidden states, rotary cos/sin, mask) and of the embedding / final norm / LM head.
  3. CLEAR replay (energy.clear_measure): call each module N times between two synced NVML reads, T trials.
     N is chosen so one trial lasts about `--target-s` seconds.

Attention and the whole layer get a "frozen" cache: its update() returns cat(cached K/V, new K/V) like DynamicCache
does in a real decode step, but stores nothing, so every replay sees exactly L cached tokens. The full step uses the
real DynamicCache and crops it back to L after each call (crop only drops a view, the real step's cat is kept).

Components: input_norm, attention, post_norm, mlp, layer (one decoder layer), embed, final_norm, lm_head,
full_step (model forward for one token with logits for that token, i.e. one decode step without sampling).

Usage:
  CUDA_VISIBLE_DEVICES=1 python expa/run_decode_components.py --source results/runs/B2b_aime/<run> \
      --problem aime2024/II-3 --contexts 256 1024 2048 4096 8192 16384 32768
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from energy import GpuBusy, Recorder, clear_measure, get_meter  # noqa: E402
from llm import qwen  # noqa: E402

COMPONENTS = ["input_norm", "attention", "post_norm", "mlp", "layer", "embed", "final_norm", "lm_head", "full_step"]


class FrozenCache:
    """Stand-in for DynamicCache inside one attention call: returns cached K/V + the new token's K/V, stores nothing."""

    def __init__(self, keys, values):
        self.keys, self.values = keys, values

    def update(self, k, v, layer_idx, cache_kwargs=None):
        return torch.cat([self.keys, k], dim=-2), torch.cat([self.values, v], dim=-2)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="B2b run dir with saved thinking-on traces")
    p.add_argument("--problem", help="problem_id of the trace (default: the longest thinking-on trace)")
    p.add_argument("--trial", type=int)
    p.add_argument("--mode", choices=["on", "off"], default="on", help="thinking mode of the trace")
    p.add_argument("--contexts", type=int, nargs="+", default=[256, 1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--components", nargs="+", default=COMPONENTS)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--target-s", type=float, default=2.0, help="length of one trial (sets N)")
    p.add_argument("--pause-s", type=float, default=0.5)
    p.add_argument("--warmup-calls", type=int, default=100)
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


def pick_trace(src: Path, problem: str | None, trial: int | None, mode: str = "on") -> dict:
    best = None
    for line in open(src / "traces.jsonl"):
        t = json.loads(line)
        if t["mode"] != mode or (problem and t["problem_id"] != problem) or (trial is not None and t["trial"] != trial):
            continue
        if best is None or len(t["output_ids"]) > len(best["output_ids"]):
            best = t
    if best is None:
        sys.exit(f"No matching thinking-{mode} trace")
    return best


@torch.inference_mode()
def capture(model, ids_L: list[int], next_id: int, L: int, layer: int, chunk: int, device: str):
    """Real cache for ids_L, then one real decode step for next_id with hooks. Returns (cache, captured inputs)."""
    cache = DynamicCache()
    for i in range(0, L, chunk):
        model(input_ids=torch.tensor([ids_L[i:i + chunk]], device=device), past_key_values=cache, use_cache=True,
              logits_to_keep=1)
    lay = model.model.layers[layer]
    got = {}

    def grab(name):
        def hook(mod, args, kwargs):
            got[name] = (args, dict(kwargs))
        return hook

    hs = [m.register_forward_pre_hook(grab(n), with_kwargs=True) for n, m in [
        ("layer", lay), ("input_norm", lay.input_layernorm), ("attention", lay.self_attn),
        ("post_norm", lay.post_attention_layernorm), ("mlp", lay.mlp), ("final_norm", model.model.norm),
        ("lm_head", model.lm_head)]]
    step_in = {"input_ids": torch.tensor([[next_id]], device=device),
               "cache_position": torch.tensor([L], device=device)}
    model(**step_in, past_key_values=cache, use_cache=True, logits_to_keep=1)
    for h in hs:
        h.remove()
    cache.crop(L)
    assert cache.get_seq_length() == L
    return cache, got, step_in


def make_fns(model, cache, got, step_in, layer: int) -> dict:
    lay = model.model.layers[layer]
    K, V = cache.layers[layer].keys, cache.layers[layer].values
    frozen = FrozenCache(K, V)

    a_args, a_kw = got["attention"]
    a_kw = {**a_kw, "past_key_values": frozen}
    l_args, l_kw = got["layer"]
    l_kw = {**l_kw, "past_key_values": frozen}

    def full_step():
        model(**step_in, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache.crop(step_in["cache_position"].item())

    return {
        "input_norm": lambda: lay.input_layernorm(*got["input_norm"][0]),
        "attention": lambda: lay.self_attn(*a_args, **a_kw),
        "post_norm": lambda: lay.post_attention_layernorm(*got["post_norm"][0]),
        "mlp": lambda: lay.mlp(*got["mlp"][0]),
        "layer": lambda: lay(*l_args, **l_kw),
        "embed": lambda: model.model.embed_tokens(step_in["input_ids"]),
        "final_norm": lambda: model.model.norm(*got["final_norm"][0]),
        "lm_head": lambda: model.lm_head(*got["lm_head"][0]),
        "full_step": full_step,
    }


def time_per_call(fn, meter, n: int = 50) -> float:
    with torch.inference_mode():
        for _ in range(10):
            fn()
        meter.sync()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        meter.sync()
    return (time.perf_counter() - t0) / n


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    src = Path(a.source)
    tr = pick_trace(src, a.problem, a.trial, a.mode)
    ids = tr["prompt_ids"] + tr["output_ids"]
    contexts = [L for L in a.contexts if L < len(ids)]
    if len(contexts) < len(a.contexts):
        print(f"Trace has {len(ids)} tokens; skipping contexts {sorted(set(a.contexts) - set(contexts))}", flush=True)

    meter = get_meter(False)
    others = meter.other_processes()
    if others and not a.allow_busy:
        sys.exit(f"GPU in use by other processes {others}; pick a free GPU or --allow-busy")
    meter.strict = not a.allow_busy
    print(f"GPU: {meter.name}; trace {tr['problem_id']} t{tr['trial']}: prompt {len(tr['prompt_ids'])} + "
          f"{len(tr['output_ids'])} generated tokens", flush=True)

    tok, model = qwen.load_model(a.model, a.dtype, a.attn, a.device)
    print(f"Measuring idle power for {a.idle_s:.0f} s ...", flush=True)
    idle = meter.measure_idle(a.idle_s)
    print(f"  idle: {idle['mean_W']:.1f} W", flush=True)

    config = {"source_run": json.loads((src / "run.json").read_text())["run_id"], "problem_id": tr["problem_id"],
              "trace_mode": tr["mode"], "trace_trial": tr["trial"], "prompt_len": len(tr["prompt_ids"]),
              "contexts": contexts,
              "layer": a.layer, "components": a.components, "T": a.T, "target_s": a.target_s,
              "pause_s": a.pause_s, "warmup_calls": a.warmup_calls, "dtype": a.dtype, "attn_impl": a.attn,
              "batch_size": 1, "n_layers": model.config.num_hidden_layers,
              "replay": "attention/layer: frozen cache (cat, no store); full_step: real DynamicCache + crop"}
    experiment = "A_decode" + (f"_{a.tag}" if a.tag else "")
    rec = Recorder(experiment, config, qwen.model_info(model, a.model_id, a.model, a.dtype), meter, idle,
                   resume_dir=a.resume)
    done = rec.done_keys(["context", "component", "trial"])
    print(f"Run dir: {rec.dir}", flush=True)

    status = "failed"
    try:
        for L in contexts:
            todo_c = [c for c in a.components if any((L, c, t) not in done for t in range(a.T))]
            if not todo_c:
                continue
            cache, got, step_in = capture(model, ids[:L], ids[L], L, a.layer, a.prefill_chunk, a.device)
            fns = make_fns(model, cache, got, step_in, a.layer)
            print(f"\n== context {L} (cache {cache.get_seq_length()} tokens) ==", flush=True)
            for comp in todo_c:
                todo = [t for t in range(a.T) if (L, comp, t) not in done]
                t_call = time_per_call(fns[comp], meter)
                N = max(10, math.ceil(a.target_s / t_call))
                vals = []
                for i, row in enumerate(clear_measure(fns[comp], meter, N=N, T=len(todo), pause_s=a.pause_s,
                                                      warmup_calls=a.warmup_calls)):
                    row["trial"] = todo[i]
                    rec.write({"experiment": experiment, "method": "clear", "context": L, "layer": a.layer,
                               "component": comp, "t_call_probe_s": t_call, **row})
                    vals.append((row["E_end_mJ"] - row["E_start_mJ"]) / N)
                m = sum(vals) / len(vals)
                sd = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
                print(f"  {comp:10s} N={N:6d} {1e6 * t_call:8.1f} us/call  {m:9.3f} mJ/call  "
                      f"(std {100 * sd / m if m else 0:4.1f}%)", flush=True)
            del cache, got, fns
            torch.cuda.empty_cache()
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
