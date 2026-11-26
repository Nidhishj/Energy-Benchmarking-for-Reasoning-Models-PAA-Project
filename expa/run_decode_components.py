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

Ablations (Phase 3):
  --cache inplace       KV cache written in place (llm/inplace_cache.py) instead of torch.cat every step
  --sdpa-backend X      force one SDPA kernel (flash / efficient / math / cudnn); --attn eager for no SDPA at all
  --no-sync             counter read without syncing the GPU first
  --cycle-layers        replay input_norm/attention/post_norm/mlp/layer over all 36 layers in model order (each with
                        its own captured inputs and cache) instead of one layer N times; energy per call is then the
                        mean over layers

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
from llm.inplace_cache import FrozenInplace, InplaceCache  # noqa: E402

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
    p.add_argument("--cache", choices=["dynamic", "inplace"], default="dynamic",
                   help="KV cache: transformers' DynamicCache (torch.cat each step) or written in place")
    p.add_argument("--sdpa-backend", choices=["flash", "efficient", "math", "cudnn"],
                   help="force one SDPA kernel (default: PyTorch chooses)")
    p.add_argument("--no-sync", action="store_true", help="read the energy counter without syncing the GPU")
    p.add_argument("--cycle-layers", action="store_true",
                   help="replay per-layer components over all layers in order instead of one layer N times")
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
def capture(model, ids_L: list[int], next_id: int, L: int, layer: int, chunk: int, device: str,
            cache_kind: str = "dynamic", all_layers: bool = False):
    """Real cache for ids_L, then one real decode step for next_id with hooks. Returns (cache, captured inputs).

    With all_layers, the per-layer modules of every layer are captured too, under "<name>@<layer index>"."""
    if cache_kind == "inplace":
        cache = InplaceCache(model.config.num_hidden_layers, L + 16)
    else:
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

    mods = [("layer", lay), ("input_norm", lay.input_layernorm), ("attention", lay.self_attn),
            ("post_norm", lay.post_attention_layernorm), ("mlp", lay.mlp), ("final_norm", model.model.norm),
            ("lm_head", model.lm_head)]
    if all_layers:
        for li, ly in enumerate(model.model.layers):
            mods += [(f"layer@{li}", ly), (f"input_norm@{li}", ly.input_layernorm), (f"attention@{li}", ly.self_attn),
                     (f"post_norm@{li}", ly.post_attention_layernorm), (f"mlp@{li}", ly.mlp)]
    hs = [m.register_forward_pre_hook(grab(n), with_kwargs=True) for n, m in mods]
    step_in = {"input_ids": torch.tensor([[next_id]], device=device),
               "cache_position": torch.tensor([L], device=device)}
    model(**step_in, past_key_values=cache, use_cache=True, logits_to_keep=1)
    for h in hs:
        h.remove()
    cache.crop(L)
    assert cache.get_seq_length() == L
    return cache, got, step_in


def make_fns(model, cache, got, step_in, layer: int, cache_kind: str = "dynamic", cycle: bool = False) -> dict:
    lay = model.model.layers[layer]
    Frozen = FrozenInplace if cache_kind == "inplace" else FrozenCache
    K, V = cache.layers[layer].keys, cache.layers[layer].values
    frozen = Frozen(K, V)

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
        **(cycled_fns(model, cache, got, Frozen) if cycle else {}),
    }


def cycled_fns(model, cache, got, Frozen) -> dict:
    """Per-layer components replayed over all layers in model order: call i runs layer i % n_layers with that layer's
    own captured inputs (and its own frozen cache for attention / layer)."""
    n = len(model.model.layers)
    calls = {c: [] for c in ("input_norm", "attention", "post_norm", "mlp", "layer")}
    for li, ly in enumerate(model.model.layers):
        fr = Frozen(cache.layers[li].keys, cache.layers[li].values)
        a_args, a_kw = got[f"attention@{li}"]
        l_args, l_kw = got[f"layer@{li}"]
        calls["input_norm"].append(lambda ly=ly, x=got[f"input_norm@{li}"][0]: ly.input_layernorm(*x))
        calls["attention"].append(lambda ly=ly, x=a_args, kw={**a_kw, "past_key_values": fr}: ly.self_attn(*x, **kw))
        calls["post_norm"].append(lambda ly=ly, x=got[f"post_norm@{li}"][0]: ly.post_attention_layernorm(*x))
        calls["mlp"].append(lambda ly=ly, x=got[f"mlp@{li}"][0]: ly.mlp(*x))
        calls["layer"].append(lambda ly=ly, x=l_args, kw={**l_kw, "past_key_values": fr}: ly(*x, **kw))

    def cycler(fs):
        state = [0]

        def fn():
            out = fs[state[0]]()
            state[0] = (state[0] + 1) % n
            return out
        return fn

    return {c: cycler(fs) for c, fs in calls.items()}


def set_sdpa_backend(name: str | None):
    """Allow only SDPA kernel `name` (None: all, PyTorch chooses). Applied only while measuring: the chunked prefill
    in capture() passes a mask, which the flash kernel does not accept."""
    want = {"efficient": "mem_efficient"}.get(name, name)
    for k in ("flash", "mem_efficient", "math", "cudnn"):
        getattr(torch.backends.cuda, f"enable_{k}_sdp")(name is None or k == want)


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
              "cache": a.cache, "sdpa_backend": a.sdpa_backend, "sync": not a.no_sync, "cycle_layers": a.cycle_layers,
              "replay": ("attention/layer: frozen cache (cat, no store); full_step: real DynamicCache + crop"
                         if a.cache == "dynamic" else
                         "attention/layer: frozen in-place buffer (write slot L, no store); full_step: InplaceCache + crop")
              + ("; per-layer components cycled over all layers" if a.cycle_layers else "")}
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
            set_sdpa_backend(None)
            cache, got, step_in = capture(model, ids[:L], ids[L], L, a.layer, a.prefill_chunk, a.device,
                                          a.cache, a.cycle_layers)
            fns = make_fns(model, cache, got, step_in, a.layer, a.cache, a.cycle_layers)
            print(f"\n== context {L} (cache {cache.get_seq_length()} tokens) ==", flush=True)
            set_sdpa_backend(a.sdpa_backend)
            for comp in todo_c:
                todo = [t for t in range(a.T) if (L, comp, t) not in done]
                try:
                    t_call = time_per_call(fns[comp], meter)
                except RuntimeError as e:  # a forced SDPA kernel that cannot run these inputs
                    print(f"  {comp:10s} cannot run: {str(e).splitlines()[0][:200]}", flush=True)
                    rec.write({"experiment": experiment, "method": "error", "context": L, "layer": a.layer,
                               "component": comp, "trial": -1, "error": str(e)[:2000]})
                    continue
                N = max(10, math.ceil(a.target_s / t_call))
                vals = []
                for i, row in enumerate(clear_measure(fns[comp], meter, N=N, T=len(todo), pause_s=a.pause_s,
                                                      warmup_calls=a.warmup_calls, sync=not a.no_sync)):
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
