"""Why does B1's %Capture(Model) fall short (73-84% on the RTX PRO 5000; slot2_plan.md, "B1 capture debugging")?

Captured(Model) = 12 x (one layer replayed alone) + embedding + head. Slot 1 showed the parts cover ~92% of the full
model's time but draw ~13% less power. This script measures, in one session and with the same inputs:

  layer_alone      layer --layer replayed alone (what B1 multiplies by 12)                     N = --N-part
  layers_cycled    all 12 layers called in sequence, each with its own captured input         N = --N-whole
                   -> H1 cache residency: one layer replayed alone keeps its ~7 MB of weights in L2; the real
                      forward pass streams 12 different layers. If layers_cycled > 12 x layer_alone, H1 holds.
  layer_flushed    layer_alone with an L2 flush (write to a buffer > L2) before every call    N = --N-flush
  flush_only       the flush alone, to subtract                                                N = --N-flush
                   -> a second, independent test of H1 (standard practice in GPU kernel benchmarking).
  encoder          model.bert.encoder (12 layers + the Python glue between them)              N = --N-whole
  backbone         model.bert (embedding + encoder; mask preparation)                          N = --N-whole
  embedding, head  as in B1                                                                    N = --N-part
  full_model       model(**inputs)                                                             N = --N-whole
                   -> H2 work outside the components: compare full_model with backbone + head and
                      encoder with layers_cycled.
  --each-layer     every layer replayed alone (is layer 5 representative?)

H3 (clocks/power state) is read from the sm_clock_MHz / power_W logged with every trial.
A torch.profiler pass over one full forward writes the top CUDA kernels and the GPU-busy fraction to the log.

Usage:
  CUDA_VISIBLE_DEVICES=0 python b1/debug_capture.py --seq-lens 32 128
  python b1/debug_capture.py --fake-meter --device cpu --dtype fp32 --seq-lens 32 --T 2 --N-part 20 --N-whole 5 ...
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from energy import GpuBusy, Recorder, clear_measure, get_meter  # noqa: E402

DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/google-bert--bert-base-uncased")
    p.add_argument("--model-id", default="google-bert/bert-base-uncased")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[32, 128])
    p.add_argument("--layer", type=int, default=5)
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--N-part", type=int, default=10000, help="repetitions for one layer / embedding / head")
    p.add_argument("--N-whole", type=int, default=1000, help="repetitions for cycled layers / encoder / model")
    p.add_argument("--N-flush", type=int, default=2000, help="repetitions for the flushed layer and flush alone")
    p.add_argument("--flush-mb", type=int, default=0, help="L2 flush buffer size; 0 = 4 x the GPU's L2 (min 256)")
    p.add_argument("--each-layer", action="store_true", help="also replay every layer alone (12 more parts)")
    p.add_argument("--no-profile", action="store_true")
    p.add_argument("--dtype", choices=DTYPES, default="fp16")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--pause-s", type=float, default=0.5)
    p.add_argument("--warmup-calls", type=int, default=50)
    p.add_argument("--idle-s", type=float, default=30.0)
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--tag", default="")
    p.add_argument("--fake-meter", action="store_true", help="dry run without an NVIDIA GPU (numbers meaningless)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _detach(x):
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, (tuple, list)):
        return type(x)(_detach(v) for v in x)
    if isinstance(x, dict):
        return {k: _detach(v) for k, v in x.items()}
    return x


def capture(model, inputs) -> dict:
    """Exact (args, kwargs) of every layer, the encoder, the embedding and the head in one real forward pass."""
    bert = model.bert
    targets = {f"layer{i}": l for i, l in enumerate(bert.encoder.layer)}
    targets |= {"encoder": bert.encoder, "embedding": bert.embeddings, "head": model.cls}
    store, handles = {}, []
    for name, mod in targets.items():
        def hook(m, args, kwargs, name=name):
            store.setdefault(name, (_detach(args), _detach(kwargs)))
        handles.append(mod.register_forward_pre_hook(hook, with_kwargs=True))
    with torch.inference_mode():
        model(**inputs)
    for h in handles:
        h.remove()
    missing = set(targets) - set(store)
    if missing:
        sys.exit(f"hooks did not fire for: {missing}")
    return {"mods": targets, "ins": store}


def profile_once(model, inputs, device: str) -> str:
    """Top kernels of one full forward pass and the share of wall time the GPU is busy."""
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    with torch.inference_mode():
        for _ in range(20):
            model(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        with profile(activities=acts) as prof:
            t0 = time.perf_counter()
            model(**inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            wall_us = (time.perf_counter() - t0) * 1e6
    events = prof.key_averages()
    lines = [f"wall {wall_us:.0f} us for one forward pass"]
    if device == "cuda":
        kernels = [e for e in prof.events() if getattr(e.device_type, "name", "") == "CUDA"]
        busy = sum(getattr(e, "device_time_total", 0) or getattr(e, "cuda_time_total", 0) for e in kernels)
        lines.append(f"{len(kernels)} CUDA kernels, {busy:.0f} us of kernel time "
                     f"({100 * busy / wall_us:.0f}% of wall: the rest is launch/CPU gaps)")
        lines.append(events.table(sort_by="self_cuda_time_total", row_limit=20))
    else:
        lines.append(events.table(sort_by="cpu_time_total", row_limit=10))
    return "\n".join(lines)


def main():
    a = parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available (use --fake-meter --device cpu for a dry run)")
    torch.manual_seed(0)
    meter = get_meter(a.fake_meter)
    others = meter.other_processes()
    if others and not a.allow_busy:
        sys.exit(f"GPU in use by other processes {others}; pick a free GPU or --allow-busy")
    meter.strict = not a.allow_busy  # from here on, stop (GpuBusy) if another process appears
    print(f"GPU: {meter.name}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForMaskedLM.from_pretrained(a.model, dtype=DTYPES[a.dtype], attn_implementation=a.attn)
    model = model.to(a.device).eval()
    n_layers = model.config.num_hidden_layers

    flush_mb = a.flush_mb
    if not flush_mb:
        l2 = torch.cuda.get_device_properties(0).L2_cache_size if a.device == "cuda" else 0
        flush_mb = max(256, 4 * l2 // 2**20)
    flush_buf = torch.empty(flush_mb * 2**20 // 4, dtype=torch.int32, device=a.device)
    print(f"L2 flush buffer: {flush_mb} MB", flush=True)

    print(f"Measuring idle power for {a.idle_s:.0f} s ...", flush=True)
    idle = meter.measure_idle(a.idle_s)
    print(f"  idle: {idle['mean_W']:.1f} W", flush=True)

    parts = ["layer_alone", "layers_cycled", "layer_flushed", "flush_only", "encoder", "backbone", "embedding",
             "head", "full_model"] + ([f"layer{i}_alone" for i in range(n_layers)] if a.each_layer else [])
    N = {"layer_alone": a.N_part, "embedding": a.N_part, "head": a.N_part, "layer_flushed": a.N_flush,
         "flush_only": a.N_flush, "layers_cycled": a.N_whole, "encoder": a.N_whole, "backbone": a.N_whole,
         "full_model": a.N_whole} | {f"layer{i}_alone": a.N_part for i in range(n_layers)}
    config = {"seq_lens": a.seq_lens, "layer": a.layer, "T": a.T, "N": {p: N[p] for p in parts}, "flush_mb": flush_mb,
              "dtype": a.dtype, "attn_impl": a.attn, "batch_size": 1, "prompt": a.prompt, "repeat": a.repeat,
              "parts": parts, "pause_s": a.pause_s, "warmup_calls": a.warmup_calls}
    model_info = {"id": a.model_id, "local_path": a.model, "class": type(model).__name__, "dtype": a.dtype,
                  "attn_impl": model.config._attn_implementation, "n_layers": n_layers}
    experiment = "B1debug" + (f"_{a.tag}" if a.tag else "")
    rec = Recorder(experiment, config, model_info, meter, idle, resume_dir=a.resume)
    done = rec.done_keys(["seq_len", "part", "trial"])
    print(f"Run dir: {rec.dir}", flush=True)

    status = "failed"
    try:
        for L in a.seq_lens:
            enc = tok(a.prompt * a.repeat, return_tensors="pt", max_length=L, truncation=True)
            if enc["input_ids"].shape[1] != L:
                sys.exit(f"prompt too short for {L} tokens; increase --repeat")
            inputs = enc.to(a.device)
            cap = capture(model, inputs)
            mods, ins = cap["mods"], cap["ins"]

            def call(name):
                args, kwargs = ins[name]
                return lambda: mods[name](*args, **kwargs)

            layer_calls = [call(f"layer{i}") for i in range(n_layers)]

            def flush():
                flush_buf.fill_(1)

            def layer_flushed(f=layer_calls[a.layer]):
                flush()
                f()

            fns = {"layer_alone": layer_calls[a.layer], "layers_cycled": lambda: [f() for f in layer_calls],
                   "layer_flushed": layer_flushed, "flush_only": flush, "encoder": call("encoder"),
                   "backbone": lambda: model.bert(**inputs), "embedding": call("embedding"), "head": call("head"),
                   "full_model": lambda: model(**inputs)}
            fns |= {f"layer{i}_alone": layer_calls[i] for i in range(n_layers)}

            if not a.no_profile:
                try:
                    print(f"--- profile, {L} tokens ---\n{profile_once(model, inputs, a.device)}", flush=True)
                except Exception as e:  # the profile is a diagnostic; never let it stop the energy measurements
                    print(f"--- profile failed: {e!r}", flush=True)

            means = {}
            for part in parts:
                todo = [t for t in range(a.T) if (L, part, t) not in done]
                vals, power, clock = [], [], []
                for i, row in enumerate(clear_measure(fns[part], meter, N=N[part], T=len(todo), pause_s=a.pause_s,
                                                      warmup_calls=a.warmup_calls) if todo else []):
                    row["trial"] = todo[i]
                    rec.write({"experiment": experiment, "method": "clear", "seq_len": L, "part": part, **row})
                    dur = row["t_end_s"] - row["t_start_s"]
                    vals.append((row["E_end_mJ"] - row["E_start_mJ"]) / N[part])
                    power.append((row["E_end_mJ"] - row["E_start_mJ"]) / 1000 / dur)
                    clock.append(row["sm_clock_MHz"])
                if vals:
                    v = torch.tensor(vals, dtype=torch.float64)
                    means[part] = float(v.mean())
                    print(f"L={L:4d} {part:14s} {v.mean():9.3f} mJ  ({100 * v.std(unbiased=False) / v.mean():4.1f}%)"
                          f"  {sum(power) / len(power):6.1f} W  sm {min(clock)}-{max(clock)} MHz", flush=True)

            m = means
            if {"layer_alone", "layers_cycled", "embedding", "head", "full_model"} <= set(m):
                print(f"--- summary, {L} tokens ---", flush=True)
                print(f"  B1 captured   12 x layer_alone + emb + head = "
                      f"{100 * (n_layers * m['layer_alone'] + m['embedding'] + m['head']) / m['full_model']:.1f}% "
                      "of full_model", flush=True)
                print(f"  H1 cycled     layers_cycled + emb + head     = "
                      f"{100 * (m['layers_cycled'] + m['embedding'] + m['head']) / m['full_model']:.1f}% of full_model;"
                      f" layers_cycled / (12 x layer_alone) = "
                      f"{m['layers_cycled'] / (n_layers * m['layer_alone']):.3f}", flush=True)
                if {"layer_flushed", "flush_only"} <= set(m):
                    print(f"  H1 flushed    layer_flushed - flush_only = {m['layer_flushed'] - m['flush_only']:.3f} mJ "
                          f"vs layer_alone {m['layer_alone']:.3f} mJ", flush=True)
                if {"encoder", "backbone"} <= set(m):
                    print(f"  H2 glue       encoder / layers_cycled = {m['encoder'] / m['layers_cycled']:.3f}; "
                          f"(backbone + head) / full_model = {(m['backbone'] + m['head']) / m['full_model']:.3f}",
                          flush=True)
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
