"""B1: reproduce CLEAR's BERT-base component energies (baselines.md section 2; CLEAR Tables 8 and 12).

Our version of clear_repo/{store_activations,module_measure,layer_measure,full_measure}.py in one script:
  Stage 1 (activation store): one real forward pass with pre-hooks that cache the exact (args, kwargs) each module
          receives, so every replay sees the true input (real token IDs for the embedding, the real GELU output
          for FFN-output, the same attention mask the full model used).
  Stage 2 (amplification):    energy.clear_measure, N back-to-back calls between two synced NVML reads, T trials.
  Stage 3 (validation):       b1/analyze_b1.py, computed from the raw rows written here.

Usage:
  CUDA_VISIBLE_DEVICES=0 python b1/run_b1.py                       # full B1 (5 lengths, T=20)
  CUDA_VISIBLE_DEVICES=0 python b1/run_b1.py --seq-lens 32 --T 2 --idle-s 5 --tag test
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from energy import GpuBusy, GpuMeter, Recorder, clear_measure  # noqa: E402

DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
COMPONENTS = ["embedding", "attention", "attention_self", "ffn_intermediate", "ffn_output", "norm_all", "block",
              "layers_cycled", "cls_head", "full_model"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/google-bert--bert-base-uncased")
    p.add_argument("--model-id", default="google-bert/bert-base-uncased", help="recorded in run.json")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[8, 32, 64, 96, 128],
                   help="input lengths incl. [CLS]/[SEP] (CLEAR Table 12: 32-128; 8 = code default, likely Table 8)")
    p.add_argument("--layer", type=int, default=5, help="encoder layer to measure (CLEAR: 5)")
    p.add_argument("--T", type=int, default=20, help="trials (CLEAR: 20)")
    p.add_argument("--N-component", type=int, default=10000, help="repetitions per trial for parts (CLEAR: 10,000)")
    p.add_argument("--N-model", type=int, default=1000, help="repetitions per trial for the full model (CLEAR: 1,000)")
    p.add_argument("--dtype", choices=DTYPES, default="fp16")
    p.add_argument("--attn", default="sdpa", help="attn_implementation (CLEAR did not set it)")
    p.add_argument("--prompt", default="The capital of France is", help="CLEAR's sentence")
    p.add_argument("--repeat", type=int, default=30,
                   help="sentence repetitions (CLEAR: 10 = 50 tokens, so >50 gets [PAD]; 30 gives ~150 tokens)")
    p.add_argument("--components", nargs="+", choices=COMPONENTS, default=COMPONENTS)
    p.add_argument("--pause-s", type=float, default=0.5, help="pause between trials (CLEAR: 0.5 s)")
    p.add_argument("--warmup-calls", type=int, default=100)
    p.add_argument("--idle-s", type=float, default=30.0, help="idle-power measurement at session start")
    p.add_argument("--allow-busy", action="store_true", help="run even if other processes use this GPU")
    p.add_argument("--resume", help="run directory to continue; already-measured trials are skipped")
    p.add_argument("--tag", default="", help="suffix for the experiment name, e.g. 'test'")
    return p.parse_args()


def build_inputs(tok, prompt: str, repeat: int, seq_len: int):
    """CLEAR's construction (sentence * repeat, truncated to seq_len incl. special tokens), but refuse padding."""
    enc = tok(prompt * repeat, return_tensors="pt", max_length=seq_len, truncation=True, padding="max_length")
    n_pad = int((enc["input_ids"] == tok.pad_token_id).sum())
    if n_pad:
        sys.exit(f"seq_len={seq_len} needs {n_pad} [PAD] tokens; increase --repeat")
    return enc.to("cuda")


def _detach(x):
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, tuple):
        return tuple(_detach(v) for v in x)
    if isinstance(x, list):
        return [_detach(v) for v in x]
    if isinstance(x, dict):
        return {k: _detach(v) for k, v in x.items()}
    return x


def capture_inputs(model, inputs, layer_idx: int) -> dict:
    """Stage 1: cache the exact (args, kwargs) each measured module receives in one real forward pass."""
    layer = model.bert.encoder.layer[layer_idx]
    targets = {
        "embedding": model.bert.embeddings,
        "block": layer,
        "attention": layer.attention,
        "attention_self": layer.attention.self,
        "ln_attention": layer.attention.output.LayerNorm,
        "ffn_intermediate": layer.intermediate,
        "ffn_output": layer.output,
        "ln_output": layer.output.LayerNorm,
        "cls_head": model.cls,
    }
    targets |= {f"layer{i}": l for i, l in enumerate(model.bert.encoder.layer)}  # for layers_cycled
    store, handles = {}, []
    for name, module in targets.items():
        def hook(mod, args, kwargs, name=name):
            store.setdefault(name, (_detach(args), _detach(kwargs)))
        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    with torch.inference_mode():
        model(**inputs)
    for h in handles:
        h.remove()
    missing = set(targets) - set(store)
    if missing:
        sys.exit(f"hooks did not fire for: {missing}")
    return {"modules": targets, "inputs": store}


def make_fns(model, inputs, cache) -> dict:
    mods, ins = cache["modules"], cache["inputs"]

    def call(name):
        args, kwargs = ins[name]
        return lambda: mods[name](*args, **kwargs)

    ln_a, ln_o = call("ln_attention"), call("ln_output")
    layer_calls = [call(f"layer{i}") for i in range(len(model.bert.encoder.layer))]
    return {
        "embedding": call("embedding"),
        "attention": call("attention"),            # self-attn + out-proj + residual + LayerNorm (as CLEAR's code)
        "attention_self": call("attention_self"),  # Q/K/V + SDPA only, no out-proj and no LayerNorm (extra)
        "ffn_intermediate": call("ffn_intermediate"),
        "ffn_output": call("ffn_output"),          # dense + residual + LayerNorm (as CLEAR's code)
        "norm_all": lambda: (ln_a(), ln_o()),      # both LayerNorms of the layer (not in CLEAR's released code)
        "block": call("block"),
        # All layers in sequence, each with its own captured input. Replaying one layer N times keeps its weights
        # in L2 and underestimates it by 5-9% (slot 2, b1/debug_capture.py); this is the B1 encoder measurement.
        "layers_cycled": lambda: [f() for f in layer_calls],
        "cls_head": call("cls_head"),
        "full_model": lambda: model(**inputs),
    }


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    torch.manual_seed(0)

    meter = GpuMeter(0)
    others = meter.other_processes()
    if others and not a.allow_busy:
        sys.exit(f"GPU {meter.nvml_index} is in use by other processes {others}; pick a free GPU or --allow-busy")
    meter.strict = not a.allow_busy  # from here on, stop (GpuBusy) if another process appears
    print(f"GPU: {meter.name} (nvml index {meter.nvml_index}, {meter.uuid})", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForMaskedLM.from_pretrained(a.model, dtype=DTYPES[a.dtype], attn_implementation=a.attn)
    model = model.cuda().eval()

    print(f"Measuring idle power for {a.idle_s:.0f} s ...", flush=True)
    idle = meter.measure_idle(a.idle_s)
    print(f"  idle: {idle['mean_W']:.1f} W at {idle['temp_C']} C", flush=True)

    config = {
        "seq_lens": a.seq_lens, "layer": a.layer, "T": a.T, "N_component": a.N_component, "N_model": a.N_model,
        "dtype": a.dtype, "attn_impl": a.attn, "batch_size": 1, "prompt": a.prompt, "repeat": a.repeat,
        "components": a.components, "pause_s": a.pause_s, "warmup_calls": a.warmup_calls,
        "sync_before_read": True, "replay_inputs": "exact args/kwargs from a real forward pass",
    }
    model_info = {"id": a.model_id, "local_path": a.model, "revision": getattr(model.config, "_commit_hash", None),
                  "class": type(model).__name__, "dtype": a.dtype, "attn_impl": model.config._attn_implementation,
                  "n_layers": model.config.num_hidden_layers}
    experiment = "B1" + (f"_{a.tag}" if a.tag else "")
    rec = Recorder(experiment, config, model_info, meter, idle, resume_dir=a.resume)
    done = rec.done_keys(["seq_len", "component", "trial"])
    print(f"Run dir: {rec.dir}", flush=True)

    status = "failed"
    try:
        for L in a.seq_lens:
            inputs = build_inputs(tok, a.prompt, a.repeat, L)
            fns = make_fns(model, inputs, capture_inputs(model, inputs, a.layer))
            for comp in a.components:
                N = a.N_model if comp in ("full_model", "layers_cycled") else a.N_component
                todo = [t for t in range(a.T) if (L, comp, t) not in done]
                if not todo:
                    continue
                vals = []
                t_start = time.time()
                for i, row in enumerate(clear_measure(fns[comp], meter, N=N, T=len(todo), pause_s=a.pause_s,
                                                      warmup_calls=a.warmup_calls)):
                    row["trial"] = todo[i]
                    rec.write({"experiment": experiment, "method": "clear", "seq_len": L, "layer": a.layer,
                               "component": comp, "input_ids": inputs["input_ids"][0].tolist(), **row})
                    vals.append((row["E_end_mJ"] - row["E_start_mJ"]) / N)
                    if row["other_processes"]:
                        print(f"  WARNING: other processes on GPU during trial {row['trial']}: "
                              f"{row['other_processes']}", flush=True)
                v = torch.tensor(vals, dtype=torch.float64)
                trial_s = (time.time() - t_start) / len(vals) - a.pause_s
                print(f"L={L:4d} {comp:17s} {v.mean():9.3f} mJ  std {v.std(unbiased=False):7.3f}  "
                      f"({100 * v.std(unbiased=False) / v.mean():4.1f}%)  ~{trial_s:.2f} s/trial", flush=True)
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
