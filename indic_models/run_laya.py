"""Component energy of Laya (ConvAI Innovations; convaiinnovations/laya), with CLEAR's method and with ours.

Laya is a non-autoregressive decision model: a ModernBERT encoder (English: ModernBERT-large, 28 layers; multilingual:
mmBERT-base, 22 layers, incl. Hindi/Bengali/Tamil/Telugu/Kannada/Malayalam) plus a 2-layer decision head. One forward
pass answers every question, so it is measured like CLEAR's BERT (B1): one forward pass, no generation.

ModernBERT alternates attention types: every 3rd layer (0, 3, 6, ...) is global, the others use a 128-token sliding
window. So two layers are measured: layer 5 (sliding; CLEAR's layer index) and layer 6 (global).

Inputs come from Laya's own pipeline: one agent.predict() call on a long text with max_len = L builds the real
sequence ([CLS] question [SEP] [MASK] option ... [SEP] text [SEP]); a pre-hook keeps the model's arguments, then one
forward pass without autocast caches the exact (args, kwargs) of every measured module, as in b1/run_b1.py.

--method clear  CLEAR's loop (clear_repo/module_measure.py): 0.5 s pause, counter read, N calls, counter read; no
                warm-up and no torch.cuda.synchronize(), so queued GPU work can fall outside the window.
--method ours   energy.clear_measure: 100 warm-up calls, sync before every counter read (b1/run_b1.py).
Both: N = 10,000 per part and 1,000 for the encoder/full model, T = 20, NVML energy counter only.

Usage:
  CUDA_VISIBLE_DEVICES=1 python indic_models/run_laya.py --checkpoint english --method ours
  CUDA_VISIBLE_DEVICES=1 python indic_models/run_laya.py --checkpoint multilingual --method clear
  CUDA_VISIBLE_DEVICES=1 python indic_models/run_laya.py --seq-lens 64 --T 2 --tag test      # quick test
"""

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use the downloaded snapshot only
import pynvml  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from energy import GpuBusy, GpuMeter, Recorder, clear_measure  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO / "models" / "convaiinnovations--laya"
REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
CHECKPOINTS = {  # subfolder, default lengths (English is trained to 512 tokens, multilingual to 1,024)
    "english": (None, [64, 128, 256, 512]),
    "multilingual": ("multilingual", [64, 128, 256, 512, 1024]),
}
QUESTION = {"department": {"type": "choice", "instructions": "Which department should handle this?",
                           "criteria": {"billing": "invoices, payments, refunds",
                                        "technical": "bugs, outages, system errors",
                                        "other": "everything else"}}}
# A support ticket, repeated until the sequence is longer than max_len (Laya then cuts the text to fit exactly).
TEXT = {
    "english": ("Hi, we were billed twice for March and the invoice still shows the old plan. Our team also cannot "
                "log in since the last update, and the dashboard shows an error when we open the reports page. "),
    "multilingual": ("नमस्ते, मार्च में हमसे दो बार शुल्क लिया गया और चालान में अब भी पुराना प्लान दिख रहा है। "
                     "पिछले अपडेट के बाद से हमारी टीम लॉग इन नहीं कर पा रही है और रिपोर्ट पेज खोलने पर त्रुटि आती है। "),
}
PART_NAMES = ["attn_norm", "attn", "mlp_norm", "mlp", "block"]
BIG = {"encoder", "encoder_cycled", "full_model"}  # measured with N_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", choices=CHECKPOINTS, default="english")
    p.add_argument("--method", choices=["clear", "ours"], default="ours")
    p.add_argument("--seq-lens", type=int, nargs="+", help="total input lengths (default per checkpoint)")
    p.add_argument("--layers", type=int, nargs="+", default=[5, 6], help="encoder layers (5 = sliding, 6 = global)")
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--N-component", type=int, default=10000)
    p.add_argument("--N-model", type=int, default=1000)
    p.add_argument("--dtype", choices=DTYPES, default="bf16", help="Laya runs in bf16 (autocast); CLEAR used fp16")
    p.add_argument("--components", nargs="+", help="subset of components (default: all)")
    p.add_argument("--pause-s", type=float, default=0.5)
    p.add_argument("--warmup-calls", type=int, default=100, help="ours only; CLEAR has no warm-up")
    p.add_argument("--idle-s", type=float, default=30.0)
    p.add_argument("--resume", help="run directory to continue")
    p.add_argument("--tag", default="")
    return p.parse_args()


def clear_loop(fn, meter, N, T, pause_s):
    """CLEAR's measure_energy (clear_repo/module_measure.py): no warm-up, no sync; plus GPU state and clock reasons."""
    with torch.inference_mode():
        for trial in range(T):
            time.sleep(pause_s)
            others = meter.other_processes()
            e0, t0 = meter.read(sync=False)
            for _ in range(N):
                fn()
            e1, t1 = meter.read(sync=False)
            yield {"trial": trial, "N": N, "E_start_mJ": e0, "E_end_mJ": e1, "t_start_s": t0, "t_end_s": t1,
                   **meter.state(), "other_processes": others}


def _detach(x):
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, (tuple, list)):
        return type(x)(_detach(v) for v in x)
    if isinstance(x, dict):
        return {k: _detach(v) for k, v in x.items()}
    return x


def model_args(agent, text, L):
    """The DecisionModel arguments Laya's own pipeline builds for `text` cut to max_len = L, and its answer."""
    store = {}
    h = agent.model.register_forward_pre_hook(lambda m, a, kw: store.setdefault("x", (_detach(a), _detach(kw))),
                                              with_kwargs=True)
    reps = 1
    while True:
        store.clear()
        res = agent.predict(text * reps, QUESTION, max_len=L)
        if store["x"][0][0].shape[1] == L:
            break
        if reps > 512:
            sys.exit(f"could not build a {L}-token input")
        reps *= 2
    h.remove()
    return store["x"], res["answers"]["department"]


class _CachedEncoder(torch.nn.Module):
    """Stands in for the encoder so the decision head runs Laya's own post-encoder code on a cached encoder output."""

    def __init__(self, h, config):
        super().__init__()
        self.h, self.config = h, config

    def forward(self, **kwargs):
        return SimpleNamespace(last_hidden_state=self.h)


def make_fns(model, args, kwargs, layers):
    """Stage 1: one forward pass caching each measured module's exact inputs; returns {component: zero-arg call}."""
    enc = model.encoder
    targets = {"embedding": enc.embeddings, "final_norm": enc.final_norm, "encoder": enc}
    for i in layers:
        lay = enc.layers[i]
        targets |= {f"L{i}.attn_norm": lay.attn_norm, f"L{i}.attn": lay.attn, f"L{i}.mlp_norm": lay.mlp_norm,
                    f"L{i}.mlp": lay.mlp, f"L{i}.block": lay}
    targets |= {f"head{j}": l for j, l in enumerate(model.head.layers)}
    targets |= {f"layer{i}": l for i, l in enumerate(enc.layers)}
    store, handles = {}, []
    for name, mod in targets.items():
        def hook(m, a, kw, name=name):
            store.setdefault(name, (_detach(a), _detach(kw)))
        handles.append(mod.register_forward_pre_hook(hook, with_kwargs=True))
    out = {}
    def keep_output(m, a, o):  # returns None, so the encoder's output is left unchanged
        out.setdefault("h", o.last_hidden_state.detach().clone())
    handles.append(enc.register_forward_hook(keep_output))
    with torch.inference_mode():
        logits, _ = model(*args, **kwargs)
    for h in handles:
        h.remove()
    if set(targets) - set(store):
        sys.exit(f"hooks did not fire for: {set(targets) - set(store)}")

    def call(name):
        a, kw = store[name]
        return lambda: targets[name](*a, **kw)

    head_only = copy.copy(model)  # same modules, encoder replaced by its cached output
    head_only._modules = dict(model._modules)
    head_only._modules["encoder"] = _CachedEncoder(out["h"], enc.config)
    layer_calls = [call(f"layer{i}") for i in range(len(enc.layers))]
    fns = {"embedding": call("embedding")}
    for i in layers:
        fns |= {f"L{i}.{p}": call(f"L{i}.{p}") for p in PART_NAMES}
    fns |= {
        "final_norm": call("final_norm"),
        "head0": call("head0"),
        "head1": call("head1"),
        "decision_head": lambda: head_only(*args, **kwargs),  # type emb + 2 head layers + scorer + act head
        "encoder_cycled": lambda: [f() for f in layer_calls],  # every layer once, each with its own input
        "encoder": call("encoder"),
        "full_model": lambda: model(*args, **kwargs),
    }
    return fns, logits.float().softmax(-1)[0].tolist()


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    torch.manual_seed(0)
    sub, default_lens = CHECKPOINTS[a.checkpoint]
    seq_lens = a.seq_lens or default_lens

    meter = GpuMeter(0)
    others = meter.other_processes()
    if others:
        sys.exit(f"GPU {meter.nvml_index} is in use by other processes {others}; pick a free GPU")
    meter.strict = True  # from here on, stop (GpuBusy) if another process appears
    print(f"GPU: {meter.name} (nvml index {meter.nvml_index})", flush=True)

    import laya
    agent = laya.load(str(MODEL_DIR), subfolder=sub, device="cuda")
    model = agent.model.to(DTYPES[a.dtype]).eval()
    enc_cfg = model.encoder.config
    n_layers = len(model.encoder.layers)
    layer_types = [("global" if i % enc_cfg.global_attn_every_n_layers == 0 else "sliding") for i in range(n_layers)]

    print(f"Measuring idle power for {a.idle_s:.0f} s ...", flush=True)
    idle = meter.measure_idle(a.idle_s)
    print(f"  idle: {idle['mean_W']:.1f} W at {idle['temp_C']} C", flush=True)

    config = {"checkpoint": a.checkpoint, "method": a.method, "seq_lens": seq_lens, "layers": a.layers, "T": a.T,
              "N_component": a.N_component, "N_model": a.N_model, "dtype": a.dtype, "batch_size": 1,
              "question": QUESTION, "text": TEXT[a.checkpoint], "pause_s": a.pause_s,
              "warmup_calls": a.warmup_calls if a.method == "ours" else 0, "sync_before_read": a.method == "ours",
              "components": a.components, "replay_inputs": "exact args/kwargs from a real forward pass"}
    model_info = {"id": "convaiinnovations/laya" + (f"/{sub}" if sub else ""), "local_path": str(MODEL_DIR),
                  "revision": REVISION, "laya_version": laya.__version__, "class": type(model).__name__,
                  "encoder": type(model.encoder).__name__, "dtype": a.dtype,
                  "attn_impl": enc_cfg._attn_implementation, "n_layers": n_layers, "layer_types": layer_types,
                  "sliding_window": enc_cfg.local_attention, "hidden_size": enc_cfg.hidden_size,
                  "n_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
    experiment = f"Laya_{a.checkpoint}_{a.method}" + (f"_{a.tag}" if a.tag else "")
    rec = Recorder(experiment, config, model_info, meter, idle, resume_dir=a.resume)
    done = rec.done_keys(["seq_len", "component", "trial"])
    print(f"Run dir: {rec.dir}", flush=True)

    status = "failed"
    try:
        for L in seq_lens:
            (args, kwargs), answer = model_args(agent, TEXT[a.checkpoint], L)
            fns, probs = make_fns(model, args, kwargs, a.layers)
            print(f"L={L}: Laya answer {answer.get('choice')}; "
                  f"replayed model {a.dtype}: option probs {[round(p, 3) for p in probs]}", flush=True)
            for comp in a.components or fns:
                N = a.N_model if comp in BIG else a.N_component
                todo = [t for t in range(a.T) if (L, comp, t) not in done]
                if not todo:
                    continue
                if a.method == "ours":
                    rows = clear_measure(fns[comp], meter, N=N, T=len(todo), pause_s=a.pause_s,
                                         warmup_calls=a.warmup_calls)
                else:
                    rows = clear_loop(fns[comp], meter, N=N, T=len(todo), pause_s=a.pause_s)
                vals, t0 = [], time.time()
                for i, row in enumerate(rows):
                    row["trial"] = todo[i]
                    row["clock_reasons"] = int(pynvml.nvmlDeviceGetCurrentClocksEventReasons(meter.handle))
                    rec.write({"experiment": experiment, "method": a.method, "seq_len": L, "component": comp,
                               "input_ids": args[0][0].tolist() if comp == "full_model" and i == 0 else None,
                               **row})
                    vals.append((row["E_end_mJ"] - row["E_start_mJ"]) / N)
                v = torch.tensor(vals, dtype=torch.float64)
                trial_s = (time.time() - t0) / len(vals) - a.pause_s
                print(f"L={L:5d} {comp:15s} {v.mean():10.4f} mJ  std {v.std(unbiased=False):8.4f}  "
                      f"({100 * v.std(unbiased=False) / v.mean():5.1f}%)  ~{trial_s:.2f} s/trial", flush=True)
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
