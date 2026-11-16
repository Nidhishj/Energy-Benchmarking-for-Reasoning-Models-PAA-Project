"""B1 glue test: measure directly the energy of BERT layer code that runs outside every measured part.

In B1 the whole layer costs ~3.6-4.7 mJ (~55 µs) more per call than attention + FFN-intermediate + FFN-output measured
separately. In BERT the residual adds and both LayerNorms are inside those parts, so the difference is code that runs
only in BertLayer.forward (apply_chunking_to_forward, output tuples). Here, in one session and with B1's inputs:
  block      the whole layer (as in B1)
  attention, ffn_intermediate, ffn_output   the three parts (as in B1)
  glue       BertLayer.forward with the three parts replaced by stand-ins that return their saved outputs
             without any GPU work, so each call runs only the layer's own code
Measured with energy.clear_measure (100 warm-up calls, sync before every counter read), N = 10,000, T = 20.

Usage:
  CUDA_VISIBLE_DEVICES=1 python b1/glue_measure.py                      # 8 and 128 tokens
  CUDA_VISIBLE_DEVICES=1 python b1/glue_measure.py --seq-lens 8 --T 2 --tag test
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import pynvml
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from energy import GpuBusy, GpuMeter, Recorder, clear_measure  # noqa: E402
from run_b1 import build_inputs, capture_inputs, make_fns  # noqa: E402

COMPONENTS = ["block", "attention", "ffn_intermediate", "ffn_output", "glue"]


class _Saved(torch.nn.Module):
    """Returns a saved output and does nothing else (no GPU work)."""

    def __init__(self, out):
        super().__init__()
        self.out = out

    def forward(self, *args, **kwargs):
        return self.out


def glue_fn(layer, args, kwargs):
    """The layer's own forward with attention / intermediate / output replaced by their saved outputs."""
    saved = {}

    def keep(name):
        def hook(m, a, o):  # returns None, so the module's output is left unchanged
            saved.setdefault(name, o)
        return hook

    handles = [getattr(layer, n).register_forward_hook(keep(n)) for n in ("attention", "intermediate", "output")]
    with torch.inference_mode():
        real = layer(*args, **kwargs)
    for h in handles:
        h.remove()
    glue = copy.copy(layer)  # same attributes, own module table
    glue._modules = dict(layer._modules)
    for name in ("attention", "intermediate", "output"):
        glue._modules[name] = _Saved(saved[name])
    with torch.inference_mode():
        out = glue(*args, **kwargs)
    if not torch.equal(out[0], real[0]):
        sys.exit("glue layer does not reproduce the real layer output")
    return lambda: glue(*args, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/google-bert--bert-base-uncased")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[8, 128])
    p.add_argument("--layer", type=int, default=5)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--N", type=int, default=10000)
    p.add_argument("--idle-s", type=float, default=30.0)
    p.add_argument("--tag", default="")
    a = p.parse_args()

    meter = GpuMeter(0)
    if meter.other_processes():
        sys.exit(f"GPU {meter.nvml_index} is in use by other processes {meter.other_processes()}")
    meter.strict = True
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForMaskedLM.from_pretrained(a.model, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    idle = meter.measure_idle(a.idle_s)
    print(f"GPU {meter.name} (nvml {meter.nvml_index}); idle {idle['mean_W']:.1f} W", flush=True)

    config = {"seq_lens": a.seq_lens, "layer": a.layer, "T": a.T, "N": a.N, "dtype": "fp16", "attn_impl": "sdpa",
              "prompt": "The capital of France is", "repeat": 30, "components": COMPONENTS, "pause_s": 0.5,
              "warmup_calls": 100, "sync_before_read": True}
    model_info = {"id": "google-bert/bert-base-uncased", "local_path": a.model, "class": type(model).__name__}
    experiment = "B1glue" + (f"_{a.tag}" if a.tag else "")
    rec = Recorder(experiment, config, model_info, meter, idle)
    print(f"Run dir: {rec.dir}", flush=True)
    status = "failed"
    try:
        for L in a.seq_lens:
            inputs = build_inputs(tok, "The capital of France is", 30, L)
            cache = capture_inputs(model, inputs, a.layer)
            fns = make_fns(model, inputs, cache)
            fns["glue"] = glue_fn(cache["modules"]["block"], *cache["inputs"]["block"])
            for comp in COMPONENTS:
                vals, secs = [], []
                for row in clear_measure(fns[comp], meter, N=a.N, T=a.T):
                    row["clock_reasons"] = int(pynvml.nvmlDeviceGetCurrentClocksEventReasons(meter.handle))
                    rec.write({"experiment": experiment, "method": "ours", "seq_len": L, "layer": a.layer,
                               "component": comp, **row})
                    vals.append((row["E_end_mJ"] - row["E_start_mJ"]) / a.N)
                    secs.append((row["t_end_s"] - row["t_start_s"]) / a.N)
                v = torch.tensor(vals, dtype=torch.float64)
                print(f"L={L:4d} {comp:17s} {v.mean():8.3f} mJ  std {v.std(unbiased=False):6.3f}  "
                      f"{1e6 * sum(secs) / len(secs):6.1f} µs/call  {v.sum() / sum(secs) / 1000:5.0f} W  "
                      f"SM {row['sm_clock_MHz']} MHz", flush=True)
        status = "complete"
    except GpuBusy:
        status = "gpu_busy"
        raise
    finally:
        rec.finish(status)
        print(f"Run {status}: {rec.dir}", flush=True)


if __name__ == "__main__":
    main()
