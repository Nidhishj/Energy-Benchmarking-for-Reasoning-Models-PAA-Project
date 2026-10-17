"""B2a: whole-model energy of BERT-base fine-tuned on MRPC, no thinking (baselines.md section 3).

Two measurements of the same model:
  1. Dataset pass: all 408 GLUE MRPC validation pairs at batch 1, --passes-per-window times in one window, read
     with raw NVML and Zeus at once (method "nvml"). Energy per example = window energy / (408 x passes). T trials. This is how dataset-level
     benchmarks such as AI Energy Score report energy.
  2. Per example: CLEAR's amplification (method "clear") on 5 examples at the 0/25/50/75/100% points of the
     token-length distribution, N calls per trial, T trials.
Accuracy and F1 come from an untimed pass that also serves as warm-up.

Usage:
  python b2/run_b2a.py
  python b2/run_b2a.py --T 2 --N 100 --idle-s 5 --tag test
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from energy import Recorder, clear_measure, get_meter  # noqa: E402
from energy.window import Window  # noqa: E402

DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/textattack--bert-base-uncased-MRPC")
    p.add_argument("--model-id", default="textattack/bert-base-uncased-MRPC")
    p.add_argument("--dtype", choices=DTYPES, default="fp16")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--N", type=int, default=1000, help="CLEAR repetitions per trial, per-example measurement")
    p.add_argument("--n-examples", type=int, default=5)
    p.add_argument("--passes-per-window", type=int, default=5,
                   help="dataset passes per energy window; one pass is ~0.65 s on an RTX PRO 5000, below the "
                        ">= 1 s needed for the energy counter to resolve it")
    p.add_argument("--pause-s", type=float, default=0.5)
    p.add_argument("--no-zeus", action="store_true")
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
    meter = get_meter(a.fake_meter)
    others = meter.other_processes()
    if others and not a.allow_busy:
        sys.exit(f"GPU in use by other processes {others}; pick a free GPU or --allow-busy")
    print(f"GPU: {meter.name}", flush=True)

    ds = load_dataset("nyu-mll/glue", "mrpc", split="validation")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, dtype=DTYPES[a.dtype],
                                                               attn_implementation=a.attn).to(a.device).eval()
    encs = [tok(r["sentence1"], r["sentence2"], return_tensors="pt").to(a.device) for r in ds]
    labels = [int(r["label"]) for r in ds]
    lengths = [int(e["input_ids"].shape[1]) for e in encs]

    def dataset_pass():
        return [int(model(**e).logits.argmax(-1)) for e in encs]

    # Untimed pass: accuracy, F1, warm-up.
    with torch.inference_mode():
        preds = dataset_pass()
    tp = sum(p == 1 and y == 1 for p, y in zip(preds, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(preds, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(preds, labels))
    acc = sum(p == y for p, y in zip(preds, labels)) / len(labels)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    print(f"MRPC validation: accuracy {100 * acc:.2f}%, F1 {100 * f1:.2f}% on {len(labels)} pairs", flush=True)

    window = Window(meter, use_zeus=not (a.no_zeus or a.fake_meter))
    print(f"Measuring idle power for {a.idle_s:.0f} s ...", flush=True)
    idle = meter.measure_idle(a.idle_s)
    print(f"  idle: {idle['mean_W']:.1f} W", flush=True)

    order = sorted(range(len(encs)), key=lambda i: (lengths[i], i))
    qs = [round(q * (len(order) - 1) / (a.n_examples - 1)) for q in range(a.n_examples)]
    picks = [order[q] for q in qs]

    config = {"dataset": "nyu-mll/glue mrpc validation", "n_pairs": len(encs), "T": a.T, "N": a.N,
              "per_example_idx": picks, "dtype": a.dtype, "attn_impl": a.attn, "batch_size": 1, "padding": False,
              "zeus": window.zeus is not None, "pause_s": a.pause_s,
              "passes_per_window": a.passes_per_window}
    model_info = {"id": a.model_id, "local_path": a.model, "revision": getattr(model.config, "_commit_hash", None),
                  "class": type(model).__name__, "dtype": a.dtype, "attn_impl": model.config._attn_implementation,
                  "n_layers": model.config.num_hidden_layers}
    experiment = "B2a" + (f"_{a.tag}" if a.tag else "")
    rec = Recorder(experiment, config, model_info, meter, idle, resume_dir=a.resume)
    done = rec.done_keys(["kind", "example_idx", "trial"])
    print(f"Run dir: {rec.dir}", flush=True)

    status = "failed"
    try:
        if ("accuracy", -1, 0) not in done:
            rec.write({"experiment": experiment, "kind": "accuracy", "example_idx": -1, "trial": 0,
                       "accuracy": acc, "f1": f1, "n": len(labels), "preds": preds})

        for trial in range(a.T):
            if ("dataset_pass", -1, trial) in done:
                continue
            time.sleep(a.pause_s)
            others = meter.other_processes()
            with torch.inference_mode(), window.measure() as w:
                for _ in range(a.passes_per_window):
                    dataset_pass()
            rec.write({"experiment": experiment, "kind": "dataset_pass", "method": "nvml", "example_idx": -1,
                       "trial": trial, "n_examples": len(encs) * a.passes_per_window,
                       "passes": a.passes_per_window, **w, **meter.state(), "other_processes": others})
            n = len(encs) * a.passes_per_window
            e = (w["E_end_mJ"] - w["E_start_mJ"]) / n
            z = f", zeus {w['zeus_mJ'] / n:.2f}" if "zeus_mJ" in w else ""
            print(f"dataset pass t{trial}: {e:.2f} mJ/example (nvml{z}), {w['t_end_s'] - w['t_start_s']:.2f} s",
                  flush=True)

        for idx in picks:
            todo = [t for t in range(a.T) if ("per_example", idx, t) not in done]
            if not todo:
                continue
            enc = encs[idx]
            vals = []
            for i, row in enumerate(clear_measure(lambda: model(**enc), meter, N=a.N, T=len(todo),
                                                  pause_s=a.pause_s, warmup_calls=100)):
                row["trial"] = todo[i]
                rec.write({"experiment": experiment, "kind": "per_example", "method": "clear", "example_idx": idx,
                           "seq_len": lengths[idx], "input_ids": enc["input_ids"][0].tolist(), **row})
                vals.append((row["E_end_mJ"] - row["E_start_mJ"]) / a.N)
            v = torch.tensor(vals, dtype=torch.float64)
            print(f"example {idx:3d} ({lengths[idx]:3d} tokens): {v.mean():.3f} mJ "
                  f"({100 * v.std(unbiased=False) / v.mean():.1f}%)", flush=True)
        status = "complete"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        rec.finish(status)
        print(f"Run {status}: {rec.dir}", flush=True)


if __name__ == "__main__":
    main()
