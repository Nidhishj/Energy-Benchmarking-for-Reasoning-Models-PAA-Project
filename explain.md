# Medserver runs (RTX 6000 Ada), 2026-10-19/20: what happened, and how our B1 method differs from CLEAR's

Author: Nidhish. Companion to `baselines_results.md` (which covers the RTX PRO 5000 runs).

## 1. What was run

| Run | GPU | Result |
|---|---|---|
| `B3short/20261019-122925_B3short_gpu1_4546c8ed` | 1 | Complete (78 configs) |
| `B2b/20261019-122926_B2b_gpu0_feea8f05` | 0 | Complete, 180/180 rows. Stopped twice by other users' jobs (13:06, 22:59) and resumed with `--resume`; the interrupted trials were discarded |
| `B1/20261019-230539_B1_gpu1_8f44f5d3` | 1 | Complete, 5 lengths, T = 20, includes `layers_cycled` |

All three were started through `scripts/run_medserver.sh`. Each counter read first checks that no other process
is on our GPU (`energy/meter.py`, `GpuBusy`). If one appears, our run stops with exit code 3 and status `gpu_busy`;
the other process is never touched. So no saved row overlaps another user's job.

Tables: `results/tables/B1_20261019-230539_B1_gpu1_8f44f5d3.md`, `results/tables/B2b_xgpu.md`, `results/tables/B3_xgpu.md`.

## 2. Results

### B1: BERT-base, %Capture of the full model (B1 has no thinking mode; it is BERT, as in CLEAR)

| Tokens | CLEAR's method (12 × one layer) | Ours (12 layers in sequence) | layers in seq. ÷ (12 × one layer) | CLEAR paper |
|---|---|---|---|---|
| 8 | 81.6% | **92.3%** | 1.14 | 96.7% |
| 32 | 85.0% | **93.0%** | 1.10 | 93.8% |
| 64 | 86.6% | **94.1%** | 1.09 | 98.6% |
| 96 | 86.0% | **92.5%** | 1.08 | 97.2% |
| 128 | 87.8% | **95.2%** | 1.09 | 97.8% |

- Our method passes the 90% check at every length on this GPU; on the RTX PRO 5000 it passed from 64 tokens up (88–93%).
- The earlier medserver run (`B1/20261018-121417`) gave only 63–77% with CLEAR's method. Its single-layer energies
  were similar, but the full-model energy was inflated and jumpy (the GPU switched between power states of about
  80 W and 130 W between trials). This run did not show that.
- Std/mean ≤ 9.5% holds for the large parts (block, full model, layers in sequence: about 5%). Two tiny parts fail at
  128 tokens: `ffn_intermediate` 12.9% and `norm_all` 16.8% (under 3 mJ each; CLEAR's Appendix D expects this).

### B2b: Qwen3-8B, 30 MATH-500 problems, 3 trials per mode

| | Ada, thinking off | Ada, thinking on | PRO 5000, thinking off | PRO 5000, thinking on |
|---|---|---|---|---|
| Accuracy | 91.1% | 86.7% | 90.0% | 90.0% |
| Energy per answer | 4.13 kJ | 28.6 kJ | 3.02 kJ | 19.9 kJ |
| Energy per token | 6.78 J | 7.31 J | 4.96 J | 5.16 J |
| Speed | 36 tok/s | 36 tok/s | 60 tok/s | 58 tok/s |
| Average power | 243 W | 260 W | 297 W | 299 W |

- Thinking on ÷ off: median **7.1×** (Ada) vs **7.0×** (PRO 5000). The cost of reasoning relative to not reasoning
  does not depend on the GPU; the absolute cost does (about 40% more per answer on the Ada).
- Zeus vs NVML: within 0.02%. The PRO 5000 thinking-on numbers in `B2b_xgpu.md` come from 2 trials (the run stored here).
- Why the Ada is slower: batch-1 decoding reads all ~16 GB of weights per token, so it is limited by memory bandwidth
  (Ada 960 GB/s GDDR6 vs PRO 5000 about 1.34 TB/s GDDR7). The two setups also differ in software
  (torch 2.6/CUDA 12.4/driver 560 vs torch 2.8/CUDA 12.8/driver 595), so hardware and software effects are not separated.

### B3: energy per token vs output length

| GPU | Mode | First token (J), shortest / middle / longest prompt | Plateau (J/token) |
|---|---|---|---|
| RTX 6000 Ada | off | 6.66 / 7.11 / 10.53 | 6.87 |
| RTX 6000 Ada | on | 6.74 / 7.03 / 11.05 | 7.11 |
| RTX PRO 5000 | off | 5.37 / 6.30 / 9.10 | 5.03 |
| RTX PRO 5000 | on | 5.60 / 5.88 / 9.13 | 5.03 |

Same shape on both GPUs. **Open question:** on the Ada, B3-short at k = 30 gives about 6.1 J/token but the B2b
plateau gives about 7.0; on the PRO 5000 both give 5.0. B3-short ran on GPU 1 and B2b on GPU 0, so the two physical
cards may differ. Rerunning B3-short on GPU 0 (about 15 min) would settle it.

## 3. CLEAR's setup vs ours

CLEAR used **the same GPU as our medserver**: "Experiments were conducted on NVIDIA AdaLovelace GPUs (RTX 6000 Ada)"
(section 4); Appendix C adds an RTX 5000 Ada for the decoder models.

| | CLEAR (paper + `clear_repo/`) | Ours (`b1/run_b1.py`, `energy/`) |
|---|---|---|
| GPU | RTX 6000 Ada (96 MB L2) | same |
| Precision, N, T | FP16, N = 10,000 / 1,000, T = 20 | same |
| `torch.cuda.synchronize()` before each counter read | **No** (`module_measure.py` lines 68–71) | Yes |
| Which input is replayed | Captured activations | Exact args and kwargs of a real forward pass (pre-hooks, `with_kwargs=True`) |
| Attention implementation | Not stated (HF default of their version) | `sdpa` (fixed by `baselines.md`) |
| Software versions | Not stated | Recorded in every `results/sessions/*.json` |
| Captured(Model) | 12 × one layer (layer 5) + embedding + head | Also 12 layers replayed **in sequence** (`layers_cycled`) + embedding + head |
| Absolute energy, 128 tokens | block 41.5 mJ, full model 557 mJ | block 25.3 mJ, full model 370 mJ |

## 4. Why we think our method is right, and where it may be wrong

### Why 12 × one layer under-counts

- One BERT-base layer has about 7.1 M parameters, about **14 MB** in FP16. All 12 layers come to about 170 MB.
- Replayed alone 10,000 times, one layer's weights stay in the 96 MB L2 cache, so every call reads them from cache.
- In the real forward pass, the 12 layers (170 MB) cannot all stay in L2, so each layer reads its weights from GDDR
  memory. That costs more energy per byte.
- So "12 × one cached layer" is cheaper than what the model really does. Replaying the 12 layers in sequence makes them
  compete for the cache, as in the real model.
- Evidence:
  - Layers in sequence cost **8–14% more** than 12 × one layer on this GPU (11–22% on the PRO 5000).
  - Every layer replayed alone costs the same (13.4–13.8 mJ at 64 tokens on the PRO 5000, `b1/debug_capture.py`). So the gap is not
    "layer 5 is untypical".
  - encoder ÷ layers in sequence = 1.00–1.04, so there is no hidden cost between layers that replay misses.

### Why we consider our measurement more correct

1. **The validation is what CLEAR itself asks for:** the parts should add up to the whole. Ours do (92–95%);
   theirs, rerun on the same GPU, do not (82–88%).
2. **Sync before every read.** Without it the end reading can come before the GPU has finished the queued calls.
   With N = 10,000 this is probably a small error, but it is a real one, and it always goes in the same direction.
3. **Exact inputs.** We replay the exact arguments a module got in a real pass (including attention mask and
   keyword arguments), so the replayed work is the real work.
4. **Shared GPU handled.** Rows are rejected if another process was on the GPU, and the GPU state (power, clock,
   temperature) is logged per trial.

### Where our method may be wrong or incomplete

1. **We do not know why CLEAR got 97% on the same GPU.** Their energies are 35–50% higher than ours. A likely
   reason is extra fixed cost per call in their setup (older HF attention code, no sync, older drivers), which would
   make the cache saving a smaller share. But we have not shown this. Until we do, "CLEAR's method is wrong" is too
   strong; "it under-counts in a modern setup where one layer fits in L2" is what we can claim.
2. **Our fix only covers the layer total.** Attention, FFN and norms are still measured replayed alone, so their
   values are still "in-cache" numbers. The split between them may be biased for BERT. For the project this matters
   less: one Qwen3-8B layer is about 400 MB, far larger than L2, so even a layer replayed alone streams its weights
   from memory. This is a hypothesis to check in Experiment A, not a result.
3. **Not 100%.** 5–8% is still missing (kernel launches and Python between modules, the GPU being idle between
   kernels at batch 1, the power state of a full pass).
4. **Different energies from the paper.** Our absolute mJ cannot be compared directly with CLEAR's tables,
   because the software differs. Within-session comparisons are reliable (spread ≤ 3%); between sessions we see 1–9% drift.
5. **Small parts are noisy.** Parts under about 5 mJ have std/mean above 9.5%.

### Proposed check

Run B1 once with CLEAR's own settings on the medserver: their scripts from `clear_repo/` unchanged (no sync, HF
default attention), same GPU, same session as one of our runs. If their method then reaches about 97%, the
difference comes from their setup, and the cache effect we found holds only for a faster setup. If it still gives
82–88%, their 97% depends on something we cannot reproduce. Takes about 45 min on a free GPU.
