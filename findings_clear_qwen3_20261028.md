# CLEAR's method on Qwen3-8B, and how much work CLEAR's loop misses (medserver, 2026-10-27 to 2026-10-28)

Author: Nidhish. Follows `findings_20261027.md`. GPU 1, RTX 6000 Ada (same card as the CLEAR paper), torch 2.8.0+cu128,
transformers 4.57.1, driver 560.35.03, FP16, layer 5, T = 20. Every number comes from a saved run; nothing is hand-entered.
All energy is a difference of the NVML energy counter (never power × time); W = energy / time is only a diagnostic.

## 1. Runs

| Run | What | Status |
|---|---|---|
| `B1clearQwen3/20261027-234148_B1clearQwen3_gpu1` | CLEAR's BERT scripts (`clear_repo/`) adapted to Qwen3-8B (`b1/clear_qwen3/`), their loop unchanged | Complete, 8/32/64/96/128 tokens |
| `B1clearQwen3tail/20261028-131634_B1clearQwen3tail_gpu1` | **Run A**: the same loop, plus a record of the GPU work it never counts (`TAIL=1`) | Complete; the 128-token full-model step was stopped by another user's job and rerun alone on GPU 1 (same settings) |

Tables: `results/tables/B1clearQwen3_20261027-234148_B1clearQwen3_gpu1.md`, `results/tables/B1clearQwen3tail_20261028-131634_B1clearQwen3tail_gpu1.md`.

CLEAR's loop (`clear_repo/module_measure.py`): `sleep(0.5)` → read counter → N calls (10,000 parts, 1,000 full model) → read counter →
divide by N; no warm-up, no `torch.cuda.synchronize()`. Each script is its own process, as the authors ran them.

## 2. How CLEAR's two checks are computed (verified against their tables)

- **%Capture(Block)** = (attention + MLP + norms, each measured alone) ÷ (whole layer measured alone).
- **%Capture(Model)** = (L × *whole measured layer* + embedding + LM head + final norm) ÷ full model.
  The model check uses the whole layer as one piece, not the sum of its parts:
  - Table 8 (BERT): 12 × 29.802 + 5.918 + 23.666 = 387.21 ✓ (with the parts it would be 360.8).
  - Table 14 (Qwen2.5-3B, 32 tokens): 36 × 113.174 + 249.853 = 4,324.12 ✓.
  - Table 15 (Llama-3.2-3B): 28 × 121.860 + 306.882 = 3,718.96 ✓.
  - Their code measures `model.bert.encoder.layer[5]` as a whole (`clear_repo/layer_measure.py` lines 38–83). The aggregation script is
    not released (`analyse_activations.py` only prints shapes), so this rule comes from the table arithmetic.
- Consequence: a block check of 80% and a model check of 91% are not a contradiction. The model check never uses the parts, so it says
  nothing about whether attention + MLP + norms add up. "36 × parts" below shows what the model check would be if it did.
- Paper claim to test (section 5): "%Capture at both the block and model level consistently remains above 90%". Qwen3-8B is not in the paper.

## 3. Results: %Capture

| Tokens | Block (run 1) | Model (run 1) | Block (Run A) | Model (Run A) | Block, Run A + missed work | Model, Run A + missed work | Model with 36 × parts (Run A) |
|---|---|---|---|---|---|---|---|
| 8 | 89.7 | 93.6 | 91.8 | 91.5 | 93.0 | 91.7 | 84.4 |
| 32 | 79.0 | 90.7 | 79.2 | 91.3 | 79.7 | 91.4 | 73.3 |
| 64 | 82.4 | 89.8 | 83.3 | 89.8 | 83.9 | 90.0 | 75.6 |
| 96 | 108.7 | 94.6 | 106.8 | 97.3 | 107.7 | 97.5 | 103.4 |
| 128 | 112.6 | 98.7 | 111.5 | 98.5 | 112.3 | 98.9 | 108.8 |

- The two runs agree within ~2 points, so the result is repeatable.
- The model check stays near 90–98%; the block check does not (79–113%). The block check is the one that tests whether the parts add up.

## 4. Run A: how much work CLEAR's loop misses

**Why work is missed.** CUDA calls are asynchronous: the CPU puts kernels in a queue (~1,000 operations deep) and moves on; when the queue
is full, the CPU waits inside the launch call. CLEAR reads `E_end` as soon as the CPU has *queued* the N calls. Whatever is still in the queue
runs after `E_end`: during the 0.5 s pause it is counted by no trial; if it lasts longer than the pause, the rest is counted by the next trial.

**How Run A measures it** (`b1/clear_qwen3/common.py`, `measure_energy_tail`). CLEAR's window is untouched (nothing waits on the GPU inside it):
- a CUDA event after the last call; during the pause we poll it (no wait) and read the counter the moment the GPU finishes;
- GPU time per call from CUDA events; calls missed = GPU time after `E_end` ÷ GPU time per call;
- SM clock, temperature and clock-limit reasons read right after `E_end`;
- after the last trial, one sync, since that tail is also lost in CLEAR.

**Calls missed per trial** (out of N = 10,000; full model out of 1,000):

| Component | 8 tok | 32 | 64 | 96 | 128 | Tail finished inside the pause | 300 W power cap active |
|---|---|---|---|---|---|---|---|
| **LM head** | **410 (4.1%)** | **356 (3.6%)** | **334 (3.3%)** | **235 (2.4%)** | **240 (2.4%)** | **1/20 trials** (rest spills into the next trial) | 20/20 |
| **MLP** | **193 (1.9%)** | **163 (1.6%)** | **119 (1.2%)** | **125 (1.3%)** | **126 (1.3%)** | 20/20 | 20/20 |
| Embedding | 26 (0.26%) | 22 | 26 | 27 | 24 | 20/20 | 0/20 |
| Each norm | ~6 (0.06%) | ~6 | ~6 | ~6 | ~6 | 20/20 | 0/20 |
| Attention | ~1 (0.01%) | 1 | 1 | 1 | 1 | 20/20 | 0/20 |
| Whole layer 5 | ~1 (0.01%) | 1 | 1 | 2 | 14 (0.14%) | 20/20 | from 32 tokens |
| Full model | 0 | 0 | 0 | 0 | 0 | 20/20 | from 64 tokens |

- Only parts with **few operations per call** lose work: the LM head is one matrix multiply (~1,000 calls fit in the queue), the MLP is
  5 operations (~200 calls). Attention, norms, the layer and the full model launch many small operations, so the CPU stays close to the GPU.
- The LM head's backlog is longer than the 0.5 s pause in 19 of 20 trials, so its measured trials also contain the previous trial's work.
- Adding the missed work back changes %Capture by **at most ~1.2 points** (section 3). **The missing sync is real but small; it does not
  explain the drift.**

## 5. What does explain the drift

1. **Power cap when parts run alone.** The 300 W limit (`sw_power_cap`) is active for the MLP and LM head in every trial (thermal flags
   inactive). Alone, they run at lower clocks than inside the layer, and the clock falls further as the card warms:
   MLP 2,662 → 2,160 → 1,080 → 765 → 720 MHz (8 → 128 tokens); LM head 2,505 → ~540 MHz; whole layer 2,745 → 2,228 MHz; attention always 2,505 MHz.
   Parts measured at a different clock than inside the layer do not add up to it → block check above 100% at 96–128 tokens.
2. **The GPU waits for the CPU on small parts.** Attention (~440 µs per call at 84–138 W) and the norms (~62 µs at 70–96 W) are launch-bound:
   measured alone, their energy is mostly "GPU awake × time". Inside the layer that waiting is shared → block check below 90% at 32–64 tokens.
3. The missing sync (section 4): ≤1.2 points.

## 6. What should be done instead (for our own component numbers)

1. `torch.cuda.synchronize()` before every counter read.
2. CUDA-graph replay for small parts, so launch time does not dominate.
3. Fixed or at least recorded clocks (`nvidia-smi -lgc` needs the GPU to ourselves; not done, other users share the box).
4. Use the parts-based check (sum of parts ÷ layer, and 36 × parts + rest ÷ model) as the main validation.
5. Measure several layers, not only layer 5.
6. Report idle and the GPU's awake base power next to every number.
7. Best: measure parts inside the running model by subtraction, E(part) ≈ E(full) − E(full with that part stubbed out), and report the
   residual as interaction.

## 7. Open items

1. Optional Run B: CLEAR's method on Qwen2.5-3B, to compare number for number with their Table 14 (~6 GB download, ~1.5 h).
2. Questions for the authors (see `findings_20261027.md` 4.5), plus: power limit and clocks during their runs; how the LM head was measured
   for decoders.
