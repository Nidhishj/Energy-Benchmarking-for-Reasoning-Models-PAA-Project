# To-Do — Baselines B1–B3

**Spec:** `baselines.md`. Section numbers (§) below refer to it.
**Budget:** ≤ 4 GPU-hours of planned work per GPU (~2 h 50 min estimated + slack). Everything in Phases 0–3 and 6 runs **off-GPU** or with minutes of GPU time only.
**Leads** (from the proposal; everyone works across tasks): **A** = Aaditya (pipeline, NVML/Zeus), **N** = Nidhish (runs, data collection), **V** = Virat (analysis, plots, report).

Each task lists who leads it, what it depends on, and what "done" means.

---

## Phase 0 — Decisions (before anything else)

- [ ] **D1: name GPU-A and GPU-B** (model, VRAM). **Lead: all.** Done when: filled into `baselines.md` §1.6.
- [ ] **D2: confirm the MRPC checkpoint** (default `textattack/bert-base-uncased-MRPC`). **Lead: N.**
- [ ] **D3: confirm the reasoning model** (default `Qwen/Qwen3-8B`, fallback `Qwen3-4B`). **Lead: all.**
  - If either GPU has < 24 GB, switch to 4B now rather than after the speed check.
- [ ] **D4/D5: confirm 30 MATH500 problems and the 8192 / 2048 token caps.** **Lead: all.**

## Phase 1 — Environment (no GPU time)

- [ ] **Create the project repo.** **Lead: A.**
  - Layout: `energy/` (measurement library), `b1/`, `b2/`, `b3/`, `configs/`, `results/`, `analysis/`.
  - Copy `CLEAR-D487/` into `b1/clear_repo/` with its original commit noted, so our fixes are visible as a diff.
- [ ] **Pin dependencies** in `requirements.txt` with exact versions. **Lead: A.**
  - Packages: `torch`, `transformers` (a version with Qwen3 support), `datasets`, `zeus`, `nvidia-ml-py`, `math-verify`, `tqdm`.
  - Use `nvidia-ml-py` (the maintained NVML binding), not the CLEAR package's `nvidia-ml-py3`.
- [ ] **Pre-download all models and datasets** on each GPU machine, so no GPU hours go to downloads. **Lead: N.**
  - Models: `bert-base-uncased`, the MRPC checkpoint, Qwen3-8B (+ 4B fallback).
  - Datasets: GLUE MRPC validation; MATH500 (`HuggingFaceH4/MATH-500`).
- [ ] **Fix the MATH500 problem list.** **Lead: N.**
  - Sample 6 problems per level, with seed 0, and save the IDs to `configs/math500_ids.json`.
  - Also save the 3 B3-short problems: the shortest, middle and longest prompt, measured after applying the chat template.
  - Done when: the file is committed, so both GPUs use identical problems.
- [ ] **Write the prompt templates** into `configs/`. **Lead: N.**
  - MATH500 template, with `enable_thinking=True/False`.
  - Decoding parameters for each mode (§1.4).

## Phase 2 — Shared measurement code (`energy/`)

- [ ] **`clear_measure(fn, N, T)`**: sync, read NVML, run `fn` N times, sync, read NVML, divide by N; 0.5 s pause between trials; one warm-up run first. **Lead: A.**
- [ ] **`window_measure(fn)`**: one run inside NVML reads *and* a Zeus window at the same time, so both numbers come from the same run. **Lead: A.**
- [ ] **`NvmlCheckpoint` StoppingCriteria** (always returns False). It syncs and logs `(k, E)` at k ∈ {64, 128, 256, 512, …}, then every 512 tokens. **Lead: A.**
  - Measure its overhead once (on vs off on one generation) and record it.
- [ ] **`run_record.py`**: writes the §1.5 JSON. **Lead: A.**
  - Fills in GPU, driver and library versions automatically, plus the git commit and the exact command line.
- [ ] **`gpu_check.py`**: confirms the NVML energy counter works, measures 30 s of idle power, and logs `nvidia-smi`. **Lead: A.**
- [ ] **`seed.py`**: one `set_seed` used everywhere. **Lead: A.**

## Phase 3 — Experiment code (written and tested off-budget)

### B1 — CLEAR on BERT-base (§2)
- [ ] **Fix the repo bugs.** **Lead: A.**
  - Remove `print(model); exit(0)` from `store_activations.py`.
  - Unify the activations filename.
  - Replace the measurement loops with `energy.clear_measure` (this adds sync and warm-up).
  - Run the embedding on the cached real `input_ids`, not random IDs.
- [ ] **Make the scripts take arguments:** `--seq-len`, `--layer`, `--dtype`, `--out`. **Lead: A.**
- [ ] **Add a separate measurement of all LayerNorms.** **Lead: A.**
- [ ] **Write the `%Capture` aggregator** (§2 formulas). **Lead: V.**
  - Done when: run on CLEAR's own Table 8 numbers, it gives 387.2 mJ captured and 96.7% capture.
- [ ] **Write one command that runs B1 end to end.** **Lead: A.**
  - It runs all 4 lengths × 8 quantities with T = 10 and writes one JSON per quantity.

### B2a — MRPC (§3)
- [ ] **Dataset-pass script:** 408 validation pairs, batch 1, no padding, inside `window_measure`, T = 10. **Lead: N.**
- [ ] **Per-example script:** `clear_measure` with N = 1,000, T = 10, on 5 examples spread across lengths. **Lead: N.**
- [ ] **Accuracy and F1 check against the model card.** **Lead: N.**
  - Done when: they match the card within 1 point. If they don't, pick another checkpoint (D2).

### B2b + B3-long — MATH500 (§3, §4)
- [ ] **Generation runner:** loads `math500_ids.json`, generates with thinking on or off, and wraps each generation in `window_measure` plus `NvmlCheckpoint`. **Lead: N.**
  - Per run, records: output tokens (thinking and answer counted separately), whether the token cap was hit, and the checkpoint `(k, E)` list.
- [ ] **Grading** with `math-verify`, recording its version. **Lead: N.**
  - Extract the answer after `</think>` (thinking on) or from the whole output (thinking off).
  - Count an unparseable answer as wrong and report how many there were.
- [ ] **Resume support:** skip `(problem, mode, trial)` combinations that already have a JSON. **Lead: N.**
  - Needed so a crash doesn't burn budget.

### B3-short (§4)
- [ ] **Forced-length runner:** `min_new_tokens = max_new_tokens = k` for k ∈ {1, …, 30}; picks the smallest N that makes a trial last ≥ 1 s and records that N; T = 5; thinking on and off. **Lead: A.**

### Pre-flight test (≤ 10 min of GPU, counted inside the setup budget)
- [ ] **Run every script once on tiny settings** on GPU-A: B1 at 1 length with T = 2; B2a on 10 examples; 1 MATH500 problem in each mode with Qwen3-0.6B. **Lead: A + N.**
  - Done when: every JSON is written, Zeus and NVML agree within 1%, and the aggregator runs on the output.

## Phase 4 — GPU session, GPU-A (target ≤ 4 h)

Run in this order (§6). Tick each step and note the actual time taken.

- [ ] **Setup:** run `gpu_check.py` (idle power, NVML check). ~10 min. **N**
- [ ] **Speed check:** 3 thinking-on problems; record tokens/s and average trace length. ~10 min. **N**
  - If a thinking-on problem averages > 120 s, apply the scaling rule (§5): either 20 problems or Qwen3-4B. Record the choice in the run log.
- [ ] **B1:** ~20 min. **A**
  - Gate: %Capture ≥ 90% and std/mean ≤ 9.5% for components > 5 mJ. **If it fails, stop and debug before going on.**
- [ ] **B2a:** ~5 min. **N**
  - Gate: Zeus and NVML within 1%.
- [ ] **B3-short:** ~10 min. **A**
- [ ] **B2b + B3-long, thinking off:** 30 problems, T = 3. ~15 min. **N**
- [ ] **B2b + B3-long, thinking on:** 30 problems, T = 2. ~90 min. **N**
  - After 5 problems, check std/mean. If it's > 5%, raise T and cut problems.
- [ ] **Reproducibility rerun:** repeat 1 B1 length, the B2a dataset pass, and 3 MATH500 problems per mode. Done when they match within the std (§1.5). **N**
- [ ] **Copy `results/`** off the machine and commit. **N**

## Phase 5 — GPU session, GPU-B (target ≤ 4 h)

- [ ] **Repeat Phase 4 exactly:** same commit, same `math500_ids.json`, same seeds. **N**
- [ ] **Log any deviation from GPU-A** (model fallback, fewer problems), with the reason. **N**

## Phase 6 — Analysis (no GPU time)

- [ ] **Table B1:** CLEAR Table 12 (BERT-base) next to GPU-A and GPU-B, with %Capture and std/mean. **V**
- [ ] **B2a summary:** energy per example from the dataset pass vs from CLEAR's method; accuracy and F1. **V**
- [ ] **B2b table,** per mode and per GPU: **V**
  - accuracy
  - mean output tokens
  - J/query
  - J/token
  - J/correct answer
  - on/off ratio, compared with Jin et al. (3.7–113×) and Oviedo et al. (~13×)
- [ ] **Fig. B3-1:** CLEAR Fig. 1 reproduction (k ≤ 30, 3 prompts, on/off, both GPUs). **V**
- [ ] **Fig. B3-2:** E(k)/k on a log x-axis up to 8k tokens, on vs off. **V**
- [ ] **Fig. B3-3:** ΔE(k) vs k for thinking-on traces. **V**
- [ ] **Table B3:** E(1), plateau E(k)/k, and k at 5% of the plateau, per GPU. **V**
- [ ] **Cross-GPU comparison at equal k,** plus trace-length histograms per GPU (§1.6 confound). **V**
- [ ] **Each figure and table lists the `run_id`s and commands behind it.** **V**

## Phase 7 — Write-up

- [ ] **Baselines section of the report:** setup, the three measurement methods, results, and deviations from CLEAR. **V**
  - Deviations to list: T = 10 vs 20; BF16 for Qwen3; natural MATH500 prompt lengths instead of fixed 64/128/256.
- [ ] **Limitations:** **V**
  - NVML counter resolution
  - GPU-only boundary
  - small problem count (30)
  - T = 2–3 for long runs
- [ ] **Hand-off note for Experiments A–C:** which code and results they reuse (`energy/`, `NvmlCheckpoint`, the B1 pipeline). **A**

---

### Quick status

| Phase | Owner(s) | GPU time | Status |
|---|---|---|---|
| 0 Decisions | all | 0 | ☐ |
| 1 Environment | A, N | 0 | ☐ |
| 2 Shared code | A | 0 | ☐ |
| 3 Experiment code + pre-flight | A, N, V | ≤ 10 min | ☐ |
| 4 GPU-A session | N, A | ≤ 4 h | ☐ |
| 5 GPU-B session | N | ≤ 4 h | ☐ |
| 6 Analysis | V | 0 | ☐ |
| 7 Write-up | V, A | 0 | ☐ |
