# Baselines — Formal Experiment Specification

**Project:** Component-Level Energy Benchmarking of Reasoning in LLMs
**Source of decisions:** `1.txt` (team decisions), formalised against CLEAR (Jain et al., arXiv:2510.02810)
**Budget:** **4 GPU-hours of planned work per GPU**, on 2 GPUs. About 6 h per GPU is available; the remaining ~2 h per GPU is reserve for reruns.
**Status:** items marked **[DECIDE]** still need a team decision. The default in brackets is assumed until then.

---

## 0. Scope and changes

The proposal (§4) had one baseline: reproduce Jin et al.'s whole-model energy for a base model vs. a reasoning (RT) model. This document replaces it with three baselines:

| ID | Baseline | Model | Dataset | Thinking | What it establishes |
|---|---|---|---|---|---|
| **B1** | CLEAR on BERT-base (authors' repo) | BERT-base | CLEAR's fixed prompt | n/a | Our component-level pipeline reproduces CLEAR's published numbers (Tables 8 and 12) |
| **B2** | Whole-model energy per query | BERT-base (B2a); Qwen3 (B2b) | MRPC (B2a); MATH500 (B2b) | B2a: none. B2b: **on and off, same problems** | Whole-model energy and accuracy, measured three ways (CLEAR-style, raw NVML, Zeus) |
| **B3** | Extended CLEAR Fig. 1: energy per token vs number of output tokens | Qwen3 | MATH500 (same runs as B2b) | on and off | How per-token energy changes from 1 output token up to full reasoning traces |

Changes from the first draft of this doc (team decision):
- **XSum removed.** It only had a thinking-off condition. Thinking on and off now always use the same dataset (MATH500). This loses CLEAR's controlled input lengths (64/128/256 tokens); input length becomes groups of MATH500's natural prompt lengths.
- **ReClor removed.** It duplicated MATH500's role (a reasoning LLM with thinking on vs off). MATH500 was kept because Jin et al. used it and it has 5 difficulty levels.
- **B2b and B3 share their generation runs.** Each thinking-on or thinking-off generation gives both energy per query (B2b) and the energy-vs-token curve (B3), by reading the energy counter at token checkpoints during the run.

Experiments A, B and C from the proposal build on these baselines and are not covered here.

---

## 1. Common protocol (applies to B1–B3)

### 1.1 Terms

| Symbol | Meaning |
|---|---|
| **T** | **Trials**: how many times the whole measurement is repeated, independently, with a 0.5 s pause between repeats. We report the mean and std across the T results. This is your "average of 8–10 runs". CLEAR uses T = 20. |
| **N** | **Repetitions within one trial** (CLEAR's amplification): the workload runs N times back to back between two energy readings, and energy per run = ΔE / N. Only used for workloads too short for NVML to resolve. |
| **k** | Number of output tokens generated so far |
| **E(k)** | Cumulative GPU energy to produce k output tokens, prefill included |

For example, "N = 10,000, T = 10" means: run the component 10,000 times, take the energy difference, divide by 10,000 — then do that whole thing 10 times and report mean ± std of the 10 values.

### 1.2 Measurement methods

Every energy number is tagged with the method that produced it.

| Tag | Method | When it applies |
|---|---|---|
| `clear` | NVML `nvmlDeviceGetTotalEnergyConsumption` read before and after **N repetitions**; energy per run = ΔE / N (CLEAR Eq. 3–4) | Workloads shorter than ~1 s: components, single forward passes, short forced-length generations |
| `nvml` | The same counter, read before and after **one** run | Workloads of ≥ 1 s: full generations, dataset passes |
| `zeus` | `ZeusMonitor.begin_window()` / `end_window()` around the same region as `nvml` | Same regions as `nvml`, as a software cross-check |

Rules:
- Call `torch.cuda.synchronize()` immediately before **every** energy read. Zeus does this by default; our NVML code must do it explicitly. The CLEAR repo's root scripts omit it.
- Do one untimed warm-up run of the exact workload before the first trial.
- `nvml` and `zeus` read the same hardware counter, so they **must agree within ~1%**. A larger gap means an instrumentation bug.
- Energy covers the **GPU only**, for every method. End-to-end boundaries can come out up to 2.4× higher (arXiv:2511.17179), which would break comparisons.

### 1.3 Consistency check

Report std/mean for every measurement. CLEAR's bar is std/mean ≤ 9.5% for measurements above 5 mJ (§5). Below 5 mJ, report std/mean but don't treat it as pass/fail (CLEAR Appendix D).

### 1.4 Fixed configuration (record in every run)

- **Batch size:** 1
- **Attention implementation:** `sdpa`, set explicitly. CLEAR measured 20–40% energy differences between implementations (Fig. 6).
- **Precision:**
  - B1/B2a: FP16, matching CLEAR.
  - Qwen3: BF16 (its native dtype). This differs from CLEAR's FP16; say so in the report.
- **Framework:** Hugging Face `transformers`. Not vLLM.
- **No other processes** on the GPU. Log `nvidia-smi` at the start of each session.
- **Idle power:** measure for 30 s at the start of each session and report it. Don't subtract it, which matches CLEAR.
- **Seeds:** fixed (`transformers.set_seed`). Same seed list on both GPUs.
- **Decoding** (Qwen3 model-card recommendations):
  - Thinking on: `temperature=0.6, top_p=0.95, top_k=20`
  - Thinking off: `temperature=0.7, top_p=0.8, top_k=20`

### 1.5 Run record (one JSON per run)

```
{ run_id, baseline (B1|B2a|B2b|B3), git_commit, exact_command,
  gpu_name, gpu_uuid, driver, cuda, torch, transformers, zeus versions,
  power_limit_W, idle_power_W, model_id, model_revision, dtype, attn_impl,
  dataset, split, example_ids, thinking (on|off), input_len, output_len,
  hit_token_cap, decoding params, seed, method (clear|nvml|zeus), N, T,
  trials_mJ[], mean_mJ, std_mJ, checkpoints [{k, E_mJ}], accuracy fields }
```

A number counts as **final** only after the same config has been run twice and agrees within the reported std. Store the exact command next to the result.

### 1.6 GPUs

| | GPU-A | GPU-B |
|---|---|---|
| Model / VRAM | **[DECIDE: fill in]** | **[DECIDE: fill in]** |
| NVML energy counter works? | Check at setup | Check at setup |

Qwen3-8B in BF16 needs ~16 GB of weights plus KV cache, so ≥ 24 GB is needed for 8k-token traces. Check that `nvmlDeviceGetTotalEnergyConsumption` returns a value on each GPU before anything else; it needs Volta or newer.

**Cross-GPU confound:** sampled generations can produce different traces on the two GPUs. Compare GPUs on **energy at equal k** (B3) and show trace-length distributions side by side, not only energy per query.

---

## 2. B1 — CLEAR on BERT-base (reproduction)

**Goal.** Reproduce CLEAR's BERT-base component energies with the authors' repo on both GPUs, to validate the pipeline that every later component-level experiment depends on.

**Setup.**
- Model: `bert-base-uncased`, `AutoModelForMaskedLM` (MLM head, as in CLEAR), FP16, batch 1.
- Input lengths: **32, 64, 96, 128 tokens** (CLEAR Table 12, BERT-base rows).
- Prompt: CLEAR's repeated prompt (`"The capital of France is"×10`), truncated to the target length.
- **T = 10.** This is lower than CLEAR's 20, to fit the budget.

**What to measure (`clear`).**

| Quantity | N | Notes |
|---|---|---|
| Embedding layer | 10,000 | Use the cached real `input_ids`. The repo currently uses random IDs. |
| Attention block (one layer) | 10,000 | `layer.attention` |
| FFN intermediate | 10,000 | `layer.intermediate` |
| FFN output | 10,000 | `layer.output` |
| Norm (all) | 10,000 | LayerNorms measured on their own. **Not in the repo yet.** |
| Measured (block) | 10,000 | The full `layer` |
| CLS + LM head | 10,000 | `model.cls` |
| Full model | 1,000 | `model(**inputs)`. Cross-check with `nvml` and `zeus` around the same loop. |

**Validation formulas** (checked against CLEAR Table 8: 12 × 29.802 + 5.918 + 23.666 = 387.2):
- Captured(Block) = Attention + FFN-int + FFN-out + Norm(all)
- %Capture(Block) = Captured(Block) / Measured(Block) × 100
- Captured(Model) = 12 × Measured(Block) + Embedding + CLS/LM head
- %Capture(Model) = Captured(Model) / Full model × 100

**Pass criteria.**
1. %Capture(Block) and %Capture(Model) ≥ 90% at every length.
2. std/mean ≤ 9.5% for components above 5 mJ.
3. **Relative** structure matches CLEAR: attention is the largest per-layer component, block energy grows with input length, and the head's share is similar. Absolute mJ will differ, because CLEAR used an RTX 6000 Ada.

**Repo fixes needed first:**
- Remove `print(model); exit(0)` in `store_activations.py`.
- Fix the `activations.pt` vs `activations2.pt` filename mismatch.
- Add the sync and warm-up from §1.2.
- Make sequence length, layer, dtype and output path parameters.
- Add the Norm measurement and the %Capture computation.

**Output.** Table B1: CLEAR Table 12 (BERT-base) side by side with GPU-A and GPU-B, plus %Capture and std/mean.

---

## 3. B2 — Whole-model energy per query

### B2a — BERT-base on MRPC (no thinking)

**Goal.** Whole-model inference energy for a small non-reasoning classifier. This is the reference point at the "no reasoning" end.

- **Model:** a BERT-base checkpoint already fine-tuned on MRPC from the Hugging Face Hub (inference only). **[DECIDE, default: `textattack/bert-base-uncased-MRPC`]**. Before using it, check that its validation accuracy matches its model card.
- **Data:** GLUE MRPC **validation split (408 pairs)**. GLUE test labels aren't public. Tokenise without padding.
- **Accuracy metrics:** accuracy and F1, the standard GLUE metrics for MRPC.

| Measurement | Method | Design |
|---|---|---|
| Dataset pass | `nvml` + `zeus` | All 408 examples at batch 1 in one window, **T = 10**. Energy per example = window energy / 408. |
| Per example | `clear`, N = 1,000 | **5 examples** spread across the token-length range, **T = 10** |

**What it shows:** whether dataset-averaged per-example energy (AI Energy Score style, via Zeus) matches CLEAR's per-example energy.

### B2b — Qwen3 on MATH500 (thinking on vs off)

**Goal.** Energy and accuracy per query for the same model, on the same problems, with thinking on vs off. This is Jin et al.'s base-vs-reasoning comparison on our hardware, with thinking as the only variable.

- **Model:** **[DECIDE, default: `Qwen/Qwen3-8B`]**. Its built-in `enable_thinking` switch means on and off use the same weights. Fallback: `Qwen3-4B`, if the speed check in §5 shows 8B won't fit the budget.
- **Data:** MATH500, **30 problems (6 per difficulty level)**, with fixed IDs. The **same 30 problems** are used for thinking on and off.
- **Max new tokens:** thinking on, 8192; thinking off, 2048. Report how often each cap was hit.
- **Grading:** the standard MATH500 answer-equivalence checker (e.g. Hugging Face `math-verify`). Record the checker and its version.

| Condition | Method | T |
|---|---|---|
| Thinking off | `nvml` + `zeus` per generation | **3** |
| Thinking on | `nvml` + `zeus` per generation | **2** |

**Why T is lower here than your 8–10:** each thinking-on generation runs for tens of seconds and uses tens to hundreds of joules. CLEAR (Appendix D) shows relative std falls as measured energy rises, so 2–3 trials are enough at this scale. **Check after the first 5 problems:** if std/mean > 5%, increase T and reduce the number of problems.

**Metrics, per condition:**
- accuracy
- output tokens (thinking and answer counted separately)
- energy per query (J)
- energy per output token
- energy per correct answer
- the **on/off ratio** of energy per query

Sanity check: Jin et al. report a 3.7–113× ratio and Oviedo et al. about 13×. A ratio outside that range is something to investigate first, not a finding.

---

## 4. B3 — Energy per token vs output tokens (extended CLEAR Fig. 1)

**Goal.** Reproduce the shape of CLEAR Fig. 1 on our GPUs and extend it from 30 output tokens to full reasoning traces. CLEAR shows energy per token E(k)/k falling as the one-time prefill cost is spread over more tokens, and levelling off at the decode cost per step. CLEAR never went past ~30 output tokens.

**Metrics for every curve:**
- **E(k)/k**: energy per token, the y-axis of CLEAR Fig. 1
- **ΔE(k)**: energy of the k-th token on its own. Required because E(k)/k is a running average, and a running average hides any growth with position.

### B3-long: checkpoints inside the B2b runs (no extra GPU time)

During each B2b generation, a `StoppingCriteria` callback (always returns False) syncs the GPU and reads the NVML counter at k ∈ {64, 128, 256, 512, …} and then every 512 tokens. That records E(k) along each real trace, thinking on and off.

### B3-short: forced-length region, k ≤ 30 (CLEAR's range)

Below ~64 tokens the counter is too coarse: a token takes ~25–30 ms, and NVML updates every 20–100 ms. So this region is measured separately:
- **3 MATH500 problems**: the shortest, middle and longest prompt among the 30. These stand in for CLEAR's three input-length curves.
- k ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30}. Force exactly k tokens with `min_new_tokens = max_new_tokens = k`.
- Thinking on and off.
- Method `clear`: repeat each generation N times so one trial lasts ≥ 1 s. **T = 5.**

### B3 outputs

1. **Fig. B3-1:** reproduction of CLEAR Fig. 1. E(k)/k vs k for k ≤ 30; 3 prompt lengths; thinking on/off; GPU-A and GPU-B.
2. **Fig. B3-2:** E(k)/k vs k on a log x-axis up to 8k tokens, thinking on vs off, combining B3-short and B3-long.
3. **Fig. B3-3:** ΔE(k) vs k for thinking-on traces. Does the cost per token grow with context? This is the whole-model precursor to Experiment A.
4. **Table B3:**
   - prefill energy E(1) per prompt
   - plateau value of E(k)/k
   - the k where E(k)/k is within 5% of the plateau
   - one row per GPU

---

## 5. Budget (per GPU, target ≤ 4 h)

Estimates assume Qwen3-8B decodes at ~30 ms/token and averages ~3k thinking tokens per MATH500 problem.

| Step | Design | Est. time |
|---|---|---|
| Setup | Idle power, NVML check, model downloads already cached | 10 min |
| **Speed check** | 3 thinking-on problems: measure tokens/s and average trace length | 10 min |
| B1 | 4 lengths × 8 quantities × T = 10 | 20 min |
| B2a | Dataset pass T = 10, plus 5 examples × T = 10 | 5 min |
| B2b + B3-long, thinking off | 30 problems × ~10 s × T = 3 | 15 min |
| B2b + B3-long, thinking on | 30 problems × ~90 s × T = 2 | 90 min |
| B3-short | 3 problems × 13 values of k × 2 modes × T = 5 × ~1.5 s | 10 min |
| Model loads, warm-ups | | 10 min |
| **Total** | | **~2 h 50 min** |
| Slack within the 4 h target | Reruns of failed configs, second run for the "final" check (§1.5) | ~1 h 10 min |

**Scaling rule after the speed check:** if a thinking-on problem averages more than 120 s, cut thinking-on to 20 problems (4 per level) or switch to Qwen3-4B. Record which option was used.

## 6. Order of execution (per GPU)

1. Setup and speed check
2. **B1**: nothing component-level is trusted until it passes
3. **B2a**: the cheapest check that `clear`, `nvml` and `zeus` agree
4. **B3-short**
5. **B2b + B3-long**, thinking off, then thinking on
6. Second run of anything that will appear in the report as final

## 7. Open decisions

| # | Decision | Default assumed here |
|---|---|---|
| D1 | GPU-A / GPU-B models | — (must fill) |
| D2 | MRPC checkpoint | `textattack/bert-base-uncased-MRPC`, after checking its accuracy |
| D3 | Reasoning model | `Qwen/Qwen3-8B` (fallback `Qwen3-4B`) |
| D4 | Number of MATH500 problems | 30 (6 per level) |
| D5 | Max new tokens | 8192 on / 2048 off |

## References

- Jain, Goyal, Pandey, Vaidhyanathan. *Dissecting Transformers: A CLEAR Perspective Towards Green AI.* arXiv:2510.02810. Methodology, Tables 8 and 12, Fig. 1, Appendix D.
- Jin, Wei, Brooks. *The Energy Cost of Reasoning.* arXiv:2505.14733. Base vs reasoning model, MATH500.
- Oviedo et al. *Energy Use of AI Inference, Efficiency Pathways, and Test-Time Scaling.* Joule 2026, arXiv:2509.20241.
- Chung et al. *Zeus.* NSDI 2023. `ZeusMonitor`.
- Luccioni et al. *AI Energy Score* (CodeCarbon, GPU-only, batch 1).
- Wang et al. *GLUE* (MRPC); Lightman et al. *Let's Verify Step by Step* (MATH500).
