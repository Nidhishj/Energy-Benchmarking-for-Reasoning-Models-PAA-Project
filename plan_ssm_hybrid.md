# Plan: energy benchmarking of Mamba / SSM and hybrid reasoning models

Status: proposal (2026-11-27). Nothing has been run yet.
Companion files: `remaining_work.md`, `plan_jepa.md`.

## 1. Why this follows from our results

Our main finding on Qwen3-8B (a dense transformer):
- A decode step's energy depends only on context length L: c(L) ≈ a + b·L, with 4.89 J at 256 → 10.87 J at 32k on the PRO 5000.
- The **only** component that grows is **attention**, because it reads the KV cache (147 KB per token, 4.8 GB at 32k). The MLP is flat.
- So a long reasoning answer costs E(T) ≈ a·T + b·T²/2, superlinear in length. Late thinking tokens are the most expensive, and wrong answers (the longest) pay most.

**State-space models (Mamba, Mamba-2) replace attention with a recurrent mixer that has a fixed-size state.**
- Each new token updates a constant-size state (conv state + SSM state per layer), instead of reading an ever-growing KV cache.
- So a pure SSM's per-token cost should be **flat in L**.
- **Hybrid** models keep a few attention layers, so their per-token cost should grow, but with a much smaller slope b.

This turns our findings into testable hypotheses:

| | Transformer (measured) | Hybrid / SSM (hypothesis) |
|---|---|---|
| J/token vs L | 2.2× from 256 to 32k | flat (SSM) or a small slope (hybrid) |
| Attention share at 32k | 67% | small, only the few attention layers |
| Energy of a T-token answer | ~a·T + b·T²/2 | ~a·T (linear) |
| Extra cost of wrong (long) answers | superlinear in length | proportional to length |
| Savings from cutting late thinking | larger than the token share | equal to the token share |
| Correct vs wrong at equal L (Exp B) | identical | identical (compute is data-independent) |

The last rows matter for the stopping idea in `remaining_work.md` §3.1. The energy-aware stopping rule only beats a token rule if c(L) rises, so on SSMs it should collapse to the token rule. That's a clean cross-architecture prediction.

## 2. What exists already (literature check, 2026-11-27)

| Work | What it measures | Energy? | Gap for us |
|---|---|---|---|
| *Characterizing SSM and Hybrid LM Performance with Long Context* ([2507.12442](https://arxiv.org/pdf/2507.12442)) | Transformer vs Mamba-2 vs hybrid (Falcon-H1) on an RTX 4090, long contexts | **Yes, whole model.** At 57k context: transformer 1,492.6 J vs Mamba-2 370.5 J. Transformers are faster at short context (~1k). | No component split, no reasoning traces, no thinking on/off |
| *Benchmarking the Computational and Representational Efficiency of SSMs* ([2601.01237](https://www.arxiv.org/pdf/2601.01237)) | Fitted scaling laws, transformer vs Mamba | Analytical. Crossover ≈ 220 tokens (memory), ≈ 370 tokens (time). | Not measured energy |
| *LLM-Induced Transients* ([2409.11416](https://arxiv.org/pdf/2409.11416)) | GPU power, Mamba-2.8B vs GPT-Neo-2.7B, vs batch size | Power traces | Not reasoning, not components |
| **M1** ([2504.10449](https://arxiv.org/abs/2504.10449), NeurIPS 2025) | Mamba hybrid reasoning model distilled from R1 distills, then RL | **No.** >3× generation speed vs a same-size transformer (vLLM); claims memory-bound decoding | No energy at all |
| **Nemotron Nano 2** ([2508.14444](https://arxiv.org/abs/2508.14444)) | Hybrid Mamba-2/Transformer reasoning model, 9B | **No.** Up to 6× throughput vs Qwen3-8B in reasoning settings, similar accuracy | No energy; throughput only |
| **Phi-4-mini-flash-reasoning** (SambaY) | Mamba-1 + sliding-window attention + gated memory units | No | Has a same-family transformer twin (Phi-4-mini-reasoning) |

**Gap we fill.** No published work, as far as these searches show:
- measures **component-level energy** (mixer vs attention vs MLP) of SSM/hybrid models during decoding at growing context;
- compares thinking on vs off, or correct vs wrong, for SSM/hybrid **reasoning** models;
- tests whether early stopping saves energy differently on SSMs than on transformers.

All the claims for M1 and Nemotron Nano 2 are throughput, not joules. On a power-capped GPU those may differ (see our "joules = 300 W × time" finding). This needs verifying with full-text reads before writing it up.

### 2b. Hardware work on Mamba (accelerators, not GPU energy measurement)

| Work | What it does |
|---|---|
| Mamba ([2312.00752](https://arxiv.org/pdf/2312.00752)) | Hardware-aware selective scan that keeps expanded states in SRAM, not HBM. Memory traffic is the design driver. |
| Mamba-X ([2508.02977](https://arxiv.org/html/2508.02977v1)) | Vision-Mamba edge accelerator with a **systolic scan array** (Kogge-Stone). Self-reported ~11.5× energy efficiency vs edge GPUs. |
| eMamba ([2508.10370](https://arxiv.org/html/2508.10370v1)) | Edge Mamba accelerator (GF 22 nm), reported against ViT/CNN accelerators |
| FastMamba ([2505.18975](https://arxiv.org/pdf/2505.18975)) | Mamba-2 accelerator on an FPGA |
| XAMBA ([2502.06924](https://arxiv.org/html/2502.06924v4)) | Running SSMs on existing NPUs |

These report self-measured or simulated energy on custom hardware, for vision or small models. **None measures component energy of SSM reasoning models on GPUs during long decoding**, so our gap stays open. They are relevant to `plan_systolic.md`, since the scan maps poorly onto a GEMM systolic array.

### 2c. Can we use the CLEAR code for Mamba?

- **The CLEAR method: yes.** Cache a module's exact input, replay it N times between synced energy reads, divide by N, and check that the parts add up to the whole. That's architecture-agnostic. The CLEAR README itself says the code "can be generalized to differences in transformer architecture" with changes.
- **The CLEAR authors' code (`clear_repo/`): not directly.**
  - It is hard-coded for BERT: `AutoModelForMaskedLM`, hooks on `layer.attention.self` and `intermediate.dense`.
  - It handles only a single fixed input, with no KV cache and no decode step.
  - It knows nothing about Mamba layers or their recurrent state.
- **Our own implementation is the one to extend:** `energy.clear_measure` (the replay loop, already general) plus `expa/run_decode_components.py` (decode-step capture with forward pre-hooks and cache handling). Changes needed:
  1. hooks for the Mamba mixer and its sub-modules (`in_proj`, `conv1d`, scan, `out_proj`), alongside attention and MLP;
  2. hybrid-cache handling: KV for the attention layers (the existing `FrozenCache` idea), plus conv and SSM state for the Mamba layers, updated in place. Validate the replay with the state evolving vs restored each call (§4.3);
  3. a `layer_type` map, so each type is measured once and multiplied by its count.
- Keep `clear_repo/` unchanged; it is the reference copy of the CLEAR authors' code.

## 3. Models (must fit one 48 GB GPU in BF16)

| Pair | Hybrid / SSM | Transformer control | Why |
|---|---|---|---|
| **P1 (main)** | `nvidia/NVIDIA-Nemotron-Nano-9B-v2`: Mamba-2 + a few attention layers, reasoning on/off | **Qwen3-8B** (all our existing data) | Same size class, both have a thinking toggle, and NVIDIA benchmarks against Qwen3-8B. Reuses all our Qwen3 runs and traces as the baseline. |
| **P2** | `microsoft/Phi-4-mini-flash-reasoning` (3.8B, SambaY) | `microsoft/Phi-4-mini-reasoning` (3.8B transformer) | Same family and data, architecture swapped: the cleanest A/B |
| P3 (optional) | M1-3B (Mamba hybrid reasoning) | DeepSeek-R1-Distill-Qwen-1.5B (the M1 paper's baseline) | Pure-Mamba-heavy end of the spectrum |

Excluded:
- Qwen3-Next-80B-A3B (Gated DeltaNet hybrid + MoE): ~160 GB in BF16, too big.
- Falcon-H1 reasoning variants: not verified yet. Check what exists.

To verify before starting:
- exact layer counts and types (how many attention layers in Nemotron Nano 2);
- chat-template thinking switches (`/think` vs `/no_think` for Nemotron);
- recommended sampling settings;
- licences.

## 4. Engineering issues (do first)

1. **Fast kernels or the numbers are meaningless.**
   - HF Mamba/Mamba-2 layers fall back to a slow pure-PyTorch scan without `mamba-ssm` and `causal-conv1d`.
   - These need a CUDA 12.x build that matches our torch 2.8 and driver 560 (Ada server).
   - Check that the fast path is actually taken: a log message or a timing check.
2. **HF support.** Confirm the transformers version supports each model natively (NemotronH, Phi4Flash); otherwise use `trust_remote_code`. Pin versions in `run.json`.
3. **The cache is different.**
   - Hybrids use a hybrid cache: KV for the attention layers, plus a conv state and SSM state per Mamba layer.
   - Our `FrozenCache` shim (`expa/run_decode_components.py`) covers attention only.
   - A Mamba mixer updates its state **in place** at decode. Replaying it N times evolves the state N steps instead of repeating the same step.
   - The compute and bytes per step are data-independent, so the cost is the same. But check it: replay with (a) the state left evolving and (b) the state restored from a snapshot each call (copy cost measured separately and subtracted). The two should agree.
4. **Rebuilding the state at L.** Prefill the first L tokens (chunked) to get the KV and SSM states, as now. Then capture module inputs with forward pre-hooks for each **layer type** (Mamba layer, attention layer, MLP).
5. **Tokenizers differ across models.** Compare per answer (J/query, J/correct) and per *position*. Never compare J/token directly between Qwen and Nemotron tokens; report characters per token if needed.

## 5. Experiments

Throughout, keep the same measurement rules as our existing work:
- NVML with sync before each read, Zeus as a cross-check;
- T = 20 and ~2 s per trial for component replays;
- idle power logged;
- a repro gate;
- energy never mixed across GPUs.

### S0. Setup and whole-model baseline (B2b analogue)
- MATH-200 and AIME 2024+2025, thinking on and off, same token limits as Qwen3 (on 32k, off 8k), same seeds scheme, graded with `math-verify`.
- Per mode: accuracy (95% CI), J/query, J/correct, tokens, J/token, energy checkpoints E(k) every 512 tokens.
- **Output:** a side-by-side table for Qwen3-8B vs Nemotron-Nano-9B-v2 on accuracy, energy per answer and energy per correct answer.
- **Cost:** AIME took ~9–11 h per copy for Qwen3. If the hybrid is 2–3× faster, ~4–5 h. MATH-200 another few hours. Run one copy first.

### S1. Component energy per decode step vs context (Exp A analogue)
- Runner: generalise `expa/run_decode_components.py` to hybrids.
- Contexts 256 … 32k on one long saved trace.
- Components:
  - Mamba layer, and inside it if possible: in_proj, conv1d, SSM scan, gate/norm, out_proj;
  - attention layer;
  - MLP;
  - norms, embedding, LM head;
  - the full step.
- Measure one layer of each type and multiply by the count of that type. Layer check: two layers of each type.
- Validation: block and model %Capture ≥ 90%, std ≤ 5%.
- **Prediction:** Mamba-layer energy flat in L, attention-layer energy growing as in Qwen3, and the whole-model slope b much smaller than Qwen3's.
- **Key figure:** J/token vs L for Qwen3-8B and the hybrid on the same GPU, with a stacked component split.
- **Cost:** ~1 h per trace, as for Exp A.

### S2. Whole-answer cost model and thinking on vs off
- Price every saved answer as Σ step(L) with S1's curves and check against measured energy (as `compare_on_off.py`).
- Report attention / Mamba / MLP shares per answer: thinking on vs off, correct vs wrong.
- **Question:** is thinking's extra cost purely linear in length for the hybrid?

### S3. Correct vs wrong at equal L (Exp B analogue)
- Same-question pairs (one correct and one wrong seed), measured at the same contexts.
- **Prediction:** identical within noise. This is a sanity check that SSM compute is data-independent.

### S4. Cutting thinking (Exp C analogue) on AIME
- Fixed caps 2k/4k/8k/16k with forced `</think>`, as `expc/run_caps.py`.
- **Prediction:** the energy saved by a cap is closer to proportional to the tokens removed (no "expensive tail"). Compare savings at equal accuracy with Qwen3's 22–26% at 16k.

### S5. Stopping rules across architectures (links to `remaining_work.md` §3.1)
- Run E2 (energy-aware vs token-based stopping) on both architectures.
- **Prediction:** the energy-aware rule helps on Qwen3 (rising c(L)) but matches the token rule on the hybrid (nearly flat c(L)).

### Optional
- **Batch size.** SSM state is tiny, so larger batches fit. Energy per token vs batch for both architectures. Out of our batch-1 scope, but cheap once S0 runs.
- **Prefill energy** vs prompt length. Transformers are faster at short context per 2507.12442.

## 6. Order and effort

| Step | GPU time | Depends on |
|---|---|---|
| Kernel install, model loading, fast-path check | — (½–1 day of setup) | — |
| S1 on one Nemotron trace (needs one long thinking trace: generate one AIME answer first) | ~1–2 h | setup |
| S0 AIME, one copy | ~4–6 h (estimate) | setup |
| S2, S3 | S2 offline; S3 ~1 h per trace | S0, S1 |
| S4 | ~same as Caps_aime per copy, if the hybrid is faster | S0 (needs live E(k) checkpoints) |
| P2 (Phi-4 pair): S0 + S1 | ~1 day | setup |

Suggested first milestone: S1 on Nemotron-Nano-9B-v2, next to our existing Qwen3-8B Exp A curve on the same GPU. That's one figure that directly tests the main hypothesis.

## 7. Risks
- Kernel and build problems with `mamba-ssm` on torch 2.8 / CUDA 12.x / driver 560. Fallback: run on the PRO 5000 machine (driver 595), or pin an older torch in a separate env.
- Accuracy differences between models confound J/correct. Report J/query, J/token-position and accuracy separately.
- The in-place state update makes component replay less clean than for attention (see §4.3). Validate with %Capture and with live checkpoints.
- Throughput claims (6×) came from vLLM / batched serving. At batch 1 in HF the gain may be much smaller. That's a finding, not a failure.
