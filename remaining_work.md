# Remaining work after Evaluation 2 (as of 2026-11-27)

What the Eval 1 and Eval 2 slides promised and is not done yet, plus new ideas.

- Eval 1 slides: `eval1_presentation-1.pdf` (slides 7–9, roadmap). Not in the repo.
- Eval 2 slides: `presentation/eval2_presentation.pdf` (slide 10 "Remaining plan", slide 11 "Still to establish").
- Status file: `eval2_status.md`.

Legend: ✅ done · 🟡 partly done · ❌ not started

---

## 1. From the Eval 1 slides

| # | Promised | Status | What is left |
|---|---|---|---|
| 1 | Component measurement inside Qwen3-8B decode steps (parts ≥ 90%) | ✅ | — Block 93.7–104.6%, model 93.5–99.6% on both GPUs |
| 2 | Thinking on/off on 200–500 problems with a higher token limit, 95% ranges | ✅ | — MATH-200 with capped answers rerun at 8k/32k (`B2b_n200_uncapped.md`): on 98.5%, off 90.0% |
| 3 | Exp C: cut thinking at 25/50/75% and 512–4k, accuracy vs energy, per MATH level | ✅ | — `C_main.md` (Ada, 81 traces), `C_pro5000.md` (52 of 54) |
| 3b | Exp C: **which component's energy drops** when thinking is cut (`CLAUDE.md`) | ❌ | Price each Exp C budget with the Exp A component curves (offline, `compare_on_off.py`-style) |
| 4 | Exp A: component energy vs position (512…8k) | ✅ | — 256…32k, both GPUs |
| 6 | Exp B: correct vs wrong, per component at equal position | 🟡 | One trace per class, from three different questions. See 2.1 |
| 7 | Ablation of the measurement fixes (sync, in-order replay, CUDA graphs) | 🟡 | Sync and in-order replay tested on BERT in Eval 1. Not yet on Qwen3 decode; CUDA graphs not built. See 2.3 |
| 8 | Nsight Compute: L2 cache hit rate, to confirm the cache explanation (Eval 1 appendix A3) | ❌ | Not started. Could also verify the attention bandwidth estimate (~0.7 TB/s at 32k) |
| 9 | A harder dataset where thinking off fails | ✅ | — AIME 2024+2025 (off 25%, full 68%) |
| 10 | GPU choice / baselines on the GPU used | ✅ | — Repro gate on the PRO 5000 passed (4.6%); key results on both cards |
| 11 | Repo layout shown on slide 9 (`src/`, `baseline/reference_impl/`) | ❌ | The real layout is `energy/`, `llm/`, `expa/`, `expc/`, `b2/`, `analysis/`. Either restructure or document the real layout in the README/report |

---

## 2. From the Eval 2 slides (slide 10)

### 2.1 Comparisons

| Item | Status | What is left |
|---|---|---|
| **More traces per class (Exp B)** | ❌ | Measure correct vs wrong answers to the **same question**. 13 AIME problems have one of each among the two seeds. First pair: `aime2024/I-8` t0 ✅ (26.4k, already measured) vs t1 ❌ (23.6k). Then 3–5 pairs plus 3–5 thinking-off traces, giving mean ± spread per class at each L. Runner: `expa/run_decode_components.py`, ~1 h per trace. The slide also lists a second wrong trace, `aime2025/II-8` t0. |
| Uncapped MATH-200 | ✅ | — Reruns in `B2b_n200_8k_off/`, `B2b_n200_32k/`, `B2b/20261118-203317_…`. Merged with `analyze_b2b.py --replace`. Slides 6/10/12 still show the old numbers. |
| **Live vs replayed step at 8–32k** | ❌ | The replay matches live (B3) within 5% up to 4k. From 512 to 8k the replay rises +27% vs +13% live. Compare on the same answer (`aime2025/I-11`) and the same GPU using the live E(k) checkpoints recorded in the AIME B2b runs (every 512 tokens up to 32k). Offline. |

### 2.2 Trade-offs

| Item | Status | What is left |
|---|---|---|
| **AIME energy vs accuracy by difficulty** | 🟡 | Numbers are in `Caps_aime.md`: the cheapest cap that keeps accuracy is 8k for problems 1–5 and 16k for 6–10 and 11–15, saving ~20–30%. Needs a table and figure, plus an evaluation of a "difficulty-aware cap". Offline. |
| **A stopping rule that pays for itself** | 🟡 | Agree-twice with full forced answers tested offline (`stop_rules.py`): 65% accuracy, but 6–17% *more* energy than full thinking. Next: cheap probes (≤64 tokens) and literature signals (entropy, hidden-state probes). See 3. |

### 2.3 Ablations (Exp A runner, ~1 h per setting)

| Item | Status | Question |
|---|---|---|
| KV cache: `DynamicCache` (`torch.cat` every step) vs `StaticCache` | ❌ | How much of attention's growth (25% → 67%) is the per-step cache copy? |
| Attention kernel: SDPA vs FlashAttention-2 vs eager | ❌ | Does the kernel change the slope of attention's growth or the launch overhead? |
| Measurement fixes off one at a time (sync, in-order replay, CUDA graphs) | ❌ | Which fix makes %Capture go from 74–88% (CLEAR) to 93–99% (ours)? |
| Host: CUDA graphs on the Ada | ❌ | Is the Ada's flat ~25 ms step up to 8k CPU launch overhead? (Inferred from timing, not yet profiled; Nsight Systems would also confirm it.) |
| Layer 0 / 18 / 35 | ✅ | — within 2.2% |

### 2.4 Other open items (`eval2_status.md`)

| Item | Status |
|---|---|
| Last 2 of 54 Exp C traces on the PRO 5000 (resume `C/20261118-093151_C_gpu0_98bf22f5`) | ❌ Needs one PRO 5000 slot (~10 min) |
| `analysis/analyze_expb.py` (equal-L table across traces, layer check, Ada vs PRO). Exp B tables were made by hand; `analyze_a_decode.py` fails on the layer-check runs, which have no `full_step`. | ❌ |
| Exp A repeat on the Ada (`I-11`, run-to-run check; also GPU 0 vs GPU 1) | ❌ |
| Per-answer component split on the PRO 5000 | ✅ `A_on_vs_off_math30_pro5000.md` |
| Server repo on the PRO machine: `git stash -u && git pull --ff-only` | ❌ |
| Update the Eval 2 slides / report with the uncapped MATH-200 numbers | ❌ |
| Eval 3 / final report | ❌ |

---

## 3. New ideas

### 3.1 Energy as a stopping signal, and energy-aware stopping

**Motivation.**
- Exp B showed a decode step's energy depends only on its position L, not on content: wrong, correct and thinking-off traces agree within 0.4% at equal L. That's expected for a dense model with fixed shapes.
- So the energy meter cannot tell *whether* to stop.
- But Exp A gives the exact *price of continuing* at each position. That price grows with L because of the KV cache: c(L) ≈ a + b·L with a ≈ 4.84 J and b ≈ 1.8×10⁻⁴ J/token² on the PRO 5000. The next 1k tokens cost ~4.9 kJ at L = 0 but ~10.8 kJ at L = 31k, and a trace costs E(T) ≈ a·T + b·T²/2, superlinear in length.
- Existing stopping papers price thinking in tokens, as if every token cost the same. None uses position-dependent joules.

**Background: the entropy signal used in the literature** (EAT, DEER, Conformal Thinking).
- At checkpoint t, append a forced ending such as `</think>\n\nThe final answer is \boxed{`.
- Compute the entropy H = −Σ p log p of the model's next-token distribution over the answer tokens.
- Low and stable H means the model has decided, so stop.
- It costs a few tokens per check; afterwards the probe tokens are cropped from the KV cache.
- This entropy is a property of the model's output (content), not of the hardware.

#### Experiment E1: does energy carry information about correctness? (expected: no)
- **Data:**
  - Live J/token per interval from the E(k) checkpoints already recorded in the AIME B2b runs (120 thinking traces).
  - Per-token logit entropy from one teacher-forced pass over each saved trace: no regeneration, ~1–2 h GPU.
- **Analysis** (within one GPU copy, never mixing GPUs):
  1. Energy residual = live J/token − c(L) from the Exp A curve of the same GPU.
  2. Correlation (Pearson and Spearman) between the energy residual and the mean entropy per interval.
  3. AUC for final correctness from energy-residual features vs entropy features.
- **Expected result:** energy AUC ≈ 0.5, entropy AUC > 0.5, correlation ≈ 0. That would be a measured negative result: *the energy meter is content-blind; stopping signals must come from the model.*
- **Extension (not planned):** in a mixture-of-experts model, energy is data-dependent through expert routing and might carry signal. The dense 8B model is the clean control.

#### Experiment E2: energy-aware vs token-based stopping
- **Signals at every 1k-token checkpoint of the 120 saved AIME thinking traces.** One GPU pass, reusing the append-then-`cache.crop` logic of `run_decode_components.py`:
  - EAT entropy of the answer tokens after a forced `</think>…\boxed{`;
  - final-layer hidden state (for a linear probe, as in *Reasoning Models Know When They're Right*, 2504.05419);
  - a ≤64-token forced answer, graded. This gives the label "correct if stopped here".
  - Cost: ~14 checkpoints × 64 tokens × 120 traces ≈ 110k tokens plus one prefill per trace, ~1–2 h per GPU.
- **Probe:** logistic regression with leave-problem-out cross-validation (both copies of a problem in the same fold).
- **Policies, simulated offline:**
  - fixed caps (have);
  - agree-twice with full probes (have);
  - entropy or probe with a **token** threshold (stop when p_t ≥ τ);
  - entropy or probe, **energy-aware**:

    continue ⇔ λ · (p̂_future − p_t) > ∫_L^{L+Δ} c(ℓ) dℓ

    The rule gets stricter as L grows. It includes a **give-up branch**: low p_t at large L means force the answer now. Wrong thinking answers are 33% of answers but 55% of thinking energy.
- **Energy accounting:** Σ c(L) up to the stop point plus the cost of each probe, priced with the Exp A curves. This pricing reproduces measured energy at 92% (Ada) and 99–105% (PRO); also check it against the live E(k) of each trace.
- **Metrics:**
  - accuracy-vs-joules frontier per policy (sweeping τ and λ);
  - joules per correct answer at matched accuracy (68%);
  - savings vs the 16k cap (currently 22–26%);
  - share of the savings that comes from the give-up branch.
- **Every outcome is reportable:**
  - energy-aware beats token-based: a new result;
  - equal: token count is a sufficient proxy for stopping decisions, and we report joule savings the papers don't measure;
  - neither beats 16k: consistent with *When Does Learning to Stop Help?* (2606.30852).
- **Caveats:**
  - only 120 thinking traces, hence cross-validation by problem and confidence intervals on the frontier;
  - forced answers can keep reasoning after `</think>` (2609.03633); unparseable probes count as wrong, and their number is reported.

**Related work to read first:**
- *Early Stopping in Large Reasoning Models: A Survey and Empirical Audit* (2611.00166)
- EAT and DEER (entropy and confidence)
- Dynasor-CoT / Certaindex (consistency)
- *Reasoning Models Know When They're Right* (2504.05419) and *Temporal Predictors of Outcome* (2511.14773) (hidden-state probes)
- *Conformal Thinking* (2602.03814)
- *When More Thinking Hurts* (2604.10739)
- *EcoThink* (2603.25498) (energy-budgeted reasoning)
