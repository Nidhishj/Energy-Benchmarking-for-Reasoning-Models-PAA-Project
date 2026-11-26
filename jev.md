# Jev and open "System One" decision models: what they are, existing work, and an energy plan

Status: literature check and proposal (2026-11-27). Nothing new has been run, but see §4: we already measured **Laya** in October.

**Source caveat.** These models are newer than the assistant's training data. Everything below comes from the Hugging Face docs, arXiv papers, model cards and blog or roundup articles found on 2026-11-27. Many numbers are vendor or third-party claims; the most uncertain ones are marked *(unverified)*. Check model cards before use.

(Not to be confused with JEPA, which `plan_jepa.md` covers. That file came from a misreading of "jev" and is now only a side idea.)

---

## 1. What Jev is

**Jev** (by **TypeSafe**) is a **"System One" decision model**.
- Instead of generating an answer token by token, it takes a **state** (the situation), a **question** and a **set of options**.
- It returns a **calibrated probability for each option**, in a **single forward pass**, without text generation.
- Three question types:
  - **Choice:** 2–16 options;
  - **Noul:** yes/no;
  - **Score:** ordered levels 0–5.

**TypeSafe's Jev itself is closed (hosted API only).**
- Weights, size and training recipe are not public.
- TypeSafe describes a "parallel sampler" and **RLCD** (Reinforcement Learning for Calibrated Decisions) *(vendor claim, unverified)*.
- Open question: one independent latency table lists the Jev API with 40 output tokens, which conflicts with "no generation". No source explains it.

### Architecture of the open Jev-style models (verifiable)
1. **Backbone:** an ordinary pretrained LM, usually a decoder-only Qwen or Gemma. Laya is an exception: a ModernBERT encoder.
2. **Input:** one prompt with structured fields, e.g. `{"state": …, "question": …, "options": [...]}`, rendered with the chat template and ending at a marker token.
3. **Decision head:**
   - take the final hidden state h at the marker or last token;
   - a small linear head scores the option slots, typically initialised from the LM-head rows of the label tokens (false/true, 0–5, A–P);
   - soft-cap the scores, `c·tanh(s/c)`, then softmax over the allowed options.
   - Example: JEV-9B's head has 24 slots, about 98k parameters.
4. **Inference = one prefill pass. No autoregressive decode, no growing KV cache.**
   - In HF Transformers: `JevForDecision`.
   - Some models (Jeff 1.1) have an optional "reasoning" mode: generate ≤ 512 tokens, then decide.
5. **Training (open models):** LoRA on the backbone plus the head, with KL or cross-entropy against soft targets, often distilled from Jev's own output distributions. Usually no RL.

**Why this matters for us.** It's the opposite of what we measured:

| | Qwen3 reasoning (our Phase 1–2) | Decision model |
|---|---|---|
| Work per query | prefill + thousands of decode steps | **one prefill pass** |
| Bottleneck | memory-bound decode (~1 FLOP/byte) | compute-bound prefill (GEMMs over the whole prompt) |
| KV cache | grows; attention up to 67% at 32k | not kept |
| Energy driven by | thinking length | prompt length × number of options |

---

## 2. Open alternatives (as reported on 2026-11-27)

| Model | Who | Backbone / size | Licence / weights | Notes |
|---|---|---|---|---|
| **Kev 1.0** | Jared Palmer | Family 0.8B–27B; Qwen3.5 bases (small sizes), Qwen3.8-27B (largest) | Apache-2.0 on HF (e.g. `jaredpalmer/kev-4b`); code `github.com/jaredpalmer/kev` | Serves the same `POST /v1/systemone` API as Jev, so the TypeSafe SDK works unchanged. An earlier release reportedly used Qwen2.5-0.5B with sizes 0.5B/0.6B/4B/8B; **sources disagree on the sizes**. One benchmark: Kev rises from ~50% (0.8B) to ~77% (27B); Kev-27B is reported level with Jev on a decision index *(unverified)*. |
| **JEV-9B / JEV-27B** | AutoTrust AI | Qwen3.5-9B (text tower) / Qwen3.8-27B, LoRA + 24-slot head | Apache-2.0, `autotrust/JEV-9B` (18 GB), `autotrust/JEV-27B` (54 GB) | A distilled student of Jev 1.13; "shares no weights or code" with TypeSafe. ~90 ms per decision, 2.5 ms per decision batched ×128 on a B200 (self-reported). |
| **Jeff 1.1** | Geuter et al. ([2610.17205](https://arxiv.org/html/2610.17205)) | Gemma 4 31B / 12B, Qwen3.6-27B, Qwen3.5-4B | HF collection `jgeuter/jeff-11-…`, code `github.com/j-geuter/Jeff` | Open recipe (LoRA + head, no RL). p50 21–57 ms per decision on an H200, vs ~190 ms for the Jev API including network. Optional ≤512-token reasoning mode. |
| **Jeva-4B** | Li et al. ([2610.18773](https://arxiv.org/html/2610.18773)) | Qwen3.5-4B encoder + one decision block | `LIQIIIII/Jeva-4B` *(link in abstract, unverified)*, code `github.com/LiQiiiii/Jeva` | "Decision-native": encodes options separately (cacheable), with a coverage gate for "none of these" |
| **Laya** | Convai Innovations | **ModernBERT encoder, 28 layers (10 global, 18 sliding-window 128), hidden 1024, 421M params**, typed decision heads | Apache-2.0, `convaiinnovations/laya` (we have it in `models/`) | **Already measured by us**, see §4 |
| **pplx-decider v1.1 / v1.2 27B** | Perplexity | 27B | `perplexity-ai/pplx-decider-v1.1-27b` | DecideBench: v1.2 at 85–99% across option-count buckets *(third-party)*. 27B in BF16 ≈ 54 GB, **does not fit one 48 GB GPU** without quantization. |
| GEV-26B-Decide | AutoTrust AI | 26B | `autotrust/GEV-26B-Decide` | Listed in the HF JEV docs |
| d3 ("Decision 3.0") | vLLM semantic-router | ? | `vllm-sr/d3` | DecisionPivot: 71.3% pair accuracy *(third-party)* |
| Clef | Cloudflare | ? | on HF *(unverified)* | Mentioned in roundups |
| SII-Decision-27B, NanoJev, LightJev, JevForge-0.8B, Instant, SubQ Snap, Mammoth-1 | various | various | various | Mentioned in benchmark papers and roundups; not checked |

---

## 3. Existing work: has anyone measured energy?

**Accuracy and latency benchmarks of decision models exist; none measures energy or power.**

| Work | Measures | Energy? |
|---|---|---|
| *General Decision Models: Benchmarking and Insights Beyond Jev* ([2610.03935](https://arxiv.org/html/2610.03935v2)) | Accuracy of Jev vs open models (Kev, Laya, NanoJev, …) | No |
| *DecisionPivot* ([2610.28522](https://arxiv.org/pdf/2610.28522)) | Diagnostic accuracy of one-pass decision models | No |
| **DecideBench** ([github.com/choyiny/decidebench](https://github.com/choyiny/decidebench)) | 400 contrastive decisions: accuracy, **cost per task, latency**; decision models vs LLMs | No (cost in $, not joules) |
| *Benchmarking System One decision models against classifiers and LMs for decision gates* ([2610.00346](https://arxiv.org/pdf/2610.00346)) | Accuracy vs trained classifiers and LLMs | No |
| *Decision models vs generative LLMs on medical MCQ* ([2610.38448](https://arxiv.org/pdf/2610.38448)) | Accuracy | No |
| *TandemBench* ([2611.05827](https://arxiv.org/pdf/2611.05827)) | Decision models in multi-agent collaboration | No |
| *Releasing the Inner Jev* (Jeff 1.1), *Jeva*, *Jev in the Wild* | Recipes, latency, ecosystem | No |
| JEV-9B model card | Latency, throughput, training GPU-hours | No |

The related general energy work all concerns **generative** models, not decision models:
- *How Hungry is AI?* ([2505.09598](https://arxiv.org/pdf/2505.09598)): reasoning raises energy per query by roughly 1–2 orders of magnitude;
- Jin et al. ([2505.14733](https://arxiv.org/pdf/2505.14733));
- *Where Do the Joules Go?* ([2601.22076](https://arxiv.org/pdf/2601.22076));
- *Profiling LLM Agent Inference on Blackwell* ([2609.29707](https://arxiv.org/pdf/2609.29707)).

**Conclusion: the energy of System One decision models (J per decision, and its component split), and how it compares with generating the decision with a reasoning LLM, appears to be unmeasured.** The only measurement we know of is ours (§4).

---

## 4. What we already have: Laya (October 2026)

From `results/tables/Laya_english.md`, `indic_models/run_laya.py` and commit `094d169`. RTX 6000 Ada (GPU 1), BF16, batch 1, prompts of 64–512 tokens with a 3-option question, CLEAR method and ours (warm-up + sync), T = 20.
- **One decision costs 1.34–2.32 J** (full model, 64 → 512 tokens), about 15.6–16.2 ms.
- **Attention is 63–76% of a layer's parts**, but the step is **launch-bound**: time per call is flat (~340 µs for attention) as tokens grow, and power is only 70–180 W, well below the 300 W cap.
- Sliding-window and global layers cost the same, because SDPA computes full attention with a mask.
- Decision head ≈ 6% of the model.
- Capture checks:
  - model, all layers in sequence: 96–111%;
  - block: 76–94% (below 90% at 64 and 512 tokens).
- **Rough scale** (different tasks, so order of magnitude only):
  - Laya: ~2 J per decision;
  - Qwen3-8B, thinking off, MATH-200: ~6.2 kJ per answer;
  - Qwen3-8B, thinking on: ~36 kJ;

  i.e. about **10³–10⁴× apart**.

---

## 5. Plan

### V0. Setup
- Separate env, because Qwen3.5 backbones and `JevForDecision` need a newer `transformers` than our 4.57. Keep `/scratch/nidhish/envs/energy` unchanged.
- Models that fit one 48 GB card in BF16:
  - Kev (0.8B, 4B, ~9B), JEV-9B, Jeff-1.1-Qwen-4B, Jeff-1.1-Medium (12B), Laya (have);
  - 27B models (Kev-27B, JEV-27B, pplx-decider-27B) only with 8-bit, or skip.
- Same measurement rules as before: NVML with sync, Zeus cross-check, idle logged, a repro gate, `nvidia-smi` check before each run.

### V1. Energy per decision: decision model vs generating the decision
- **Task:** a decision benchmark with fixed options, e.g. DecideBench (400 contrastive decisions, if the data is public) and a multiple-choice set (MMLU-style or the medical MCQ set).
- **Arms on the same GPU:**
  - Kev-4B / Kev-9B, JEV-9B, Jeff-1.1-Qwen-4B, Laya (one pass each);
  - **Qwen3-8B thinking off** (generate the letter);
  - **Qwen3-8B thinking on** (reason, then the letter);
  - optionally Jeff's ≤512-token reasoning mode.
- **Metrics:**
  - accuracy (95% CI), **J per decision**, **J per correct decision**;
  - latency, power;
  - batch 1 and a batched setting (decision models batch well).
- **Question:** how much energy does "deciding in one pass" save vs reasoning, and at what accuracy cost? This is the decision-model counterpart of our thinking on/off result.

### V2. Component energy of the single pass (CLEAR's own setting)
- One forward pass on a fixed input is exactly what CLEAR was designed for (its BERT example; our Laya run).
- Components: attention, MLP, norms, embedding, decision head; layer checks; %Capture ≥ 90%.
- **Sweeps:** prompt length (128 → 4k tokens) and number of options (2 → 16).
- **Same-backbone control:** Kev-4B (Qwen3.5-4B + head) vs Qwen3.5-4B generating the answer. Same weights, so it isolates "decide in one pass" vs "generate" (prefill only vs prefill + decode).
- **Question:** prefill is compute-bound. Does the card reach its power cap, as decode does? What's the attention share vs prompt length? (Compare with Laya's launch-bound short prompts.)

### V3. Decision with a small thinking budget
Jeff 1.1's optional reasoning (≤ 512 tokens, then decide), measured as energy vs accuracy. It links decision models back to our Exp C (cutting thinking).

### V4. Link to `plan_systolic.md`
Prefill-only decision models are large-M GEMMs, which map well onto systolic arrays, unlike batch-1 decode. Simulate one decision on the same SCALE-Sim configs as a contrast case.

### Effort
- V0: ~½–1 day (env, downloads: ~18 GB per 9B model).
- V1: hours of GPU (one pass per decision is cheap). The Qwen3-8B thinking-on arm dominates the cost.
- V2: ~1 h per model with our CLEAR runner, adapted from `indic_models/run_laya.py` and `energy.clear_measure`.

### Risks
- **Naming and version confusion** (Kev releases and sizes differ between sources; "JEV" vs Jev). Pin exact repo IDs and revisions in `run.json`.
- Many accuracy claims are third-party or marketing. Measure accuracy ourselves on the same task.
- TypeSafe's Jev is API-only, so its energy can't be measured; only open models.
- The 27B models don't fit in BF16 on one card.
- Decision tasks differ from AIME/MATH, so comparisons with our reasoning results are across tasks. V1 avoids this by putting every arm on the same task.

---

## Sources (2026-11-27)
- [HF Transformers: JEV](https://huggingface.co/docs/transformers/en/model_doc/jev)
- [autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B)
- [Releasing the Inner Jev (Jeff 1.1)](https://arxiv.org/html/2610.17205)
- [Jev in the Wild](https://arxiv.org/html/2609.30216)
- [Jeva](https://arxiv.org/html/2610.18773)
- [Introducing Kev (Jared Palmer)](https://jaredpalmer.com/blog/introducing-kev)
- [Kev specs (systemonemodels.org)](https://systemonemodels.org/models/kev/)
- [Kev on DEV](https://dev.to/ayush7614/kev-an-open-source-jev-alternative-i-ran-locally-gek)
- [Kev model page (laya-ai.com)](https://laya-ai.com/system-one-models/kev)
- [Open-source replicas article](https://ai-engineering-trend.medium.com/just-two-days-after-jev-went-viral-two-open-source-replicas-are-here-one-9b-one-0-5b-57024ceef03e)
- [DataCamp: open-source Jev alternatives](https://www.datacamp.com/blog/top-open-source-jev-alternatives)
- [DataCamp: top AI decision models](https://www.datacamp.com/blog/top-ai-decision-models)
- [Layer3 Labs guide](https://www.layer3labs.io/guides/system-one-decision-models)
- [Respan: Jev alternatives](https://www.respan.ai/articles/jev-alternatives)
- [Pinggy: self-hosted decision models](https://pinggy.io/blog/best_open_source_jev_alternatives_self_hosted_decision_models/)
- [General Decision Models](https://arxiv.org/html/2610.03935v2)
- [DecisionPivot](https://arxiv.org/pdf/2610.28522)
- [DecideBench](https://github.com/choyiny/decidebench)
- [System One vs classifiers and LMs](https://arxiv.org/pdf/2610.00346)
- [Decision models on medical MCQ](https://arxiv.org/pdf/2610.38448)
- [TandemBench](https://arxiv.org/pdf/2611.05827)
- [How Hungry is AI?](https://arxiv.org/pdf/2505.09598)
- [Profiling LLM Agent Inference on Blackwell](https://arxiv.org/pdf/2609.29707)
