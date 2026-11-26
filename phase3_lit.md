# Phase 3 reading list: only the important papers

Short list for the work after Eval 2 (`remaining_work.md`, `plan_ssm_hybrid.md`, `plan_jepa.md`, `plan_systolic.md`).
- About 20 papers, in reading order within each topic.
- ★ = read in full; the others need only abstract, method and main figure.
- The notes come from abstracts and search snippets (2026-11-27). Check numbers in the paper before citing them.

Already known from Phase 1–2, so not repeated:
- CLEAR (2510.02810)
- Jin et al., reasoning vs non-reasoning energy (2505.14733)

---

## A. When to stop thinking (stopping rules)

| | Paper | Why read it | What to take for us |
|---|---|---|---|
| ★ | **Early Stopping in Large Reasoning Models: A Survey and Empirical Audit** ([2611.00166](https://arxiv.org/pdf/2611.00166)) | The single entry point: taxonomy of stopping signals plus an audit of how reliable they are | Which signals to implement for E2 (`remaining_work.md` §3.1); baselines to compare against |
| ★ | **Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification** ([2504.05419](https://arxiv.org/abs/2504.05419)) | A linear probe on hidden states predicts whether an intermediate answer is correct; up to 24% fewer tokens with no accuracy loss | The cheapest probe (one dot product); its training and labelling recipe for our AIME traces |
| ★ | **EAT** (entropy after `</think>`) and **DEER** (confidence at reasoning transitions), both 2025 | The entropy and confidence signals | How to compute the answer-token entropy after a forced `</think>` and the stopping threshold, for E1 and E2 |
| | **Dynasor-CoT / Certaindex** (2025) | Answer-consistency stopping with *short* probes | Why our agree-twice failed (4k-token probes); the probe length and interval they use |
| ★ | **When Does Learning to Stop Help? Risk-Controlled Early Exit** ([2606.30852](https://arxiv.org/pdf/2606.30852)) | When an adaptive rule actually beats one fixed token budget | Exactly our "does anything beat the 16k cap?" question; their calibration procedure |
| | **Conformal Thinking** ([2602.03814](https://arxiv.org/pdf/2602.03814)) | Stopping with a guaranteed accuracy-loss budget | How to state E2's results with risk control instead of an ad-hoc threshold |
| | **Temporal Predictors of Outcome in Reasoning LMs** ([2511.14773](https://arxiv.org/html/2511.14773v1)) | Correctness is decodable after only a few tokens (AUC > 0.8); longer traces are harder items | The selection-effect caveat for our correct-vs-wrong comparisons |
| | **`</think>` Doesn't Stop Reasoning** ([2609.03633](https://arxiv.org/pdf/2609.03633)) | Forced `</think>` often doesn't end reasoning | A known failure mode of our forced-answer probes; what to count as an invalid probe |
| | **When More Thinking Hurts** ([2604.10739](https://arxiv.org/pdf/2604.10739)) | Budget forcing: diminishing returns, and overthinking that abandons correct answers | A related result for our AIME caps; it frames this as energy but counts tokens |

## B. Energy of LLM inference (measurement methodology)

| | Paper | Why read it | What to take for us |
|---|---|---|---|
| ★ | **Where Do the Joules Go? Diagnosing Inference Energy Consumption** ([2601.22076](https://arxiv.org/html/2601.22076)) | The largest recent study: 46 models, H100/B200, Zeus | State of the art in whole-model energy diagnosis; positions our component-level work |
| | **The ML.ENERGY Benchmark** ([2505.06371](https://arxiv.org/html/2505.06371v2)) | Their measurement protocol and tooling (Zeus) | Methodology to cite, and checks we should match |
| | **Characterizing LLM Inference Energy-Performance Tradeoffs** ([2501.08219](https://arxiv.org/pdf/2501.08219)) | Decode is dominated by weight and KV-cache access | Independent support for our "joules = power cap × time" and memory-bound findings |

## C. Mamba / SSM and hybrid models (`plan_ssm_hybrid.md`)

| | Paper | Why read it | What to take for us |
|---|---|---|---|
| ★ | **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** ([2312.00752](https://arxiv.org/pdf/2312.00752)) | The architecture and its hardware-aware scan (state in SRAM, not HBM) | What the Mamba mixer's sub-modules are (in_proj, conv1d, scan, out_proj), i.e. what to hook |
| ★ | **NVIDIA Nemotron Nano 2: Hybrid Mamba-Transformer Reasoning Model** ([2508.14444](https://arxiv.org/abs/2508.14444)) | Our main candidate: 9B hybrid with a thinking toggle, benchmarked against Qwen3-8B (up to 6× throughput) | Layer layout (how many attention layers), thinking switch, sampling settings; whether the throughput gain survives at batch 1 in joules |
| ★ | **Characterizing SSM and Hybrid LM Performance with Long Context** ([2507.12442](https://arxiv.org/pdf/2507.12442)) | The only measured energy comparison found: transformer 1,493 J vs Mamba-2 371 J at 57k (RTX 4090); transformers win at short context | Prior work to cite and extend (whole-model only → components and reasoning) |
| | **M1: Towards Scalable Test-Time Compute with Mamba Reasoning Models** ([2504.10449](https://arxiv.org/abs/2504.10449)) | Mamba reasoning model, >3× generation speed vs a same-size transformer | Optional third model pair; their memory-bound decoding argument |
| | **Phi-4-mini-flash-reasoning** (SambaY; Microsoft model card and paper) | Hybrid with a same-family transformer twin (Phi-4-mini-reasoning) | The cleanest architecture A/B; Mamba-1 vs Mamba-2 choice |

## D. JEPA (`plan_jepa.md`)

| | Paper | Why read it | What to take for us |
|---|---|---|---|
| ★ | **V-JEPA 2** (Meta, 2025; [HF docs](https://huggingface.co/docs/transformers/model_doc/vjepa2), [blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)) | Official weights on HF; at inference a ViT video encoder | Input tokenisation (tubelets, frames × patches), so J0 can sweep input size; model sizes |
| ★ | **LLM-JEPA: Large Language Models Meet JEPAs** ([2509.14252](https://arxiv.org/abs/2509.14252), ICLR 2026) | JEPA objective for LLMs; the inference architecture is unchanged | Whether fine-tuned checkpoints exist (J1); which tasks (GSM8K) |
| | **VL-JEPA** ([2512.10942](https://arxiv.org/pdf/2512.10942)) | Predicts text embeddings, with selective decoding only when needed | The only JEPA with a real inference-time compute change; check whether weights are released (J2) |

## E. Systolic arrays and accelerators (`plan_systolic.md`)

| | Paper | Why read it | What to take for us |
|---|---|---|---|
| ★ | **SCALE-Sim v3** ([2504.15377](https://arxiv.org/pdf/2504.15377)) | The simulator we'd use; notes how it differs from Timeloop/Accelergy | Input format (layer topology), memory model, how to get energy |
| ★ | **LLMCompass** (ISCA 2024; [PDF](https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2024_2025/papers/LLMCOMPASS_ISCA_2024.pdf)) | LLM inference hardware evaluation on SCALE-Sim; more compute barely helps decode | How to model a full LLM decode step on a systolic array, as a template for Y1 |
| | **SPAD: Specialized Prefill and Decode Hardware** ([2510.08544](https://arxiv.org/pdf/2510.08544)) | Small (16×16) arrays plus vector units suit decode | The second hardware config for Y1/Y2 |
| | **Mamba-X: An End-to-End Vision Mamba Accelerator** ([2508.02977](https://arxiv.org/html/2508.02977v1)) | Systolic *scan* array for the selective scan | How to map the Mamba scan in Y3 |

---

## Suggested reading order (about 1 week)
1. **Stopping:** survey 2611.00166 → 2504.05419 → EAT/DEER → 2606.30852 (needed for E1/E2).
2. **SSM:** Mamba → Nemotron Nano 2 → 2507.12442 (needed before the first hybrid run).
3. **Energy methodology:** *Where Do the Joules Go?* (for the related-work section).
4. **JEPA:** V-JEPA 2 → LLM-JEPA (before J0/J1).
5. **Systolic:** SCALE-Sim v3 → LLMCompass (before Y1).
