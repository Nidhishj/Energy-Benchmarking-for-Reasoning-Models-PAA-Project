# Eval 2: speaker notes for slides 9–12 (Nidhish)

Slides: `presentation/eval2_presentation.pdf`.
- Slide 9: Analysis
- Slide 10: Remaining plan
- Slide 11: Summary
- Slide 12: Backup

Suggested time: about 2 min on slide 9, 1.5 min on slide 10, 45 s on slide 11. Slide 12 is only for questions.

**Updates since the slides were made.** Say these out loud if the slides haven't been fixed.
- **Uncapped MATH-200 is done.** We reran the 47 token-capped answers with the same seeds; the regenerated tokens match the originals up to the old limit.
  - Thinking on: 85.5% → **98.5%**. Thinking off: 86.0% → **90.0%**.
  - Energy: 35.8 vs 6.2 kJ per answer (5.7×).
  - Still capped: 1 thinking-on answer and 3 thinking-off answers out of 200.
  - Table: `results/tables/B2b_n200_uncapped.md`.
  - Affected slides: slide 6 (it says 86.0 / 85.5, 6.1×), slide 10 (it lists this as still to do), and slide 12 (the 14.5% / 9% capped figures and the MATH-200 column of the table).

---

## Background you need for these slides

### AIME, the harder dataset
- **AIME = American Invitational Mathematics Examination.** It's the second round of the US high-school olympiad pipeline: AMC 10/12 → **AIME** (roughly the top 5–7% of AMC scorers) → USAMO → IMO team.
- **Format:**
  - 15 problems in 3 hours.
  - Every answer is an **integer from 0 to 999**, so grading is exact and automatic.
  - Two papers per year (I and II), so 30 problems a year. Problems get harder in order within each paper.
- **Why it's a standard benchmark:**
  - Strong human competitors average about 5–6 out of 15.
  - Exact-match grading.
  - A new set every year.
  - Every reasoning model reports AIME pass@1 (o1/o3, DeepSeek-R1, Qwen3).
- **Our set:** AIME 2024 (`HuggingFaceH4/aime_2024`) + AIME 2025 (`MathArena/aime_2025`) = **60 problems**. Config: `configs/aime_subset.json`, IDs like `aime2025/I-11`.
  - Thinking limited to 32k tokens, thinking off to 8k.
  - 2 copies (one per Ada GPU, different seeds), so 120 answers per mode.
- **Why we added it:** on MATH, thinking off already scores 86–90%, so there's no real trade-off to study. On AIME, thinking off gets **25%** vs **68%** for full thinking. Its traces are also long (median 14k tokens, up to 32k), which is the regime where attention dominates.
- **Difficulty groups:** there's no official label, so we use the problem number within its paper (1–5, 6–10, 11–15). Thinking-on accuracy across the groups is 82% → 68% → 52%.
- **Caveats:**
  - Both years were held before Qwen3's release (April 2025), so contamination is possible. It affects every arm of our comparisons equally, and we don't claim absolute capability.
  - 120 answers per mode gives 95% intervals of about ±8–9 points, so we only claim large differences.

### Step energy and step time
- A **decode step** is one forward pass of the whole model that produces one new token, so a 10,000-token answer takes 10,000 steps.
- In Exp A we replay the full step (`model(input_ids=[[next]], past_key_values=cache)` with L tokens already cached) N times between two synced NVML reads.
  - **Step energy** = ΔE_NVML / N, in J per token.
  - **Step time** = Δt / N, after `cuda.synchronize()`, in s per token.
- Step energy ÷ step time = average power during the step.

| L | Step energy (PRO 5000) | Step time | Energy / time |
|---|---|---|---|
| 256 | 4.89 J | 16.4 ms | 298 W |
| 32k | 10.87 J | 36.6 ms | 297 W |

### Words on the slides
- **"Dearer"** = more expensive (British English). When speaking, say "costs more".
  - "Ada is 25% dearer at 256 tokens" means the Ada uses 25% more energy per token there (6.10 vs 4.89 J).
  - "Its late tokens are dearer" means tokens late in a long answer cost more than early ones (10.9 J at 32k vs 4.9 J at 256).
- **Trace** = one saved answer (prompt + generated token IDs).
- **Class** = kind of answer: correct, wrong, or thinking off.
- **Probe** = an intermediate answer forced in the middle of thinking (force `</think>`, let the model answer).
- **Trade-off** = energy saved vs accuracy lost when the model thinks less.

### Live vs replay: two ways we measure a token's energy (both are ours)
- **Live (B3, our Eval 1 baseline):** the model generates normally with `generate()`. A callback syncs and reads the NVML counter every 512 tokens, so energy of tokens 7,680–8,192 = E(8192) − E(7680), and ÷ 512 gives J per token around 8k.
  - It's the true cost, but whole-model only: NVML measures the whole card and can't split by component.
- **Replay (Exp A, Eval 2):** freeze at position L (rebuild the KV cache from a saved trace) and run only the step for token L+1 (or one module) N times in a loop; energy ÷ N.
  - It's the only way to get the component split, because a single module runs for microseconds, far below NVML's resolution.
  - It's an artificial setup, so it must be checked against live.
- The CLEAR paper only does replay, on short fixed inputs. It never did live decoding.

| Position | Live (B3) | Replay (Exp A) | Difference |
|---|---|---|---|
| ~512 | 5.03 J | 4.95 J | −2% ✅ |
| ~4k | 5.33 J | 5.61 J | +5% ✅ |
| ~8k | 5.68 J | 6.30 J | +11% ⚠️ |

(Whole model, PRO 5000.)

- Wording to use: the replay is **"validated up to 4k, being checked beyond that"**. Don't say it's wrong: we don't know yet which side is off.
- **Why the 8k gap may be the comparison, not the replay:**
  - Only 10 of the 90 Eval 1 answers reached 8k, from different problems than the replay trace.
  - Live counts output tokens only; the replay's L includes the prompt.
  - The cache is built differently: one token at a time live vs chunked prefill in the replay.
- **Check, no new GPU time:** the AIME runs recorded live checkpoints every 512 tokens up to 32k, so we can compare live vs replay on the same answer (`aime2025/I-11`) and the same GPU.

### Exp B recap (slide 7, your teammates present it; slide 9 builds on it)
- **Question:** wrong answers cost far more. Is that because each wrong token costs more, or because wrong answers are longer?
- **Equal-L test:** the same CLEAR decode-step replay on three traces, measured at every L.
  - Traces: I-11 (wrong), I-8 (correct), II-6 (thinking off). These are **three different questions**.
  - They agree within 0.4% at every L (Ada, on vs off: within 3%), and the component split matches too.
- **So step cost depends only on L.** That's expected for a dense model with fixed shapes: no data-dependent compute.
- **Per-answer pricing:**
  - An answer's energy = Σᵢ step(prompt + i), from the measured cost curve and each answer's length.
  - On AIME, wrong thinking answers (median 28.6k tokens) are 2.7× longer than correct ones (10.6k), cost 2.5× more, and spend 48% vs 40% on attention.
  - The 33% of thinking answers that are wrong use 55% of all thinking energy.

---

## Slide 9: Analysis, what profiling and the experiments taught us

**Opening line:** *"The previous slides showed what the energy does. This slide explains why, from the hardware side."*

### Box 1: On the PRO 5000, joules = 300 W × time

**Say:**
- "Step energy divided by step time is 298 W at 256 tokens and 297 W at 32k, so the card is at its 300 W power limit in every decode step."
- "Energy per token is therefore latency × a constant. 16.4 → 36.6 ms per token is 2.23× in time, and 4.89 → 10.87 J is 2.22× in energy."
- "At batch 1, saving energy and saving latency are the same thing: every millisecond saved per token saves about 0.3 J."

**If asked about throttling:**
- We log SM clock and temperature with every run.
- The card runs at its power cap, and consistently: the rerun of the October problems is within 4.6% of Eval 1.

### Box 2: Weights are a fixed cost, the KV cache grows

**Step 1: decode is memory-bound.**
- Qwen3-8B is about 8.2B parameters × 2 bytes (BF16) ≈ **16.4 GB of weights**, all read once per token.
- At batch 1, each weight is used for one multiply-add: about 2 FLOPs per 2 bytes, i.e. **~1 FLOP per byte**.
- The GPU needs hundreds of FLOPs per byte before compute becomes the limit, so step time ≈ bytes read ÷ memory bandwidth.
- Check: 16.4 GB in 16.4 ms ≈ 1.0 TB/s, about 75% of the PRO 5000's 1.34 TB/s.

**Step 2: why the MLP is flat.**
- One MLP layer has gate and up (4096×12288 each) plus down (12288×4096): 151M parameters ≈ **302 MB**.
- The MLP only touches the current token, so it reads the same bytes at any L.
- Measured: 263 µs and 78 mJ per layer at every context, which is 302 MB / 263 µs ≈ 1.15 TB/s (86% of peak).

**Step 3: why attention grows.**
- **Projection weights** (q, k, v, o): ≈ 42M parameters ≈ **84 MB per layer**. These are constant.
- **KV cache:** per layer per token, 8 KV heads × 128 dims × 2 (K and V) × 2 bytes = **4 KB**.
  - Across 36 layers that is **147 KB per token**: 1.2 GB at 8k and 4.8 GB at 32k.
- Attention is the only component whose bytes depend on L.

**Step 4: the `torch.cat` copy.**
HF's `DynamicCache.update()` appends the new token with:

```python
self.keys = torch.cat([self.keys, key_states], dim=-2)   # same for values
```

A tensor can't grow in place, so `cat` allocates a buffer of length L+1 and copies all L old entries into it. Per layer per step:

| Traffic | Bytes |
|---|---|
| `cat` reads the old cache | 1 × cache |
| `cat` writes the new cache | 1 × cache |
| SDPA reads the cache | 1 × cache |
| **Total** | **≈ 3 × cache** (the minimum needed is 1×) |

**Step 5: numbers for one layer (PRO 5000).**

| | L = 256 | L = 32k |
|---|---|---|
| Projection weights | 84 MB | 84 MB |
| KV cache, this layer | ~1 MB | 134 MB |
| Bytes moved (weights + 3 × cache) | ~87 MB | 486 MB |
| Measured attention time | 152 µs | 681 µs |
| Effective bandwidth | ~0.57 TB/s | ~0.71 TB/s |

The 0.71 TB/s on the slide (they write "~0.75 TB/s, about half of 1.34") comes from this calculation. It's an estimate; we haven't measured it with Nsight Compute.

**Say:**
- "Decode is memory-bound, so time tracks bytes. MLP bytes are fixed, attention bytes grow with the KV cache, and HF's `DynamicCache` triples the cache traffic, because `torch.cat` reallocates and copies the whole cache every step."
- "So part of attention's growth is the library, not the algorithm. A static cache (pre-allocated, written in place at slot L) removes the copy. That's the first ablation on slide 10."
- "The growth itself stays even then, because SDPA still has to read the full cache."

### Box 3: The Ada server waits on its CPU at short context

**What happens:**
- One decode step is a few hundred small kernels. For each of the 36 layers: norm, q/k/v projections, q/k norms, RoPE, cat, SDPA, o projection, residual add, MLP, and so on.
- The host (Python → PyTorch → CUDA driver) launches each kernel, at a few µs each.
- When a kernel runs for less time than the next launch takes, the GPU sits idle. Step time is then set by **launch speed on the host**, not by the GPU.

**Evidence:**
- On the Ada, step time is **flat at ~25.4 ms from 256 to 8k tokens** while the cache grows by 1.2 GB.
- If the GPU were the bottleneck, that 1.2 GB × 3 at 0.96 TB/s would add about 3.7 ms per step. It doesn't, because it fits into the idle gaps.
- The PRO 5000 rises from the start (16.4 → 21.2 ms by 8k), so it is GPU-bound.
- Attention, which is many small kernels, suffers most: 401 µs per layer on the Ada vs 152 µs on the PRO 5000 at 256 tokens.
- The Ada still draws about 240 W while it waits.
- So the Ada costs about 25% more at 256 tokens, but the two cards are equal at 32k (10.75 vs 10.87 J), where both are GPU-bound.

**Where 401 µs comes from (µs, not ms: 0.4 ms for one attention call of one layer).** It's the per-call time from the CLEAR replay in Exp A:
- `sync; t0; E0` → call layer 18's `self_attn(*captured_inputs)` N times (about 5,000 here, so the loop lasts ~2 s) → `sync; t1; E1`.
- Time per call = (t1 − t0) / N = 401 µs, and energy per call = (E1 − E0) / N = 54.7 mJ.
- Averaged over T = 20 trials.
- Tables: `results/tables/A_decode.md` (Ada) and `A_decode_pro5000.md`.

| One call of a layer-18 module | Ada L=256 | Ada L=4k | PRO L=256 | PRO L=4k |
|---|---|---|---|---|
| input_norm (RMSNorm on 4,096 values) | 58 µs | 57 µs | 25 µs | 25 µs |
| attention | 401 µs | 399 µs | 152 µs | 185 µs |
| MLP | 335 µs | 335 µs | 263 µs | 262 µs |
| full step | 25.5 ms | 25.3 ms | 16.4 ms | 18.9 ms |

- **An RMSNorm on 8 KB of data takes 58 µs on the Ada.** The GPU work is a few µs; the rest is host launch time. The PRO host does the same call in 25 µs.
- **Ada attention is flat (401 → 399 µs) while its cache grows 16×** (256 → 4k). On the PRO it rises (152 → 185 µs). Launch time hides the extra GPU work.
- **Attention (~10 small kernels) is 2.6× slower than on the PRO; the MLP (a few big matmuls) only 1.3×.** Many small kernels suffer most from launch overhead.
- Energy still rises on the Ada (54.7 → 63.8 mJ from 256 to 4k): the GPU does more work in the same wall time, filling the idle gaps.
- **If asked "is that a replay artefact?":** "The real full forward step shows the same flat ~25.4 ms up to 8k on the Ada."

**Say:**
- "These are per-call times from the replay: one module called N times between synchronized timestamps, divided by N. On the Ada, one attention call takes 401 µs at 256 tokens and still 399 µs at 4k, flat while the cache grows 16×. Even an RMSNorm on 4,096 values takes 58 µs, against 25 µs on the PRO machine. That's launch overhead, not GPU work."
- "This isn't the Ada being slow. Its kernels finish faster than the host can issue the next one. The bottleneck is the host: CPU single-thread speed, driver (560 on the Ada server vs 595), Python overhead."
- "Part of Eval 1's gap between the Ada and the PRO 5000 was the host, not the GPU."
- "This is inferred from timing and not yet profiled. The check is CUDA graphs on the Ada: one launch per step instead of hundreds. If the flat region disappears, the explanation is confirmed."

**If asked "then what is the Ada good for?":**

| | RTX 6000 Ada (2022) | RTX PRO 5000 Blackwell (2025) |
|---|---|---|
| Memory | 48 GB GDDR6, **0.96 TB/s** | 48 GB GDDR7, **1.34 TB/s** (1.4×) |
| SMs | **142** | 110 |
| Power limit | 300 W | 300 W |

- At batch-1 decode (memory-bound), the PRO 5000 should win and does: it's faster at every L (16.4 vs 25.5 ms at 256, and 36.6 vs 40.8 ms at 32k).
- The Ada's ~30% more SMs help only compute-bound work: prefill of long prompts, large-batch serving (more FLOPs per byte) and training. None of these is our workload.
- At 32k the energies are equal even though the PRO is 11% faster, because the Ada averages about 263 W there and the PRO about 297 W.
- We used the Ada because we had continuous access to it. We compare only within one card, component shares match across cards, and absolute joules are quoted per card.

### Box 4: Answer energy ≈ its length × the average step cost

**Say:**
- "Exp B showed a step's cost depends only on its position L: within 0.4% across wrong, correct and thinking-off traces. That's expected for a dense model with fixed shapes and no data-dependent compute."
- "So an answer's energy is Σᵢ step(prompt + i). That reproduces measured whole-answer energy at 99–105% on the PRO 5000 and 92% on the Ada."
  - "The missing ~8% on the Ada is sampling and `generate()` bookkeeping outside the forward pass, which is larger on the launch-bound host."
- "Thinking costs about 6× because it's about 6× longer and its late tokens cost more. Wrong answers are the longest, so they pay most for attention."

### Footer line: three validity notes

> "one layer stands for all 36 (within 2.2%); 'glue' between modules is 0–11% of a step; the Ada and PRO 5000 split energy the same way, so shares transfer across GPUs but joules do not."

1. **One layer represents all 36.** We measure layer 18 and multiply by 36. All layers have identical shapes and full attention. Layers 0 and 35 measured at 1k and 16k are within 2.2% of layer 18.
2. **Glue = full step − sum of the measured modules.** That is residual adds, building the mask and RoPE cos/sin, and Python between module calls.
   - It's 8–11% of a step at short L and about 3% at 16–32k.
   - So the named components account for 89–100% of the step.
3. **Percentages transfer across GPUs, joules don't.**
   - J per token differs by card: 4.89 vs 6.10 at 256.
   - The split has the same shape on both: attention ≈ 25–32% → 63–67%.
   - So "attention is two thirds of a token's energy at 32k" holds on either card. "A token costs 4.89 J" holds only on the PRO 5000.

---

## Slide 10: Remaining plan

**Opening line:** *"What's left falls into three groups: make the claims statistically solid, turn the savings into a usable rule, and separate algorithmic cost from implementation cost."*

### Status of each item

| Group | Item | Status | Detail |
|---|---|---|---|
| Comparisons | More traces per class (Exp B) | ❌ | One correct, one wrong and one thinking-off trace so far, from three different questions. Next: correct and wrong answers to the **same question** (13 AIME problems have one of each among our two seeds; first pair `aime2024/I-8` t0 ✅ 26.4k vs t1 ❌ 23.6k), then 3–5 pairs. About 1 h of GPU per trace, using saved traces, so no new generation. |
| Comparisons | Uncapped MATH-200 | ✅ **Done** | On 98.5%, off 90.0%. The Eval 1 result where thinking didn't help was a truncation artefact. |
| Comparisons | Live vs replayed step at 8–32k | ❌ | The replay is validated up to 4k (within 5% of live); at 8k it's 11% above live (+27% vs +13% from 512 to 8k). Check offline with the live AIME checkpoints on the same answer and GPU. See "Live vs replay" in the background section. |
| Trade-offs | AIME energy vs accuracy by difficulty | 🟡 Data exists | From `Caps_aime.md`, see the table below. It still needs a proper figure and a "difficulty-aware cap" evaluation (offline). |
| Trade-offs | A stopping rule that pays for itself | 🟡 Partial | The expensive version was tested offline. The cheap-probe version needs GPU runs. |
| Ablations | KV cache: dynamic vs static | ❌ | About 1 h of GPU with the Exp A runner |
| Ablations | Attention kernel: SDPA vs FlashAttention-2 vs eager | ❌ | About 1 h each |
| Ablations | Measurement fixes (sync, in-order replay, CUDA graphs) one at a time | ❌ | Partly tested on BERT in Eval 1; not yet on Qwen3 decode |
| Ablations | Host: CUDA graphs on the Ada | ❌ | Confirms the launch-bound explanation |
| Ablations | Layer 0 / 18 / 35 | ✅ **Done** | Within 2.2% |
| — | Last 2 of 54 Exp C traces on the PRO 5000 | ❌ | About 10 min in the next PRO slot |

### AIME by difficulty (from existing data, copy 1 energy)

| Problems | Cheapest cap that keeps accuracy | Accuracy, cap vs full | kJ, cap vs full | Saved |
|---|---|---|---|---|
| 1–5 | 8k | 82% vs 82% | 55 vs 79 | ~30% |
| 6–10 | 16k | 68% vs 68% | 89 vs 110 | ~19% (31% on copy 2) |
| 11–15 | 16k | 50% vs 52% | 114 vs 163 | ~30% |

### What to say for each group

**Opening:** "Here's what remains, in three groups: strengthening our comparisons, turning the savings into a usable rule, and ablations that separate the algorithm's cost from the implementation's."

**Comparisons**
- **Exp B, more answers per class.** This is a future run of the same Exp B measurement on more saved answers.
  - **Why:** so far it's one wrong vs one correct answer, from different questions. That's a single pair, so we don't know how much two answers of the same kind differ. Correctness and the question also change together.
  - **Say:** "Exp B so far compares a wrong, a correct and a thinking-off answer from three different AIME problems. They agree within 0.4% at every context, but correctness and the question change together, and one pair doesn't give us the spread within a class. On 13 AIME problems, one of our two seeds got it right and the other got it wrong. For example, on 2024 I-8 the correct answer is 26.4k tokens and the wrong one 23.6k. Measuring both at the same contexts holds the prompt fixed and varies only correctness. We'll do three to five such pairs, which gives a mean and spread per class at every context. It's the same runner on saved answers, about an hour each."
- **Uncapped MATH-200, done.**
  - **Say:** "We reran the 47 answers that hit the token limit, with the same seeds and higher limits. Thinking on goes from 85.5% to 98.5% and thinking off from 86% to 90%. So the Eval 1 result where thinking didn't help was just truncation."
- **Live vs replay.**
  - **Say:** "We measure a token's cost two ways. Live: during normal generation we read the energy counter every 512 tokens, which is the true cost but whole-model only. Replay: we freeze at position L and repeat that one step, which is the only way to get the component split. Up to 4k tokens they agree within 5%. At 8k the replay is 11% higher, but the live point there rests on only ten MATH answers. Our AIME runs recorded live checkpoints up to 32k, so we'll compare the two on the same answer and GPU, without new runs."
  - Don't say "the replay is wrong". Say "validated up to 4k, being checked beyond".

**Trade-offs** ("trade-off" = energy saved vs accuracy lost when thinking less). Don't mention the oracle.
- "A fixed 16k cap on AIME keeps accuracy and saves 22–26%, but it stops every answer at the same point. Easy problems (1–5) already reach full accuracy at 8k, so a cap set by difficulty saves about 30% on them."
- "Better is an adaptive stopping rule, which decides per answer when it has thought enough. We tried a simple one:
  - at 2k, 4k, 8k and 16k thinking tokens, force `</think>` and get an intermediate answer (a 'probe');
  - stop when two probes in a row give the same answer.

  Accuracy stayed close (65% vs 68%), but each forced answer can be up to 4,000 tokens, so overall it used 6–17% *more* energy than just thinking to the end. The checks have to be cheap."
- "Several recent papers propose cheaper stopping signals. We'll test them on our traces with energy measured:
  - **Dynasor-CoT / Certaindex**: short forced answers at intervals; stop when they're consistent.
  - **DEER, EAT**: stop when the model is confident, measured by the entropy of the answer after a forced `</think>`.
  - **Hidden-state probes** (*Reasoning Models Know When They're Right*): a linear classifier on the hidden state predicts whether the current answer is already correct, at the cost of one dot product."
- "Those papers report token savings. We add the energy side: late tokens are the most expensive, so cutting them saves more energy than the token count suggests. Wrong thinking answers are 33% of answers but 55% of thinking energy, so stopping those early matters most."
- **If asked why our rule failed:** "Our probes were full answers of up to 4k tokens. All of these methods use checks that cost a few tokens or less."

**Ablations**
- "Static cache: does attention's growth survive without the per-step `torch.cat`? If attention's share drops a lot, part of the 67% is the HF implementation."
- "Attention kernel: at decode every kernel must still read the full cache, so the linear growth should remain, but the slope and the launch overhead may change. Eager is included because hooks can see inside it."
- "Measurement fixes: which one is responsible for %Capture going from 74–88% (CLEAR's method) to 93–99% (ours)."
- "Host: CUDA graphs on the Ada test the launch-bound explanation."

**Budget line:** "Each ablation reuses the Exp A runner at about 1 h per setting. The trace and cap analyses reuse saved answers, so they need no GPU time."

---

## Slide 11: Summary

**Key question:** where does a reasoning model's decoding energy go as the answer grows and for wrong answers, and how much does cutting thinking short save?

Read the three columns as one sentence each.

- **Built:** "A component-energy pipeline for real decode steps at any context. Parts add up to 93.5–99.6% of the whole step, and per-answer pricing reproduces Eval 1's whole-model energy within 5%. Thinking-budget runners, MATH-200 and AIME, and repro-gated, resumable runs on two GPUs."
- **Learned:** "Attention grows from 25% to 67% of a token's energy between 256 and 32k tokens while the MLP stays flat. Per token, wrong = correct = thinking off, so wrong answers cost more only through length: 55% of AIME thinking energy. Half the thinking is enough on MATH (−43%); a 16k cap on AIME saves 22–26% at no accuracy cost."
- **Still to establish:** "Exp B over more traces; whether attention's growth survives a static cache and FlashAttention; the live vs replay gap at long context; a stopping rule that beats a fixed cap on AIME." Uncapped MATH-200 is now done.

**Closing line:**
> *"The cost of a reasoning token is set by its position. Thinking is expensive because it is long, and the long tail is attention-dominated, so the lever is cutting length adaptively, especially on answers heading towards wrong."*

**If asked "so you don't need to think much?":**
- "Not unconditionally. Thinking past a point is redundant: on MATH, 50% of each trace's own thinking length gives the same accuracy, and on AIME nothing beyond 16k helps."
- "But that point depends on the problem. On AIME every cap below 16k loses accuracy, and the hardest problems go from 2% (off) to 52% (full). On MATH-200, letting truncated answers finish raised thinking-on accuracy from 85.5% to 98.5%."
- "The 50% result also assumes we know each trace's natural length in advance. So the savings are real, but they need an adaptive stopping rule, which we don't have yet."

---

## Slide 12: Backup, caveats and numbers for questions

**What this slide is:** a reference slide for questions. Don't present it. After slide 11, say "we have a backup slide with caveats and numbers if needed", and go to it only if a question matches a caveat or asks for an exact number.

- **Left side, caveats:** known limitations, each with a ready answer (table below):
  - one answer per class;
  - replay vs live beyond 4k;
  - Exp C reads ~2% low;
  - MATH-200 truncation (now fixed);
  - the two machines have different hosts and drivers;
  - GPU-only energy, idle not subtracted.
- **Right side, numbers:** Exp A J/token and ms/token per context for both GPUs, and the whole-model thinking on vs off table.
  - The MATH-200 column there is outdated; the corrected values are below.
  - **kJ/correct** = total energy ÷ number of correct answers, i.e. the energy cost of one right answer.
  - On AIME, a correct answer costs 191 kJ with thinking vs 84 kJ without, 2.3× more per correct answer, but thinking solves 2.7× more problems (67.5% vs 25%).

**On live vs replay:** say "validated up to 4k, being checked beyond that". Don't say the replay is wrong.

| Likely question | Answer |
|---|---|
| Only one trace per class? | Yes so far, from three different questions. Step cost depends only on L (within 0.4% across the three), so more should agree. Next: correct vs wrong answers to the same question (13 AIME problems have both; first pair 2024 I-8), 3–5 pairs. |
| Is the replay faithful to real decoding? | Validated up to 4k: within about 5% of live, and whole-answer energy within 5%. At 8k it's 11% above live, but the live 8k point is only 10 MATH answers. We're checking it on the same AIME answer and GPU using the live checkpoints up to 32k. |
| Why isn't idle power subtracted? | Idle is logged and both values are in the tables. Under load the card sits at its power cap, so "dynamic" energy isn't well defined. Raw is the honest number. |
| Is NVML accurate? | It measures the GPU only. Zeus agrees within 0.04%, but it reads the same counter, so that's a consistency check rather than independent. There's no wall meter, so host energy isn't included. |
| Why compare two different machines? | They differ in host and driver (560 vs 595); both run torch 2.8 and transformers 4.57. Joules mix GPU and host, but the component shares don't depend on it, since the split is the same on both. |
| Why batch size 1? | Single-user / local inference, where decode is memory-bound. At larger batches decode becomes compute-bound and the split will change. That's out of scope so far. |
| Exp C thinking energy? | It comes from a teacher-forced replay of the saved tokens, which runs about 2% faster than `generate()`, so it reads about 2% low. |
| Truncated MATH-200 answers? | Fixed. After the reruns only 1 thinking-on and 3 thinking-off answers out of 200 hit the limit (32k on / 8k off). The slide's 14.5% / 9% are from before the reruns. |
| How is 147 KB per token derived? | 36 layers × 8 KV heads × 128 head dim × 2 (K and V) × 2 bytes (BF16) = 147,456 B. |
| How is ~0.7 TB/s derived? | Per layer at 32k: 84 MB of projection weights + 3 × 134 MB of cache (cat read, cat write, SDPA read) = 486 MB in 681 µs ≈ 0.71 TB/s. Over 36 layers: ~17.5 GB in ~24.5 ms. |

### Updated numbers for the backup table (MATH-200 column)

| | MATH-200 off | MATH-200 on | AIME off | AIME on |
|---|---|---|---|---|
| Accuracy | **90.0%** (slide: 86.0%) | **98.5%** (slide: 85.5%) | 25.0% | 67.5% |
| kJ / answer | **6.2** (slide: 4.8) | **35.8** (slide: 29.1) | 20.9 | 128.7 |
| kJ / correct | **6.9** | **36.3** | 83.7 | 190.7 |

Exp A per-token numbers (unchanged):

| Context | PRO 5000 J/token | ms | Ada J/token | ms |
|---|---|---|---|---|
| 256 | 4.89 | 16.4 | 6.10 | 25.5 |
| 1k | 5.04 | 17.0 | 6.16 | 25.4 |
| 4k | 5.61 | 18.9 | 6.54 | 25.3 |
| 8k | 6.30 | 21.2 | 6.69 | 25.4 |
| 16k | 7.74 | 26.0 | 7.64 | 29.0 |
| 32k | 10.87 | 36.6 | 10.75 | 40.8 |
