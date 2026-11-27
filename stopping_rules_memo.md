# Energy-aware deterministic early stopping for Qwen3-8B thinking — research memo

*Scope:* batch-1 HF decoding, RTX PRO 5000 / RTX 6000 Ada, AIME 24+25 (60 problems × 2 seeds). All project numbers are our own measurements. Literature checked 2026-11-27. "Verified" means I opened the arXiv abstract or HTML page; for most papers I read only the abstract and the method section.

---

## 0. The objection, answered honestly

**Fact 1: energy is not proportional to tokens.** Marginal cost is c(L) ≈ a + bL (PRO 5000: a≈4.84 J, b≈1.8e-4 J/tok²). So E(T) = aT + bT²/2, the 31–32k tokens cost about 2.2× the first 1k, and the second half of a 32k trace holds **59% of its energy** (E(16k)≈100 kJ, E(32k)≈247 kJ).

**Fact 2: energy is a deterministic, content-free, increasing function of L.** Correct, wrong and no-think traces agree within 0.4% at equal L. Therefore:

- **Energy can never be a stopping *signal*.** It carries no information about correctness beyond L. It can only be a *cost*.
- **Any rule that only thresholds cumulative energy is a token rule:** E(L) ≥ B ⇔ L ≥ E⁻¹(B). On one GPU with short AIME prompts, a joule cap *is* a token cap.
- **Every energy-aware rule below is equivalent to *some* position-dependent token rule.** Energy-awareness contributes three things:
  1. **The physics picks the schedule.** In a cost–benefit rule, the exchange rate between "confidence" and "one more chunk" scales with c(L). Measuring c(L) fixes the *shape* of the threshold schedule with zero extra free parameters. A token-rule designer would have to tune that shape.
  2. **One budget means different token caps on different hardware.** b/a differs per GPU: attention's share of step energy is 25% → 67% on the PRO 5000. Prompt length P also shifts L = P+T. A budget stated in joules therefore becomes a different token cap per GPU and per context.
  3. **Accounting.** Probes and thinking go on one scale. Tail tokens are weighted up to ~2.2×, so rules that cut long traces earn up to ~1.2× more relative savings than token counts suggest: 59% vs 50% for halving a 32k trace, 53% vs 50% for an 8k trace.
- **Expected size of the effect: modest.** c(L) spans only ~2.2× over 32k, so energy-shaped and token-shaped schedules differ by ≤2.2× in error tolerance. Energy-awareness should matter most for **give-up rules**, because the expensive tail is concentrated in wrong traces (33% of answers, 55% of thinking energy, median 28.6k tokens).
- **Collapse is most complete in our regime** (batch 1, short prompts, fixed clocks). It would weaken under batched serving, long RAG/agent prompts (P ≫ T) or DVFS/power capping. Those are out of scope; we say so in the paper.

---

## 1. Literature scan (2024–2026)

| Work | Signal / idea | Key numbers | Status |
|---|---|---|---|
| **EAT** — Wang, McInerney, Wang, Kallus, [arXiv:2509.26522](https://arxiv.org/abs/2509.26522) | Entropy of next token after an appended `</think>`. Stop when the EMA variance of that entropy < δ. | Checked at every `\n\n` line (every 50–200 tokens works similarly). EMA α≈0.2, warm-up 4/α=20 lines, δ swept over 2^0..2^39. Cost ≈ 1 token per check. 13–21% fewer tokens at no loss on MATH-500/AIME-25. Beats a fixed per-question token cap on AUC. Models include R1-0528-Qwen3-8B. | Verified |
| **DEER** — Yang et al., [arXiv:2504.15895](https://arxiv.org/abs/2504.15895) | At "Wait" transitions, force a trial answer. Exit if the geometric mean of per-token max-prob ≥ λ. | λ = 0.95 (the entropy-monitor variant uses 0.672). Qwen3-14B AIME-24: 10.9k → 7.6k tokens, 70.0 → 76.7%. Overall 19–80% shorter. Trial-answer length not stated. | Verified (abstract + §4) |
| **Dynasor-CoT / Certaindex** — Fu et al., [arXiv:2412.20993](https://arxiv.org/abs/2412.20993), [blog](https://haoailab.com/blogs/dynasor-cot) | Probe-in-the-middle every 32/64/128 tokens with "…Final Answer: \boxed{". Stop after N consecutive identical answers; discard answers containing "wait"/"hmm". | Up to 29% tokens saved overall on AIME/MATH-500 (blog wording inconsistent). 12% on AIME for DeepSeek-R1. N is swept, not fixed. Probe length not stated. | Verified (NeurIPS'25) |
| **Reasoning Models Know When They're Right** — Zhang et al., [arXiv:2504.05419](https://arxiv.org/abs/2504.05419) | Linear/2-layer MLP probe on last-layer hidden state at the end of each chunk ("wait", "verify", … delimiters). | Thresholds 0.8 / 0.85 / 0.9. At 0.85: 24% fewer tokens at equal accuracy (R1-Distill-Llama-8B, MATH). ECE 0.03–0.10. | Verified |
| **CoDE-Stop** — Hosseini et al., [arXiv:2604.04930](https://arxiv.org/abs/2604.04930) | Trajectory of intermediate-answer confidence. Wrong rollouts are long, with unstable confidence. | 25–50% fewer tokens. | Verified (abstract) |
| **Conformal Thinking** — Wang et al., ICML'26, [arXiv:2602.03814](https://arxiv.org/abs/2602.03814) | Upper threshold (stop when confident) plus a **rising lower threshold** λ₋(t)=σ(c(ω_t−sB), l, u) that gives up on unsolvable items. Risk-controlled with a Hoeffding UCB. | Signals: DEER-style confidence, EAT, an MLP probe (Qwen3-8B only), token count. Checked per `\n\n` chunk. Savings shown only in figures. | Verified |
| **When Does Learning to Stop Help?** — Dong, Qin, Shah, [arXiv:2606.30852](https://arxiv.org/abs/2606.30852) | Learned stop rule vs static token budget, both certified with Learn-then-Test. | No universal winner. The better family flips with the risk target and **with the price of a provisional answer**. At 5% risk, the learned rule saves 10–17 pp more tokens on Qwen3 math. Fixing a family in advance costs up to 27 pp. Qwen3-8B latency ∝ tokens in vLLM. | Verified (abstract) |
| **Early-stopping survey + audit** — Nguyen et al., [arXiv:2611.00166](https://arxiv.org/abs/2611.00166) | Replay protocol; TITE metric. | Correctness can *drop* at later exit points. Forced termination does not always end reasoning. | Verified (abstract) |
| **EcoThink** — Li & Lu, WWW'26, [arXiv:2603.25498](https://arxiv.org/abs/2603.25498) | A router decides whether to reason at all. | −40.4% energy on average (−81.9% on web retrieval). **Metering method not verified** from the abstract. Not a within-trace stopping rule. | Partly verified |
| **SIMPLE law** — Cavagna et al., [arXiv:2611.03330](https://arxiv.org/abs/2611.03330) | Energy law with a separate KV-cache term: each token raises the cost of later tokens. | Intelligence per joule peaks at an *intermediate* reasoning budget and drops up to 3.14× over a 4× budget range (H100). Independently matches our c(L). | Verified (abstract) |
| **Energy of TTC** — Jin, Wei, Brooks, [arXiv:2505.14733](https://arxiv.org/abs/2505.14733); **Energy-per-token** — Wilhelm et al., [arXiv:2603.20224](https://arxiv.org/abs/2603.20224) | Accuracy/energy trade-off of test-time compute. Wilhelm *proposes* energy-curve-controlled CoT depth. | No stopping rule evaluated. Wilhelm's mechanism is **proposed only (not verified as implemented)**. | Abstracts |
| **Optimal stopping / VOI** — Kalayci, Raman, Dughmi, [arXiv:2510.01394](https://arxiv.org/abs/2510.01394) (Pandora's box for Best-of-N); De Sabbata et al., [arXiv:2410.05563](https://arxiv.org/abs/2410.05563) (value-of-computation reward); Ferguson, *Optimal Stopping*, [ch. 5 Monotone problems](https://www.math.ucla.edu/~tom/Stopping/sr5.pdf); Russell & Wefald, "Principles of Metareasoning," AIJ 1991 | Stop when the expected gain of more computation is ≤ its cost. The one-stage look-ahead rule is optimal for *monotone* problems. | Kalayci et al.: 15–35% fewer generations. De Sabbata et al.: 20–37% fewer tokens. | arXiv: verified. Ferguson: TOC verified. **Russell & Wefald: cited via secondary sources, not opened.** |

No paper I found uses a **joule** budget or a **c(L)-shaped** threshold inside a reasoning trace. "Energy-Conditioned Thinking" ([ICLR'26 workshop](https://iclr.cc/virtual/2026/10016661)) uses "energy" only as a proxy for a compute budget. The gap is real but narrow, given §0.

**Why the one-stage look-ahead rule is justified on our data.** At the population level, the marginal benefit per kJ of the caps falls monotonically:

| Step | pp per kJ |
|---|---|
| 2k → 4k | 13/12 = 1.08 |
| 4k → 8k | 15/24 = 0.63 |
| 8k → 16k | 12/33 = 0.36 |
| 16k → full | 0 |

Marginal cost c(L) rises. Falling benefit plus rising cost is Ferguson's monotone case, so the **myopic rule "stop when expected gain of the next chunk ≤ its joule cost" is optimal** under that model.

---

## 2. Signals and cost model (shared by all rules)

**Per probe k at position L_k (grid Δ = 512 or 1k)**, after appending `</think>\n\nThe final answer is \boxed{`, record:

- H_k: entropy at the first answer token.
- **s_k: geometric mean of per-token max-prob over the forced answer** (DEER Eq. 4). Qwen splits digits into separate tokens, so first-token top-1 probability covers only the first digit. Log all answer-token probabilities.
- ŷ_k: the forced answer, ≤64 tokens, graded. A missing `}` counts as wrong (survey: forced stops sometimes keep reasoning).
- h_k: final-layer hidden state.
- Also log the entropy right after `</think>`, which is **EAT proper** and costs one forward pass.
  - **Measured 2026-11-27 (1,737 probes, PRO 5000): EAT is degenerate for Qwen3-8B with this template.** After `</think>` the next token is always the same blank line, so H_eat ≈ 0 (max 0.55). Use H_k and s_k instead; EAT would have to be read one position later (after `</think>\n\n`).
  - Per-probe NVML energy is noisy (13–58 J for the same 3-token answer): a probe is shorter than the counter's update interval. Price probes with π_k, as below.

**Cost model (deterministic).**

- Thinking: Ê(L) = aL + bL²/2, with a and b fitted per GPU on the E(k) checkpoints.
- Probe: π_k = (1 + m_k)·c(L_k). That is one suffix prefill plus m_k answer tokens; probe tokens are discarded via `cache.crop`.
- For AIME, m_k ≈ 4–5, so π ≈ 30 J against ~5 kJ per 1k thinking tokens.
- **Rules use Ê (the model) so they are reproducible. Results are reported on both Ê and the measured NVML E(k) plus measured probe energy.** A live-meter variant is an ablation only, because it is not deterministic.

---

## 3. Proposed rules

Each rule has **one** free scalar, chosen leave-one-problem-out (LOPO). On stopping at probe k, output ŷ_k.

**R1 — Joule-budget cap (per GPU).**
- *Rule:* stop at the first L with Ê(P+L) − Ê(P) ≥ B. Closed form: T_cap = (−a + √(a² + 2bB))/b for P≈0.
- *Example (PRO 5000):* B = 100 kJ → 15.9k tokens. B = 50 kJ → 8.9k tokens (linear accounting at 5.04 J/token would give 9.9k).
- *Justification:* the operator budgets energy per request, and the SIMPLE-law intelligence-per-joule peak lies at an intermediate budget.
- *Joules vs tokens:* a single B transfers across GPUs and prompt lengths as different token caps.
- *Collapse:* on a single GPU with short prompts it **is** the 16k cap. Expect a 22–26% saving at no loss.
- *Failure mode:* none beyond those of a token cap, which ignores the problem entirely.

**R2 — Cost-scaled confidence threshold (one-stage look-ahead).**
- *Rule:* continue only if the value of the possible fix exceeds the cost of the next chunk: λ·(1 − s_k) > Δ·c(L_k). Equivalently, **stop iff s_k ≥ τ(L_k) = 1 − (1 − τ₀)·c(L_k)/c(L₀)**. The single parameter is τ₀.
- *Example:* with τ₀ = 0.95, the stopping bar falls to τ ≈ 0.89 at 32k.
- *Note on direction:* rising marginal cost makes it *easier to stop*. Put differently, the evidence needed to *justify continuing* rises with c(L).
- *Justification:* Ferguson ch. 5, the monotonicity table in §1, DEER's confidence signal.
- *Collapse:* if c is constant, τ is constant, which is **DEER with a fixed λ**.
- *Failure modes:* the model can be over-confident on wrong answers (ECE 0.10 on AIME in 2504.05419); with digit tokenization, s_k can be dominated by the confidence of a single digit.

**R3 — Joule-window answer stability (consistency with short probes).**
- *Rule:* stop at the first k where ŷ is identical over all probes in the last **W joules** of thinking, i.e. probes j with Ê(L_k) − Ê(L_j) ≤ W, with at least 2 probes in the window and none containing "wait"/"hmm".
- *Justification:* Dynasor-CoT/Certaindex consistency; "agree twice" already reaches 65%.
- *Joules vs tokens:* a fixed-joule window spans fewer tokens later in the trace (W/c(L)). The stability requirement therefore relaxes as cost rises, matching R2.
- *Collapse:* Dynasor's fixed N-probe or fixed-token window.
- *Expected behaviour:* close to "agree twice" accuracy at roughly its "free-probe" savings (13–21%), because probes now cost ~0.6% (1k grid) to ~1.2% (512 grid) of thinking energy rather than up to 4k tokens each.
- *Failure mode:* a wrong answer that repeats stably (a confident attractor).

**R4 — Give-up (futility) rule for long, unsettled traces.**
- *Rule:* at probe k with Ê(L_k) ≥ B_g, **give up** if the last 3 probe answers are pairwise distinct, or if max s over them is < 0.5. Output the modal probe answer, breaking ties by latest. The single parameter is B_g.
- *Value-of-information form:* give up iff λ·P(eventually correct | unsettled) < Ê(T_max) − Ê(L). The tail Ê(32k) − Ê(16k) = 147 kJ is 59% of the trace.
- *Justification:* the lower threshold in Conformal Thinking (rising with ω_t; here it rises with joules), CoDE-Stop's "wrong traces are long and unstable", and our data (55% of energy is spent on wrong answers, 16k → full adds +0 pp).
- *Where joules matter:* this is the **largest lever for energy-awareness**, because the saved segment is the most expensive one.
- *Collapse:* a token threshold L_g.
- *Failure modes:* late solvers (oracle 73% vs 68% means some problems flip late); the cost of a give-up answer that is wrong but was a coin flip anyway.

**R5 — Self-funding probe schedule.**
- *Rule:* probe at the next grid point L only if Σ_{j<k} π_j + π_k ≤ f·Ê(L), with f fixed at 2%. Plug in each probe's *actual* length m_k.
- *Collapse:* in steady state the spacing is Δ ≈ (1+m)/f tokens, i.e. constant (with m=5, f=0.02, Δ≈300 → the 512 grid). Because probe cost and thinking cost both scale with c(L), **the rule collapses to fixed token spacing whenever m is fixed**. It does not collapse when m varies.
- *Why it matters:* it explains and fixes "agree twice". Four 4k-token probes are about 16k probe tokens, so +6–17% instead of the 13–21% saving. R5 would have skipped probes after the first expensive one.
- *Usage:* this is the schedule for R2–R4, not a stopping rule on its own.

**R6 — Composite "energy funnel" (recommended headline).**
- *Rule:* use R5's schedule. Stop if R2 fires (upper bar falling with c(L)) **or** R4 fires (lower bar rising with Ê). Apply R1's hard cap B.
- *Structure:* the two-threshold structure of Conformal Thinking, with joules replacing tokens.
- *Parameters:* set τ₀, B_g and B by LOPO on a 3-D grid. With 60 problems this risks overfitting, so tie them to one exchange rate λ (J per unit P(correct)) if possible.

**Reference only (not deterministic):** a LOPO logistic probe on h_k, following 2504.05419. It serves as an upper bound for what a learned signal adds. The oracle (73%, −18–28%) is the ceiling.

---

## 4. Evaluation protocol

1. **Replay setup.**
   - Every rule is simulated on the same 120 traces at the 512-token grid.
   - **All probe energy is charged:** measured in the GPU pass, with π_k as a cross-check. Thinking energy is Ê and measured E(k) at the stopping point. The answer phase is charged for every policy, including full thinking.
   - Accuracy is the grade of the emitted answer.
2. **Cross-validation.**
   - LOPO, holding out both seeds of a problem together.
   - On the 59 training problems, choose the cheapest parameter whose accuracy is ≥ full-thinking accuracy on those 59. Apply it to the held-out problem.
   - Report this single out-of-sample point; in-sample frontiers are illustration only.
   - Transfer check: calibrate on AIME-24, test on AIME-25.
3. **Metrics.**
   - (i) Accuracy vs mean kJ/answer frontier, plus area under it.
   - (ii) **kJ per correct answer at matched accuracy (≥68%).** Unmatched, this metric is degenerate: the 4k cap minimises it at 100 kJ/correct but at 41% accuracy, versus 144 for the 16k cap and 190 for full thinking.
   - (iii) % energy saved vs full thinking and vs the 16k cap (the bar to beat is 22–26%).
   - (iv) **Energy amplification** = %J saved / %tokens saved.
   - Uncertainty: problem-cluster bootstrap 95% CIs and paired McNemar tests vs full thinking. One answer is 0.83 pp; problem-level SE ≈ 4 pp, so treat differences < ~5 pp as noise.
4. **Decisive energy-awareness tests.**
   - **(a) γ-ablation.** Replace c(L)/c(L₀) in R2/R3/R4 with (c(L)/c(L₀))^γ: γ=0 is the pure token rule, γ=1 the physics shape, γ∈{2,4} exaggerated. Also run Conformal-Thinking-style λ₋ in ω_t vs Ê.
   - **(b) Cross-GPU transfer.** Calibrate B, B_g and W *in joules* on Ada, apply on the PRO 5000 (and the reverse). Compare against the same parameters calibrated *in tokens*. Measure regret against each GPU's own LOPO optimum.
   - **(c) Ranking test.** Check whether the best rule ranked by tokens equals the best rule ranked by joules.
5. **Reading the outcome.**
   - **Energy-awareness matters** if:
     - γ=1 beats γ=0 on the LOPO joule frontier outside the CIs, or
     - joule-calibrated budgets transfer across GPUs with lower regret, or
     - amplification is materially > 1 for R4 (prediction: ~1.1–1.2), or
     - the rule rankings change.
   - **It does not matter** if γ=0 ≈ γ=1, amplification ≈ 1 and the rankings agree. Then the honest claim is: *"energy-aware stopping on one GPU at batch 1 is position-weighted token stopping. The contribution is correct accounting (probes charged at c(L), tail weighting) and a hardware-portable budget, not a new signal."*
   - **My prior:** mostly the second outcome. The visible wins should come from short probes (R3/R5) and give-up (R4), which help in tokens too. Any energy-specific margin should be largest for R4.
