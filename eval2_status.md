# Eval 2 status (2026-11-18)

What has been run for Evaluation 2 ("Working Prototype"), what it showed, and what is still open. Run plans:
`eval2_rtxpro5000_plan.md`, `eval2_rtx6000ada_plan.md`.

## Done

### RTX PRO 5000 Blackwell slot, 2026-11-18 06:00–12:00 (the Eval 1 baseline card)

| Step | Run | Result |
|---|---|---|
| 1. Repro vs Oct 18 | `B2b_repro/20261118-060056_B2b_repro_gpu0_8826de02` | **PASS**: mJ/token within 4.6% of October (limit 5%); thinking on within 1%. Driver 595.84 → 595.99.02. Table: `results/logs/eval2_pro5000_20261118/repro_check.md` |
| 2. Exp A, `aime2025/I-11` t0 (wrong answer), 8 contexts | `A_decode/20261118-060530_A_decode_gpu0_2973a123` | Complete. Block %Capture 93.7–98.0%, Model 93.5–99.6%. `results/tables/A_decode_pro5000.md` |
| 3. Exp B, `aime2024/I-8` t0 (correct answer), 7 contexts | `A_decode_B_correct/20261118-070708_A_decode_B_correct_gpu0_9e3882bc` | Complete. Block 93.8–97.9%, Model 94.0–99.2%. `results/tables/A_decode_B_correct_pro5000.md` |
| 4a. Thinking-off trace `aime2025/II-6` t0, 6 contexts | `A_decode_off/20261118-075958_A_decode_off_gpu0_146d9e91` | Complete. Block 93.9–95.5%, Model 93.5–98.4%. `results/tables/A_decode_off_pro5000.md` |
| 4b. Layer check, layers 0 and 35 at 1k and 16k | `A_decode_layer0/…_cffee253`, `A_decode_layer35/…_53883e81` | Complete. Within 2.2% of layer 18 (table below) |
| 5. Exp C on the Eval 1 MATH traces | `C/20261118-093151_C_gpu0_98bf22f5` | **52 of 54 traces** (status `stopped`, resumable). `results/tables/C_pro5000.md` |

Logs: `results/logs/eval2_pro5000_20261118/` (`launcher.log` is the whole session as seen from the Mac).

**Cost of one new token vs context (J/token, full decode step), PRO 5000**

| Context | Wrong trace (I-11) | Correct trace (I-8) | Thinking off (II-6) | Attention share | MLP share | Ada, same I-11 trace |
|---|---|---|---|---|---|---|
| 256 | 4.89 | 4.90 | 4.88 | 25% | 57% | 6.10 |
| 1k | 5.04 | 5.05 | 5.03 | 27% | 56% | 6.16 |
| 2k | 5.25 | 5.26 | 5.25 | 28% | 54% | 6.22 |
| 4k | 5.61 | 5.62 | 5.61 | 31% | 50% | 6.54 |
| 8k | 6.30 | 6.31 | 6.32 | 37% | 45% | 6.69 |
| 16k | 7.74 | 7.74 | — | 54% | 36% | 7.64 |
| 32k | 10.87 | — | — | 67% | 26% | 10.75 |

- A token costs the same whether the answer ends up right or wrong, and with thinking on or off (within 0.4% at
  equal context). Wrong and thinking-on answers cost more only because they are longer and so decode at longer
  context (AIME wrong thinking-on answers: median ~28k tokens vs ~10.5k for correct ones).
- Attention grows from 25% to 67% of the step between 256 and 32k tokens; MLP stays at ~78 mJ per layer.
- The PRO 5000 is ~20% cheaper per token than the Ada at short context; at 32k the two cards are equal.

**Layer check (mJ per call, median of 20 trials, vs layer 18 from step 2)**

| Context | Component | Layer 0 | Layer 35 | Layer 18 |
|---|---|---|---|---|
| 1k | attention | 37.4 (−0.2%) | 37.1 (−1.0%) | 37.5 |
| 1k | MLP | 79.1 (+1.4%) | 77.9 (−0.0%) | 78.0 |
| 1k | whole layer | 128.5 (+1.2%) | 126.9 (−0.1%) | 126.9 |
| 16k | attention | 118.3 (+1.4%) | 119.3 (+2.2%) | 116.7 |
| 16k | MLP | 78.2 (+0.8%) | 78.4 (+1.0%) | 77.6 |
| 16k | whole layer | 205.4 (+0.2%) | 206.3 (+0.6%) | 205.1 |

So measuring layer 18 and multiplying by 36 is justified.

**Exp C, PRO 5000 vs Ada (MATH, cut thinking at a share of its natural length)**

| | PRO 5000 (52 traces) | Ada (`C_main.md`, 81 traces) |
|---|---|---|
| Keep 50% of thinking | 100% accuracy, 43% energy saved | 100% accuracy, 43% energy saved |
| Keep 25% | 98.1%, 65% saved | see `C_main.md` |
| Thinking off | 98.1%, 86% saved | 95.1%, 85% saved |

### Earlier, on the RTX 6000 Ada (already on `main`)

- Exp A on `aime2025/I-11` t0: `A_decode/20261117-152101_A_decode_gpu1_21ae0d3c`, `results/tables/A_decode.md`.
- Exp C on MATH (81 traces): `results/tables/C_main.md`.
- AIME 2024+2025 thinking on/off, fixed caps (`Caps_aime.md`), offline stopping rules.
- MATH n200 thinking on/off (`B2b_n200`).

## Not done

- **Exp C on the PRO 5000: 2 traces left.** Resume in any later PRO 5000 slot:
  `python -u expc/run_expc.py --source results/runs/B2b/20261017-193728_B2b_gpu0_ba965317 --stop-at HH:MM --resume results/runs/C/20261118-093151_C_gpu0_98bf22f5`
- **Exp B second wrong trace** (`aime2025/II-8` t0, 8 contexts, ~70 min): not run (Ada plan, GPU 1 queue).
- **Ada queue not run:** Exp A repeat on I-11 (run-to-run check on the Ada), MATH n200 reruns of the 18 thinking-off
  answers capped at 2,048 tokens (at 8k) and the 29 thinking-on answers capped at 8,192 (at 32k).
- **`analysis/analyze_expb.py` not written**: the tables above (equal-context comparison, layer check, Ada vs PRO
  5000) were put together by hand from the per-run tables. `analysis/analyze_a_decode.py` fails on the layer-check
  runs because they do not measure the full step.
- **Per-answer component split on the PRO 5000** (Exp A cost curve × token counts of the saved answers): done for
  the Ada only.
- **Eval 2 slides/report**: not started.
- **Run scripts** (`scripts/run_eval2_pro5000.sh`, `scripts/run_eval2_ada.sh`, `scripts/eval2_lib.sh`,
  `analysis/compare_repro.py`) that produced these runs are not on `main` yet.
- **Server repo** (`/data/cs3406/team02/Energy-Benchmarking-for-Reasoning-Models-PAA-Project`): `git pull` is
  blocked by uncommitted edits left from October (B3/B2b tables and figures); today's code was copied in from a
  local clone instead. Clean it up at the start of the next slot (`git stash -u && git pull --ff-only`).

## Lessons from the slot

The server kills programs when the SSH session ends, and the session was held from a laptop. It dropped twice:
lid closed on battery at 08:40 (~35 min of GPU time lost) and a network drop at 10:51 (Exp C resumed at 11:25).
No data was lost; every runner resumes with `--resume`. Next time: laptop on AC, lid open, wired network.
