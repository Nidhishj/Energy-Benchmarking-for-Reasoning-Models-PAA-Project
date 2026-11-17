# Eval 2 runs — RTX 6000 Ada (server, GPU 0 and GPU 1)

Goal: finish the Eval 2 roadmap (Exp A + Exp B results) on the GPU that already holds Exp A, Exp C, AIME and
the 200-problem run. Companion plan for the RTX PRO 5000: `eval2_rtxpro5000_plan.md`.

## Where we are (as of 17 Nov, commit `199e176`)

| Roadmap item | Status | Run dir (`results/runs/...`) |
|---|---|---|
| Component pipeline on Qwen3 decode steps (parts ≥ 90%) | Done: Block 99–105%, Model 94–98% | `A_decode/20261117-152101_A_decode_gpu1_21ae0d3c` |
| **Exp A**: component energy vs context (256..32k) | Done, **one trace** (`aime2025/I-11` t0, a **wrong** answer that hit 32k), layer 18 × 36, GPU 1 | same |
| **Exp B**: correct vs wrong, per component | **Not started** | — |
| Exp C (cut thinking short) | Done early: MATH (28 problems) and AIME caps (2 copies) + stopping rules | `C/…`, `Caps_aime/…` |
| More problems, higher token limit | AIME done (32k). MATH n200: **29 thinking-on and 18 thinking-off answers hit the token limit** (8,192 / 2,048) | `B2b_n200/20261116-150325_…` |

What Exp B needs: Exp A shows a token's cost depends on how many tokens are already in the context. If that also
holds regardless of *which* trace (correct or wrong), then a wrong answer's component split follows from its
length: on AIME, wrong thinking-on answers have median ~28k tokens vs ~10.5k for correct ones, so more of their
tokens fall where attention dominates. Runs 1–2 below test the "regardless of which trace" part; the per-answer
split is then computed offline (no GPU).

## Quick start: one script per GPU

```bash
git pull --ff-only
mkdir -p results/logs
# both GPUs, until 21:30:
CUDA_VISIBLE_DEVICES=1 setsid nohup scripts/run_eval2_ada.sh gpu1 21:30 > results/logs/eval2_gpu1.out 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=0 setsid nohup scripts/run_eval2_ada.sh gpu0 21:30 > results/logs/eval2_gpu0.out 2>&1 < /dev/null &
# only one GPU: everything in priority order on it
CUDA_VISIBLE_DEVICES=1 setsid nohup scripts/run_eval2_ada.sh all 21:30 > results/logs/eval2_all.out 2>&1 < /dev/null &
```

`scripts/run_eval2_ada.sh` runs the queues below in order, skips a step that would not finish before the stop time,
lets the n200 reruns stop themselves at it, and writes the Exp A/B tables at the end. Rerun the same command
later: finished steps are skipped, a stopped one resumes. `DRY_RUN=1` prints the plan without touching the GPU.

### With a 6-hour allocation

| Allocation | What finishes in 6 h | Left for later |
|---|---|---|
| **Both GPUs** | GPU 1: runs 1–4 (~3.4 h). GPU 0: Exp A repeat (~1 h), thinking-off reruns (≤ 1 h), then ~4 h of the 32k reruns (roughly half of the 29) | The rest of the 32k reruns |
| **One GPU** (`all`) | Runs 1–5 (~4.4 h) and the thinking-off reruns (≤ 1 h) | All of the 32k reruns (needs another ~5–8 h) |

Either way, everything Eval 2 needs (Exp B equal-t check, thinking-off check, layer check, Exp A repeat) fits in
6 h. Only the n200 thinking-on reruns don't: until they finish, report n200 thinking on with the 29 capped
answers marked, as now.

The commands below are what the script runs, for running a step by hand.

## Before starting

- Pull the commit that adds `--mode` to `expa/run_decode_components.py` (needed for run 3).
- `nvidia-smi`: both GPUs free, no other processes (the runners refuse a busy GPU).
- Run everything detached so it survives the terminal closing, one log per run:

```bash
PY=/scratch/nidhish/envs/energy/bin/python
AIME=results/runs/B2b_aime/20261117-000045_B2b_aime_gpu1_725358c0      # traces generated on GPU 1, trial 0
mkdir -p results/logs
```

## GPU 1 queue (~3.5 h): Exp B and checks on the GPU that measured Exp A

The script runs them in the order 1, 3, 4, 2: the second wrong trace goes last as the least critical.
All on GPU 1 so the comparison with the existing `I-11` run is within one card (the two Ada cards differ by ~8–10%).
Same settings as the Exp A run (T = 20, ~2 s trials, layer 18). Contexts above a trace's length are skipped
automatically.

| # | Run | Trace | Why | Est. |
|---|---|---|---|---|
| 1 | **Exp B, correct trace** | `aime2024/I-8` t0: **correct**, 26,518 tokens (longest correct) | Same contexts as the wrong `I-11` run: does a correct trace cost the same per component at equal t? | ~1 h |
| 2 | **Exp B, second wrong trace** | `aime2025/II-8` t0: **wrong**, hit 32k | Spread between two wrong traces = how big a "difference" must be to count | ~1 h |
| 3 | **Thinking-off trace** | `aime2025/II-6` t0, thinking **off**, 8,987 tokens | Thinking-off tokens were never measured; checks they cost the same as thinking-on tokens at equal t | ~40 min |
| 4 | **Layer check** | `I-11` t0, layers 0 and 35 | Exp A measures layer 18 and multiplies by 36; checks the first and last layers cost the same | ~20 min |

```bash
CUDA_VISIBLE_DEVICES=1 setsid nohup bash -c "
$PY -u expa/run_decode_components.py --source $AIME --problem aime2024/I-8 --trial 0 \
    --contexts 256 512 1024 2048 4096 8192 16384 --tag B_correct &&
$PY -u expa/run_decode_components.py --source $AIME --problem aime2025/II-8 --trial 0 \
    --contexts 256 512 1024 2048 4096 8192 16384 32768 --tag B_wrong &&
$PY -u expa/run_decode_components.py --source $AIME --problem aime2025/II-6 --trial 0 --mode off \
    --contexts 256 512 1024 2048 4096 8192 --tag off &&
$PY -u expa/run_decode_components.py --source $AIME --problem aime2025/I-11 --trial 0 --layer 0 \
    --contexts 1024 16384 --components attention mlp layer --tag layer0 &&
$PY -u expa/run_decode_components.py --source $AIME --problem aime2025/I-11 --trial 0 --layer 35 \
    --contexts 1024 16384 --components attention mlp layer --tag layer35
" > results/logs/eval2_gpu1.log 2>&1 < /dev/null &
```

Gates (check in the log after each run):
- Every component's trial spread (`std`) ≤ 5% (the Exp A run had 3–13%; the 13% ones are the sub-1 mJ embed/norm calls).
- Run 1, 2: `attention` and `mlp` at each shared context within ~5% of the `I-11` run. A bigger gap is itself a
  result (cost depends on content), so rerun that context once before believing it.
- Run 4: layers 0 and 35 within ~5% of layer 18 at both contexts.

If a run stops: rerun the same command with `--resume results/runs/A_decode_<tag>/<run dir>`.

## GPU 0 queue (~1 h, then overnight)

| # | Run | Why | Est. |
|---|---|---|---|
| 5 | **Exp A repeat** on `I-11` t0, same settings as the original | Same config run twice must give the same number (our reproducibility rule); also gives GPU 0 vs GPU 1 | ~1 h |
| 6 | **MATH n200, thinking off, at 8,192 tokens**: the 18 answers that hit 2,048 | Thinking off got cut at 2,048; AIME already uses 8,192 for thinking off | ~1 h |
| 7 | **MATH n200, thinking on, at 32,768 tokens**: the 29 answers that hit 8,192 | They were graded wrong for running out of tokens; this is why thinking on (85.5%) scored below off (86%) | ~5–8 h |

Runs 6–7 use the same seeds as the n200 run (seed depends on problem, mode and trial), so each answer should
regenerate the same tokens up to the old limit and then continue. Check this on the first answer: its first 8,192
tokens must match `B2b_n200/.../traces.jsonl`.

```bash
CAP_OFF="test/geometry/434.json test/intermediate_algebra/894.json test/precalculus/117.json \
test/intermediate_algebra/362.json test/intermediate_algebra/1779.json test/intermediate_algebra/90.json \
test/precalculus/1313.json test/intermediate_algebra/345.json test/geometry/880.json test/geometry/229.json \
test/intermediate_algebra/158.json test/geometry/172.json test/intermediate_algebra/2022.json \
test/counting_and_probability/1003.json test/intermediate_algebra/582.json test/intermediate_algebra/1806.json \
test/counting_and_probability/181.json test/intermediate_algebra/1279.json"
CAP_ON="test/geometry/434.json test/intermediate_algebra/607.json test/intermediate_algebra/894.json \
test/precalculus/117.json test/intermediate_algebra/1151.json test/prealgebra/1733.json test/prealgebra/631.json \
test/counting_and_probability/25149.json test/number_theory/417.json test/intermediate_algebra/1779.json \
test/counting_and_probability/238.json test/intermediate_algebra/90.json test/prealgebra/465.json \
test/precalculus/989.json test/prealgebra/260.json test/intermediate_algebra/345.json test/intermediate_algebra/966.json \
test/intermediate_algebra/2196.json test/geometry/880.json test/geometry/229.json test/intermediate_algebra/232.json \
test/intermediate_algebra/1350.json test/intermediate_algebra/2022.json test/prealgebra/1139.json \
test/intermediate_algebra/582.json test/intermediate_algebra/1806.json test/algebra/2176.json \
test/intermediate_algebra/1279.json test/counting_and_probability/159.json"

CUDA_VISIBLE_DEVICES=0 setsid nohup bash -c "
$PY -u expa/run_decode_components.py --source $AIME --problem aime2025/I-11 --trial 0 \
    --contexts 256 1024 2048 4096 8192 16384 32768 --tag repro &&
$PY -u b2/run_b2b.py --subset configs/math500_subset200.json --modes off --T-off 1 --max-new-off 8192 \
    --problems $CAP_OFF --tag n200_8k_off &&
$PY -u b2/run_b2b.py --subset configs/math500_subset200.json --modes on --T-on 1 --max-new-on 32768 \
    --problems $CAP_ON --tag n200_32k
" > results/logs/eval2_gpu0.log 2>&1 < /dev/null &
```

## Analysis after the runs (no GPU)

```bash
$PY analysis/analyze_a_decode.py results/runs/A_decode_B_correct/<run> --out A_decode_B_correct   # one per new run
$PY analysis/analyze_a_decode.py results/runs/A_decode_repro/<run>     --out A_decode_repro
```

Still to write (Aaditya):
- `analysis/analyze_expb.py`:
  - **Equal-t table**: correct vs wrong vs thinking-off trace, per component, at each shared context.
  - **Per-answer split**: Exp A's per-component cost at each t, summed over every token of every saved answer
    (AIME both copies, MATH n200 with the 32k reruns merged in). Reports attention / MLP / norms / head per answer,
    for correct vs wrong and on vs off, and checks the sum against the measured J/query (first version
    covered 86–92%).
- Exp C per component: the same per-answer split at each thinking budget, i.e. which component's energy drops
  when thinking is cut.

## Done when

- [ ] Runs 1–5 complete, gates checked, tables generated
- [ ] Runs 6–7 complete; n200 table regenerated with the reruns replacing the capped rows
- [ ] `analyze_expb.py` tables and figure: equal-t check + per-answer split, correct vs wrong
- [ ] Logs copied to `results/logs/`, everything committed and pushed to `main`
