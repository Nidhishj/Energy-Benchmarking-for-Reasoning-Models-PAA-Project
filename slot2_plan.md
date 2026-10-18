# Slot 2 plan — RTX PRO 5000, 4 hours

Follows `slot_plan.md` (slot 1, 2026-10-17 18:00–20:00). Times below are from the start of the slot (T+0:00).
**Login:** `ssh -o ServerAliveInterval=30 team02@10.4.25.63`. Work in `/data/cs3406/team02`.
Logging out or a dropped connection kills every program; stop GPU work 15 min before the slot ends.

## Where slot 1 left us

| Baseline | Status | Run dir (`results/runs/...`) |
|---|---|---|
| B1 (CLEAR on BERT, T=20, 5 lengths) | Done, **%Capture(Model) fails** (73–84% vs ≥ 90%) | `B1/20261017-185510_B1_gpu0_a489156e` |
| B2a (MRPC) | Done, but the dataset pass was 0.65 s per window (too short); **rerun** | `B2a/20261017-191221_B2a_gpu0_883b6182` |
| B2b speed check | Done: 3/3 correct, ~60 tok/s, ~27 s per thinking-on problem | `B2b_speedcheck/20261017-192358_...` |
| B3-short | Done, reproduces CLEAR Fig. 1 | `B3short/20261017-192548_B3short_gpu0_4546c8ed` |
| B2b + B3-long | **Partial:** thinking-off trial 0 (30/30), trial 1 (27/30); trial 2 and all thinking-on missing | `B2b/20261017-193728_B2b_gpu0_ba965317` |

Already in place on the server (no setup needed): env `~/envs/energy` (torch 2.8.0+cu128), all models in
`Energy-Benchmarking-for-Reasoning-Models-PAA-Project/models/`, uv in `~/.local/bin`.

## Before the slot (no GPU) — done 2026-10-18

- [x] **B2b analysis: per-token spread column.** On slot 1's thinking-off rows: J/query spread 7.5% (answer lengths
      differ between sampled trials), **mJ/token spread 0.6% (max 2.5%)**, so the 5% check passes on mJ/token.
- [x] **`b1/debug_capture.py`** for the B1 capture debugging (below). Dry-run on CPU with the fake meter.
- [x] **`--trials` for `b2/run_b2b.py`**: runs only the listed trial indices into a new run dir. It is only added to
      the config when used, so yesterday's run still resumes: checked by resuming a copy of it with today's code
      (same config hash once the dry run's Zeus-off flag is accounted for).

## Slot timeline

| Time | Step | Est. | Gate / check |
|---|---|---|---|
| T+0:00 | **Sync code** (below), `nvidia-smi` (GPU free, no other processes) | 10 min | Server repo at the latest `main` |
| T+0:10 | **Resume B2b** (below): finishes thinking-off (33 gens), then all thinking-on (60 gens) | ~50 min | Zeus vs NVML within 1%; cap-hit rate; accuracy on ≈ 90%+ |
| T+1:00 | **B2a rerun** with 5 dataset passes per window | 3 min | Dataset-pass std/mean now ≲ 3% |
| T+1:05 | **Reproducibility reruns** (baselines.md 1.5) | 15 min | Each matches slot 1 within the reported std |
| T+1:20 | **Checkpoint overhead**: same problem and seed, with and without checkpoints | 5 min | Energy difference < 1% |
| T+1:25 | **B1 capture debugging** | 60–75 min | Find which hypothesis explains the ~18% gap |
| T+2:40 | Optional: third thinking-on trial (below), or more B1 debugging | ≤ 50 min | Stop by T+3:30 |
| T+3:30 | **Analyses, copy results to the Mac, commit and push** | 15 min | Everything on GitHub before T+3:45 |

### Sync code (T+0:00)

The server has slot 1's run dirs as untracked files that are now also on GitHub, so a plain `git pull` refuses.
Results are identical on GitHub, so drop the local copies and pull:

```bash
cd /data/cs3406/team02/Energy-Benchmarking-for-Reasoning-Models-PAA-Project
rm -rf results && git checkout -- . && git pull --ff-only
ls results/runs/B2b/20261017-193728_B2b_gpu0_ba965317     # traces.jsonl, measurements.jsonl, run.json must be back
source ~/envs/energy/bin/activate && export HF_HOME=~/hf_cache
```

If `git pull` hangs (GitHub was slow from the server in slot 1), push the code from the Mac instead:
`rsync -az --exclude models --exclude .git ./ team02@10.4.25.63:/data/cs3406/team02/Energy-Benchmarking-for-Reasoning-Models-PAA-Project/`.

### Resume B2b (T+0:10)

```bash
python -u b2/run_b2b.py --resume results/runs/B2b/20261017-193728_B2b_gpu0_ba965317 --idle-s 10 --stop-at <HH:MM> 2>&1 | tee ~/b2b_resume.log
```

- Same arguments as slot 1 (defaults: T-off 3, T-on 2, caps 2048/8192), otherwise `--resume` refuses because
  the config hash differs. `--stop-at` is not part of the config; set it to T+1:00 or later.
- Estimate: 33 thinking-off gens × ~9 s ≈ 5 min, then 60 thinking-on gens. Level 1 took ~27 s; harder levels think
  longer, so budget ~40 s average ≈ 40 min. Watch for `CAP` (8192 tokens) on level 5.

### B2a rerun (T+1:00)

```bash
python -u b2/run_b2a.py 2>&1 | tee ~/b2a_rerun.log      # --passes-per-window 5 is the default now
python analysis/analyze_b2a.py results/runs/B2a/<new run>
```

### Reproducibility reruns (T+1:05)

A number counts as final only after a second run of the same config agrees within the reported std
(baselines.md 1.5). New run dirs (tag `repro`), so they don't mix with the main runs:

```bash
python -u b1/run_b1.py --seq-lens 64 --tag repro                                        # ~5 min
python -u b2/run_b2b.py --problems test/algebra/858.json test/algebra/246.json test/algebra/733.json \
    --T-on 1 --T-off 1 --tag repro --stop-at <HH:MM>                                    # ~4 min
python -u b3/run_b3_short.py --ks 1 5 30 --T 5 --tag repro                              # ~3 min
```

Compare B1 L=64 per component, B2b mJ/token per problem (sampled outputs differ, so compare energy per token,
not per query), and B3-short E(k) at k = 1, 5, 30.

### Checkpoint overhead (T+1:20)

The checkpoint callback syncs the GPU ~20 times per generation. Check that this doesn't change the energy:

```bash
python -u b2/run_b2b.py --problems test/algebra/1214.json --modes on --T-on 1 --tag ckpt_on
python -u b2/run_b2b.py --problems test/algebra/1214.json --modes on --T-on 1 --tag ckpt_off --no-checkpoints
```

Same seed, so the traces should be identical (check `output_len`); compare J/token.

## B1 capture debugging (T+1:25)

**The gap.** Captured(Model) = 12 × Block + Embedding + CLS head reaches only 73–84% of the full model. From
slot 1's raw rows (32 / 128 tokens):

| | Time per call | Energy per call | Average power |
|---|---|---|---|
| One block (replayed alone) | 117 µs | 13.3 / 18.1 mJ | 114 / 155 W |
| Full model | 1614 / 1617 µs | 213 / 290 mJ | 132 / 180 W |
| 12 blocks + embedding + head (sum) | ~1.47–1.51 ms (≈ 92% of full) | ≈ 80–83% of full | |

So about 8 points of the gap are **time** (the full model spends ~110–150 µs on things no component covers) and
the rest is **power** (a block replayed on its own draws ~13% less power than inside the full model).

**Hypotheses and the measurement that tests each:**

1. **Cache residency.** Replaying one layer 10,000 times keeps its ~7 MB of weights in the GPU's L2 cache, so
   replays read less from DRAM than the real forward pass, which streams 12 different layers.
   *Test:* replay all 12 layers in sequence per call (each with its own captured input), N = 1,000.
   If Σ(12 layers in sequence) ≈ full model − embedding − head, while 12 × (one layer alone) is lower, the gap is
   cache residency, and component numbers need cycling replays.
2. **Work outside the components.** The full forward pass also builds the attention mask, runs Python glue between
   layers and may launch extra small kernels. *Test:* PyTorch profiler on one full forward pass (kernel list and
   time not inside any measured module); also time `model.bert(**inputs)` vs `model(**inputs)`.
3. **Power state.** A 117 µs kernel sequence repeated back to back may not hold the same clocks/power as a 1.6 ms
   forward pass. *Test:* compare `sm_clock_MHz` and `power_W` logged per trial (already in the rows) between block
   and full-model trials; optionally lock clocks if permitted (`nvidia-smi -lgc`, needs root, probably not).

All three are measured by one script (~6 min for two lengths; `--each-layer` adds ~3 min per length and checks that
layer 5 is representative):

```bash
python -u b1/debug_capture.py --seq-lens 32 128 2>&1 | tee ~/b1_debug.log
```

It prints a profile of one forward pass (kernel count, GPU-busy share of wall time), one line per part
(mJ, std/mean, average W, SM clock range), and a summary per length:
- `B1 captured`: B1's definition, should reproduce slot 1's ~80%.
- `H1 cycled`: the same sum with the 12 layers run in sequence. Close to 100% → cache residency explains the gap.
- `H1 flushed`: one layer with the L2 cache flushed before each call, flush energy subtracted. Above `layer_alone`
  → the cached replay underestimates.
- `H2 glue`: `encoder / layers_cycled` and `(backbone + head) / full_model`. Well above 1 → work between layers.

Start with H1 (biggest expected effect, cheapest to test), then H2 from the profile. Record the result in the B1
table notes either way: CLEAR's own released code replays a single layer, so this is also a finding about CLEAR.

## Optional third thinking-on trial (T+2:40)

```bash
python -u b2/run_b2b.py --modes on --T-on 3 --trials 2 --tag extra --stop-at <HH:MM>
```

New run dir; give it to the analysis together with the main run. ~30 gens × ~40 s ≈ 20 min.

## Analyses and push (T+3:30)

```bash
python analysis/analyze_b2b.py results/runs/B2b/20261017-193728_B2b_gpu0_ba965317 --out B2b_final
python analysis/analyze_b3.py --short results/runs/B3short/20261017-192548_B3short_gpu0_4546c8ed \
    --long results/runs/B2b/20261017-193728_B2b_gpu0_ba965317 --out B3_final
python analysis/analyze_b2a.py results/runs/B2a/<new run>
```

From the Mac:

```bash
rsync -az team02@10.4.25.63:/data/cs3406/team02/Energy-Benchmarking-for-Reasoning-Models-PAA-Project/results/ results/
rsync -az "team02@10.4.25.63:/data/cs3406/team02/*.log" results/logs/<date>_rtxpro5000/
git add results && git commit -m "Slot 2 results" && git push
```

## Done when

- [ ] B2b: 30 problems × (3 thinking-off + 2 thinking-on) trials, analysed, on/off ratio reported
- [ ] B3: figures B3-1/2/3 and Table B3 from the complete B2b run
- [ ] B2a rerun with a stable dataset-pass number; Zeus vs NVML within 1%
- [ ] Reproducibility reruns agree with slot 1 (or the disagreement is documented)
- [ ] Checkpoint overhead measured
- [ ] B1 capture gap explained, or the tested hypotheses and their results written up
- [ ] All results, logs and tables pushed to `main`
