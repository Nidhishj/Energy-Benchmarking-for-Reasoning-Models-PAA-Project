# Slot plan — RTX PRO 5000 (48 GB)

**Slots:** today 18:00–20:00 (2 h), tomorrow 4 h on the same GPU.
**Login:** `ssh -o ServerAliveInterval=30 team02@10.4.25.63` (only during the slot). Work in `/data/cs3406/team02`.
**Hard rule:** logging out or a dropped SSH connection kills every program (nohup/tmux don't help). Keep two
terminals open for the whole slot, with `ServerAliveInterval` set, and stop runs ~10 min before the slot ends.
All runners write each measurement to disk as soon as it's taken and support `--resume`, so an interrupted run
loses at most one measurement.

**Goal today:** test the whole pipeline on the real GPU and finish everything except the long thinking-on runs:
B1 (rerun, new GPU), B2a, the speed check, B3-short, B2b thinking-off, and the start of thinking-on. Tomorrow: B2b thinking-on (+B3-long),
reproducibility reruns.

## Today, 18:00–20:00

| Time | Terminal 1 (GPU work) | Terminal 2 (downloads, checks) |
|---|---|---|
| 18:00 | Clone `main` and run setup: `bash scripts/slot_setup.sh` (see "Setup" below) | — |
| ~18:08 | `source envs/energy/bin/activate` etc. (printed at the end of setup) | `bash scripts/slot_download.sh` (BERT first, Qwen3-8B last) |
| ~18:10 | `python b1/analyze_b1.py --selftest` (formula check, no GPU), then **B1**: `python b1/run_b1.py` | Downloads continue |
| ~18:40 | `python b1/analyze_b1.py results/runs/B1/<run_id>` → **gate: %Capture ≥ 90%, std/mean ≤ 9.5% (>5 mJ)** | |
| ~18:42 | **B2a**: `python b2/run_b2a.py` (~5 min). **Gate:** accuracy close to the model card; Zeus vs NVML within 1% | |
| ~18:50 | **Speed check**: `python b2/run_b2b.py --modes on --T-on 1 --limit 3 --tag speedcheck` (~5 min) | Look at `traces.jsonl` and the printed lines: answers sane, `</think>` found, tokens/s |
| ~18:58 | **B3-short**: `python b3/run_b3_short.py --stop-at 19:45` (~12 min) | |
| ~19:12 | **B2b**: `python b2/run_b2b.py --stop-at 19:48`. Runs all thinking-off generations first (~15 min), then starts thinking-on until 19:48 | |
| 19:48 | Stop GPU work. Run the analyses (below) on the server, or copy results back first | |
| 19:50 | **Copy results off the server** (below). The 10-min warning arrives at 19:50. | |

If something fails, fix it and rerun with `--resume <run_dir>`; don't start a new run directory for the same config.
If a gate fails (B1 capture, Zeus/NVML gap), stop and debug that first: every later number depends on it.

### Setup (Terminal 1, 18:00)

```bash
cd /data/cs3406/team02
git clone https://github.com/Nidhishj/Energy-Benchmarking-for-Reasoning-Models-PAA-Project
bash Energy-Benchmarking-for-Reasoning-Models-PAA-Project/scripts/slot_setup.sh
```

Setup prints the driver, torch's CUDA architectures, a bf16 matmul test and a 1-second NVML counter reading.
**Check:** `sm_120` is in the arch list (Blackwell), the driver is ≥ 570, and the counter reading is non-zero.
If the server has no internet: copy the models in from a machine that has them, e.g. from the Mac
`rsync -avz models/ team02@10.4.25.63:/data/cs3406/team02/Energy-Benchmarking-for-Reasoning-Models-PAA-Project/models/`.

### Analyses

```bash
python b1/analyze_b1.py results/runs/B1/<run_id>
python analysis/analyze_b2a.py results/runs/B2a/<run_id>
python analysis/analyze_b2b.py results/runs/B2b/<run_id>            # thinking-off only, today
python analysis/analyze_b3.py --short results/runs/B3short/<run_id> --long results/runs/B2b/<run_id>
```

### Copying results back (19:50, from the Mac, not from the server)

```bash
rsync -avz team02@10.4.25.63:/data/cs3406/team02/Energy-Benchmarking-for-Reasoning-Models-PAA-Project/results/ results/
```

Then commit `results/` from the Mac. Files also stay in `/data/cs3406/team02` for tomorrow, but don't rely on that
alone. `traces.jsonl` lives inside each B2b run directory (not in the git-ignored `results/traces/`), so it is
committed with the results; Experiment C needs it.

## Tomorrow, 4 h

| Order | Run | Est. |
|---|---|---|
| 1 | Setup check only (`git pull`, env exists): ~2 min | 5 min |
| 2 | **B2b thinking on**: `python b2/run_b2b.py --resume results/runs/B2b/<today's run> --stop-at <end − 15 min>` | ~90 min |
| 3 | Reproducibility reruns (baselines.md 1.5): one B1 length, B2a dataset pass, 3 MATH500 problems per mode | ~20 min |
| 4 | Checkpoint overhead: one problem with and without `--no-checkpoints`, same seed | ~5 min |
| 5 | Anything that failed today; extra trials if the trial spread was > 5% | rest |
| end | Analyses + copy results back | 15 min |

`--resume` continues the same run directory as long as the arguments that go into the config are the same
(modes, T, token caps, problems); `--stop-at` is not part of the config. So tomorrow's command is today's command
plus `--resume`, and thinking-off and thinking-on rows end up in one run.

**Scaling rule** (baselines.md 5): if the speed check shows a thinking-on problem averaging > 120 s, run 20 problems
(`--limit` is not stratified; use `--problems` with 4 IDs per level instead) or keep T-on = 1.

## Dry-run commands (no GPU, any machine)

Every runner takes `--fake-meter --device cpu` (energy = 100 W × time, meaningless numbers) and writes to
`$ENERGY_RESULTS_DIR` if set, so dry runs don't land in `results/`. Used to test the code before the slot.
