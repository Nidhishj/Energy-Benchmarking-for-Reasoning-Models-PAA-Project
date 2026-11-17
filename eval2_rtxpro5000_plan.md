# Eval 2 runs — RTX PRO 5000 Blackwell (`team02@10.4.25.63`)

Goal: the Eval 1 baselines (B1, B2b, B3) were measured on this card, but every Eval 2 result so far (Exp A, Exp C,
AIME, MATH n200) is on the RTX 6000 Ada. This session puts the component results on the same card as the
baselines, and gives a second GPU for Exp A/B. Companion plan: `eval2_rtx6000ada_plan.md`.

The session length is not known yet, so **the steps are in priority order**: each one is useful on its own; stop
wherever time runs out. Steps 1–4 need ~3.5 h. Stop GPU work 15 min before the session ends to copy results.

## Setup (T+0:00, ~10 min)

```bash
ssh -o ServerAliveInterval=30 team02@10.4.25.63
cd /data/cs3406/team02/Energy-Benchmarking-for-Reasoning-Models-PAA-Project
git status --short | head          # all Oct 17–18 PRO 5000 runs are already on GitHub
git pull --ff-only                 # if untracked run dirs block it: rm -rf results && git checkout -- . && git pull --ff-only
source /data/cs3406/team02/envs/energy/bin/activate && export HF_HOME=~/hf_cache
PY=/data/cs3406/team02/envs/energy/bin/python
AIME=results/runs/B2b_aime/20261117-000045_B2b_aime_gpu1_725358c0
nvidia-smi                          # GPU free; note the driver
$PY -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

- In October this machine had driver 595.84, torch 2.8.0+cu128 and Qwen3-8B under `models/`. If the driver or
  torch changed, note it in the log; step 1 then tells us whether the numbers moved.
- Logging out or a dropped connection killed every program last time. Keep the SSH session open (use `tmux` if it
  is allowed). Every runner below can continue with `--resume <run dir>`.
- `pull` must bring the commit with `--mode` in `expa/run_decode_components.py` (step 4) and the AIME traces
  (`results/runs/B2b_aime/*/traces.jsonl`). The traces are token IDs, so they replay on any GPU.

## 1. Sanity check against Eval 1 (~5 min)

Same command as the October repro run (`B2b_repro/20261018-162557_B2b_repro_gpu0_a5135e7f`):

```bash
$PY -u b2/run_b2b.py --problems test/algebra/858.json test/algebra/246.json test/algebra/733.json \
    --T-on 1 --T-off 1 --idle-s 10 --tag repro 2>&1 | tee ~/eval2_repro.log
$PY analysis/analyze_b2b.py results/runs/B2b_repro/<new run> --out B2b_repro_eval2
```

Gate: mJ/token within 5% of the October run, in both modes. If not, stop and compare driver, clocks and power
limit first: every later number on this card depends on it.

## 2. Exp A on the PRO 5000 (~1 h)

Same trace and settings as the Ada run (`A_decode/20261117-152101_A_decode_gpu1_21ae0d3c`), so the two GPUs are
compared on identical work:

```bash
$PY -u expa/run_decode_components.py --source $AIME --problem aime2025/I-11 --trial 0 \
    --contexts 256 512 1024 2048 4096 8192 16384 32768 2>&1 | tee ~/eval2_expa.log
$PY analysis/analyze_a_decode.py results/runs/A_decode/<new run> --out A_decode_pro5000
```

Gates: Block %Capture and Model %Capture ≥ 90% at every context; trial spread ≤ 5% for attention, MLP, LM head and
the full step. At 32k the KV cache is ~4.8 GB on top of ~16 GB of weights, which fits in 48 GB.

Expected from the Eval 1 baselines: lower J/token than the Ada (B3 gave ~5.0 J/token at short context here).

## 3. Exp B, correct trace (~1 h)

The trace measured in step 2 is a **wrong** answer. Same contexts on a **correct** one:

```bash
$PY -u expa/run_decode_components.py --source $AIME --problem aime2024/I-8 --trial 0 \
    --contexts 256 512 1024 2048 4096 8192 16384 --tag B_correct 2>&1 | tee ~/eval2_expb.log
$PY analysis/analyze_a_decode.py results/runs/A_decode_B_correct/<new run> --out A_decode_B_correct_pro5000
```

Gate: attention and MLP at each shared context within ~5% of step 2. A bigger gap is a result in itself; rerun
that one context before believing it.

## 4. Thinking-off trace and layer check (~1 h)

```bash
$PY -u expa/run_decode_components.py --source $AIME --problem aime2025/II-6 --trial 0 --mode off \
    --contexts 256 512 1024 2048 4096 8192 --tag off 2>&1 | tee ~/eval2_off.log
$PY -u expa/run_decode_components.py --source $AIME --problem aime2025/I-11 --trial 0 --layer 0 \
    --contexts 1024 16384 --components attention mlp layer --tag layer0 2>&1 | tee ~/eval2_layer0.log
$PY -u expa/run_decode_components.py --source $AIME --problem aime2025/I-11 --trial 0 --layer 35 \
    --contexts 1024 16384 --components attention mlp layer --tag layer35 2>&1 | tee ~/eval2_layer35.log
```

Gates: thinking-off tokens cost the same as thinking-on tokens at equal context (within ~5%); layers 0 and 35
within ~5% of layer 18.

## 5. Optional, if ≥ 3 h remain: Exp C on the Eval 1 traces (~2–3 h)

Reproduces the MATH "cut thinking short" result on the baseline card, from this card's own Eval 1 thinking-on
traces (30 problems × 2 trials, 6 hit the token limit and are skipped). The same run on the Ada took 4 h for
81 traces; this card is ~1.7× faster per token.

```bash
$PY -u expc/run_expc.py --source results/runs/B2b/20261017-193728_B2b_gpu0_ba965317 \
    --stop-at <HH:MM> 2>&1 | tee ~/eval2_expc.log
$PY analysis/analyze_expc.py results/runs/C/<new run> \
    --off results/runs/B2b/20261017-193728_B2b_gpu0_ba965317 --out C_pro5000
```

`--stop-at` = 15 min before the session ends; continue later with `--resume`. The thinking-off comparison uses the
October thinking-off answers from the same run.

## Not on this card

- AIME thinking on/off and fixed caps: ~9–11 h per copy, and `run_caps.py` needs the source run's live
  checkpoints from the same GPU. Stays on the Ada.
- MATH n200 reruns at higher token limits: on the Ada (runs 6–7 of the Ada plan).

## Copy back and push (last 15 min)

```bash
# on the server
git add results && git commit -m "Eval 2 on RTX PRO 5000: repro, Exp A, Exp B correct trace, off/layer checks"
git push            # if GitHub is slow from the server, rsync to the Mac instead:
# on the Mac
rsync -az team02@10.4.25.63:/data/cs3406/team02/Energy-Benchmarking-for-Reasoning-Models-PAA-Project/results/ results/
rsync -az "team02@10.4.25.63:~/eval2_*.log" results/logs/<date>_rtxpro5000/
```

## Done when

- [ ] Step 1 matches October within 5% (or the change is explained)
- [ ] Exp A on the PRO 5000 with %Capture ≥ 90%: Ada vs PRO 5000 table, same trace
- [ ] Exp B correct trace at equal contexts; thinking-off and layer checks
- [ ] (optional) Exp C on the PRO 5000
- [ ] Results and logs on `main`
