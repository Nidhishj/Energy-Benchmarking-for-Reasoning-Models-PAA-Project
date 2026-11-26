#!/usr/bin/env bash
# Phase 3 on the RTX PRO 5000 Blackwell (6-hour slot): ablations and the open gaps from the Eval 2 slides, in
# priority order. Same mechanics as run_eval2_pro5000.sh: a step that would not finish before STOP is skipped, and
# rerunning the same command skips finished steps and resumes a stopped one.
#   0. repro gate (vs Oct 18)                       1. last 2 Exp C traces (resume the Nov 18 run)
#   2. live decode of aime2025/I-11, DynamicCache    (live vs replayed step at 8-32k)
#   3. Exp A same-session baseline, then the KV cache written in place (replay and live)
#   4. attention kernel: eager; measurement fixes: no sync, layers cycled; SDPA kernels forced one at a time
#   5. Exp B: correct vs wrong answers to the same AIME question (5 pairs), then 2 more thinking-off traces
#   6. optional: stopping signals (forced short answers every 1k tokens) for the energy-aware stopping rules
#
# Usage, from the repo root (keep the SSH session open):
#   scripts/run_phase3_pro5000.sh 11:45
# Env: PY, CUDA_VISIBLE_DEVICES (default 0), IGNORE_REPRO=1, DRY_RUN=1 (print the plan only).
set -uo pipefail
cd "$(dirname "$0")/.."
STOP=${1:?usage: scripts/run_phase3_pro5000.sh HH:MM}
PY=${PY:-/data/cs3406/team02/envs/energy/bin/python}
GPU_MATCH="PRO 5000"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
LOGDIR=results/logs/phase3_pro5000_$(date +%Y%m%d)
mkdir -p "$LOGDIR"
source scripts/eval2_lib.sh
SINCE=20261127   # only today's runs count as done (step 1 overrides this to find the Nov 18 Exp C run)

AIME1=results/runs/B2b_aime/20261117-000045_B2b_aime_gpu1_725358c0   # copy 1: trial 0
AIME0=results/runs/B2b_aime/20261117-000045_B2b_aime_gpu0_4d9566a3   # copy 2: trial 1
REF_REPRO=results/runs/B2b_repro/20261018-162557_B2b_repro_gpu0_a5135e7f
EVAL1_B2B=results/runs/B2b/20261017-193728_B2b_gpu0_ba965317
EXPA="$PY -u expa/run_decode_components.py"
LIVE="$PY -u expa/run_live_trace.py --source $AIME1 --problem aime2025/I-11 --trial 0"
I11="--source $AIME1 --problem aime2025/I-11 --trial 0"
CTX6="256 1024 4096 8192 16384 32768"
T10="--T 10"

if [[ ${DRY_RUN:-0} == 1 ]]; then
    run_step() { echo "[plan] $1 (~$2 min): ${*:3}"; }
fi

echo "GPU work until $STOP ($(minutes_left) min); logs in $LOGDIR"
nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader | tee "$LOGDIR/gpu.txt"

# 0. Repro gate (~8 min)
run_step B2b_repro 8 $PY -u b2/run_b2b.py --problems test/algebra/858.json test/algebra/246.json \
    test/algebra/733.json --T-on 1 --T-off 1 --idle-s 10 --tag repro --stop-at "$STOP"
if [[ ${DRY_RUN:-0} != 1 ]]; then
    NEW_REPRO=$(complete_run B2b_repro)
    if [[ -z $NEW_REPRO ]]; then
        echo "Repro run did not complete; stopping (rerun to resume)."
        summary
        exit 1
    fi
    if ! $PY analysis/compare_repro.py "$NEW_REPRO" "$REF_REPRO" | tee "$LOGDIR/repro_check.md"; then
        if [[ ${IGNORE_REPRO:-0} != 1 ]]; then
            echo "Repro gate FAILED: compare driver, clocks and power limit with October before going on."
            echo "To continue anyway: IGNORE_REPRO=1 scripts/run_phase3_pro5000.sh $STOP"
            summary
            exit 1
        fi
        echo "Repro gate failed, continuing because IGNORE_REPRO=1"
    fi
fi

# 1. Last 2 of 54 Exp C traces: resume the stopped Nov 18 run (~10 min)
SINCE=20261118 run_step C 12 $PY -u expc/run_expc.py --source $EVAL1_B2B --stop-at "$STOP"

# 2. Live decode, DynamicCache: J/token every 512 tokens up to 32.9k, to compare with the replayed step (~17 min)
run_step Live_trace_dyn 18 $LIVE --cache dynamic --tag dyn

# 3. KV cache: same-session Exp A baseline, then in place (replay), then in place (live)
run_step A_decode_p3base 16 $EXPA $I11 --contexts $CTX6 --components attention mlp layer full_step $T10 --tag p3base
run_step A_decode_inplace 16 $EXPA $I11 --contexts $CTX6 --components attention mlp layer full_step $T10 \
    --cache inplace --tag inplace
run_step Live_trace_inplace 16 $LIVE --cache inplace --tag inplace

# 4. Attention kernel and measurement fixes
run_step A_decode_eager 14 $EXPA $I11 --contexts $CTX6 --components attention layer full_step $T10 \
    --attn eager --prefill-chunk 1024 --tag eager   # eager keeps the full score matrix: small prefill chunks
run_step A_decode_nosync 12 $EXPA $I11 --contexts 1024 16384 $T10 --no-sync --tag nosync
run_step A_decode_cycled 13 $EXPA $I11 --contexts 1024 16384 $T10 --cycle-layers --tag cycled
for b in math efficient flash cudnn; do
    run_step A_decode_sdpa_$b 7 $EXPA $I11 --contexts 1024 8192 32768 --components attention full_step $T10 \
        --sdpa-backend $b --tag sdpa_$b
done

# 5. Exp B: correct vs wrong answers to the SAME question (copy 1 = trial 0, copy 2 = trial 1), 4 contexts each
B="--contexts 1024 4096 8192 16384 --components attention mlp full_step $T10"
pair() {  # <problem> ; both copies
    local p=$1 s=${1//\//_}
    run_step A_decode_B_${s}_t0 9 $EXPA --source $AIME1 --problem $p --trial 0 $B --tag B_${s}_t0
    run_step A_decode_B_${s}_t1 9 $EXPA --source $AIME0 --problem $p --trial 1 $B --tag B_${s}_t1
}
pair aime2024/I-8      # t0 correct 26.4k, t1 wrong 23.6k
pair aime2025/I-9      # t0 correct 16.5k, t1 wrong 25.4k
pair aime2025/II-11    # t0 wrong 17.5k,   t1 correct 20.3k
pair aime2025/I-7      # t0 correct 20.0k, t1 wrong 30.2k
pair aime2025/II-15    # t0 correct 19.3k, t1 wrong 32.6k
# Two more thinking-off traces (the longest in copy 1 after II-6, measured on Nov 18; II-15 also has a pair above)
for p in aime2025/II-15 aime2024/II-3; do
    s=${p//\//_}
    run_step A_decode_Boff_${s} 7 $EXPA --source $AIME1 --problem $p --trial 0 --mode off \
        --contexts 1024 4096 8192 --components attention mlp full_step $T10 --tag Boff_${s}
done

# 6. Optional: stopping signals every 1k thinking tokens on all 120 AIME traces (~50 min; stops itself at STOP and
#    resumes on the next call; the signals do not depend on the GPU, so the rest can be finished on the Ada)
run_step Probe_signals 15 $PY -u expc/run_probe_signals.py --source $AIME1 $AIME0 --stop-at "$STOP"

summary
echo "Results are copied back to the Mac by the launcher; analysis runs there."
