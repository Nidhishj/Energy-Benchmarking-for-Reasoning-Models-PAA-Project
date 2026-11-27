#!/usr/bin/env bash
# Phase 3 PRO 5000 slot, spare time after run_phase3_pro5000.sh: more live decodes, to check that the in-place KV cache
# saving and the live-vs-replay gap hold beyond aime2025/I-11. Same mechanics as the main script (skip finished
# steps, resume stopped ones, skip what would not finish before STOP).
#   1. live decode of a correct/wrong pair (aime2025/I-7: t0 correct 20.0k, t1 wrong 30.2k), both caches
#   2. I-11 repeated with both caches (run-to-run spread of the live numbers)
#   3. eager attention live up to 16k (does the kernel ablation hold in a real decode)
#
# Usage, from the repo root: scripts/run_phase3_extra.sh 11:45
set -uo pipefail
cd "$(dirname "$0")/.."
STOP=${1:?usage: scripts/run_phase3_extra.sh HH:MM}
PY=${PY:-/data/cs3406/team02/envs/energy/bin/python}
GPU_MATCH="PRO 5000"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
LOGDIR=results/logs/phase3_pro5000_$(date +%Y%m%d)
mkdir -p "$LOGDIR"
source scripts/eval2_lib.sh
SINCE=20261127

AIME1=results/runs/B2b_aime/20261117-000045_B2b_aime_gpu1_725358c0   # copy 1: trial 0
AIME0=results/runs/B2b_aime/20261117-000045_B2b_aime_gpu0_4d9566a3   # copy 2: trial 1
LIVE="$PY -u expa/run_live_trace.py"

if [[ ${DRY_RUN:-0} == 1 ]]; then
    run_step() { echo "[plan] $1 (~$2 min): ${*:3}"; }
fi

echo "Extra GPU work until $STOP ($(minutes_left) min)"

# 1. Correct/wrong pair, live, both caches
for c in dynamic inplace; do
    run_step Live_trace_I7_t0_$c 10 $LIVE --source $AIME1 --problem aime2025/I-7 --trial 0 --cache $c --tag I7_t0_$c
    run_step Live_trace_I7_t1_$c 14 $LIVE --source $AIME0 --problem aime2025/I-7 --trial 1 --cache $c --tag I7_t1_$c
done

# 2. I-11 again, both caches
for c in dynamic inplace; do
    run_step Live_trace_I11_rep_$c 14 $LIVE --source $AIME1 --problem aime2025/I-11 --trial 0 --cache $c \
        --tag I11_rep_$c
done

# 3. Eager attention, live, up to 16k (eager keeps the full score matrix: small prefill chunks)
run_step Live_trace_I11_eager 10 $LIVE --source $AIME1 --problem aime2025/I-11 --trial 0 --attn eager \
    --prefill-chunk 1024 --max-len 16384 --tag I11_eager

summary
