#!/usr/bin/env bash
# Eval 2 on the RTX PRO 5000 Blackwell, steps in priority order (eval2_rtxpro5000_plan.md):
#   1. repro check vs the Oct 18 run on this card (gate: mJ/token within 5%, else stop)
#   2. Exp A on aime2025/I-11 t0 (the trace measured on the Ada)     3. Exp B: correct trace aime2024/I-8 t0
#   4. thinking-off trace + layer 0/35 check                           5. Exp C on the Eval 1 traces (optional)
#   then the analysis tables for whatever completed.
# A step that would not finish before STOP is skipped; rerun the same command later (or after a dropped
# connection) and finished steps are skipped while a stopped one resumes.
#
# Usage, from the repo root (keep the SSH session open, or run inside tmux):
#   scripts/run_eval2_pro5000.sh 17:45            # 17:45 = when GPU work must stop (15 min before the session ends)
# Env: PY (default: the team02 energy env), CUDA_VISIBLE_DEVICES (default 0), IGNORE_REPRO=1 to go on after a failed
# step-1 gate, DRY_RUN=1 to print the plan without touching the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
STOP=${1:?usage: scripts/run_eval2_pro5000.sh HH:MM}
PY=${PY:-/data/cs3406/team02/envs/energy/bin/python}
GPU_MATCH="PRO 5000"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
LOGDIR=results/logs/eval2_pro5000_$(date +%Y%m%d)
mkdir -p "$LOGDIR"
source scripts/eval2_lib.sh

AIME=results/runs/B2b_aime/20261117-000045_B2b_aime_gpu1_725358c0
REF_REPRO=results/runs/B2b_repro/20261018-162557_B2b_repro_gpu0_a5135e7f
EVAL1_B2B=results/runs/B2b/20261017-193728_B2b_gpu0_ba965317
EXPA="$PY -u expa/run_decode_components.py --source $AIME"

if [[ ${DRY_RUN:-0} == 1 ]]; then
    run_step() { echo "[plan] $1 (~$2 min): ${*:3}"; }
fi

echo "GPU work until $STOP ($(minutes_left) min); logs in $LOGDIR"
nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader | tee "$LOGDIR/gpu.txt"

# 1. Repro gate (~10 min)
run_step B2b_repro 10 $PY -u b2/run_b2b.py --problems test/algebra/858.json test/algebra/246.json \
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
            echo "To continue anyway: IGNORE_REPRO=1 scripts/run_eval2_pro5000.sh $STOP"
            summary
            exit 1
        fi
        echo "Repro gate failed, continuing because IGNORE_REPRO=1"
    fi
fi

# 2. Exp A, same trace as the Ada run (8 contexts)
run_step A_decode "$(expa_minutes 8 9)" $EXPA --problem aime2025/I-11 --trial 0 \
    --contexts 256 512 1024 2048 4096 8192 16384 32768
# 3. Exp B, correct trace (7 contexts: the trace has 26,518 tokens)
run_step A_decode_B_correct "$(expa_minutes 7 9)" $EXPA --problem aime2024/I-8 --trial 0 \
    --contexts 256 512 1024 2048 4096 8192 16384 --tag B_correct
# 4. Thinking-off trace (8,987 tokens) and the layer check
run_step A_decode_off "$(expa_minutes 6 9)" $EXPA --problem aime2025/II-6 --trial 0 --mode off \
    --contexts 256 512 1024 2048 4096 8192 --tag off
run_step A_decode_layer0 "$(expa_minutes 2 3)" $EXPA --problem aime2025/I-11 --trial 0 --layer 0 \
    --contexts 1024 16384 --components attention mlp layer --tag layer0
run_step A_decode_layer35 "$(expa_minutes 2 3)" $EXPA --problem aime2025/I-11 --trial 0 --layer 35 \
    --contexts 1024 16384 --components attention mlp layer --tag layer35
# 5. Optional: Exp C on this card's Eval 1 traces; stops itself at STOP and resumes on the next call
if (( $(minutes_left) >= 30 )) || [[ ${DRY_RUN:-0} == 1 ]]; then
    run_step C 30 $PY -u expc/run_expc.py --source $EVAL1_B2B --stop-at "$STOP"
else
    SKIPPED+=("C")
fi

# Analysis of whatever completed (CPU only)
if [[ ${DRY_RUN:-0} != 1 ]]; then
    for exp in A_decode A_decode_B_correct A_decode_off A_decode_layer0 A_decode_layer35; do
        d=$(complete_run $exp)
        [[ -n $d ]] && $PY analysis/analyze_a_decode.py "$d" --out "${exp}_pro5000" | tail -3
    done
    d=$(complete_run C)
    [[ -n $d ]] && $PY analysis/analyze_expc.py "$d" --off $EVAL1_B2B --out C_pro5000 | tail -3
fi
summary
echo "Then: git add results && git commit -m 'Eval 2 on RTX PRO 5000' && git push   (or rsync results/ to the Mac)"
