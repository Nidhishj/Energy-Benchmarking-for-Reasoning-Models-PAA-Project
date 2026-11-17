#!/usr/bin/env bash
# Eval 2 on the RTX 6000 Ada server (eval2_rtx6000ada_plan.md). Queues, in priority order:
#   gpu1: Exp B correct trace, thinking-off trace, layer 0/35 check, Exp B second wrong trace     (~3.5 h)
#   gpu0: Exp A repeat on I-11, MATH n200 reruns at 8k (thinking off) and 32k (thinking on)      (~2 h + reruns)
#   all:  gpu1 then gpu0 on ONE GPU, for a single-GPU allocation (~5.5 h before the 32k reruns)
# A step that would not finish before STOP is skipped; the n200 reruns stop themselves at STOP. Rerun the same
# command later: finished steps are skipped and a stopped one resumes.
#
# Usage, from the repo root:
#   CUDA_VISIBLE_DEVICES=1 setsid nohup scripts/run_eval2_ada.sh gpu1 21:30 > results/logs/eval2_gpu1.out 2>&1 < /dev/null &
#   CUDA_VISIBLE_DEVICES=0 setsid nohup scripts/run_eval2_ada.sh gpu0 21:30 > results/logs/eval2_gpu0.out 2>&1 < /dev/null &
# Env: PY (default: Nidhish's energy env), DRY_RUN=1 to print the plan without touching the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
QUEUE=${1:?usage: scripts/run_eval2_ada.sh gpu1|gpu0|all HH:MM}
STOP=${2:?usage: scripts/run_eval2_ada.sh gpu1|gpu0|all HH:MM}
PY=${PY:-/scratch/nidhish/envs/energy/bin/python}
GPU_MATCH="RTX 6000 Ada"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the GPU this queue may use}"
LOGDIR=results/logs/eval2_ada_$(date +%Y%m%d)
mkdir -p "$LOGDIR"
source scripts/eval2_lib.sh

AIME=results/runs/B2b_aime/20261117-000045_B2b_aime_gpu1_725358c0
EXPA="$PY -u expa/run_decode_components.py --source $AIME"
N200="$PY -u b2/run_b2b.py --subset configs/math500_subset200.json"
# n200 answers that hit the token limit in B2b_n200/20261116-150325_B2b_n200_gpu0_9421301e
CAP_OFF="test/geometry/434.json test/intermediate_algebra/894.json test/precalculus/117.json
test/intermediate_algebra/362.json test/intermediate_algebra/1779.json test/intermediate_algebra/90.json
test/precalculus/1313.json test/intermediate_algebra/345.json test/geometry/880.json test/geometry/229.json
test/intermediate_algebra/158.json test/geometry/172.json test/intermediate_algebra/2022.json
test/counting_and_probability/1003.json test/intermediate_algebra/582.json test/intermediate_algebra/1806.json
test/counting_and_probability/181.json test/intermediate_algebra/1279.json"
CAP_ON="test/geometry/434.json test/intermediate_algebra/607.json test/intermediate_algebra/894.json
test/precalculus/117.json test/intermediate_algebra/1151.json test/prealgebra/1733.json test/prealgebra/631.json
test/counting_and_probability/25149.json test/number_theory/417.json test/intermediate_algebra/1779.json
test/counting_and_probability/238.json test/intermediate_algebra/90.json test/prealgebra/465.json
test/precalculus/989.json test/prealgebra/260.json test/intermediate_algebra/345.json test/intermediate_algebra/966.json
test/intermediate_algebra/2196.json test/geometry/880.json test/geometry/229.json test/intermediate_algebra/232.json
test/intermediate_algebra/1350.json test/intermediate_algebra/2022.json test/prealgebra/1139.json
test/intermediate_algebra/582.json test/intermediate_algebra/1806.json test/algebra/2176.json
test/intermediate_algebra/1279.json test/counting_and_probability/159.json"

if [[ ${DRY_RUN:-0} == 1 ]]; then
    run_step() { echo "[plan] $1 (~$2 min): ${*:3}"; }
fi

queue_gpu1() {
    run_step A_decode_B_correct "$(expa_minutes 7 9)" $EXPA --problem aime2024/I-8 --trial 0 \
        --contexts 256 512 1024 2048 4096 8192 16384 --tag B_correct
    run_step A_decode_off "$(expa_minutes 6 9)" $EXPA --problem aime2025/II-6 --trial 0 --mode off \
        --contexts 256 512 1024 2048 4096 8192 --tag off
    run_step A_decode_layer0 "$(expa_minutes 2 3)" $EXPA --problem aime2025/I-11 --trial 0 --layer 0 \
        --contexts 1024 16384 --components attention mlp layer --tag layer0
    run_step A_decode_layer35 "$(expa_minutes 2 3)" $EXPA --problem aime2025/I-11 --trial 0 --layer 35 \
        --contexts 1024 16384 --components attention mlp layer --tag layer35
    run_step A_decode_B_wrong "$(expa_minutes 8 9)" $EXPA --problem aime2025/II-8 --trial 0 \
        --contexts 256 512 1024 2048 4096 8192 16384 32768 --tag B_wrong
}

queue_gpu0() {
    run_step A_decode_repro "$(expa_minutes 7 9)" $EXPA --problem aime2025/I-11 --trial 0 \
        --contexts 256 1024 2048 4096 8192 16384 32768 --tag repro
    # Same seeds as the n200 run, so each answer regenerates its old tokens and then continues past the old limit.
    run_step B2b_n200_8k_off 15 $N200 --modes off --T-off 1 --max-new-off 8192 --problems $CAP_OFF \
        --tag n200_8k_off --stop-at "$STOP"
    run_step B2b_n200_32k 20 $N200 --modes on --T-on 1 --max-new-on 32768 --problems $CAP_ON \
        --tag n200_32k --stop-at "$STOP"
}

echo "Queue $QUEUE on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES until $STOP ($(minutes_left) min); logs in $LOGDIR"
case $QUEUE in
    gpu1) queue_gpu1 ;;
    gpu0) queue_gpu0 ;;
    all) queue_gpu1; queue_gpu0 ;;
    *) echo "unknown queue $QUEUE (gpu1, gpu0 or all)"; exit 2 ;;
esac

# Analysis of whatever completed (CPU only)
if [[ ${DRY_RUN:-0} != 1 ]]; then
    for exp in A_decode_B_correct A_decode_off A_decode_layer0 A_decode_layer35 A_decode_B_wrong A_decode_repro; do
        d=$(complete_run $exp)
        [[ -n $d ]] && $PY analysis/analyze_a_decode.py "$d" --out "$exp" | tail -3
    done
fi
summary
