#!/usr/bin/env bash
# Run baseline steps one after another on ONE medserver GPU (RTX 6000 Ada), and stop as soon as the GPU is not ours.
#
#   bash scripts/run_medserver.sh <gpu> <step> [<step> ...]        steps: b1 b2a b2b b3short
#   tmux new -d -s energy_gpu0 'bash scripts/run_medserver.sh 0 b2b'                    # survives logout
#   tmux new -d -s energy_gpu1 'bash scripts/run_medserver.sh 1 b3short b1'
#
# Extra arguments per step come from env vars (e.g. to resume):
#   B2B_ARGS="--T-on 3 --resume results/runs/B2b/<run_id>" bash scripts/run_medserver.sh 0 b2b
#
# Busy GPU handling:
#   - Before each step: refuses to start (exit 3) if the GPU has another process, > 500 MiB used or > 5% utilisation,
#     or if /scratch has < 3 GB free.
#   - During a step: the runners check for other processes at every energy-counter read (energy/meter.py, GpuBusy).
#     If one appears, only OUR run stops (exit 3, run status "gpu_busy"); the unfinished trial is not saved, and the
#     remaining steps are skipped. Other users' processes are never touched. Resume later with --resume <run dir>.
set -uo pipefail

GPU=${1:?usage: run_medserver.sh <gpu> <step> [<step> ...]}
shift
[ $# -gt 0 ] || { echo "no steps given (b1 b2a b2b b3short)"; exit 2; }

REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/scratch/nidhish/envs/energy/bin/python}
LOGDIR=$REPO/results/logs/$(date +%Y%m%d)_medserver
mkdir -p "$LOGDIR"
MAIN_LOG=$LOGDIR/run_medserver_gpu${GPU}.log
cd "$REPO"
export CUDA_VISIBLE_DEVICES=$GPU

log() { echo "$(date '+%F %T') [gpu$GPU] $*" | tee -a "$MAIN_LOG"; }

gpu_free() {
    local procs used util free_gb
    local uuid
    uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
    procs=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep -c "$uuid")
    used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    util=$(nvidia-smi -i "$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    free_gb=$(df -BG --output=avail /scratch | tail -1 | tr -dc '0-9')
    log "check: processes=$procs memory_used=${used}MiB util=${util}% /scratch_free=${free_gb}G load=$(cut -d' ' -f1-3 /proc/loadavg)"
    if [ "$procs" -gt 0 ] || [ "$used" -gt 500 ] || [ "$util" -gt 5 ]; then
        log "GPU $GPU NOT FULLY FREE:"
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader | grep "$uuid" | tee -a "$MAIN_LOG"
        return 1
    fi
    if [ "$free_gb" -lt 3 ]; then
        log "/scratch has only ${free_gb}G free (need 3G)"
        return 1
    fi
}

for step in "$@"; do
    case $step in
        b1)      cmd="b1/run_b1.py ${B1_ARGS:-}" ;;
        b2a)     cmd="b2/run_b2a.py ${B2A_ARGS:-}" ;;
        b2b)     cmd="b2/run_b2b.py ${B2B_ARGS:---T-on 3}" ;;
        b3short) cmd="b3/run_b3_short.py ${B3_ARGS:-}" ;;
        *) log "unknown step '$step' (b1 b2a b2b b3short)"; exit 2 ;;
    esac
    # utilization.gpu is averaged over the last sample period, so let our own previous step's activity fade first
    [ "$step" = "$1" ] || sleep 15
    gpu_free || { log "not starting $step; stopping (remaining steps skipped)"; exit 3; }

    step_log=$LOGDIR/${step}_gpu${GPU}_$(date +%H%M%S).log
    log "start $step: python -u $cmd  (log: $step_log)"
    $PY -u $cmd > "$step_log" 2>&1
    code=$?
    run_dir=$(grep -m1 '^Run dir:' "$step_log" | cut -d' ' -f3-)
    case $code in
        0) log "done  $step: $(tail -1 "$step_log")" ;;
        3) log "STOPPED $step: GPU $GPU not fully free (another process appeared). $(grep -m1 'NOT FREE' "$step_log")"
           [ -n "$run_dir" ] && log "resume later with: --resume ${run_dir#$REPO/}"
           exit 3 ;;
        *) log "FAILED $step (exit $code). Last lines:"; tail -5 "$step_log" | tee -a "$MAIN_LOG"
           [ -n "$run_dir" ] && log "run dir: ${run_dir#$REPO/}"
           exit "$code" ;;
    esac
done
log "all steps done: $*"
