#!/usr/bin/env bash
# CLEAR's own method (clear_repo/ BERT skeleton, adapted to a decoder) on Qwen3-8B, FP16, one GPU, one session.
#
#   tmux new -d -s clear_qwen 'GPU=1 bash b1/clear_qwen3/run.sh'
#   GPU=1 LENS="8 32 64 96 128" bash b1/clear_qwen3/run.sh
#   GPU=1 LENS=8 TRIALS=2 ENERGY_RESULTS_DIR=<scratch> bash b1/clear_qwen3/run.sh      # quick test
#
# Per input length, each as its own python process (as the CLEAR authors ran their scripts):
#   store_activations.py -> layer_measure.py (layer 5) -> module_measure.py (parts) -> full_measure.py
# %Capture as in CLEAR Table 1: Captured(Block) = attention + MLP + both norms; Captured(Model) = 36 x block +
# embedding + LM head + final norm. Analyse with: python b1/clear_qwen3/analyze.py <run dir>
#
# Busy GPU: checked before each script and every second while it runs; if another process appears on our GPU,
# only our script is killed, the run stops and the log says GPU NOT FULLY FREE.
set -uo pipefail

GPU=${GPU:-1}
LENS=${LENS:-"8 32 64 96 128"}
export TRIALS=${TRIALS:-20}
export TAIL=${TAIL:-0}                     # 1 = Run A: also record the queued GPU work CLEAR's loop never counts
EXP=$([ "$TAIL" = 1 ] && echo B1clearQwen3tail || echo B1clearQwen3)

REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-/scratch/nidhish/envs/energy/bin/python}
RESULTS=${ENERGY_RESULTS_DIR:-$REPO/results}
RUN_ID=$(date +%Y%m%d-%H%M%S)_${EXP}_gpu$GPU
OUT=$RESULTS/runs/$EXP/$RUN_ID
mkdir -p "$OUT"
LOG=$OUT/run.log
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
export GPU CUDA_VISIBLE_DEVICES=$GPU MODEL=$REPO/models/Qwen--Qwen3-8B
cd "$REPO"

log() { echo "$(date '+%F %T') [gpu$GPU] $*" | tee -a "$LOG"; }

others_on_gpu() {  # compute processes on our GPU other than pid $1
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$UUID" |
        awk -F', ' -v me="${1:-0}" '$2 != me {print "pid " $2 " (" $3 ")"}' | paste -sd ';'
}

set_status() {
    "$PY" -c "import json,sys,time; p=sys.argv[1]; r=json.load(open(p)); r['status']=sys.argv[2]; r['finished']=time.strftime('%Y%m%d-%H%M%S'); json.dump(r,open(p,'w'),indent=2)" "$OUT/run.json" "$1"
}

stop_monitor() { [ -n "${MON_PID:-}" ] && kill "$MON_PID" 2>/dev/null; }
trap stop_monitor EXIT

used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
util=$(nvidia-smi -i "$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
busy=$(others_on_gpu)
log "run dir: ${OUT#$REPO/}"
log "check: processes='${busy}' memory_used=${used}MiB util=${util}% load=$(cut -d' ' -f1-3 /proc/loadavg)"
if [ -n "$busy" ] || [ "$used" -gt 500 ] || [ "$util" -gt 5 ]; then
    log "GPU $GPU NOT FULLY FREE; not starting"
    exit 3
fi

nvidia-smi --query-gpu=timestamp,index,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used,pstate,clocks_event_reasons.sw_power_cap \
    --format=csv -lms 1000 > "$OUT/gpu_state.csv" &
MON_PID=$!

OUT=$OUT LENS=$LENS EXP=$EXP "$PY" - <<'EOF' 2>&1 | tee -a "$LOG"
import json, os, sys, subprocess, socket, time
sys.path.insert(0, os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]
from energy import GpuMeter
from energy.record import versions
m = GpuMeter(0)
idle = m.measure_idle(30)
run = {"run_id": os.path.basename(os.environ["OUT"]), "experiment": os.environ["EXP"], "host": socket.gethostname(),
       "started": time.strftime("%Y%m%d-%H%M%S"), **m.info(), "idle": idle, "versions": versions(),
       "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
       "model": "Qwen/Qwen3-8B", "dtype": "fp16", "attn_impl": "sdpa", "layer": 5,
       "seq_lens": [int(x) for x in os.environ["LENS"].split()], "trials": int(os.environ["TRIALS"]),
       "N": {"parts": 10000, "full_model": 1000},
       "method": "CLEAR clear_repo/ BERT scripts adapted to Qwen3 (b1/clear_qwen3/)", "tail_tracking": os.environ["TAIL"] == "1",
       "status": "running"}
json.dump(run, open(os.path.join(os.environ["OUT"], "run.json"), "w"), indent=2)
print(f"idle: {idle['mean_W']:.1f} W at {idle['temp_C']} C; {m.name}")
EOF

run_py() {  # $1 = label, $2 = log file, rest = script and args; kill only our script if another process appears
    local label=$1 lf=$2 pid others code
    shift 2
    others=$(others_on_gpu)
    if [ -n "$others" ]; then log "GPU $GPU NOT FULLY FREE before $label: $others; stopping"; set_status gpu_busy; exit 3; fi
    ( cd b1/clear_qwen3 && exec "$PY" -u "$@" ) > "$lf" 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        others=$(others_on_gpu "$pid")
        if [ -n "$others" ]; then
            log "GPU $GPU NOT FULLY FREE during $label: $others; killing our process $pid only"
            kill "$pid"; wait "$pid" 2>/dev/null
            echo "$label" > "$OUT/STOPPED"
            set_status gpu_busy
            exit 3
        fi
        sleep 1
    done
    wait "$pid"; code=$?
    if [ "$code" -ne 0 ]; then log "FAILED $label (exit $code):"; tail -5 "$lf" | tee -a "$LOG"; set_status failed; exit "$code"; fi
}

for L in $LENS; do
    W=$OUT/L$L
    mkdir -p "$W"
    t0=$(date +%s)
    run_py "L$L store" "$W/store.log" store_activations.py "$L" "$W/act.pt"
    run_py "L$L layer" "$W/layer.log" layer_measure.py "$W/act.pt" "$W/block.json"
    run_py "L$L parts" "$W/parts.log" module_measure.py "$W/act.pt" "$W/parts.json"
    run_py "L$L full"  "$W/full.log"  full_measure.py "$L" "$W/full.json"
    rm -f "$W/act.pt"
    log "done L=$L in $(( $(date +%s) - t0 )) s: $("$PY" b1/clear_qwen3/analyze.py --one "$W")"
done

set_status complete
log "all done: ${OUT#$REPO/}"
"$PY" b1/clear_qwen3/analyze.py "$OUT" > /dev/null && log "table: results/tables/${EXP}_$RUN_ID.md"
