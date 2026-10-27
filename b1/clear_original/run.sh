#!/usr/bin/env bash
# Run the CLEAR authors' own BERT scripts (clear_repo/) back to back in one session on one GPU, to see whether
# their code reproduces their %Capture (~97%) on our RTX 6000 Ada, or gives ~85% like b1/run_b1.py.
#
#   tmux new -d -s clear_gpu0 'bash b1/clear_original/run.sh'
#   GPU=0 ROUNDS=3 LENS="8 32 64 96 128" CONTROL=1 bash b1/clear_original/run.sh
#   TRIALS=2 LENS=8 ROUNDS=1 CONTROL=0 ENERGY_RESULTS_DIR=<scratch> bash b1/clear_original/run.sh    # quick test
#
# For each round and each input length, in this order, each as its own python process (as the authors ran them):
#   store_activations.py -> layer_measure.py (layer 5) -> module_measure.py (parts) -> full_measure.py
# Afterwards (CONTROL=1) our b1/run_b1.py runs on the same GPU, so both methods share one session.
#
# The scripts are copied to <run dir>/scripts/ and patched only where they cannot run as released; every change
# is saved as a diff next to them:
#   - store_activations.py: remove `print(model); exit(0)` (as released it exits before saving anything)
#   - seq_len (hard-coded 8) -> this length; model_name -> local copy of bert-base-uncased (same weights)
#   - activation and output file names -> files in the run dir (the scripts disagree: activations.pt vs activations2.pt)
#   - device_index -> $GPU; num_trials only if TRIALS is set (tests)
# Measurement code (repeat counts, 0.5 s pause, no warm-up, no sync, separate processes) is left as it is.
#
# Their scripts do not check for other processes, so this wrapper does: before each script, and every second while
# it runs. If another process appears on our GPU, only our current script is killed, the run stops, and the log
# says GPU NOT FULLY FREE. GPU power, clocks and temperature of both GPUs are logged every second (gpu_state.csv).
set -uo pipefail

GPU=${GPU:-0}
ROUNDS=${ROUNDS:-3}
LENS=${LENS:-"8 32 64 96 128"}
TRIALS=${TRIALS:-}
CONTROL=${CONTROL:-1}

REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-/scratch/nidhish/envs/energy/bin/python}
MODEL=$REPO/models/google-bert--bert-base-uncased
RESULTS=${ENERGY_RESULTS_DIR:-$REPO/results}
RUN_ID=$(date +%Y%m%d-%H%M%S)_B1clear_gpu$GPU
OUT=$RESULTS/runs/B1clear/$RUN_ID
mkdir -p "$OUT/scripts" "$OUT/logs"
LOG=$OUT/run.log
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES=$GPU  # store_activations.py does not set it; the others set it to device_index
cd "$REPO"

log() { echo "$(date '+%F %T') [gpu$GPU] $*" | tee -a "$LOG"; }

others_on_gpu() {  # compute processes on our GPU other than pid $1
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$UUID" |
        awk -F', ' -v me="${1:-0}" '$2 != me {print "pid " $2 " (" $3 ")"}' | paste -sd ';'
}

set_status() {  # run.json status: complete / gpu_busy
    "$PY" -c "import json,sys,time; p=sys.argv[1]; r=json.load(open(p)); r['status']=sys.argv[2]; r['finished']=time.strftime('%Y%m%d-%H%M%S'); json.dump(r,open(p,'w'),indent=2)" "$OUT/run.json" "$1"
}

stop_monitor() { [ -n "${MON_PID:-}" ] && kill "$MON_PID" 2>/dev/null; }
trap stop_monitor EXIT

# ---- start checks ----
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
util=$(nvidia-smi -i "$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
busy=$(others_on_gpu)
log "run dir: ${OUT#$REPO/}"
log "check: processes='${busy}' memory_used=${used}MiB util=${util}% load=$(cut -d' ' -f1-3 /proc/loadavg)"
if [ -n "$busy" ] || [ "$used" -gt 500 ] || [ "$util" -gt 5 ]; then
    log "GPU $GPU NOT FULLY FREE; not starting"
    exit 3
fi

nvidia-smi --query-gpu=timestamp,index,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used,pstate \
    --format=csv -lms 1000 > "$OUT/gpu_state.csv" &
MON_PID=$!

# ---- metadata + idle power (our meter, 30 s) ----
GPU=$GPU OUT=$OUT ROUNDS=$ROUNDS LENS=$LENS TRIALS=$TRIALS "$PY" - <<'EOF' 2>&1 | tee -a "$LOG"
import json, os, sys, hashlib, subprocess, socket, time
sys.path.insert(0, os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU"]
from energy import GpuMeter
from energy.record import versions
m = GpuMeter(0)
idle = m.measure_idle(30)
src = {f: hashlib.sha256(open(f"clear_repo/{f}", "rb").read()).hexdigest()[:16]
       for f in ["store_activations.py", "layer_measure.py", "module_measure.py", "full_measure.py"]}
run = {"run_id": os.path.basename(os.environ["OUT"]), "experiment": "B1clear", "host": socket.gethostname(),
       "started": time.strftime("%Y%m%d-%H%M%S"), **m.info(), "idle": idle, "versions": versions(),
       "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
       "rounds": int(os.environ["ROUNDS"]), "seq_lens": [int(x) for x in os.environ["LENS"].split()],
       "trials_override": os.environ["TRIALS"] or None, "clear_scripts_sha256_prefix": src,
       "source": "https://anonymous.4open.science/r/CLEAR-D487 (clear_repo/, see clear_repo/SOURCE.md)",
       "status": "running"}
json.dump(run, open(os.path.join(os.environ["OUT"], "run.json"), "w"), indent=2)
print(f"idle: {idle['mean_W']:.1f} W at {idle['temp_C']} C; {m.name}")
EOF

patch_scripts() {  # $1 = work dir, $2 = seq_len
    local W=$1 L=$2 f
    mkdir -p "$W"
    for f in store_activations layer_measure module_measure full_measure; do cp "clear_repo/$f.py" "$W/$f.py"; done
    sed -i -e '/^print(model)$/d' -e '/^exit(0)$/d' \
           -e "s|^seq_len = 8\$|seq_len = $L  # patched|" \
           -e "s|^output_file = \"activations2.pt\"\$|output_file = \"$W/act.pt\"  # patched|" "$W/store_activations.py"
    sed -i -e "s|^seq_len = 8\$|seq_len = $L  # patched|" \
           -e "s|^output_file = f\"16_full.json\"\$|output_file = \"$W/full.json\"  # patched|" "$W/full_measure.py"
    sed -i -e "s|^activations_file = \"activations2.pt\"\$|activations_file = \"$W/act.pt\"  # patched|" \
           -e "s|^output_file = f\"16_block.json\"\$|output_file = \"$W/block.json\"  # patched|" "$W/layer_measure.py"
    sed -i -e "s|^activations_file = \"activations.pt\" .*|activations_file = \"$W/act.pt\"  # patched|" \
           -e "s|^output_file = f\"32_parts.json\"\$|output_file = \"$W/parts.json\"  # patched|" "$W/module_measure.py"
    for f in store_activations layer_measure module_measure full_measure; do
        sed -i -e "s|^model_name = \"bert-base-uncased\".*|model_name = \"$MODEL\"  # patched|" \
               -e "s|^device_index = 0\$|device_index = $GPU  # patched|" "$W/$f.py"
        [ -n "$TRIALS" ] && sed -i "s|^num_trials = 20\$|num_trials = $TRIALS  # patched (test)|" "$W/$f.py"
        diff -u "clear_repo/$f.py" "$W/$f.py" > "$W/$f.diff"
    done
    # every intended change must have applied: store 3 lines (seq_len, output, model), the others 4 (+ trials)
    local n want
    for f in store_activations layer_measure module_measure full_measure; do
        n=$(grep -c '# patched' "$W/$f.py")
        want=4; [ "$f" = store_activations ] && want=3
        [ -n "$TRIALS" ] && [ "$f" != store_activations ] && want=$((want + 1))
        [ "$n" -eq "$want" ] || { log "PATCH FAILED: $f.py has $n patched lines, want $want"; exit 4; }
    done
    grep -q '^exit(0)$' "$W/store_activations.py" && { log "PATCH FAILED: exit(0) still in store_activations.py"; exit 4; }
}

run_py() {  # $1 = label, $2 = work dir, $3 = script; kill only our script if another process appears
    local label=$1 W=$2 s=$3 pid others code
    others=$(others_on_gpu)
    if [ -n "$others" ]; then log "GPU $GPU NOT FULLY FREE before $label: $others; stopping"; exit 3; fi
    ( cd "$W" && exec "$PY" -u "$s.py" ) > "$W/$s.log" 2>&1 &
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
    if [ "$code" -ne 0 ]; then log "FAILED $label (exit $code):"; tail -5 "$W/$s.log" | tee -a "$LOG"; exit "$code"; fi
}

for r in $(seq 1 "$ROUNDS"); do
    for L in $LENS; do
        W=$OUT/scripts/r$r/L$L
        patch_scripts "$W" "$L"
        t0=$(date +%s)
        for s in store_activations layer_measure module_measure full_measure; do run_py "r$r L$L $s" "$W" "$s"; done
        log "done r$r L=$L in $(( $(date +%s) - t0 )) s: $("$PY" b1/clear_original/analyze.py --one "$W")"
    done
done

if [ "$CONTROL" = 1 ]; then
    others=$(others_on_gpu)
    if [ -n "$others" ]; then log "GPU $GPU NOT FULLY FREE before control: $others; stopping"; exit 3; fi
    log "control: our b1/run_b1.py --seq-lens $LENS ${TRIALS:+--T $TRIALS} (log: logs/control_b1.log)"
    CUDA_VISIBLE_DEVICES=$GPU ENERGY_RESULTS_DIR=$RESULTS "$PY" -u b1/run_b1.py --seq-lens $LENS ${TRIALS:+--T $TRIALS} \
        > "$OUT/logs/control_b1.log" 2>&1
    code=$?
    ctrl=$(grep -m1 '^Run dir:' "$OUT/logs/control_b1.log" | cut -d' ' -f3-)
    log "control exit $code, run dir: ${ctrl#$REPO/}"
    echo "$ctrl" > "$OUT/control_run_dir"
    [ "$code" -eq 0 ] || exit "$code"
fi

"$PY" - "$OUT/run.json" <<'EOF'
import json, sys, time
p = sys.argv[1]; r = json.load(open(p)); r["status"] = "complete"; r["finished"] = time.strftime("%Y%m%d-%H%M%S")
json.dump(r, open(p, "w"), indent=2)
EOF
log "all done: ${OUT#$REPO/}"
log "analyse: python b1/clear_original/analyze.py ${OUT#$REPO/}"
