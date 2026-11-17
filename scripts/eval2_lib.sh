# Shared helpers for scripts/run_eval2_*.sh (source it, don't run it).
# Expects: PY (python of the energy env), STOP (HH:MM, when GPU work must end), GPU_MATCH (part of the GPU name
# recorded in results/sessions/, e.g. "PRO 5000"), LOGDIR. Optional: SINCE (YYYYMMDD; older runs are ignored).

SINCE=${SINCE:-20261117}
SKIPPED=()
FAILED=()

# Minutes left until STOP. A STOP more than 12 h in the past means tomorrow (sessions that cross midnight).
minutes_left() {
    "$PY" - "$STOP" <<'EOF'
import datetime as dt, sys
now = dt.datetime.now()
h, m = map(int, sys.argv[1].split(":"))
end = now.replace(hour=h, minute=m, second=0, microsecond=0)
if (now - end).total_seconds() > 12 * 3600:
    end += dt.timedelta(days=1)
print(int((end - now).total_seconds() // 60))
EOF
}

# Latest run of an experiment dir on this GPU since SINCE: prints "<status> <run dir>", or nothing.
find_run() {
    "$PY" - "$1" "$GPU_MATCH" "$SINCE" <<'EOF'
import glob, json, os, sys
exp, gpu, since = sys.argv[1:4]
best = None
for f in glob.glob(f"results/runs/{exp}/*/run.json"):
    r = json.load(open(f))
    if r.get("started", "") < since:
        continue
    s = f"results/sessions/{r['session_id']}.json"
    if not os.path.exists(s) or gpu not in json.load(open(s))["gpu_name"]:
        continue
    if best is None or r["started"] > best[0]:
        best = (r["started"], r["status"], os.path.dirname(f))
if best:
    print(best[1], best[2])
EOF
}

# Estimated minutes for one Exp A run: ~8.5 min per context with all 9 components at T = 20 and ~2.5 s per trial
# (the Ada run took 53 min for 7 contexts), scaled by the number of components, plus ~3 min for load and idle.
expa_minutes() {  # <n contexts> <n components>
    echo $(( $1 * 85 * $2 / 90 + 3 ))
}

# run_step <experiment dir> <est. minutes> <command...>
# Skips a step already complete on this GPU, resumes one that stopped, and skips (without failing) a step that would
# not finish before STOP. A failed step is recorded and the queue goes on.
run_step() {
    local exp=$1 est=$2
    shift 2
    local found left
    found=$(find_run "$exp")
    if [[ $found == complete* ]]; then
        echo "[skip] $exp: already complete in ${found#* }"
        return 0
    fi
    left=$(minutes_left)
    if (( est > left )); then
        echo "[time] $exp needs ~$est min, only $left min left before $STOP: skipped"
        SKIPPED+=("$exp")
        return 0
    fi
    local resume=()
    if [[ -n $found ]]; then
        resume=(--resume "${found#* }")
        echo "[run ] $exp: resuming ${found#* } (status ${found%% *})"
    fi
    echo "[run ] $exp (~$est min, $left min left) started $(date +%H:%M)"
    "$@" ${resume[@]+"${resume[@]}"} 2>&1 | tee -a "$LOGDIR/$exp.log"
    local rc=${PIPESTATUS[0]}
    if (( rc != 0 )); then
        echo "[FAIL] $exp (exit $rc): see $LOGDIR/$exp.log"
        FAILED+=("$exp")
    fi
}

# Run dir of the latest complete run of an experiment on this GPU (for the analysis step), or nothing.
complete_run() {
    local found
    found=$(find_run "$1")
    [[ $found == complete* ]] && echo "${found#* }"
    return 0
}

summary() {
    echo
    echo "== Summary $(date +%H:%M) =="
    echo "Skipped for time: ${SKIPPED[*]:-none}"
    echo "Failed:           ${FAILED[*]:-none}"
    echo "Logs:             $LOGDIR"
    echo "Rerun the same command to continue: finished steps are skipped, stopped ones resume."
}
