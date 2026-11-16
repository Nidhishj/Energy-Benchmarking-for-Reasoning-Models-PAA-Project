#!/usr/bin/env bash
# AIME 2024+2025 (60 problems), one trial per GPU, everything for that trial on the same GPU:
#   1. thinking off (max 8,192 tokens) and thinking on (max 32,768 tokens), with E(k) checkpoints every 512 tokens
#   2. fixed thinking caps 2k/4k/8k/16k (+ forced answer at the limit for unfinished thinking) on those traces
# Usage (detached, survives the terminal closing):
#   CUDA_VISIBLE_DEVICES=1 setsid nohup scripts/run_aime.sh 0 > results/logs/aime_t0_gpu1.log 2>&1 < /dev/null &
#   CUDA_VISIBLE_DEVICES=0 setsid nohup scripts/run_aime.sh 1 > results/logs/aime_t1_gpu0.log 2>&1 < /dev/null &
# To continue after a stop, rerun the b2b/caps commands below by hand with --resume <run dir>.
set -euo pipefail
cd "$(dirname "$0")/.."
TRIAL=${1:?trial index}
PY=${PY:-/scratch/nidhish/envs/energy/bin/python}
T=$((TRIAL + 1))

$PY b2/run_b2b.py --subset configs/aime_subset.json --modes off on --T-off $T --T-on $T --trials $TRIAL \
    --max-new-off 8192 --max-new-on 32768 --tag aime

SRC=$($PY - "$TRIAL" <<'EOF'
import glob, json, sys
trial = int(sys.argv[1])
runs = []
for d in glob.glob("results/runs/B2b_aime/*/"):
    r = json.load(open(d + "run.json"))
    if r["config"].get("trials") == [trial] or (trial == 0 and r["config"].get("trials") in (None, [0])):
        runs.append((r["started"], d))
print(sorted(runs)[-1][1])
EOF
)
echo "Caps on source $SRC"
$PY expc/run_caps.py --source "$SRC" --caps 2048 4096 8192 16384 --tag aime
