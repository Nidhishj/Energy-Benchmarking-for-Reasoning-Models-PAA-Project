#!/usr/bin/env bash
# Download models into the repo's models/ folder (git-ignored). Run in a second terminal while setup or B1 runs:
#   bash scripts/slot_download.sh
# Uses curl with resume (scripts/hf_curl.sh), so an interrupted download continues where it stopped.
# Bandwidth on the slot server is ~5 MB/s in total: Qwen3-8B (16 GB) takes ~55 min, and starves pip installs
# running at the same time, so start it after setup has finished installing torch.
set -euo pipefail
BASE=${BASE:-/data/cs3406/team02}
REPO=$BASE/Energy-Benchmarking-for-Reasoning-Models-PAA-Project
H=$REPO/scripts/hf_curl.sh
cd "$REPO/models"
# Smallest first, so B1 and B2a can start while Qwen3-8B is still downloading.
$H google-bert/bert-base-uncased google-bert--bert-base-uncased
$H textattack/bert-base-uncased-MRPC textattack--bert-base-uncased-MRPC
$H Qwen/Qwen3-8B Qwen--Qwen3-8B
echo "All downloads done."
