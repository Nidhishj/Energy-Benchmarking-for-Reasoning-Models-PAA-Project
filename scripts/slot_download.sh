#!/usr/bin/env bash
# Download models and datasets into the repo's models/ folder (git-ignored). Run in a second terminal while
# setup or B1 runs:  bash scripts/slot_download.sh
# Downloads use the network and disk, not the GPU, so they don't disturb energy measurements.
set -euo pipefail
BASE=${BASE:-/data/cs3406/team02}
REPO=$BASE/Energy-Benchmarking-for-Reasoning-Models-PAA-Project
export HF_HOME=$BASE/hf_cache
HF=$BASE/envs/energy/bin/hf
cd "$REPO"
# Smallest first, so B1 and B2a can start while Qwen3-8B (~16 GB) is still downloading.
$HF download google-bert/bert-base-uncased --local-dir models/google-bert--bert-base-uncased
$HF download textattack/bert-base-uncased-MRPC --local-dir models/textattack--bert-base-uncased-MRPC
$BASE/envs/energy/bin/python -c "from datasets import load_dataset; load_dataset('nyu-mll/glue', 'mrpc', split='validation'); print('MRPC cached')"
$HF download Qwen/Qwen3-8B --local-dir models/Qwen--Qwen3-8B
echo "All downloads done."
