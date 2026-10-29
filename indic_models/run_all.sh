#!/usr/bin/env bash
# Laya, both methods, one GPU, one after another. Stops at the first failure (e.g. another user's job on the GPU).
#   GPU=1 CHECKPOINTS="english multilingual" nohup bash indic_models/run_all.sh > results/logs/laya_gpu1.log 2>&1 &
set -u
GPU=${GPU:-1}
CHECKPOINTS=${CHECKPOINTS:-"english multilingual"}
METHODS=${METHODS:-"ours clear"}
COMPONENTS="embedding L5.attn_norm L5.attn L5.mlp_norm L5.mlp L5.block L6.attn_norm L6.attn L6.mlp_norm L6.mlp L6.block final_norm head0 head1 decision_head encoder_cycled full_model"
cd "$(dirname "$0")/.."
for ck in $CHECKPOINTS; do
  for m in $METHODS; do
    echo "$(date '+%F %T') start $ck $m on GPU $GPU; GPU state: $(nvidia-smi -i $GPU --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
    CUDA_VISIBLE_DEVICES=$GPU /scratch/nidhish/envs/energy/bin/python -u indic_models/run_laya.py \
        --checkpoint "$ck" --method "$m" --components $COMPONENTS || { echo "$(date '+%F %T') STOPPED at $ck $m (exit $?)"; exit 1; }
  done
done
echo "$(date '+%F %T') all done"
