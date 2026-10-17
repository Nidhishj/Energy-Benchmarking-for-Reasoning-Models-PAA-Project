#!/usr/bin/env bash
# One-time setup on the slot server (RTX PRO 5000). Run from anywhere:  bash scripts/slot_setup.sh
# Idempotent: rerunning skips what is already done. Takes ~5-10 min, mostly the torch download.
set -euo pipefail
BASE=${BASE:-/data/cs3406/team02}
REPO=$BASE/Energy-Benchmarking-for-Reasoning-Models-PAA-Project
ENV=$BASE/envs/energy
export HF_HOME=$BASE/hf_cache

echo "== GPU and disk"
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv
df -h "$BASE" | tail -1
echo "== Internet"
curl -sS -o /dev/null -w "huggingface.co HTTP %{http_code}\n" --max-time 10 https://huggingface.co || echo "NO INTERNET: copy models in with rsync (see slot_plan.md)"

echo "== Repo"
if [ -d "$REPO/.git" ]; then git -C "$REPO" pull --ff-only; else git clone https://github.com/Nidhishj/Energy-Benchmarking-for-Reasoning-Models-PAA-Project "$REPO"; fi

echo "== Python 3.11 env (uv)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -x "$ENV/bin/python" ] || uv venv --python 3.11 "$ENV"
# unsafe-best-match: the torch index also hosts tqdm/numpy etc. at other versions; take pins from PyPI too.
VIRTUAL_ENV=$ENV uv pip install --index-strategy unsafe-best-match -r "$REPO/requirements-cu128.txt"

echo "== Checks"
cd "$REPO"
"$ENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("arch list", torch.cuda.get_arch_list())
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); print("bf16 matmul ok", float((x @ x).float().abs().mean()) > 0)
from energy import GpuMeter
m = GpuMeter(0); e0, _ = m.read(); import time; time.sleep(1); e1, _ = m.read()
print("NVML energy counter ok:", m.name, f"{e1 - e0} mJ in 1 s")
from zeus.monitor import ZeusMonitor; print("zeus import ok")
PY
echo "Setup done. Activate with: source $ENV/bin/activate && export HF_HOME=$HF_HOME && cd $REPO"
