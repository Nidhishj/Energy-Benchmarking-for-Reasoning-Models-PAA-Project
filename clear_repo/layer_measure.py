import os
import time
import json
import torch
from transformers import AutoModelForMaskedLM
from pynvml import *
from tqdm import tqdm

# ------------------------------
# Settings
# ------------------------------
model_name = "bert-base-uncased"
cache_dir = "./.hfCache"
activations_file = "activations2.pt"
repeat_count = 10000
num_trials = 20
device_index = 0
output_file = f"16_block.json"

# ------------------------------
# NVML Init
# ------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(device_index)

def get_gpu_energy_mj():
    try:
        return nvmlDeviceGetTotalEnergyConsumption(handle)  # millijoules since boot
    except Exception as e:
        print(f"NVML Error: {e}")
        return 0

# ------------------------------
# Load activations
# ------------------------------
data = torch.load(activations_file)
layer_idx = int(data["layer_idx"])
attn_in = torch.cat(data["attn_in"], dim=0).cuda()
if attn_in.dim() == 2:
    attn_in = attn_in.unsqueeze(0)

# ------------------------------
# Load model
# ------------------------------
model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype = torch.float16).cuda()
model.eval()
dtype = model.dtype

# Select target layer
chosen_layer = model.bert.encoder.layer[layer_idx]

# ------------------------------
# Measure complete layer
# ------------------------------
def measure_layer_energy(layer, inputs, repeat_count, num_trials, label):
    energy_usages = []
    for trial in tqdm(range(num_trials), desc=label):
        with torch.no_grad():
            time.sleep(0.5)
            start_energy = get_gpu_energy_mj()
            for _ in range(repeat_count):
                _ = layer(inputs)
            end_energy = get_gpu_energy_mj()
        energy_used_per_call_mJ = (end_energy - start_energy) / repeat_count
        energy_usages.append(energy_used_per_call_mJ)
    vals = torch.tensor(energy_usages, dtype=torch.float64)
    return {
        "all_trials": [round(x, 6) for x in energy_usages],
        "average_mJ": round(vals.mean().item(), 6),
        "variance_mJ2": round(vals.var(unbiased=False).item(), 6),
        "std_dev_mJ": round(vals.std(unbiased=False).item(), 6),
    }

results = {
    f"layer_{layer_idx}": measure_layer_energy(
        chosen_layer,
        inputs=attn_in,
        repeat_count=repeat_count,
        num_trials=num_trials,
        label=f"Layer {layer_idx}"
    )
}

# ------------------------------
# Save results
# ------------------------------
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nLayer energy results saved to {output_file}")

# Shutdown NVML
nvmlShutdown()
