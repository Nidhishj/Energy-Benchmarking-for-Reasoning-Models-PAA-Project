#!/usr/bin/env python3
"""
Example script showing how to use gpu_energy_meter package
to replace the measurement code in haecer.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gpu_energy_meter import GPUEnergyMeter, EnergyStats

# ----------------------------------------------------------------------
# Configuration (same as original)
# ----------------------------------------------------------------------
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CACHE_DIR = "../.hfCache"
ACTIVATIONS_FILE = "activations2.pt"
REPEAT_COUNT = 100
NUM_TRIALS_MLP = 100
NUM_TRIALS_BLOCK = 100
DEVICE_INDEX = 0

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    cache_dir=CACHE_DIR, 
    torch_dtype=torch.float16
).cuda()
model.eval()

print("Loading activations...")
data = torch.load(ACTIVATIONS_FILE)
layer_idx = int(data["layer_idx"])
mlp_inputs = data["mlp_in"]

# ----------------------------------------------------------------------
# 1️⃣ MLP Energy Measurement (using the package)
# ----------------------------------------------------------------------
def measure_mlp_energy():
    """Measure MLP layer energy using gpu_energy_meter package."""
    print("\n" + "="*60)
    print("Measuring MLP Layer Energy")
    print("="*60)
    
    # Prepare input
    mlp_batch_input = torch.cat(mlp_inputs, dim=0).unsqueeze(0).cuda()
    mlp = model.model.layers[layer_idx].mlp
    
    # Define the function to measure
    def mlp_forward():
        with torch.no_grad():
            return mlp(mlp_batch_input)
    
    # Use the package to measure
    with GPUEnergyMeter(device_index=DEVICE_INDEX) as meter:
        results = meter.measure_function(
            mlp_forward,
            num_trials=NUM_TRIALS_MLP,
            repeat_count=REPEAT_COUNT,
            warmup=True,
            sleep_between_trials=0.0
        )
        
        # Save results
        stats = EnergyStats(results['all_trials_mJ'])
        stats.save("mlp_energy_results.json", "MLP Layer")
    
    return results

# ----------------------------------------------------------------------
# 2️⃣ Transformer Block Energy Measurement (using the package)
# ----------------------------------------------------------------------
def measure_block_energy():
    """Measure full transformer block energy using gpu_energy_meter package."""
    print("\n" + "="*60)
    print("Measuring Transformer Block Energy")
    print("="*60)
    
    # Prepare input
    attn_batch_input = torch.cat(mlp_inputs, dim=0)
    if attn_batch_input.dim() == 2:
        attn_batch_input = attn_batch_input.unsqueeze(0)
    
    dtype = model.dtype
    block = model.model.layers[layer_idx]
    attn_batch_input = attn_batch_input.to(device="cuda", dtype=dtype)
    
    B, S, _ = attn_batch_input.shape
    position_ids = torch.arange(S, device=attn_batch_input.device).unsqueeze(0).expand(B, -1)
    cos, sin = model.model.rotary_emb(attn_batch_input, position_ids)
    position_embeddings = (cos, sin)
    
    # Define the function to measure
    def block_forward():
        with torch.no_grad():
            return block(
                hidden_states=attn_batch_input,
                attention_mask=None,
                position_embeddings=position_embeddings,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
    
    # Use the package to measure
    with GPUEnergyMeter(device_index=DEVICE_INDEX) as meter:
        results = meter.measure_function(
            block_forward,
            num_trials=NUM_TRIALS_BLOCK,
            repeat_count=REPEAT_COUNT,
            warmup=True,
            sleep_between_trials=0.0
        )
        
        # Save results
        stats = EnergyStats(results['all_trials_mJ'])
        stats.save("block_energy_results.json", "Transformer Block")
    
    return results

# ----------------------------------------------------------------------
# 3️⃣ Alternative: Manual measurement using context manager
# ----------------------------------------------------------------------
def measure_with_context_manager():
    """Example showing manual measurement using context manager."""
    print("\n" + "="*60)
    print("Example: Manual Measurement with Context Manager")
    print("="*60)
    
    mlp_batch_input = torch.cat(mlp_inputs, dim=0).unsqueeze(0).cuda()
    mlp = model.model.layers[layer_idx].mlp
    
    with GPUEnergyMeter(device_index=DEVICE_INDEX) as meter:
        energies = []
        
        for trial in range(10):  # Just 10 trials for demo
            with meter.measure() as m:
                with torch.no_grad():
                    for _ in range(REPEAT_COUNT):
                        _ = mlp(mlp_batch_input)
                torch.cuda.synchronize()
            
            energy_per_iter = m.energy_mj / REPEAT_COUNT
            energies.append(energy_per_iter)
            print(f"Trial {trial+1}: {energy_per_iter:.4f} mJ")
        
        # Calculate stats
        stats = EnergyStats(energies)
        print(f"\nResults: {stats}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Run all measurements
        mlp_results = measure_mlp_energy()
        block_results = measure_block_energy()
        
        # Optional: show manual approach
        # measure_with_context_manager()
        
        print("\n" + "="*60)
        print("All measurements complete!")
        print("="*60)
        print(f"\nMLP Mean Energy: {mlp_results['mean_mJ']:.2f} mJ")
        print(f"Block Mean Energy: {block_results['mean_mJ']:.2f} mJ")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
