# CLEAR

A Python package for measuring GPU energy consumption using NVIDIA Management Library (NVML). Designed for researchers and developers who need to measure the energy efficiency of GPU operations, particularly useful for machine learning workload due to the time sensitivity of the operations which GPU power sensors can't directly measure

## Features

- 🔋 Measure energy consumption of arbitrary GPU operations
- 📊 Automatic statistical analysis (mean, variance, std, min, max)
- 🎯 Context manager for easy measurement
- 📁 Save results to JSON for later analysis
- 🔄 Support for multiple trials and repeated measurements
- 🎨 Clean, simple API

## Installation

### From source

```bash
git clone https://github.com/yourusername/gpu_energy_meter.git
cd gpu_energy_meter
pip install -e .
```

### Dependencies

- Python >= 3.8
- PyTorch >= 1.9.0
- nvidia-ml-py3 >= 7.352.0

## Quick Start

### Basic Usage with Context Manager

```python
from gpu_energy_meter import measure_energy
import torch

# Simple measurement
with measure_energy() as measurement:
    # Your GPU code here
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    torch.cuda.synchronize()

print(f"Energy consumed: {measurement.energy_mj} mJ")
```

### Using GPUEnergyMeter Class

```python
from gpu_energy_meter import GPUEnergyMeter
import torch

# Initialize meter
meter = GPUEnergyMeter(device_index=0)

# Measure a code block
with meter.measure() as m:
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    torch.cuda.synchronize()

print(f"Energy: {m.energy_mj} mJ")

# Don't forget to shutdown
meter.shutdown()
```

### Measuring Functions with Multiple Trials

```python
from gpu_energy_meter import GPUEnergyMeter
import torch

def my_gpu_function():
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    return y

# Measure with statistics
with GPUEnergyMeter() as meter:
    results = meter.measure_function(
        my_gpu_function,
        num_trials=10,        # Run 10 trials
        repeat_count=100,     # Repeat 100 times per trial
        warmup=True           # Run warmup iteration
    )
    
    print(f"Mean energy: {results['mean_mJ']:.2f} mJ")
    print(f"Std dev: {results['std_mJ']:.2f} mJ")
    print(f"Min: {results['min_mJ']:.2f} mJ")
    print(f"Max: {results['max_mJ']:.2f} mJ")
```

### Saving Results to File

```python
from gpu_energy_meter import GPUEnergyMeter
from gpu_energy_meter.utils import save_energy_results

def model_forward():
    # Your model code
    pass

with GPUEnergyMeter() as meter:
    results = meter.measure_function(
        model_forward,
        num_trials=100,
        repeat_count=50
    )

# Save to JSON
from gpu_energy_meter import EnergyStats
stats = EnergyStats(results['all_trials_mJ'])
stats.save("results.json", "Model Forward Pass")
```

## Real-World Example: Measuring LLM Components

Here's how to use it with your original code for measuring transformer components:

```python
from gpu_energy_meter import GPUEnergyMeter
import torch
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.float16
).cuda()

# Load activations
data = torch.load("activations.pt")
layer_idx = int(data["layer_idx"])
mlp_inputs = data["mlp_in"]
mlp_batch_input = torch.cat(mlp_inputs, dim=0).unsqueeze(0).cuda()

# Measure MLP energy
mlp = model.model.layers[layer_idx].mlp

def mlp_forward():
    with torch.no_grad():
        return mlp(mlp_batch_input)

with GPUEnergyMeter() as meter:
    results = meter.measure_function(
        mlp_forward,
        num_trials=100,
        repeat_count=100,
        warmup=True
    )
    
    # Save results
    from gpu_energy_meter import EnergyStats
    stats = EnergyStats(results['all_trials_mJ'])
    stats.save("mlp_energy.json", "MLP Layer")
```

### Measuring Full Transformer Block

```python
def block_forward():
    block = model.model.layers[layer_idx]
    B, S, _ = attn_batch_input.shape
    position_ids = torch.arange(S, device="cuda").unsqueeze(0).expand(B, -1)
    cos, sin = model.model.rotary_emb(attn_batch_input, position_ids)
    
    with torch.no_grad():
        return block(
            hidden_states=attn_batch_input,
            attention_mask=None,
            position_embeddings=(cos, sin),
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )

with GPUEnergyMeter() as meter:
    results = meter.measure_function(
        block_forward,
        num_trials=100,
        repeat_count=100
    )
    
    stats = EnergyStats(results['all_trials_mJ'])
    stats.save("block_energy.json", "Transformer Block")
```

## API Reference

### GPUEnergyMeter

Main class for GPU energy measurements.

#### Methods

- `__init__(device_index=0, set_visible_devices=True)`: Initialize meter
- `measure()`: Context manager for measuring a code block
- `measure_function(func, *args, num_trials=1, repeat_count=1, warmup=True, **kwargs)`: Measure function with statistics
- `get_energy_mj()`: Get current total energy consumption
- `shutdown()`: Shutdown NVML

### measure_energy(device_index=0)

Simple context manager for quick measurements.

### EnergyStats

Class for storing and analyzing energy statistics.

#### Methods

- `to_dict()`: Convert to dictionary
- `save(filepath, name)`: Save to JSON file
- `__repr__()`: String representation

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "measurement_name": "MLP Layer",
  "all_trials_mJ": [100.5, 102.3, 101.8, ...],
  "mean_mJ": 101.2,
  "variance_mJ2": 1.5,
  "std_mJ": 1.22,
  "min_mJ": 99.8,
  "max_mJ": 103.1,
  "num_trials": 100
}
```

## Requirements

- NVIDIA GPU with NVML support
- CUDA installed
- Appropriate NVIDIA drivers

## Tips for Accurate Measurements

1. **Warm up the GPU** before taking measurements to avoid cold start effects
2. **Use multiple trials** (e.g., 100) to get stable statistics
3. **Repeat operations** within each trial for better accuracy
4. **Minimize background GPU usage** during measurements
5. **Use torch.cuda.synchronize()** to ensure operations complete before measuring
6. **Consider adding small sleep intervals** between trials if needed

## Common Issues

### NVML Error

If you get NVML errors, ensure:
- NVIDIA drivers are properly installed
- You have permissions to access GPU monitoring
- The device index is correct

### Inaccurate Measurements

If measurements seem off:
- Add warmup iterations
- Increase the number of trials
- Ensure GPU is not being used by other processes
- Use `torch.cuda.synchronize()` after operations



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
