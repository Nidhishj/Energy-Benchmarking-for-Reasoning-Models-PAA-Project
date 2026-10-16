# GPU Energy Meter - Complete Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Basic Usage](#basic-usage)
4. [Advanced Usage](#advanced-usage)
5. [Replacing Your Original Code](#replacing-your-original-code)
6. [API Documentation](#api-documentation)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Installation

### Option 1: Install from source (Recommended)

```bash
# Clone or download the package
cd gpu_energy_meter

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

### Option 2: Direct installation of dependencies

```bash
pip install torch nvidia-ml-py3 transformers
```

Then copy the `gpu_energy_meter` folder to your project.

## Quick Start

### 1. Simple Single Measurement

```python
from gpu_energy_meter import measure_energy
import torch

with measure_energy() as m:
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    torch.cuda.synchronize()

print(f"Energy: {m.energy_mj} mJ")
```

### 2. Function with Statistics

```python
from gpu_energy_meter import GPUEnergyMeter

def my_operation():
    x = torch.randn(1000, 1000).cuda()
    return torch.matmul(x, x)

with GPUEnergyMeter() as meter:
    results = meter.measure_function(
        my_operation,
        num_trials=10,
        repeat_count=100
    )
    print(f"Mean: {results['mean_mJ']:.2f} mJ")
```

## Basic Usage

### Using Context Manager

The simplest way to measure energy:

```python
from gpu_energy_meter import GPUEnergyMeter

meter = GPUEnergyMeter(device_index=0)

with meter.measure() as m:
    # Your GPU code
    model(inputs)

print(f"Energy used: {m.energy_mj} mJ")
meter.shutdown()
```

### Using as Context Manager (Auto-cleanup)

```python
with GPUEnergyMeter() as meter:
    with meter.measure() as m:
        model(inputs)
    print(f"Energy: {m.energy_mj} mJ")
# Automatically shuts down NVML
```

## Advanced Usage

### Multiple Trials with Statistics

```python
from gpu_energy_meter import GPUEnergyMeter, EnergyStats

def forward_pass():
    with torch.no_grad():
        return model(inputs)

with GPUEnergyMeter() as meter:
    # Run 100 trials, each repeating 50 times
    results = meter.measure_function(
        forward_pass,
        num_trials=100,
        repeat_count=50,
        warmup=True,  # Run warmup iteration first
        sleep_between_trials=0.1  # Optional: sleep between trials
    )
    
    # Results include:
    # - all_trials_mJ: list of all measurements
    # - mean_mJ: average energy
    # - std_mJ: standard deviation
    # - variance_mJ2: variance
    # - min_mJ, max_mJ: min/max values
    # - num_trials: number of trials

    # Save to file
    stats = EnergyStats(results['all_trials_mJ'])
    stats.save("results.json", "My Measurement")
```

### Manual Loop for Fine Control

```python
with GPUEnergyMeter() as meter:
    energies = []
    
    for i in range(100):
        # Optional: custom logic between trials
        if i % 10 == 0:
            print(f"Progress: {i}/100")
        
        with meter.measure() as m:
            # Your code
            for _ in range(50):
                output = model(inputs)
            torch.cuda.synchronize()
        
        energy_per_iter = m.energy_mj / 50
        energies.append(energy_per_iter)
    
    # Calculate your own statistics
    stats = EnergyStats(energies)
    print(stats)
```

## Replacing Your Original Code

### Original haecer.py Pattern

Your original code looked like:

```python
# Original
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(DEVICE_INDEX)

def get_gpu_energy_mj():
    return nvmlDeviceGetTotalEnergyConsumption(handle)

energy_usages = []
for trial in range(NUM_TRIALS):
    start_energy = get_gpu_energy_mj()
    for _ in range(REPEAT_COUNT):
        _ = mlp(mlp_batch_input)
    end_energy = get_gpu_energy_mj()
    energy_usages.append((end_energy - start_energy) / REPEAT_COUNT)

# Calculate stats
vals = torch.tensor(energy_usages, dtype=torch.float64)
stats = {
    "mean_mJ": float(vals.mean().item()),
    # ... etc
}
```

### New Pattern with Package

Replace with:

```python
from gpu_energy_meter import GPUEnergyMeter, EnergyStats

def mlp_forward():
    with torch.no_grad():
        return mlp(mlp_batch_input)

with GPUEnergyMeter(device_index=DEVICE_INDEX) as meter:
    results = meter.measure_function(
        mlp_forward,
        num_trials=NUM_TRIALS,
        repeat_count=REPEAT_COUNT,
        warmup=True
    )
    
    # Save results
    stats = EnergyStats(results['all_trials_mJ'])
    stats.save("mlp_results.json", "MLP Layer")
```

### Complete Conversion Example

See `examples/example_usage.py` for a complete conversion of your haecer.py code.

Key changes:
1. Replace NVML initialization with `GPUEnergyMeter()`
2. Replace manual measurement loops with `measure_function()`
3. Replace manual statistics with `EnergyStats`
4. Automatic cleanup (no need to call `nvmlShutdown()`)

## API Documentation

### GPUEnergyMeter

```python
class GPUEnergyMeter:
    def __init__(self, device_index=0, set_visible_devices=True)
    def measure() -> ContextManager[Measurement]
    def measure_function(func, *args, num_trials=1, repeat_count=1, 
                        warmup=True, sleep_between_trials=0.0, **kwargs) -> dict
    def get_energy_mj() -> float
    def shutdown()
```

**Parameters:**
- `device_index`: GPU device index (default: 0)
- `set_visible_devices`: Whether to set CUDA_VISIBLE_DEVICES (default: True)

### measure_function() Parameters

- `func`: Function to measure (should not take any arguments that change between calls)
- `num_trials`: Number of independent measurement trials (default: 1)
- `repeat_count`: How many times to repeat the function per trial (default: 1)
- `warmup`: Run one warmup iteration before measuring (default: True)
- `sleep_between_trials`: Seconds to sleep between trials (default: 0.0)

**Returns:** Dictionary with statistics

### EnergyStats

```python
class EnergyStats:
    def __init__(self, measurements: List[float])
    def to_dict() -> dict
    def save(filepath: str, name: str)
```

## Best Practices

### 1. Always Use Warmup

```python
# GOOD: Warmup enabled (default)
results = meter.measure_function(
    my_func,
    num_trials=100,
    warmup=True  # GPU is warmed up first
)

# AVOID: No warmup
results = meter.measure_function(
    my_func,
    num_trials=100,
    warmup=False  # First trial may be cold start
)
```

### 2. Use Sufficient Trials

```python
# GOOD: Enough trials for stable statistics
results = meter.measure_function(my_func, num_trials=100)

# AVOID: Too few trials
results = meter.measure_function(my_func, num_trials=3)
```

### 3. Synchronize CUDA Operations

```python
def my_gpu_operation():
    output = model(inputs)
    torch.cuda.synchronize()  # IMPORTANT: Wait for GPU to finish
    return output
```

### 4. Minimize Background GPU Usage

```bash
# Check what's using your GPU
nvidia-smi

# Kill unnecessary processes before measurement
```

### 5. Use torch.no_grad() for Inference

```python
def inference():
    with torch.no_grad():  # Don't compute gradients
        return model(inputs)
```

## Troubleshooting

### Error: "Failed to initialize NVML"

**Solution:**
- Check NVIDIA drivers: `nvidia-smi`
- Ensure you have permissions
- Try with sudo if needed

### Measurements seem unstable

**Solution:**
- Increase `num_trials` (e.g., 100)
- Use `warmup=True`
- Close other GPU applications
- Add `sleep_between_trials=0.1`

### GPU not found

**Solution:**
```python
# Check available devices
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Specify correct device
meter = GPUEnergyMeter(device_index=0)  # or 1, 2, etc.
```

### Function takes arguments

If your function needs arguments that should remain constant:

```python
# GOOD: Use a closure
inputs = prepare_inputs()
def forward():
    return model(inputs)

results = meter.measure_function(forward)

# OR: Use lambda
results = meter.measure_function(
    lambda: model(inputs)
)
```

### Very small energy values

If measurements are too small (< 1 mJ):

```python
# Increase repeat_count to get measurable values
results = meter.measure_function(
    my_func,
    repeat_count=1000  # Repeat more times per trial
)
```

## Comparison with Original Code

### Lines of Code

**Original:**
```python
# ~50 lines of measurement code
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
energy_usages = []
for trial in range(100):
    start = get_gpu_energy_mj()
    for _ in range(REPEAT_COUNT):
        _ = mlp(input)
    end = get_gpu_energy_mj()
    energy_usages.append((end - start) / REPEAT_COUNT)
vals = torch.tensor(energy_usages)
stats = {
    "mean_mJ": float(vals.mean().item()),
    # ... more stats
}
with open("results.json", "w") as f:
    json.dump(stats, f)
nvmlShutdown()
```

**With Package:**
```python
# ~8 lines
from gpu_energy_meter import GPUEnergyMeter, EnergyStats

def mlp_forward():
    return mlp(input)

with GPUEnergyMeter() as meter:
    results = meter.measure_function(mlp_forward, num_trials=100, repeat_count=REPEAT_COUNT)
    stats = EnergyStats(results['all_trials_mJ'])
    stats.save("results.json", "MLP")
```

### Benefits

1. **Less code**: 85% reduction in measurement code
2. **Automatic cleanup**: No need to remember `nvmlShutdown()`
3. **Better error handling**: Built-in error messages
4. **Reusable**: Use across multiple projects
5. **Well-tested**: Focused, tested codebase
6. **Documented**: Clear API and examples

## Examples Directory

Run the included examples:

```bash
# Quick start with 6 simple examples
python examples/quick_start.py

# Full example matching your original code
python examples/example_usage.py
```

## Support

For issues or questions:
1. Check this guide
2. Review the examples
3. Check the main README.md
4. Open an issue on GitHub

## Version History

- v0.1.0 (2024): Initial release
  - Basic energy measurement
  - Statistical analysis
  - JSON export
  - Context managers
  - Multiple trial support
