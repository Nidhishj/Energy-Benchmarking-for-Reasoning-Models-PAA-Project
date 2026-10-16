# GPU Energy Meter Package


A complete, production-ready Python package for measuring GPU energy consumption. This package wraps all the NVML (NVIDIA Management Library) measurement code .

## 📂 Package Structure

```
gpu_energy_meter_package/
│
├── gpu_energy_meter/              # Core package directory
│   ├── __init__.py               # Package initialization and exports
│   ├── meter.py                  # Main GPUEnergyMeter class
│   └── utils.py                  # Statistics and utility functions
│
├── examples/                      # Usage examples
│   ├── quick_start.py            # 6 simple examples for beginners
│   └── example_usage.py          # Your haecer.py converted to use package
│
├── setup.py                       # Package installation configuration
├── requirements.txt              # Python dependencies
├── MANIFEST.in                   # Files to include in distribution
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
│
├── README.md                     # Main documentation (detailed)
├── USAGE_GUIDE.md               # Complete usage guide (comprehensive)
└── QUICKSTART.md                # Get started in 5 minutes
```

## 🎯 Key Features

### 1. **Simple API**
Three main components:
- `GPUEnergyMeter` - Main measurement class
- `measure_energy()` - Quick context manager for single measurements
- `EnergyStats` - Statistical analysis helper

### 2. **Automatic Management**
- NVML initialization/shutdown handled automatically
- Context managers for safe resource cleanup
- Built-in error handling

### 3. **Statistical Analysis**
Automatic calculation of:
- Mean energy consumption
- Standard deviation
- Variance  
- Min/Max values
- All individual measurements

### 4. **Easy Data Export**
- Save results to JSON
- Load and analyze previous results
- Formatted output for reports

### 5. **Flexible Measurement Options**
- Single measurements
- Multiple trials with statistics
- Repeated operations
- Custom warmup
- Sleep between trials

## 🚀 How to Use

### Installation

```bash
cd gpu_energy_meter_package
pip install -e .
```

### Basic Usage

```python
from gpu_energy_meter import measure_energy
import torch

# Quick measurement
with measure_energy() as m:
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    torch.cuda.synchronize()

print(f"Energy: {m.energy_mj} mJ")
```

### Advanced Usage with Statistics

```python
from gpu_energy_meter import GPUEnergyMeter, EnergyStats

def my_operation():
    x = torch.randn(1000, 1000).cuda()
    return torch.matmul(x, x)

with GPUEnergyMeter() as meter:
    # Run 100 trials, each repeating 50 times
    results = meter.measure_function(
        my_operation,
        num_trials=100,
        repeat_count=50,
        warmup=True
    )
    
    print(f"Mean: {results['mean_mJ']:.2f} mJ")
    print(f"Std:  {results['std_mJ']:.2f} mJ")
    
    # Save to file
    stats = EnergyStats(results['all_trials_mJ'])
    stats.save("results.json", "Matrix Multiplication")
```

## 🔄 Converting Your Original Code

### Before (haecer.py pattern):

```python
import os
from pynvml import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

def get_gpu_energy_mj():
    return nvmlDeviceGetTotalEnergyConsumption(handle)

# Measure MLP
energy_usages = []
for trial in range(100):
    start_energy = get_gpu_energy_mj()
    for _ in range(100):
        _ = mlp(mlp_batch_input)
    end_energy = get_gpu_energy_mj()
    energy_usages.append((end_energy - start_energy) / 100)

# Calculate statistics
vals = torch.tensor(energy_usages, dtype=torch.float64)
stats = {
    "mean_mJ": float(vals.mean().item()),
    "variance_mJ2": float(vals.var(unbiased=False).item()),
    "std_mJ": float(vals.std(unbiased=False).item()),
}

# Save results
with open("results.json", "w") as f:
    json.dump(stats, f, indent=2)

nvmlShutdown()
```

### After (with this package):

```python
from gpu_energy_meter import GPUEnergyMeter, EnergyStats

def mlp_forward():
    with torch.no_grad():
        return mlp(mlp_batch_input)

with GPUEnergyMeter(device_index=0) as meter:
    results = meter.measure_function(
        mlp_forward,
        num_trials=100,
        repeat_count=100,
        warmup=True
    )
    
    stats = EnergyStats(results['all_trials_mJ'])
    stats.save("results.json", "MLP Layer")
```

## 📊 Output Format

Results are saved as JSON:

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

## 📚 Documentation Files

### For Different Needs:

1. **QUICKSTART.md** - Start here! Get running in 5 minutes
   - Installation
   - First measurement
   - Basic conversion example

2. **README.md** - Main documentation
   - Features overview
   - Installation methods
   - API reference
   - Multiple examples
   - Tips for accuracy

3. **USAGE_GUIDE.md** - Comprehensive guide
   - Detailed API documentation
   - Advanced usage patterns
   - Best practices
   - Troubleshooting
   - Complete conversion guide

4. **examples/** - Working code
   - `quick_start.py` - 6 simple examples
   - `example_usage.py` - Your haecer.py fully converted

## 🎓 Learning Path

1. **Complete Beginners:**
   - Read `QUICKSTART.md`
   - Run `examples/quick_start.py`
   - Try modifying the examples

2. **Converting Existing Code:**
   - Look at `examples/example_usage.py`
   - Read the "Converting Your Original Code" section in `USAGE_GUIDE.md`
   - Replace your measurement code step by step

3. **Advanced Users:**
   - Read `USAGE_GUIDE.md` for all options
   - Check API reference in `README.md`
   - Customize for your specific needs

## 🔧 Main API Components

### 1. GPUEnergyMeter Class

```python
meter = GPUEnergyMeter(device_index=0, set_visible_devices=True)
```

**Methods:**
- `measure()` - Context manager for code blocks
- `measure_function(func, ...)` - Measure with statistics
- `get_energy_mj()` - Get current energy reading
- `shutdown()` - Cleanup NVML

### 2. measure_energy() Function

```python
with measure_energy() as m:
    # code
print(m.energy_mj)
```

Simple context manager for quick measurements.

### 3. EnergyStats Class

```python
stats = EnergyStats(energy_list)
stats.save("file.json", "name")
stats.to_dict()
```

Statistical analysis and data export.

## 💡 Common Use Cases

### 1. Single Operation Measurement
```python
with measure_energy() as m:
    model(inputs)
```

### 2. Compare Operations
```python
with GPUEnergyMeter() as meter:
    r1 = meter.measure_function(op_a, num_trials=10)
    r2 = meter.measure_function(op_b, num_trials=10)
    ratio = r2['mean_mJ'] / r1['mean_mJ']
```

### 3. Multiple Trials for Statistics
```python
with GPUEnergyMeter() as meter:
    results = meter.measure_function(
        my_func,
        num_trials=100,
        repeat_count=50
    )
```

### 4. Manual Loop with Custom Logic
```python
with GPUEnergyMeter() as meter:
    for i in range(100):
        # custom logic
        with meter.measure() as m:
            my_operation()
        energies.append(m.energy_mj)
```

## 🏆 Best Practices

1. **Always use warmup** - First GPU operation may be slower
2. **Use sufficient trials** - At least 10, preferably 100
3. **Synchronize CUDA** - Use `torch.cuda.synchronize()`
4. **Minimize background** - Close other GPU applications
5. **Use torch.no_grad()** - For inference measurements

## 🔍 What Makes This Package Better?

### vs. Manual NVML Code:

| Aspect | Manual NVML | This Package |
|--------|-------------|--------------|
| Lines of code | ~50 per measurement | ~8 per measurement |
| Cleanup | Manual nvmlShutdown() | Automatic |
| Statistics | Manual calculation | Built-in |
| Error handling | DIY | Built-in |
| Reusability | Copy-paste | Import |
| Documentation | Self-document | Comprehensive |

## 📦 Dependencies

- Python >= 3.8
- torch >= 1.9.0
- nvidia-ml-py3 >= 7.352.0

## 🚀 Getting Started Now

1. **Install:**
   ```bash
   cd gpu_energy_meter_package
   pip install -e .
   ```

2. **Test:**
   ```bash
   python examples/quick_start.py
   ```

3. **Convert your code:**
   - Open `examples/example_usage.py`
   - See how your haecer.py is converted
   - Apply same pattern to your code




---
