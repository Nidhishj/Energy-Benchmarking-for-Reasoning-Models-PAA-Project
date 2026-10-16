# GPU Energy Meter - Quick Start

Get started in 5 minutes! ⚡

## Installation

```bash
# Navigate to the package directory
cd gpu_energy_meter_package

# Install the package
pip install -e .
```

## Verify Installation

```python
python -c "from gpu_energy_meter import GPUEnergyMeter; print('✅ Installation successful!')"
```

## Your First Measurement (30 seconds)

Create a file `test_measurement.py`:

```python
from gpu_energy_meter import measure_energy
import torch

# Simple measurement
with measure_energy() as m:
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    torch.cuda.synchronize()

print(f"✅ Energy consumed: {m.energy_mj:.2f} mJ")
```

Run it:
```bash
python test_measurement.py
```

## Convert Your Existing Code (2 minutes)

### Before (Your haecer.py style):

```python
from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

energy_usages = []
for trial in range(100):
    start = nvmlDeviceGetTotalEnergyConsumption(handle)
    for _ in range(100):
        output = mlp(input)
    end = nvmlDeviceGetTotalEnergyConsumption(handle)
    energy_usages.append((end - start) / 100)

# Calculate stats...
```

### After (With this package):

```python
from gpu_energy_meter import GPUEnergyMeter, EnergyStats

def mlp_forward():
    return mlp(input)

with GPUEnergyMeter() as meter:
    results = meter.measure_function(
        mlp_forward,
        num_trials=100,
        repeat_count=100
    )
    
    stats = EnergyStats(results['all_trials_mJ'])
    stats.save("results.json", "MLP Layer")
```

**That's it!** ✨

## Package Structure

```
gpu_energy_meter/
├── gpu_energy_meter/          # Main package
│   ├── __init__.py           # Package exports
│   ├── meter.py              # Core measurement class
│   └── utils.py              # Statistics and helpers
├── examples/                  # Usage examples
│   ├── quick_start.py        # 6 simple examples
│   └── example_usage.py      # Your haecer.py converted
├── setup.py                   # Installation script
├── README.md                  # Full documentation
├── USAGE_GUIDE.md            # Detailed usage guide
└── requirements.txt          # Dependencies
```

## What You Get

✅ **Simple API**: 3 main components
- `GPUEnergyMeter` - Main measurement class  
- `measure_energy()` - Quick context manager
- `EnergyStats` - Statistics helper

✅ **Automatic Features**:
- NVML initialization/cleanup
- Statistical analysis
- JSON export
- Error handling

✅ **85% Less Code**: Focus on your research, not measurement boilerplate

## Next Steps

1. **Try the examples**:
   ```bash
   python examples/quick_start.py
   python examples/example_usage.py
   ```

2. **Read the full guide**: `USAGE_GUIDE.md` for advanced usage

3. **Check the API**: `README.md` for complete documentation

## Common Use Cases

### Measure a single operation:
```python
with measure_energy() as m:
    model(inputs)
print(m.energy_mj)
```

### Compare operations:
```python
with GPUEnergyMeter() as meter:
    r1 = meter.measure_function(operation_a, num_trials=10)
    r2 = meter.measure_function(operation_b, num_trials=10)
    print(f"B uses {r2['mean_mJ']/r1['mean_mJ']:.2f}x the energy")
```

### Save results:
```python
with GPUEnergyMeter() as meter:
    results = meter.measure_function(my_func, num_trials=100)
    EnergyStats(results['all_trials_mJ']).save("results.json", "My Test")
```

## Need Help?

- 📖 Full docs: `README.md`
- 📚 Usage guide: `USAGE_GUIDE.md`  
- 💡 Examples: `examples/` directory
- 🐛 Issues: Open an issue on GitHub

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.9.0
- nvidia-ml-py3 ≥ 7.352.0
- NVIDIA GPU with NVML support

**Happy Measuring! 🚀**
