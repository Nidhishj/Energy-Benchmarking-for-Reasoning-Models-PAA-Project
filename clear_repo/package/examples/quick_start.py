#!/usr/bin/env python3
"""
Quick start examples for gpu_energy_meter package.
Simple examples to get you started quickly.
"""

import torch
from gpu_energy_meter import GPUEnergyMeter, measure_energy

# ----------------------------------------------------------------------
# Example 1: Simplest usage - single measurement
# ----------------------------------------------------------------------
def example_1_simple_measurement():
    """Simplest way to measure energy."""
    print("\n=== Example 1: Simple Measurement ===")
    
    with measure_energy() as m:
        # Your GPU code here
        x = torch.randn(1000, 1000).cuda()
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
    
    print(f"Energy consumed: {m.energy_mj:.2f} mJ")

# ----------------------------------------------------------------------
# Example 2: Measure a function with statistics
# ----------------------------------------------------------------------
def example_2_function_measurement():
    """Measure a function multiple times and get statistics."""
    print("\n=== Example 2: Function with Statistics ===")
    
    def my_gpu_operation():
        x = torch.randn(2000, 2000).cuda()
        y = torch.matmul(x, x)
        return y
    
    with GPUEnergyMeter() as meter:
        results = meter.measure_function(
            my_gpu_operation,
            num_trials=5,      # 5 measurement trials
            repeat_count=10,   # Repeat 10 times per trial
            warmup=True        # Run warmup first
        )
        
        print(f"Mean: {results['mean_mJ']:.2f} mJ")
        print(f"Std:  {results['std_mJ']:.2f} mJ")
        print(f"Min:  {results['min_mJ']:.2f} mJ")
        print(f"Max:  {results['max_mJ']:.2f} mJ")

# ----------------------------------------------------------------------
# Example 3: Manual loop with context manager
# ----------------------------------------------------------------------
def example_3_manual_loop():
    """Manual measurement loop for more control."""
    print("\n=== Example 3: Manual Loop ===")
    
    with GPUEnergyMeter() as meter:
        energies = []
        
        for i in range(5):
            with meter.measure() as m:
                # Your code here
                x = torch.randn(1500, 1500).cuda()
                y = torch.matmul(x, x)
                torch.cuda.synchronize()
            
            energies.append(m.energy_mj)
            print(f"Iteration {i+1}: {m.energy_mj:.2f} mJ")
        
        avg_energy = sum(energies) / len(energies)
        print(f"\nAverage: {avg_energy:.2f} mJ")

# ----------------------------------------------------------------------
# Example 4: Save results to file
# ----------------------------------------------------------------------
def example_4_save_results():
    """Measure and save results to JSON file."""
    print("\n=== Example 4: Save Results ===")
    
    from gpu_energy_meter import EnergyStats
    
    def my_operation():
        x = torch.randn(1000, 1000).cuda()
        return torch.matmul(x, x)
    
    with GPUEnergyMeter() as meter:
        results = meter.measure_function(
            my_operation,
            num_trials=10,
            repeat_count=20
        )
        
        # Create stats object and save
        stats = EnergyStats(results['all_trials_mJ'])
        stats.save("my_results.json", "Matrix Multiplication")
        
        print("Results saved to my_results.json")

# ----------------------------------------------------------------------
# Example 5: Compare different operations
# ----------------------------------------------------------------------
def example_5_compare_operations():
    """Compare energy consumption of different operations."""
    print("\n=== Example 5: Compare Operations ===")
    
    def operation_a():
        x = torch.randn(1000, 1000).cuda()
        return torch.matmul(x, x)
    
    def operation_b():
        x = torch.randn(2000, 2000).cuda()
        return x * x  # Element-wise multiplication
    
    with GPUEnergyMeter() as meter:
        # Measure operation A
        results_a = meter.measure_function(
            operation_a,
            num_trials=10,
            repeat_count=50
        )
        
        # Measure operation B
        results_b = meter.measure_function(
            operation_b,
            num_trials=10,
            repeat_count=50
        )
        
        print(f"\nOperation A (matmul 1000x1000):")
        print(f"  Mean: {results_a['mean_mJ']:.2f} mJ")
        
        print(f"\nOperation B (element-wise 2000x2000):")
        print(f"  Mean: {results_b['mean_mJ']:.2f} mJ")
        
        ratio = results_b['mean_mJ'] / results_a['mean_mJ']
        print(f"\nOperation B uses {ratio:.2f}x the energy of Operation A")

# ----------------------------------------------------------------------
# Example 6: Measure model inference
# ----------------------------------------------------------------------
def example_6_model_inference():
    """Measure energy of a simple neural network."""
    print("\n=== Example 6: Model Inference ===")
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 500),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10)
    ).cuda()
    
    # Create input
    input_data = torch.randn(32, 1000).cuda()
    
    def forward_pass():
        with torch.no_grad():
            return model(input_data)
    
    with GPUEnergyMeter() as meter:
        results = meter.measure_function(
            forward_pass,
            num_trials=20,
            repeat_count=100,
            warmup=True
        )
        
        print(f"Model inference energy:")
        print(f"  Mean: {results['mean_mJ']:.2f} mJ")
        print(f"  Std:  {results['std_mJ']:.2f} mJ")

# ----------------------------------------------------------------------
# Run all examples
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("GPU Energy Meter - Quick Start Examples")
    print("=" * 60)
    
    try:
        example_1_simple_measurement()
        example_2_function_measurement()
        example_3_manual_loop()
        example_4_save_results()
        example_5_compare_operations()
        example_6_model_inference()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
