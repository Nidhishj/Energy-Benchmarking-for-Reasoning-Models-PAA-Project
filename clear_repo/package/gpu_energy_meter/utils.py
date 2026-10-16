"""
Utility functions for energy measurement statistics and data storage.
"""
import json
from typing import List, Dict, Any
import torch


class EnergyStats:
    """Container for energy measurement statistics."""
    
    def __init__(self, measurements: List[float]):
        """
        Initialize with a list of energy measurements.
        
        Args:
            measurements: List of energy values in millijoules
        """
        self.measurements = measurements
        self._calculate_stats()
    
    def _calculate_stats(self):
        """Calculate statistical measures from measurements."""
        if not self.measurements:
            self.mean_mj = 0.0
            self.variance_mj2 = 0.0
            self.std_mj = 0.0
            self.min_mj = 0.0
            self.max_mj = 0.0
            return
        
        vals = torch.tensor(self.measurements, dtype=torch.float64)
        self.mean_mj = float(vals.mean().item())
        self.variance_mj2 = float(vals.var(unbiased=False).item())
        self.std_mj = float(vals.std(unbiased=False).item())
        self.min_mj = float(vals.min().item())
        self.max_mj = float(vals.max().item())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert statistics to dictionary.
        
        Returns:
            Dictionary containing all statistics
        """
        return {
            "all_trials_mJ": [float(round(x, 4)) for x in self.measurements],
            "mean_mJ": float(round(self.mean_mj, 4)),
            "variance_mJ2": float(round(self.variance_mj2, 4)),
            "std_mJ": float(round(self.std_mj, 4)),
            "min_mJ": float(round(self.min_mj, 4)),
            "max_mJ": float(round(self.max_mj, 4)),
            "num_trials": len(self.measurements),
        }
    
    def save(self, filepath: str, name: str = "Energy Measurement"):
        """
        Save statistics to JSON file.
        
        Args:
            filepath: Path to save JSON file
            name: Name of the measurement (for display)
        """
        data = self.to_dict()
        data["measurement_name"] = name
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\n=== {name} Results ===")
        for k, v in data.items():
            if k != "all_trials_mJ":  # Don't print all trials
                print(f"{k:20}: {v}")
        print(f"Saved results to {filepath}\n")
    
    def __repr__(self):
        return (f"EnergyStats(mean={self.mean_mj:.2f}mJ, "
                f"std={self.std_mj:.2f}mJ, n={len(self.measurements)})")


def calculate_energy_stats(energy_values: List[float]) -> Dict[str, Any]:
    """
    Calculate statistics from energy measurements.
    
    Args:
        energy_values: List of energy measurements in millijoules
        
    Returns:
        Dictionary containing statistical measures
    """
    stats = EnergyStats(energy_values)
    return stats.to_dict()


def save_energy_results(
    name: str,
    energy_values: List[float],
    output_file: str
):
    """
    Calculate statistics and save to JSON file.
    
    Args:
        name: Name of the measurement
        energy_values: List of energy measurements in millijoules
        output_file: Path to output JSON file
        
    Example:
        >>> energies = [100.5, 102.3, 101.8, 99.7]
        >>> save_energy_results("MLP Forward", energies, "mlp_results.json")
    """
    stats = EnergyStats(energy_values)
    stats.save(output_file, name)


def load_energy_results(filepath: str) -> Dict[str, Any]:
    """
    Load energy measurement results from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary containing energy statistics
    """
    with open(filepath, "r") as f:
        return json.load(f)
