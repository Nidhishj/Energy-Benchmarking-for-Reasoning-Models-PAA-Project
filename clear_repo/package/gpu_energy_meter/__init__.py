"""
GPU Energy Meter - A Python package for measuring GPU energy consumption.

This package provides utilities to measure energy consumption of GPU operations
using NVIDIA Management Library (NVML).
"""

from .meter import GPUEnergyMeter, measure_energy
from .utils import EnergyStats

__version__ = "0.1.0"
__all__ = ["GPUEnergyMeter", "measure_energy", "EnergyStats"]
