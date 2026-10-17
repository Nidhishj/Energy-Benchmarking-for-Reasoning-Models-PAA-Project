"""Shared energy-measurement code (NVML meter, CLEAR-style amplification, result records)."""

from energy.meter import GpuMeter
from energy.clear import clear_measure
from energy.record import Recorder


def get_meter(fake: bool = False, torch_device: int = 0):
    """GpuMeter for real runs; FakeMeter (energy = 100 W x time) for dry runs without an NVIDIA GPU."""
    if fake:
        from energy.fake import FakeMeter

        return FakeMeter()
    return GpuMeter(torch_device)


__all__ = ["GpuMeter", "clear_measure", "Recorder", "get_meter"]
