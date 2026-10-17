"""NVML energy meter bound to the same physical GPU that torch is using.

CLEAR's scripts open NVML device 0 and force CUDA_VISIBLE_DEVICES=0, which silently measures the wrong card
when another GPU is selected. Here the NVML handle is looked up by the UUID of torch's current device, so torch
and NVML always point at the same card.
"""

import os
import time

import pynvml
import torch


class GpuMeter:
    def __init__(self, torch_device: int = 0):
        pynvml.nvmlInit()
        self.torch_device = torch_device
        props = torch.cuda.get_device_properties(torch_device)
        self.uuid = "GPU-" + str(props.uuid)
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(self.uuid)
        self.nvml_index = pynvml.nvmlDeviceGetIndex(self.handle)
        self.name = pynvml.nvmlDeviceGetName(self.handle)

    def sync(self):
        torch.cuda.synchronize(self.torch_device)

    def read(self, sync: bool = True) -> tuple[int, float]:
        """Return (total energy since boot in mJ, wall time in s). Syncs the GPU first so queued work is counted."""
        if sync:
            self.sync()
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle), time.time()

    def state(self) -> dict:
        """Temperature, clocks and power at this moment (throttling changes energy, so we log it per trial)."""
        h = self.handle
        return {
            "temp_C": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            "sm_clock_MHz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "mem_clock_MHz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
            "power_W": pynvml.nvmlDeviceGetPowerUsage(h) / 1000,
        }

    def other_processes(self) -> list[dict]:
        """Processes on this GPU that are not us. Their work would be counted in our energy readings."""
        me = os.getpid()
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        return [{"pid": p.pid, "used_MiB": (p.usedGpuMemory or 0) // 2**20} for p in procs if p.pid != me]

    def info(self) -> dict:
        h = self.handle
        return {
            "gpu_name": self.name,
            "gpu_uuid": self.uuid,
            "nvml_index": self.nvml_index,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "power_limit_W": pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000,
            "memory_total_MiB": pynvml.nvmlDeviceGetMemoryInfo(h).total // 2**20,
        }

    def measure_idle(self, seconds: float = 30.0) -> dict:
        """Average idle power over `seconds`, from the same energy counter used for measurements."""
        e0, t0 = self.read()
        time.sleep(seconds)
        e1, t1 = self.read()
        return {
            "duration_s": t1 - t0,
            "E_start_mJ": e0,
            "E_end_mJ": e1,
            "mean_W": (e1 - e0) / 1000 / (t1 - t0),
            **self.state(),
        }
