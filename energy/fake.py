"""Stand-in for GpuMeter on machines without an NVIDIA GPU. For dry runs of the scripts only.

Energy is a fixed power times wall time, so dry-run numbers mean nothing; they only exercise the code paths
(recording, resume, checkpoints, analysis). Never report anything produced with this meter.
"""

import time


class FakeMeter:
    torch_device = None

    def __init__(self, watts: float = 100.0):
        self.watts = watts
        self._t0 = time.time()
        self.uuid = "FAKE"
        self.nvml_index = -1
        self.name = "FAKE meter (dry run)"

    def sync(self):
        pass

    def read(self, sync: bool = True) -> tuple[int, float]:
        t = time.time()
        return int((t - self._t0) * self.watts * 1000), t

    def state(self) -> dict:
        return {"temp_C": 0, "sm_clock_MHz": 0, "mem_clock_MHz": 0, "power_W": self.watts}

    def other_processes(self) -> list[dict]:
        return []

    def info(self) -> dict:
        return {"gpu_name": self.name, "gpu_uuid": self.uuid, "nvml_index": self.nvml_index,
                "cuda_visible_devices": None, "driver": "none", "power_limit_W": 0, "memory_total_MiB": 0}

    def measure_idle(self, seconds: float = 30.0) -> dict:
        e0, t0 = self.read()
        time.sleep(seconds)
        e1, t1 = self.read()
        return {"duration_s": t1 - t0, "E_start_mJ": e0, "E_end_mJ": e1,
                "mean_W": (e1 - e0) / 1000 / max(t1 - t0, 1e-9), **self.state()}
