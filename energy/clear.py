"""CLEAR's amplification step (Jain et al., arXiv:2510.02810, Eq. 3-4).

Run `fn` N times back to back between two energy reads; energy per run = (E_end - E_start) / N. Repeat T times.
Differences from the authors' scripts: the GPU is synced before every read, there is an untimed warm-up, and
every trial's raw counter values are returned so any energy definition can be recomputed later.
"""

import time
from typing import Callable, Iterator

import torch

from energy.meter import GpuMeter


def clear_measure(
    fn: Callable[[], object],
    meter: GpuMeter,
    N: int,
    T: int,
    pause_s: float = 0.5,
    warmup_calls: int = 100,
) -> Iterator[dict]:
    """Yield one raw row per trial: E_start/E_end (mJ), t_start/t_end (s), N, GPU state and other processes."""
    with torch.inference_mode():
        for _ in range(warmup_calls):
            fn()
        meter.sync()

        for trial in range(T):
            time.sleep(pause_s)
            others = meter.other_processes()
            e0, t0 = meter.read()
            for _ in range(N):
                fn()
            e1, t1 = meter.read()
            yield {
                "trial": trial,
                "N": N,
                "E_start_mJ": e0,
                "E_end_mJ": e1,
                "t_start_s": t0,
                "t_end_s": t1,
                **meter.state(),
                "other_processes": others,
            }
