"""Measure one run of a long workload (>= 1 s) two ways at once: raw NVML counter reads and a Zeus window.

Both read the same hardware counter on the same GPU, so they must agree within ~1% (baselines.md section 1.2).
A larger gap points at an instrumentation bug, not a finding.
"""

from contextlib import contextmanager


class Window:
    def __init__(self, meter, use_zeus: bool = True):
        self.meter = meter
        self.zeus = None
        self._n = 0
        if use_zeus:
            from zeus.monitor import ZeusMonitor

            # Zeus indexes GPUs relative to CUDA_VISIBLE_DEVICES, like torch. CPU (RAPL) monitoring is not wanted:
            # every energy number in this project is GPU-only.
            try:
                self.zeus = ZeusMonitor(gpu_indices=[meter.torch_device], cpu_indices=[])
            except TypeError:
                self.zeus = ZeusMonitor(gpu_indices=[meter.torch_device])

    @contextmanager
    def measure(self):
        """Yields a dict that is filled with E_start/E_end (mJ), t_start/t_end (s) and zeus_mJ/zeus_s on exit."""
        key = f"window{self._n}"
        self._n += 1
        out = {}
        if self.zeus:
            self.zeus.begin_window(key)  # syncs the GPU by default
        e0, t0 = self.meter.read()
        try:
            yield out
        except BaseException:
            if self.zeus:
                try:
                    self.zeus.end_window(key)
                except Exception:
                    pass
            raise
        e1, t1 = self.meter.read()
        out.update(E_start_mJ=e0, E_end_mJ=e1, t_start_s=t0, t_end_s=t1)
        if self.zeus:
            m = self.zeus.end_window(key)
            out["zeus_mJ"] = m.gpu_energy[self.meter.torch_device] * 1000
            out["zeus_s"] = m.time
