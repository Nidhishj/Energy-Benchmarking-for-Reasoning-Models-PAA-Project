"""Read the energy counter at fixed output-token counts k during one `generate()` call (B3-long, baselines.md 4).

Implemented as a StoppingCriteria that never stops: generate() calls it once after every new token, with the
prompt plus all tokens so far. At the checkpoint k values it syncs the GPU and reads the counter, so E(k) minus the
reading taken just before generate() is the energy to produce k tokens, prefill included.
"""

import torch
from transformers import StoppingCriteria


def checkpoint_grid(max_k: int, dense_until: int = 512, step: int = 512) -> list[int]:
    """1, 2, 4, ..., dense_until, then every `step` tokens up to max_k.

    Below ~64 tokens one checkpoint interval is shorter than the counter's update period (20-100 ms), so those
    points are coarse; B3-short measures that region properly. They are recorded anyway because they cost nothing.
    """
    ks, k = [], 1
    while k <= min(dense_until, max_k):
        ks.append(k)
        k *= 2
    k = dense_until + step
    while k <= max_k:
        ks.append(k)
        k += step
    return ks


class EnergyCheckpoints(StoppingCriteria):
    def __init__(self, meter, prompt_len: int, ks: list[int]):
        self.meter = meter
        self.prompt_len = prompt_len
        self.ks = set(ks)
        self.rows: list[dict] = []

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> torch.BoolTensor:
        k = input_ids.shape[1] - self.prompt_len
        if k in self.ks:
            e, t = self.meter.read()  # syncs first
            self.rows.append({"k": k, "E_mJ": e, "t_s": t})
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
