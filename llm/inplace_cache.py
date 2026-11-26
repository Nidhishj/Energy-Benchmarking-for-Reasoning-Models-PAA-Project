"""A KV cache that writes new tokens in place instead of copying the whole cache every step (KV-cache ablation).

transformers' DynamicLayer appends with torch.cat, so every decode step reads and rewrites the entire cache of every
layer. InplaceLayer pre-allocates [batch, kv_heads, max_len, head_dim] once, copies only the new token(s) into the
next free slots, and returns a view of the filled part. The attention maths and the mask are unchanged (the view has
exactly the cached length, so decode steps still skip the mask and SDPA still shares K/V heads via enable_gqa).
"""

import torch
from transformers import DynamicCache
from transformers.cache_utils import Cache, DynamicLayer


class InplaceLayer(DynamicLayer):
    def __init__(self, max_len: int):
        super().__init__()
        self.max_len = max_len
        self.n = 0

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        b, h, _, d = key_states.shape
        self.kbuf = torch.empty(b, h, self.max_len, d, dtype=self.dtype, device=self.device)
        self.vbuf = torch.empty(b, h, self.max_len, d, dtype=self.dtype, device=self.device)
        self.keys, self.values = self.kbuf[:, :, :0], self.vbuf[:, :, :0]
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        m = key_states.shape[-2]
        if self.n + m > self.max_len:
            raise RuntimeError(f"InplaceLayer full: {self.n} + {m} > {self.max_len}")
        self.kbuf[:, :, self.n:self.n + m].copy_(key_states)
        self.vbuf[:, :, self.n:self.n + m].copy_(value_states)
        self.n += m
        self.keys, self.values = self.kbuf[:, :, :self.n], self.vbuf[:, :, :self.n]
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self.n

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.n - abs(max_length)
        if self.n <= max_length:
            return
        self.n = max_length
        self.keys, self.values = self.kbuf[:, :, :self.n], self.vbuf[:, :, :self.n]


class InplaceCache(DynamicCache):
    """DynamicCache with InplaceLayer layers (isinstance DynamicCache, so model code treats it the same)."""

    def __init__(self, n_layers: int, max_len: int):
        Cache.__init__(self, layers=[InplaceLayer(max_len) for _ in range(n_layers)])


class FrozenInplace:
    """In-place counterpart of FrozenCache for one attention call at context L: K/V for L tokens sit in a buffer of
    L + 1 slots; update() writes the new token into slot L and returns the whole buffer (no copy, nothing stored)."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        b, h, L, d = keys.shape
        self.L = L
        self.kbuf = torch.empty(b, h, L + 1, d, dtype=keys.dtype, device=keys.device)
        self.vbuf = torch.empty(b, h, L + 1, d, dtype=values.dtype, device=values.device)
        self.kbuf[:, :, :L].copy_(keys)
        self.vbuf[:, :, :L].copy_(values)

    def update(self, k, v, layer_idx, cache_kwargs=None):
        self.kbuf[:, :, self.L:].copy_(k)
        self.vbuf[:, :, self.L:].copy_(v)
        return self.kbuf, self.vbuf
