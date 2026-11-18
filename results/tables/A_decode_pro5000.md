# Experiment A: component energy of one decoding step vs context length (Qwen3-8B)

Run `20261118-060530_A_decode_gpu0_2973a123` (complete), GPU NVIDIA RTX PRO 5000 Blackwell (`…327428f0`), idle 46.3 W. Trace `aime2025/I-11` (thinking on), layer 18 of 36, bf16, sdpa, T = 20, ~2.0 s per trial.

## Energy per call (mJ, raw; std/mean over trials) and time per call

| Context | input_norm | attention | post_norm | mlp | layer | embed | final_norm | lm_head | full_step |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 2.3 (3%), 25 µs | 34.1 (2%), 152 µs | 2.4 (4%), 26 µs | 77.9 (2%), 263 µs | 124.4 (1%), 419 µs | 0.5 (5%), 5 µs | 2.3 (4%), 26 µs | 310.6 (2%), 1043 µs | 4893.1 (1%), 16448 µs |
| 512 | 2.6 (5%), 26 µs | 35.8 (2%), 154 µs | 2.4 (4%), 26 µs | 77.9 (2%), 263 µs | 125.4 (1%), 422 µs | 0.5 (5%), 5 µs | 2.3 (3%), 26 µs | 311.0 (2%), 1043 µs | 4950.1 (2%), 16611 µs |
| 1024 | 2.5 (6%), 25 µs | 37.5 (2%), 152 µs | 2.4 (4%), 25 µs | 78.4 (2%), 263 µs | 127.2 (1%), 429 µs | 0.5 (5%), 5 µs | 2.3 (3%), 26 µs | 310.6 (2%), 1043 µs | 5039.0 (2%), 16951 µs |
| 2048 | 2.5 (5%), 26 µs | 41.1 (2%), 158 µs | 2.5 (5%), 26 µs | 78.2 (2%), 263 µs | 132.4 (1%), 445 µs | 0.5 (5%), 5 µs | 2.3 (3%), 26 µs | 310.8 (2%), 1043 µs | 5251.7 (2%), 17662 µs |
| 4096 | 2.1 (2%), 25 µs | 49.0 (2%), 185 µs | 2.4 (4%), 25 µs | 78.3 (2%), 262 µs | 140.7 (1%), 471 µs | 0.5 (5%), 5 µs | 2.3 (3%), 26 µs | 309.9 (2%), 1043 µs | 5607.2 (2%), 18876 µs |
| 8192 | 2.5 (6%), 25 µs | 64.7 (2%), 234 µs | 2.4 (4%), 25 µs | 78.3 (2%), 262 µs | 155.0 (1%), 522 µs | 0.5 (5%), 5 µs | 2.3 (3%), 26 µs | 310.3 (2%), 1043 µs | 6303.7 (2%), 21210 µs |
| 16384 | 2.5 (6%), 25 µs | 117.0 (2%), 395 µs | 2.5 (5%), 25 µs | 77.9 (2%), 262 µs | 205.2 (2%), 689 µs | 0.5 (5%), 5 µs | 2.3 (3%), 26 µs | 310.6 (2%), 1043 µs | 7736.0 (2%), 25970 µs |
| 32768 | 2.5 (6%), 25 µs | 202.2 (2%), 681 µs | 2.5 (5%), 25 µs | 78.3 (2%), 263 µs | 291.3 (2%), 979 µs | 0.5 (5%), 5 µs | 2.3 (3%), 26 µs | 309.8 (2%), 1043 µs | 10874.3 (1%), 36585 µs |

## Where one new token's energy goes (whole model)

| Context | Full step (J/token) | ms/token | Attention (all layers) | MLP | Norms | LM head | Other | Block %Capture | Model %Capture |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 4.89 | 16.4 | 25% | 57% | 3% | 6% | +8% | 93.8% | 97.9% |
| 512 | 4.95 | 16.6 | 26% | 57% | 4% | 6% | +7% | 94.6% | 97.6% |
| 1024 | 5.04 | 17.0 | 27% | 56% | 4% | 6% | +7% | 95.0% | 97.1% |
| 2048 | 5.25 | 17.7 | 28% | 54% | 3% | 6% | +9% | 93.8% | 96.8% |
| 4096 | 5.61 | 18.9 | 31% | 50% | 3% | 6% | +10% | 93.7% | 95.9% |
| 8192 | 6.30 | 21.2 | 37% | 45% | 3% | 5% | +11% | 95.5% | 93.5% |
| 16384 | 7.74 | 26.0 | 54% | 36% | 2% | 4% | +3% | 97.4% | 99.6% |
| 32768 | 10.87 | 36.6 | 67% | 26% | 2% | 3% | +3% | 98.0% | 99.3% |

Notes:
- One call = one module for one new token with L tokens already in the KV cache. Attention includes q/k/v/o projections, q/k norms, RoPE, the DynamicCache concatenation and SDPA.
- Shares multiply layer-18 values by the number of layers (all layers have the same shapes; Qwen3-8B uses full attention in every layer).
- 'Other' = full step minus the parts (glue: residual adds, mask/rotary set-up, Python overhead); negative means the parts sum to more than the whole.
- Full step = model forward for one token with logits; it excludes sampling and generate() bookkeeping.
