# Experiment A: component energy of one decoding step vs context length (Qwen3-8B)

Run `20261118-075958_A_decode_off_gpu0_146d9e91` (complete), GPU NVIDIA RTX PRO 5000 Blackwell (`…327428f0`), idle 44.8 W. Trace `aime2025/II-6` (thinking on), layer 18 of 36, bf16, sdpa, T = 20, ~2.0 s per trial.

## Energy per call (mJ, raw; std/mean over trials) and time per call

| Context | input_norm | attention | post_norm | mlp | layer | embed | final_norm | lm_head | full_step |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 2.3 (3%), 26 µs | 33.8 (3%), 150 µs | 2.4 (4%), 26 µs | 77.9 (2%), 263 µs | 123.9 (1%), 419 µs | 0.5 (5%), 5 µs | 2.3 (3%), 26 µs | 309.9 (2%), 1043 µs | 4883.8 (1%), 16454 µs |
| 512 | 2.5 (6%), 26 µs | 35.6 (2%), 152 µs | 2.4 (4%), 26 µs | 78.0 (2%), 263 µs | 126.1 (1%), 422 µs | 0.5 (5%), 5 µs | 2.3 (3%), 26 µs | 310.1 (2%), 1043 µs | 4933.1 (1%), 16617 µs |
| 1024 | 2.5 (6%), 25 µs | 37.4 (2%), 150 µs | 2.4 (4%), 25 µs | 78.2 (2%), 263 µs | 127.0 (1%), 429 µs | 0.5 (6%), 5 µs | 2.3 (3%), 26 µs | 310.7 (2%), 1043 µs | 5032.6 (2%), 16957 µs |
| 2048 | 2.5 (5%), 26 µs | 42.2 (1%), 158 µs | 2.4 (4%), 26 µs | 78.1 (2%), 263 µs | 132.3 (1%), 445 µs | 0.5 (5%), 5 µs | 2.3 (3%), 26 µs | 310.0 (2%), 1043 µs | 5247.1 (1%), 17670 µs |
| 4096 | 2.5 (5%), 25 µs | 49.6 (2%), 185 µs | 2.4 (4%), 25 µs | 78.1 (2%), 263 µs | 139.7 (1%), 471 µs | 0.5 (5%), 5 µs | 2.3 (3%), 26 µs | 309.5 (2%), 1043 µs | 5608.6 (1%), 18893 µs |
| 8192 | 2.5 (5%), 26 µs | 65.5 (2%), 234 µs | 2.5 (5%), 25 µs | 78.0 (2%), 263 µs | 155.4 (2%), 523 µs | 0.5 (5%), 5 µs | 2.3 (3%), 26 µs | 310.9 (2%), 1043 µs | 6319.7 (2%), 21218 µs |

## Where one new token's energy goes (whole model)

| Context | Full step (J/token) | ms/token | Attention (all layers) | MLP | Norms | LM head | Other | Block %Capture | Model %Capture |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 4.88 | 16.5 | 25% | 57% | 3% | 6% | +8% | 93.9% | 97.7% |
| 512 | 4.93 | 16.6 | 26% | 57% | 4% | 6% | +7% | 93.9% | 98.4% |
| 1024 | 5.03 | 17.0 | 27% | 56% | 4% | 6% | +8% | 94.9% | 97.1% |
| 2048 | 5.25 | 17.7 | 29% | 54% | 3% | 6% | +8% | 94.7% | 96.7% |
| 4096 | 5.61 | 18.9 | 32% | 50% | 3% | 6% | +9% | 95.1% | 95.2% |
| 8192 | 6.32 | 21.2 | 37% | 44% | 3% | 5% | +10% | 95.5% | 93.5% |

Notes:
- One call = one module for one new token with L tokens already in the KV cache. Attention includes q/k/v/o projections, q/k norms, RoPE, the DynamicCache concatenation and SDPA.
- Shares multiply layer-18 values by the number of layers (all layers have the same shapes; Qwen3-8B uses full attention in every layer).
- 'Other' = full step minus the parts (glue: residual adds, mask/rotary set-up, Python overhead); negative means the parts sum to more than the whole.
- Full step = model forward for one token with logits; it excludes sampling and generate() bookkeeping.
