# Experiment A: component energy of one decoding step vs context length (Qwen3-8B)

Run `20261118-070708_A_decode_B_correct_gpu0_9e3882bc` (complete), GPU NVIDIA RTX PRO 5000 Blackwell (`…327428f0`), idle 44.2 W. Trace `aime2024/I-8` (thinking on), layer 18 of 36, bf16, sdpa, T = 20, ~2.0 s per trial.

## Energy per call (mJ, raw; std/mean over trials) and time per call

| Context | input_norm | attention | post_norm | mlp | layer | embed | final_norm | lm_head | full_step |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 2.3 (3%), 26 µs | 33.9 (2%), 151 µs | 2.4 (3%), 26 µs | 78.1 (2%), 263 µs | 124.4 (1%), 419 µs | 0.5 (5%), 5 µs | 2.3 (3%), 26 µs | 310.4 (2%), 1043 µs | 4900.3 (1%), 16452 µs |
| 512 | 2.6 (5%), 26 µs | 35.7 (2%), 153 µs | 2.4 (4%), 26 µs | 78.2 (2%), 263 µs | 125.5 (1%), 422 µs | 0.5 (5%), 5 µs | 2.3 (3%), 26 µs | 310.6 (2%), 1043 µs | 4951.6 (2%), 16608 µs |
| 1024 | 2.5 (6%), 26 µs | 37.4 (2%), 151 µs | 2.5 (4%), 26 µs | 78.2 (2%), 263 µs | 127.3 (1%), 429 µs | 0.5 (6%), 5 µs | 2.3 (3%), 26 µs | 309.9 (2%), 1043 µs | 5047.3 (2%), 16953 µs |
| 2048 | 2.6 (5%), 26 µs | 41.5 (2%), 158 µs | 2.5 (4%), 26 µs | 78.1 (2%), 263 µs | 132.9 (1%), 445 µs | 0.5 (5%), 5 µs | 2.3 (3%), 26 µs | 311.0 (2%), 1043 µs | 5255.5 (2%), 17658 µs |
| 4096 | 2.5 (5%), 26 µs | 49.6 (2%), 185 µs | 2.5 (4%), 26 µs | 78.1 (2%), 263 µs | 140.2 (1%), 471 µs | 0.5 (6%), 5 µs | 2.3 (3%), 26 µs | 309.8 (2%), 1043 µs | 5622.7 (2%), 18888 µs |
| 8192 | 2.6 (5%), 26 µs | 66.7 (2%), 234 µs | 2.5 (5%), 26 µs | 78.5 (2%), 263 µs | 156.1 (1%), 523 µs | 0.5 (5%), 5 µs | 2.3 (3%), 26 µs | 311.0 (2%), 1043 µs | 6312.7 (2%), 21224 µs |
| 16384 | 2.5 (5%), 26 µs | 117.2 (2%), 395 µs | 2.5 (5%), 26 µs | 78.1 (2%), 263 µs | 204.6 (1%), 689 µs | 0.5 (6%), 5 µs | 2.3 (3%), 26 µs | 310.7 (2%), 1043 µs | 7739.8 (2%), 25989 µs |

## Where one new token's energy goes (whole model)

| Context | Full step (J/token) | ms/token | Attention (all layers) | MLP | Norms | LM head | Other | Block %Capture | Model %Capture |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 4.90 | 16.5 | 25% | 57% | 3% | 6% | +8% | 93.8% | 97.8% |
| 512 | 4.95 | 16.6 | 26% | 57% | 4% | 6% | +7% | 94.8% | 97.5% |
| 1024 | 5.05 | 17.0 | 27% | 56% | 4% | 6% | +8% | 94.7% | 97.0% |
| 2048 | 5.26 | 17.7 | 28% | 54% | 3% | 6% | +9% | 93.8% | 97.0% |
| 4096 | 5.62 | 18.9 | 32% | 50% | 3% | 6% | +9% | 94.7% | 95.3% |
| 8192 | 6.31 | 21.2 | 38% | 45% | 3% | 5% | +9% | 96.3% | 94.0% |
| 16384 | 7.74 | 26.0 | 55% | 36% | 2% | 4% | +3% | 97.9% | 99.2% |

Notes:
- One call = one module for one new token with L tokens already in the KV cache. Attention includes q/k/v/o projections, q/k norms, RoPE, the DynamicCache concatenation and SDPA.
- Shares multiply layer-18 values by the number of layers (all layers have the same shapes; Qwen3-8B uses full attention in every layer).
- 'Other' = full step minus the parts (glue: residual adds, mask/rotary set-up, Python overhead); negative means the parts sum to more than the whole.
- Full step = model forward for one token with logits; it excludes sampling and generate() bookkeeping.
