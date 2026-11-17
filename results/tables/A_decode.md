# Experiment A: component energy of one decoding step vs context length (Qwen3-8B)

Run `20261117-152101_A_decode_gpu1_21ae0d3c` (complete), GPU NVIDIA RTX 6000 Ada Generation (`…73beecc5`), idle 47.0 W. Trace `aime2025/I-11` (thinking on), layer 18 of 36, bf16, sdpa, T = 20, ~2.0 s per trial.

## Energy per call (mJ, raw; std/mean over trials) and time per call

| Context | input_norm | attention | post_norm | mlp | layer | embed | final_norm | lm_head | full_step |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 3.8 (6%), 58 µs | 54.7 (6%), 401 µs | 4.0 (10%), 58 µs | 88.3 (4%), 335 µs | 153.0 (4%), 643 µs | 0.9 (11%), 12 µs | 4.0 (5%), 59 µs | 353.8 (5%), 1328 µs | 6096.2 (5%), 25516 µs |
| 1024 | 4.3 (7%), 57 µs | 58.0 (5%), 401 µs | 4.0 (7%), 57 µs | 88.4 (3%), 335 µs | 153.9 (4%), 643 µs | 0.9 (13%), 13 µs | 3.9 (6%), 58 µs | 355.0 (4%), 1327 µs | 6164.9 (4%), 25445 µs |
| 2048 | 4.3 (9%), 58 µs | 60.9 (6%), 402 µs | 4.0 (9%), 57 µs | 88.8 (4%), 335 µs | 156.6 (5%), 638 µs | 0.9 (9%), 12 µs | 3.9 (5%), 58 µs | 352.4 (4%), 1327 µs | 6224.0 (4%), 25287 µs |
| 4096 | 4.3 (7%), 57 µs | 63.8 (5%), 399 µs | 4.0 (8%), 57 µs | 88.7 (4%), 335 µs | 160.9 (5%), 640 µs | 0.9 (13%), 12 µs | 3.9 (7%), 58 µs | 353.9 (4%), 1327 µs | 6536.0 (4%), 25279 µs |
| 8192 | 4.2 (8%), 58 µs | 75.2 (6%), 401 µs | 4.1 (10%), 58 µs | 89.4 (3%), 335 µs | 165.4 (5%), 639 µs | 0.9 (10%), 12 µs | 3.9 (6%), 58 µs | 355.4 (4%), 1327 µs | 6690.7 (4%), 25401 µs |
| 16384 | 4.2 (9%), 57 µs | 105.7 (5%), 400 µs | 4.2 (8%), 57 µs | 88.7 (3%), 335 µs | 198.7 (5%), 759 µs | 0.9 (10%), 12 µs | 3.8 (5%), 58 µs | 355.7 (4%), 1328 µs | 7635.5 (4%), 29042 µs |
| 32768 | 4.3 (13%), 57 µs | 187.3 (5%), 704 µs | 4.3 (10%), 57 µs | 89.0 (5%), 335 µs | 283.0 (5%), 1073 µs | 0.9 (9%), 12 µs | 4.0 (6%), 58 µs | 352.7 (5%), 1327 µs | 10745.8 (4%), 40772 µs |

## Where one new token's energy goes (whole model)

| Context | Full step (J/token) | ms/token | Attention (all layers) | MLP | Norms | LM head | Other | Block %Capture | Model %Capture |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 6.10 | 25.5 | 32% | 52% | 5% | 6% | +5% | 98.6% | 96.2% |
| 1024 | 6.16 | 25.4 | 34% | 52% | 5% | 6% | +4% | 100.5% | 95.7% |
| 2048 | 6.22 | 25.3 | 35% | 51% | 5% | 6% | +3% | 100.8% | 96.3% |
| 4096 | 6.54 | 25.3 | 35% | 49% | 5% | 5% | +6% | 99.9% | 94.1% |
| 8192 | 6.69 | 25.4 | 40% | 48% | 5% | 5% | +2% | 104.6% | 94.4% |
| 16384 | 7.64 | 29.0 | 50% | 42% | 4% | 5% | -0% | 102.0% | 98.4% |
| 32768 | 10.75 | 40.8 | 63% | 30% | 3% | 3% | +1% | 100.7% | 98.1% |

Notes:
- One call = one module for one new token with L tokens already in the KV cache. Attention includes q/k/v/o projections, q/k norms, RoPE, the DynamicCache concatenation and SDPA.
- Shares multiply layer-18 values by the number of layers (all layers have the same shapes; Qwen3-8B uses full attention in every layer).
- 'Other' = full step minus the parts (glue: residual adds, mask/rotary set-up, Python overhead); negative means the parts sum to more than the whole.
- Full step = model forward for one token with logits; it excludes sampling and generate() bookkeeping.
