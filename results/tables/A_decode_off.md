# Experiment A: component energy of one decoding step vs context length (Qwen3-8B)

Run `20261117-231352_A_decode_off_gpu1_131f580f` (complete), GPU NVIDIA RTX 6000 Ada Generation (`…73beecc5`), idle 40.9 W. Trace `aime2024/I-10` (thinking on), layer 18 of 36, bf16, sdpa, T = 20, ~2.0 s per trial.

## Energy per call (mJ, raw; std/mean over trials) and time per call

| Context | input_norm | attention | post_norm | mlp | layer | embed | final_norm | lm_head | full_step |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 3.8 (5%), 59 µs | 55.8 (6%), 411 µs | 4.1 (7%), 59 µs | 88.3 (4%), 335 µs | 154.1 (5%), 658 µs | 0.9 (7%), 13 µs | 4.0 (6%), 60 µs | 352.7 (4%), 1327 µs | 6152.9 (6%), 25798 µs |
| 1024 | 4.4 (7%), 58 µs | 58.7 (6%), 407 µs | 4.2 (8%), 58 µs | 88.9 (4%), 335 µs | 156.3 (5%), 650 µs | 0.9 (13%), 13 µs | 4.0 (5%), 59 µs | 354.9 (4%), 1327 µs | 6270.2 (5%), 25736 µs |
| 2048 | 4.3 (8%), 59 µs | 61.5 (5%), 405 µs | 4.2 (10%), 58 µs | 89.4 (3%), 335 µs | 156.7 (5%), 649 µs | 0.9 (10%), 13 µs | 4.1 (5%), 59 µs | 351.0 (4%), 1328 µs | 6402.6 (5%), 25737 µs |
| 4096 | 4.4 (8%), 59 µs | 66.5 (5%), 406 µs | 4.2 (7%), 59 µs | 88.6 (3%), 335 µs | 164.3 (5%), 650 µs | 1.0 (12%), 13 µs | 4.1 (7%), 59 µs | 354.7 (3%), 1327 µs | 6690.0 (5%), 25687 µs |
| 8192 | 4.4 (6%), 58 µs | 76.4 (6%), 406 µs | 4.2 (10%), 58 µs | 88.5 (4%), 335 µs | 170.6 (4%), 650 µs | 0.9 (12%), 13 µs | 3.9 (5%), 59 µs | 352.7 (3%), 1327 µs | 6732.4 (4%), 25609 µs |

## Where one new token's energy goes (whole model)

| Context | Full step (J/token) | ms/token | Attention (all layers) | MLP | Norms | LM head | Other | Block %Capture | Model %Capture |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 6.15 | 25.8 | 33% | 52% | 5% | 6% | +5% | 98.6% | 96.0% |
| 1024 | 6.27 | 25.7 | 34% | 51% | 5% | 6% | +5% | 99.9% | 95.5% |
| 2048 | 6.40 | 25.7 | 35% | 50% | 5% | 5% | +5% | 101.7% | 93.7% |
| 4096 | 6.69 | 25.7 | 36% | 48% | 5% | 5% | +7% | 99.5% | 93.8% |
| 8192 | 6.73 | 25.6 | 41% | 47% | 5% | 5% | +2% | 101.7% | 96.5% |

Notes:
- One call = one module for one new token with L tokens already in the KV cache. Attention includes q/k/v/o projections, q/k norms, RoPE, the DynamicCache concatenation and SDPA.
- Shares multiply layer-18 values by the number of layers (all layers have the same shapes; Qwen3-8B uses full attention in every layer).
- 'Other' = full step minus the parts (glue: residual adds, mask/rotary set-up, Python overhead); negative means the parts sum to more than the whole.
- Full step = model forward for one token with logits; it excludes sampling and generate() bookkeeping.
