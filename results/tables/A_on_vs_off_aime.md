# Thinking on vs off, component level (Qwen3-8B)

- Thinking on: `20261117-152101_A_decode_gpu1_21ae0d3c`, trace `aime2025/I-11`
- Thinking off: `20261117-231352_A_decode_off_gpu1_131f580f`, trace `aime2024/I-10`
- Same physical GPU: yes (NVIDIA RTX 6000 Ada Generation …73beecc5); layer 18, T = 20

## 1. Same context, on vs off

| Context | Attention % of one layer (on / off) | Attention % of a decode step (on / off) | MLP % of a decode step (on / off) | J per token (on / off) |
|---|---|---|---|---|
| 256 | 36% / 36% | 32% / 33% | 52% / 52% | 6.10 / 6.15 |
| 1024 | 38% / 38% | 34% / 34% | 52% / 51% | 6.16 / 6.27 |
| 2048 | 39% / 39% | 35% / 35% | 51% / 50% | 6.22 / 6.40 |
| 4096 | 40% / 40% | 35% / 36% | 49% / 48% | 6.54 / 6.69 |
| 8192 | 46% / 45% | 40% / 41% | 48% / 47% | 6.69 / 6.73 |

Thinking on also measured at [16384, 32768] (thinking-off answers are never that long).

## 2. Where each answer's generation energy goes (every saved answer, priced token by token)

| Answers | n | Median tokens | attention | MLP | norms | LM head + embed | other | Attention, mean per answer | Predicted kJ/answer | Predicted ÷ measured |
|---|---|---|---|---|---|---|---|---|---|---|
| thinking off | 120 | 2290 | 35% | 50% | 5% | 6% | 5% | 34% | 19.3 | 92% |
| thinking on | 120 | 14372 | 45% | 44% | 4% | 5% | 2% | 41% | 118.5 | 92% |
| on, correct | 81 | 10572 | 40% | 47% | 4% | 5% | 3% | 39% | 79.0 | 92% |
| on, wrong | 39 | 28614 | 48% | 42% | 4% | 5% | 1% | 47% | 200.5 | 92% |
| off, correct | 30 | 1354 | 34% | 51% | 5% | 6% | 5% | 33% | 9.6 | 93% |
| off, wrong | 90 | 2662 | 35% | 50% | 5% | 6% | 5% | 34% | 22.6 | 92% |

Notes:
- Shares in section 2 are energy-weighted (sum over answers); 'mean per answer' averages each answer's own share.
- Predicted ÷ measured: priced generation energy vs the energy measured for the whole generate() call (which also includes prefill, sampling and generate() bookkeeping), so it is below 100%.
- Thinking-off per-token costs beyond the longest measured off context are held at that value.
