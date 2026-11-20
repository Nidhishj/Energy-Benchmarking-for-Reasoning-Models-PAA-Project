# Thinking on vs off, component level (Qwen3-8B)

- Thinking on: `20261118-060530_A_decode_gpu0_2973a123`, trace `aime2025/I-11`
- Thinking off: `20261118-075958_A_decode_off_gpu0_146d9e91`, trace `aime2025/II-6`
- Same physical GPU: yes (NVIDIA RTX PRO 5000 Blackwell …327428f0); layer 18, T = 20

## 1. Same context, on vs off

| Context | Attention % of one layer (on / off) | Attention % of a decode step (on / off) | MLP % of a decode step (on / off) | J per token (on / off) |
|---|---|---|---|---|
| 256 | 27% / 27% | 25% / 25% | 57% / 57% | 4.89 / 4.88 |
| 512 | 29% / 28% | 26% / 26% | 57% / 57% | 4.95 / 4.93 |
| 1024 | 29% / 29% | 27% / 27% | 56% / 56% | 5.04 / 5.03 |
| 2048 | 31% / 32% | 28% / 29% | 54% / 54% | 5.25 / 5.25 |
| 4096 | 35% / 36% | 31% / 32% | 50% / 50% | 5.61 / 5.61 |
| 8192 | 42% / 42% | 37% / 37% | 45% / 44% | 6.30 / 6.32 |

Thinking on also measured at [16384, 32768] (thinking-off answers are never that long).

## 2. Where each answer's generation energy goes (every saved answer, priced token by token)

| Answers | n | Median tokens | attention | MLP | norms | LM head + embed | other | Attention, mean per answer | Predicted kJ/answer | Predicted ÷ measured |
|---|---|---|---|---|---|---|---|---|---|---|
| thinking off | 90 | 457 | 26% | 57% | 4% | 6% | 8% | 25% | 3.0 | 99% |
| thinking on | 90 | 3122 | 29% | 53% | 3% | 6% | 9% | 28% | 20.2 | 102% |
| on, correct | 81 | 2986 | 29% | 54% | 3% | 6% | 9% | 28% | 17.4 | 102% |
| on, wrong | 9 | 8192 | 32% | 50% | 3% | 6% | 9% | 32% | 46.0 | 105% |
| off, correct | 81 | 424 | 26% | 57% | 4% | 6% | 8% | 25% | 2.5 | 99% |
| off, wrong | 9 | 1461 | 27% | 56% | 4% | 6% | 8% | 26% | 7.2 | 100% |

Notes:
- Shares in section 2 are energy-weighted (sum over answers); 'mean per answer' averages each answer's own share.
- Predicted ÷ measured: priced generation energy vs the energy measured for the whole generate() call (which also includes prefill, sampling and generate() bookkeeping), so it is below 100%.
- Thinking-off per-token costs beyond the longest measured off context are held at that value.
