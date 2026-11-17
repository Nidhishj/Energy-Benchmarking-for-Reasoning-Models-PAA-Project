# Fixed thinking caps on AIME 2024+2025 (Qwen3-8B), partial: copy 2 still running

- Copy 1: `20261117-083259_Caps_aime_gpu1_0e93701e` (complete), GPU NVIDIA RTX 6000 Ada Generation …73beecc5: 60 of 60 traces done
- Copy 2: `20261117-104542_Caps_aime_gpu0_34f66e20` (running), GPU NVIDIA RTX 6000 Ada Generation …13ab3888: 16 of 60 traces done

## Accuracy and energy per thinking budget

| Budget | Accuracy, all copies (95% range) | Copy 1: acc / J per query / J per correct | Copy 2: acc / J per query / J per correct | Mean thinking tok | Mean answer tok | Answer hit cap |
|---|---|---|---|---|---|---|
| off | 21/76 = 28% (19–39) | 28% / 19.1 kJ / 67.2 kJ | 25% / 24.3 kJ / 97.1 kJ | 0 | 2957 | 0 |
| ≤2k | 18/76 = 24% (16–34) | 22% / 29.1 kJ / 134.4 kJ | 31% / 29.1 kJ / 93.1 kJ | 2048 | 2230 | 22 |
| ≤4k | 34/76 = 45% (34–56) | 42% / 39.5 kJ / 94.7 kJ | 56% / 44.2 kJ / 78.5 kJ | 4091 | 1784 | 8 |
| ≤8k | 45/76 = 59% (48–70) | 57% / 62.2 kJ / 109.8 kJ | 69% / 67.2 kJ / 97.8 kJ | 7539 | 1536 | 5 |
| ≤16k | 54/76 = 71% (60–80) | 70% / 90.7 kJ / 129.6 kJ | 75% / 94.4 kJ / 125.9 kJ | 11651 | 1217 | 0 |
| full | 57/76 = 75% (64–83) | 75% / 117.0 kJ / 155.9 kJ | 75% / 105.7 kJ / 140.9 kJ | 14450 | 1050 | 0 |
| full+forced | 57/76 = 75% (64–83) | 75% / 117.9 kJ / 157.2 kJ | 75% / 105.7 kJ / 140.9 kJ | 14450 | 1121 | 0 |

## Accuracy by difficulty group (all copies)

| Group | Traces | off | ≤2k | ≤4k | ≤8k | ≤16k | full | full+forced |
|---|---|---|---|---|---|---|---|---|
| 1-5 (easier) | 26 | 54% | 46% | 69% | 81% | 88% | 88% | 88% |
| 6-10 (medium) | 25 | 24% | 20% | 44% | 60% | 72% | 80% | 80% |
| 11-15 (hardest) | 25 | 4% | 4% | 20% | 36% | 52% | 56% | 56% |

## Energy per query by difficulty group (kJ, per copy)

| Group | Copy | off | ≤2k | ≤4k | ≤8k | ≤16k | full | full+forced |
|---|---|---|---|---|---|---|---|---|
| 1-5 (easier) | 1 | 14.3 | 25.0 | 35.8 | 54.8 | 69.8 | 78.5 | 78.5 |
| 1-5 (easier) | 2 | 14.1 | 25.2 | 38.5 | 60.2 | 70.1 | 70.1 | 70.1 |
| 6-10 (medium) | 1 | 19.1 | 28.8 | 40.1 | 62.1 | 88.9 | 109.8 | 110.1 |
| 6-10 (medium) | 2 | 27.2 | 27.4 | 48.1 | 67.3 | 94.9 | 104.9 | 104.9 |
| 11-15 (hardest) | 1 | 23.8 | 33.6 | 42.4 | 69.7 | 113.6 | 162.6 | 165.0 |
| 11-15 (hardest) | 2 | 33.6 | 35.5 | 47.0 | 75.6 | 123.1 | 149.2 | 149.2 |

Notes:
- ≤k: thinking stopped after k tokens, then `</think>` forced and the answer sampled (max 4,096 tokens). Traces that finish thinking before k are not cut and keep their own answer.
- Thinking energy at a cap is the energy counter read at token k during the uncapped run itself (same GPU), plus the energy of the forced answer.
- full+forced: runs whose thinking never ended get an answer forced at the token limit instead of no answer.
- Energy is only compared within a copy (each copy ran on one GPU; GPUs differ by ~8–10%).
