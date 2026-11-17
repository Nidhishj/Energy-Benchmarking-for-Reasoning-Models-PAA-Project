# Fixed thinking caps on AIME 2024+2025 (Qwen3-8B), 2 copies

- Copy 1: `20261117-083259_Caps_aime_gpu1_0e93701e` (complete), GPU NVIDIA RTX 6000 Ada Generation …73beecc5: 60 of 60 traces done
- Copy 2: `20261117-104542_Caps_aime_gpu0_34f66e20` (complete), GPU NVIDIA RTX 6000 Ada Generation …13ab3888: 60 of 60 traces done

## Accuracy and energy per thinking budget

| Budget | Accuracy, all copies (95% range) | Copy 1: acc / J per query / J per correct | Copy 2: acc / J per query / J per correct | Mean thinking tok | Mean answer tok | Answer hit cap |
|---|---|---|---|---|---|---|
| off | 30/120 = 25% (18–33) | 28% / 19.1 kJ / 67.2 kJ | 22% / 22.8 kJ / 105.2 kJ | 0 | 3012 | 0 |
| ≤2k | 33/120 = 28% (20–36) | 22% / 29.1 kJ / 134.4 kJ | 33% / 29.2 kJ / 87.7 kJ | 2048 | 2150 | 29 |
| ≤4k | 49/120 = 41% (32–50) | 42% / 39.5 kJ / 94.7 kJ | 40% / 43.0 kJ / 107.5 kJ | 4069 | 1781 | 11 |
| ≤8k | 67/120 = 56% (47–64) | 57% / 62.2 kJ / 109.8 kJ | 55% / 67.1 kJ / 121.9 kJ | 7484 | 1528 | 8 |
| ≤16k | 81/120 = 68% (59–75) | 70% / 90.7 kJ / 129.6 kJ | 65% / 104.3 kJ / 160.5 kJ | 11882 | 1248 | 0 |
| full | 81/120 = 68% (59–75) | 75% / 117.0 kJ / 155.9 kJ | 60% / 140.5 kJ / 234.2 kJ | 15403 | 985 | 0 |
| full+forced | 83/120 = 69% (60–77) | 75% / 117.9 kJ / 157.2 kJ | 63% / 143.2 kJ / 226.1 kJ | 15403 | 1140 | 0 |

## Accuracy by difficulty group (all copies)

| Group | Traces | off | ≤2k | ≤4k | ≤8k | ≤16k | full | full+forced |
|---|---|---|---|---|---|---|---|---|
| 1-5 (easier) | 40 | 52% | 50% | 68% | 82% | 85% | 82% | 82% |
| 6-10 (medium) | 40 | 20% | 25% | 40% | 50% | 68% | 68% | 72% |
| 11-15 (hardest) | 40 | 2% | 8% | 15% | 35% | 50% | 52% | 52% |

## Energy per query by difficulty group (kJ, per copy)

| Group | Copy | off | ≤2k | ≤4k | ≤8k | ≤16k | full | full+forced |
|---|---|---|---|---|---|---|---|---|
| 1-5 (easier) | 1 | 14.3 | 25.0 | 35.8 | 54.8 | 69.8 | 78.5 | 78.5 |
| 1-5 (easier) | 2 | 16.6 | 25.4 | 38.2 | 59.1 | 80.0 | 89.8 | 90.3 |
| 6-10 (medium) | 1 | 19.1 | 28.8 | 40.1 | 62.1 | 88.9 | 109.8 | 110.1 |
| 6-10 (medium) | 2 | 24.6 | 29.1 | 44.7 | 67.7 | 105.2 | 151.8 | 155.7 |
| 11-15 (hardest) | 1 | 23.8 | 33.6 | 42.4 | 69.7 | 113.6 | 162.6 | 165.0 |
| 11-15 (hardest) | 2 | 27.1 | 33.3 | 46.1 | 74.4 | 127.7 | 180.0 | 183.5 |

Notes:
- ≤k: thinking stopped after k tokens, then `</think>` forced and the answer sampled (max 4,096 tokens). Traces that finish thinking before k are not cut and keep their own answer.
- Thinking energy at a cap is the energy counter read at token k during the uncapped run itself (same GPU), plus the energy of the forced answer.
- full+forced: runs whose thinking never ended get an answer forced at the token limit instead of no answer.
- Energy is only compared within a copy (each copy ran on one GPU; GPUs differ by ~8–10%).
