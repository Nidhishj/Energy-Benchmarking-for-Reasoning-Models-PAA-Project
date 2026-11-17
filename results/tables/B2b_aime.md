# B2b: Qwen3 on AIME 2024+2025, thinking on vs off

Runs: 20261117-000045_B2b_aime_gpu1_725358c0, 20261117-000045_B2b_aime_gpu0_4d9566a3

## NVIDIA RTX 6000 Ada Generation

| Mode | Rows | Problems | Accuracy | Output tok | Thinking tok | Answer tok | J/query | J/query idle-sub | mJ/token | J/correct | Token cap hit | Trial spread J/query | Trial spread mJ/token | Zeus vs NVML |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 120 | 60 | 25.0% | 3012 | 0 | 3012 | 20922.4 | 16823.1 | 6832.4 | 83689.8 | 8.3% | 25.9% (max 81.3%) | 2.9% (max 6.7%) | +0.01% (max 0.23%) |
| on | 120 | 60 | 67.5% | 16388 | 15403 | 985 | 128742.7 | 104014.7 | 7555.7 | 190730.0 | 11.7% | 16.7% (max 57.7%) | 6.2% (max 15.3%) | +0.00% (max 0.07%) |

**On/off energy ratio** (per problem, mean over trials, 60 problems): median 5.8x, range 1.1x to 27.0x, ratio of means 6.2x. Published reference: Jin et al. 3.7-113x, Oviedo et al. ~13x.

| Mode | Level | Accuracy | J/query | Output tok |
|---|---|---|---|---|
| off | 2024 | 28% | 23015.4 | 3307 |
| off | 2025 | 22% | 18829.5 | 2717 |
| on | 2024 | 75% | 108113.8 | 14087 |
| on | 2025 | 60% | 149371.7 | 18689 |

Trial spread = std/mean across trials of the same problem. Sampled answers differ in length between trials,
so J/query spread mostly reflects length; the 5% check (baselines.md) applies to mJ/token.
Token-capped runs are included in the averages; their J/query understates the true cost.
