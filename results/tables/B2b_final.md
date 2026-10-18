# B2b: Qwen3 on MATH500, thinking on vs off

Runs: 20261017-193728_B2b_gpu0_ba965317

## NVIDIA RTX PRO 5000 Blackwell

| Mode | Rows | Problems | Accuracy | Output tok | Thinking tok | Answer tok | J/query | J/query idle-sub | mJ/token | J/correct | Token cap hit | Trial spread J/query | Trial spread mJ/token | Zeus vs NVML |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 90 | 30 | 90.0% | 606 | 0 | 606 | 3023.8 | 2394.5 | 4958.9 | 3359.8 | 2.2% | 10.2% (max 37.2%) | 1.0% (max 5.3%) | +0.02% (max 1.66%) |
| on | 60 | 30 | 90.0% | 3822 | 3344 | 478 | 19915.6 | 15802.9 | 5159.4 | 22128.5 | 10.0% | 8.0% (max 19.2%) | 0.2% (max 0.8%) | +0.00% (max 0.00%) |

**On/off energy ratio** (per problem, mean over trials, 30 problems): median 7.0x, range 3.4x to 12.9x, ratio of means 6.6x. Published reference: Jin et al. 3.7-113x, Oviedo et al. ~13x.

| Mode | Level | Accuracy | J/query | Output tok |
|---|---|---|---|---|
| off | 1 | 100% | 1316.0 | 268 |
| off | 2 | 83% | 2805.5 | 564 |
| off | 3 | 100% | 2299.3 | 462 |
| off | 4 | 94% | 3293.5 | 658 |
| off | 5 | 72% | 5404.9 | 1081 |
| on | 1 | 100% | 11478.5 | 2247 |
| on | 2 | 92% | 23919.7 | 4569 |
| on | 3 | 100% | 13116.9 | 2564 |
| on | 4 | 83% | 22288.6 | 4276 |
| on | 5 | 75% | 28774.4 | 5454 |

Trial spread = std/mean across trials of the same problem. Sampled answers differ in length between trials,
so J/query spread mostly reflects length; the 5% check (baselines.md) applies to mJ/token.
Token-capped runs are included in the averages; their J/query understates the true cost.
