# B2b: Qwen3 on MATH500, thinking on vs off

Runs: 20261017-193728_B2b_gpu0_ba965317, 20261018-163010_B2b_extra_gpu0_eeec9b0d

## NVIDIA RTX PRO 5000 Blackwell

| Mode | Rows | Problems | Accuracy | Output tok | Thinking tok | Answer tok | J/query | J/query idle-sub | mJ/token | J/correct | Token cap hit | Trial spread J/query | Trial spread mJ/token | Zeus vs NVML |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 90 | 30 | 90.0% | 606 | 0 | 606 | 3023.8 | 2394.5 | 4958.9 | 3359.8 | 2.2% | 10.2% (max 37.2%) | 1.0% (max 5.3%) | +0.02% (max 1.66%) |
| on | 90 | 30 | 90.0% | 3794 | 3321 | 473 | 19771.8 | 15494.1 | 5155.9 | 21968.6 | 11.1% | 11.3% (max 27.8%) | 0.4% (max 0.9%) | +0.00% (max 0.35%) |

**On/off energy ratio** (per problem, mean over trials, 30 problems): median 7.1x, range 3.5x to 11.7x, ratio of means 6.5x. Published reference: Jin et al. 3.7-113x, Oviedo et al. ~13x.

| Mode | Level | Accuracy | J/query | Output tok |
|---|---|---|---|---|
| off | 1 | 100% | 1316.0 | 268 |
| off | 2 | 83% | 2805.5 | 564 |
| off | 3 | 100% | 2299.3 | 462 |
| off | 4 | 94% | 3293.5 | 658 |
| off | 5 | 72% | 5404.9 | 1081 |
| on | 1 | 100% | 10977.9 | 2154 |
| on | 2 | 94% | 23631.6 | 4516 |
| on | 3 | 100% | 12768.6 | 2496 |
| on | 4 | 83% | 22281.4 | 4276 |
| on | 5 | 72% | 29199.3 | 5530 |

Trial spread = std/mean across trials of the same problem. Sampled answers differ in length between trials,
so J/query spread mostly reflects length; the 5% check (baselines.md) applies to mJ/token.
Token-capped runs are included in the averages; their J/query understates the true cost.
