# B2b: Qwen3 on MATH500 (200 problems, token-capped answers rerun at 8k off / 32k on), thinking on vs off

Runs: 20261116-150325_B2b_n200_gpu0_9421301e, 20261118-203317_B2b_gpu0_b42156ac, 20261118-204428_B2b_n200_8k_off_gpu0_04ee8b31, 20261118-211548_B2b_n200_32k_gpu0_29c903cf

## NVIDIA RTX 6000 Ada Generation

| Mode | Rows | Problems | Accuracy | Output tok | Thinking tok | Answer tok | J/query | J/query idle-sub | mJ/token | J/correct | Token cap hit | Trial spread J/query | Trial spread mJ/token | Zeus vs NVML |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 200 | 200 | 90.0% | 896 | 0 | 896 | 6233.4 | 5040.9 | 6690.4 | 6926.0 | 1.5% |  |  | +0.04% (max 1.77%) |
| on | 200 | 200 | 98.5% | 4719 | 4111 | 609 | 35791.5 | 28293.1 | 7233.9 | 36336.6 | 0.5% |  |  | +0.01% (max 0.24%) |

**On/off energy ratio** (per problem, mean over trials, 200 problems): median 6.8x, range 0.6x to 51.0x, ratio of means 5.7x. Published reference: Jin et al. 3.7-113x, Oviedo et al. ~13x.

| Mode | Level | Accuracy | J/query | Output tok |
|---|---|---|---|---|
| off | 1 | 100% | 2479.2 | 386 |
| off | 2 | 95% | 3057.1 | 454 |
| off | 3 | 95% | 5292.2 | 773 |
| off | 4 | 88% | 8347.1 | 1191 |
| off | 5 | 72% | 11991.5 | 1677 |
| on | 1 | 98% | 19198.8 | 2682 |
| on | 2 | 100% | 20665.1 | 2906 |
| on | 3 | 100% | 30931.9 | 4237 |
| on | 4 | 100% | 44904.1 | 5828 |
| on | 5 | 95% | 63257.8 | 7944 |

Trial spread = std/mean across trials of the same problem. Sampled answers differ in length between trials,
so J/query spread mostly reflects length; the 5% check (baselines.md) applies to mJ/token.
Token-capped runs are included in the averages; their J/query understates the true cost.
