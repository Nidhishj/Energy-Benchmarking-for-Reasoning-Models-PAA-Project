# B2b: Qwen3 on MATH500, thinking on vs off

Runs: 20261018-162557_B2b_repro_gpu0_a5135e7f

## NVIDIA RTX PRO 5000 Blackwell

| Mode | Rows | Problems | Accuracy | Output tok | Thinking tok | Answer tok | J/query | J/query idle-sub | mJ/token | J/correct | Token cap hit | Trial spread J/query | Trial spread mJ/token | Zeus vs NVML |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 3 | 3 | 100.0% | 759 | 0 | 759 | 3721.7 | 3098.3 | 4827.5 | 3721.7 | 0.0% |  |  | +0.00% (max 0.00%) |
| on | 3 | 3 | 100.0% | 3837 | 3113 | 724 | 19820.2 | 16556.7 | 5158.3 | 19820.2 | 0.0% |  |  | +0.03% (max 0.10%) |

**On/off energy ratio** (per problem, mean over trials, 3 problems): median 6.3x, range 3.6x to 11.5x, ratio of means 5.3x. Published reference: Jin et al. 3.7-113x, Oviedo et al. ~13x.

| Mode | Level | Accuracy | J/query | Output tok |
|---|---|---|---|---|
| off | 4 | 100% | 2089.9 | 440 |
| off | 5 | 100% | 6985.3 | 1398 |
| on | 4 | 100% | 17310.5 | 3374 |
| on | 5 | 100% | 24839.6 | 4763 |

Trial spread = std/mean across trials of the same problem. Sampled answers differ in length between trials,
so J/query spread mostly reflects length; the 5% check (baselines.md) applies to mJ/token.
Token-capped runs are included in the averages; their J/query understates the true cost.
