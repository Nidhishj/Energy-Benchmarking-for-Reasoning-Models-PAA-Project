# B3: energy per token vs output tokens

Short runs: 20261017-192548_B3short_gpu0_4546c8ed
Long runs (B2b checkpoints): 20261017-193728_B2b_gpu0_ba965317, 20261018-163010_B2b_extra_gpu0_eeec9b0d

![B3-1](../figures/B3_final-1_short.png)

| GPU | Mode | Prompt | k | E(k) mJ | std | E(k)/k mJ |
|---|---|---|---|---|---|---|
| NVIDIA RTX PRO 5000 Blackwell | off | longest | 1 | 9102.3 | 273.5 | 9102.3 |
| NVIDIA RTX PRO 5000 Blackwell | off | longest | 2 | 13885.5 | 394.0 | 6942.8 |
| NVIDIA RTX PRO 5000 Blackwell | off | longest | 3 | 18818.4 | 571.0 | 6272.8 |
| NVIDIA RTX PRO 5000 Blackwell | off | longest | 4 | 23404.6 | 714.4 | 5851.1 |
| NVIDIA RTX PRO 5000 Blackwell | off | longest | 5 | 28996.8 | 730.7 | 5799.4 |
| NVIDIA RTX PRO 5000 Blackwell | off | longest | 6 | 33217.9 | 974.8 | 5536.3 |
| NVIDIA RTX PRO 5000 Blackwell | off | longest | 8 | 43595.9 | 1428.6 | 5449.5 |
| NVIDIA RTX PRO 5000 Blackwell | off | longest | 10 | 52243.8 | 1587.0 | 5224.4 |
| NVIDIA RTX PRO 5000 Blackwell | off | longest | 12 | 62220.6 | 1732.4 | 5185.1 |
| NVIDIA RTX PRO 5000 Blackwell | off | longest | 15 | 76700.8 | 2326.4 | 5113.4 |
| NVIDIA RTX PRO 5000 Blackwell | off | longest | 20 | 101535.9 | 3253.5 | 5076.8 |
| NVIDIA RTX PRO 5000 Blackwell | off | longest | 25 | 127384.3 | 3370.0 | 5095.4 |
| NVIDIA RTX PRO 5000 Blackwell | off | longest | 30 | 149016.7 | 4332.0 | 4967.2 |
| NVIDIA RTX PRO 5000 Blackwell | off | middle | 1 | 6298.0 | 138.9 | 6298.0 |
| NVIDIA RTX PRO 5000 Blackwell | off | middle | 2 | 11147.9 | 308.7 | 5573.9 |
| NVIDIA RTX PRO 5000 Blackwell | off | middle | 3 | 15984.5 | 539.8 | 5328.2 |
| NVIDIA RTX PRO 5000 Blackwell | off | middle | 4 | 20859.0 | 566.2 | 5214.7 |
| NVIDIA RTX PRO 5000 Blackwell | off | middle | 5 | 26026.9 | 739.6 | 5205.4 |
| NVIDIA RTX PRO 5000 Blackwell | off | middle | 6 | 30775.1 | 911.0 | 5129.2 |
| NVIDIA RTX PRO 5000 Blackwell | off | middle | 8 | 39531.5 | 111.3 | 4941.4 |
| NVIDIA RTX PRO 5000 Blackwell | off | middle | 10 | 49362.6 | 1577.7 | 4936.3 |
| NVIDIA RTX PRO 5000 Blackwell | off | middle | 12 | 59146.5 | 1242.4 | 4928.9 |
| NVIDIA RTX PRO 5000 Blackwell | off | middle | 15 | 73253.8 | 1630.7 | 4883.6 |
| NVIDIA RTX PRO 5000 Blackwell | off | middle | 20 | 98492.3 | 3508.2 | 4924.6 |
| NVIDIA RTX PRO 5000 Blackwell | off | middle | 25 | 123271.1 | 3262.2 | 4930.8 |
| NVIDIA RTX PRO 5000 Blackwell | off | middle | 30 | 151205.7 | 2679.8 | 5040.2 |
| NVIDIA RTX PRO 5000 Blackwell | off | shortest | 1 | 5366.6 | 7.0 | 5366.6 |
| NVIDIA RTX PRO 5000 Blackwell | off | shortest | 2 | 9905.6 | 293.2 | 4952.8 |
| NVIDIA RTX PRO 5000 Blackwell | off | shortest | 3 | 14812.4 | 582.7 | 4937.5 |
| NVIDIA RTX PRO 5000 Blackwell | off | shortest | 4 | 19619.7 | 594.8 | 4904.9 |
| NVIDIA RTX PRO 5000 Blackwell | off | shortest | 5 | 24638.3 | 651.5 | 4927.7 |
| NVIDIA RTX PRO 5000 Blackwell | off | shortest | 6 | 29464.9 | 870.7 | 4910.8 |
| NVIDIA RTX PRO 5000 Blackwell | off | shortest | 8 | 39155.0 | 1093.3 | 4894.4 |
| NVIDIA RTX PRO 5000 Blackwell | off | shortest | 10 | 48297.7 | 1745.5 | 4829.8 |
| NVIDIA RTX PRO 5000 Blackwell | off | shortest | 12 | 59649.0 | 356.1 | 4970.7 |
| NVIDIA RTX PRO 5000 Blackwell | off | shortest | 15 | 72294.6 | 1818.6 | 4819.6 |
| NVIDIA RTX PRO 5000 Blackwell | off | shortest | 20 | 97155.9 | 2007.6 | 4857.8 |
| NVIDIA RTX PRO 5000 Blackwell | off | shortest | 25 | 120711.1 | 3070.0 | 4828.4 |
| NVIDIA RTX PRO 5000 Blackwell | off | shortest | 30 | 146213.7 | 467.7 | 4873.8 |
| NVIDIA RTX PRO 5000 Blackwell | on | longest | 1 | 9131.8 | 236.2 | 9131.8 |
| NVIDIA RTX PRO 5000 Blackwell | on | longest | 2 | 13967.4 | 470.8 | 6983.7 |
| NVIDIA RTX PRO 5000 Blackwell | on | longest | 3 | 18759.4 | 561.0 | 6253.1 |
| NVIDIA RTX PRO 5000 Blackwell | on | longest | 4 | 23363.1 | 778.4 | 5840.8 |
| NVIDIA RTX PRO 5000 Blackwell | on | longest | 5 | 28830.2 | 700.8 | 5766.0 |
| NVIDIA RTX PRO 5000 Blackwell | on | longest | 6 | 33188.3 | 983.5 | 5531.4 |
| NVIDIA RTX PRO 5000 Blackwell | on | longest | 8 | 43615.7 | 1410.3 | 5452.0 |
| NVIDIA RTX PRO 5000 Blackwell | on | longest | 10 | 52772.7 | 1434.2 | 5277.3 |
| NVIDIA RTX PRO 5000 Blackwell | on | longest | 12 | 62901.1 | 1675.6 | 5241.8 |
| NVIDIA RTX PRO 5000 Blackwell | on | longest | 15 | 77374.9 | 2338.7 | 5158.3 |
| NVIDIA RTX PRO 5000 Blackwell | on | longest | 20 | 100734.3 | 2940.0 | 5036.7 |
| NVIDIA RTX PRO 5000 Blackwell | on | longest | 25 | 125935.9 | 3380.3 | 5037.4 |
| NVIDIA RTX PRO 5000 Blackwell | on | longest | 30 | 148407.1 | 2875.2 | 4946.9 |
| NVIDIA RTX PRO 5000 Blackwell | on | middle | 1 | 5879.6 | 55.6 | 5879.6 |
| NVIDIA RTX PRO 5000 Blackwell | on | middle | 2 | 10666.6 | 293.5 | 5333.3 |
| NVIDIA RTX PRO 5000 Blackwell | on | middle | 3 | 15682.8 | 494.8 | 5227.6 |
| NVIDIA RTX PRO 5000 Blackwell | on | middle | 4 | 20001.2 | 513.6 | 5000.3 |
| NVIDIA RTX PRO 5000 Blackwell | on | middle | 5 | 24951.7 | 675.5 | 4990.3 |
| NVIDIA RTX PRO 5000 Blackwell | on | middle | 6 | 29966.8 | 948.9 | 4994.5 |
| NVIDIA RTX PRO 5000 Blackwell | on | middle | 8 | 40282.9 | 472.8 | 5035.4 |
| NVIDIA RTX PRO 5000 Blackwell | on | middle | 10 | 50028.5 | 1264.2 | 5002.9 |
| NVIDIA RTX PRO 5000 Blackwell | on | middle | 12 | 58818.6 | 1567.4 | 4901.6 |
| NVIDIA RTX PRO 5000 Blackwell | on | middle | 15 | 72235.2 | 831.0 | 4815.7 |
| NVIDIA RTX PRO 5000 Blackwell | on | middle | 20 | 97971.7 | 642.4 | 4898.6 |
| NVIDIA RTX PRO 5000 Blackwell | on | middle | 25 | 121000.1 | 3158.9 | 4840.0 |
| NVIDIA RTX PRO 5000 Blackwell | on | middle | 30 | 149410.8 | 2605.3 | 4980.4 |
| NVIDIA RTX PRO 5000 Blackwell | on | shortest | 1 | 5599.9 | 167.4 | 5599.9 |
| NVIDIA RTX PRO 5000 Blackwell | on | shortest | 2 | 10687.1 | 240.9 | 5343.6 |
| NVIDIA RTX PRO 5000 Blackwell | on | shortest | 3 | 15377.0 | 466.5 | 5125.7 |
| NVIDIA RTX PRO 5000 Blackwell | on | shortest | 4 | 20199.9 | 668.7 | 5050.0 |
| NVIDIA RTX PRO 5000 Blackwell | on | shortest | 5 | 24842.5 | 810.4 | 4968.5 |
| NVIDIA RTX PRO 5000 Blackwell | on | shortest | 6 | 29804.6 | 876.6 | 4967.4 |
| NVIDIA RTX PRO 5000 Blackwell | on | shortest | 8 | 39767.3 | 1179.7 | 4970.9 |
| NVIDIA RTX PRO 5000 Blackwell | on | shortest | 10 | 48227.3 | 606.4 | 4822.7 |
| NVIDIA RTX PRO 5000 Blackwell | on | shortest | 12 | 60426.4 | 1105.5 | 5035.5 |
| NVIDIA RTX PRO 5000 Blackwell | on | shortest | 15 | 73025.7 | 2135.4 | 4868.4 |
| NVIDIA RTX PRO 5000 Blackwell | on | shortest | 20 | 98547.0 | 2585.3 | 4927.4 |
| NVIDIA RTX PRO 5000 Blackwell | on | shortest | 25 | 123311.8 | 3054.9 | 4932.5 |
| NVIDIA RTX PRO 5000 Blackwell | on | shortest | 30 | 146724.9 | 4625.4 | 4890.8 |

![B3-2](../figures/B3_final-2_long.png)

![B3-3](../figures/B3_final-3_marginal.png)

## Table B3

| GPU | Mode | E(1) shortest / middle / longest (J) | Plateau (J/token) | k within 5% of plateau |
|---|---|---|---|---|
| NVIDIA RTX PRO 5000 Blackwell | off | 5.37 / 6.30 / 9.10 | 5.025 | 4 |
| NVIDIA RTX PRO 5000 Blackwell | on | 5.60 / 5.88 / 9.13 | 5.033 | 3 |
