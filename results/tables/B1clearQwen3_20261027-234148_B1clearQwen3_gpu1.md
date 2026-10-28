# CLEAR's method on Qwen3-8B (FP16): `20261027-234148_B1clearQwen3_gpu1`

GPU NVIDIA RTX 6000 Ada Generation (nvml 1), driver 560.35.03, idle 17.1 W; torch 2.8.0+cu128, transformers 4.57.1; status complete.
Method: CLEAR's BERT scripts (`clear_repo/`) adapted to a decoder (`b1/clear_qwen3/`); their measurement loop unchanged (no warm-up, no sync, 0.5 s pause, N = 10,000 parts / 1,000 model, T = 20). Layer 5. Layout as CLEAR Table 1. Energy in mJ per call.


| Component | 8 tokens: Avg. | Std. Dev. | 32 tokens: Avg. | Std. Dev. | 64 tokens: Avg. | Std. Dev. | 96 tokens: Avg. | Std. Dev. | 128 tokens: Avg. | Std. Dev. |
|---|---|---|---|---|---|---|---|---|---|---|
| Attention Block | 37.280 | 2.057 | 42.490 | 2.302 | 47.723 | 2.518 | 55.139 | 2.932 | 59.527 | 2.977 |
| MLP | 95.802 | 2.177 | 93.251 | 2.390 | 101.025 | 2.117 | 141.656 | 6.699 | 159.148 | 7.851 |
| Norm. (All) | 9.300 | 0.740 | 9.598 | 0.579 | 10.160 | 0.609 | 10.410 | 0.566 | 11.671 | 0.740 |
| Captured (Block) | 142.383 | - | 145.339 | - | 158.907 | - | 207.205 | - | 230.347 | - |
| Block | 158.702 | 5.084 | 183.986 | 5.882 | 192.963 | 6.638 | 190.578 | 4.798 | 204.650 | 6.713 |
| % Capture (Block) | 89.7% | - | 79.0% | - | 82.4% | - | 108.7% | - | 112.6% | - |
| Embedding Layer | 0.950 | 0.380 | 1.143 | 0.315 | 1.305 | 0.309 | 1.172 | 0.349 | 1.176 | 0.342 |
| LM Head | 346.745 | 5.972 | 377.581 | 3.086 | 404.885 | 4.845 | 689.222 | 30.531 | 664.995 | 30.994 |
| Final Layer Norm. | 4.482 | 0.350 | 4.584 | 0.446 | 4.977 | 0.459 | 4.918 | 0.485 | 5.189 | 0.422 |
| Captured (Model) | 6065.442 | - | 7006.813 | - | 7357.817 | - | 7556.107 | - | 8038.774 | - |
| Model | 6479.180 | 43.168 | 7724.580 | 59.162 | 8191.311 | 65.034 | 7983.610 | 77.376 | 8145.092 | 42.155 |
| % Capture (Model) | 93.6% | - | 90.7% | - | 89.8% | - | 94.6% | - | 98.7% | - |

## Time per call and average power

| Component | 8 tokens: µs | W | 32 tokens: µs | W | 64 tokens: µs | W | 96 tokens: µs | W | 128 tokens: µs | W |
|---|---|---|---|---|---|---|---|---|---|---|
| attention | 443 | 84 | 447 | 95 | 451 | 106 | 429 | 129 | 431 | 138 |
| mlp | 358 | 268 | 350 | 266 | 378 | 267 | 528 | 268 | 591 | 269 |
| input_layernorm | 62 | 80 | 61 | 83 | 63 | 83 | 61 | 91 | 62 | 96 |
| post_attention_layernorm | 62 | 70 | 61 | 74 | 62 | 79 | 60 | 81 | 63 | 91 |
| block | 722 | 220 | 714 | 258 | 742 | 260 | 718 | 266 | 765 | 268 |
| embeddings | 14 | 69 | 16 | 69 | 18 | 73 | 17 | 71 | 17 | 71 |
| lm_head | 1289 | 269 | 1404 | 269 | 1504 | 269 | 2568 | 268 | 2482 | 268 |
| final_norm | 62 | 72 | 63 | 72 | 63 | 79 | 61 | 80 | 63 | 82 |
| full_model | 29718 | 218 | 30646 | 252 | 31113 | 263 | 29926 | 267 | 30382 | 268 |
