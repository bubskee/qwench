# Numerical summary

Final checkpoints; means over five activations and three initialization seeds. Noise std 0.35. MSE targets remain clean logical labels.

| Task / training | Raw MSE | Clipped MSE | Projected + clipped MSE | Valid within fits | Valid across fits |
|---|---:|---:|---:|---:|---:|
| and_clean | 0.217444 | 0.101878 | 0.064454 | 0/15 | 15/15 |
| and_denoise | 0.058498 | 0.057389 | 0.057386 | 0/15 | 15/15 |
| xor_clean | 0.500006 | 0.197055 | 0.144804 | 0/15 | 7/15 |
| xor_denoise | 0.110538 | 0.107952 | 0.107933 | 0/15 | 15/15 |

Noise-trained activations; each cell uses three initialization seeds. Geometry uses the across pair at the prespecified 00/background-3 slice. Only unflagged p values contribute to p means; validity counts are shown.

| Activation | AND p (valid) | AND projected/clipped MSE | XOR p (valid) | XOR projected/clipped MSE |
|---|---:|---:|---:|---:|
| ReLU | 3.746 (3/3) | 0.056956 | 4.538 (3/3) | 0.106145 |
| GELU | 3.117 (3/3) | 0.057499 | 2.835 (3/3) | 0.108095 |
| SiLU | 2.635 (3/3) | 0.058973 | 2.748 (3/3) | 0.108917 |
| Leaky ReLU | 4.287 (3/3) | 0.056474 | 4.459 (3/3) | 0.106241 |
| Squared ReLU | 2.686 (3/3) | 0.057028 | 2.717 (3/3) | 0.110268 |

Projected/clipped class errors:

| Task / training | Positive MSE | Negative MSE | Class-balanced MSE |
|---|---:|---:|---:|
| and_clean | 0.110102 | 0.049148 | 0.079625 |
| and_denoise | 0.127187 | 0.033983 | 0.080585 |
| xor_clean | 0.139254 | 0.150314 | 0.144784 |
| xor_denoise | 0.107629 | 0.108235 | 0.107932 |
