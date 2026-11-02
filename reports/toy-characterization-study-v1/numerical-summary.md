# Final checkpoint descriptive summaries

Mean over three initialization seeds; p is each network’s median across two fixed pairs.
Nominal p includes flagged fits; the good-fit column shows how many of the six pair fits are unflagged.

| Family | Task | Activation | p | Null p | Good fits / 6 | Noisy MSE | Null MSE | Bayes MSE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| coupled | context_clean | gelu | 2.0051 | 2.0192 | 6 | 0.3636 | 0.1833 | 0.0405 |
| coupled | context_denoise | gelu | 4.5344 | 4.5825 | 6 | 0.0431 | 0.0535 | 0.0405 |
| coupled | linear_clean | gelu | 2.0547 | 2.0764 | 6 | 0.2419 | 0.1685 | 0.0394 |
| coupled | linear_denoise | gelu | 5.1724 | 5.1490 | 0 | 0.0402 | 0.0541 | 0.0394 |
| coupled | context_clean | leaky_relu | 1.9848 | 1.9991 | 6 | 0.2759 | 0.1717 | 0.0405 |
| coupled | context_denoise | leaky_relu | 4.1593 | 4.2142 | 6 | 0.0441 | 0.0539 | 0.0405 |
| coupled | linear_clean | leaky_relu | 1.9990 | 1.9991 | 6 | 0.1890 | 0.1420 | 0.0394 |
| coupled | linear_denoise | leaky_relu | 6.3902 | 6.3329 | 6 | 0.0403 | 0.0542 | 0.0394 |
| coupled | context_clean | relu | 1.9837 | 1.9979 | 6 | 0.2614 | 0.1676 | 0.0405 |
| coupled | context_denoise | relu | 4.1999 | 4.3109 | 6 | 0.0437 | 0.0538 | 0.0405 |
| coupled | linear_clean | relu | 1.9990 | 1.9991 | 6 | 0.1810 | 0.1349 | 0.0394 |
| coupled | linear_denoise | relu | 6.2557 | 6.2054 | 6 | 0.0405 | 0.0541 | 0.0394 |
| coupled | context_clean | relu2 | 2.2520 | 2.2636 | 6 | 0.5435 | 0.1551 | 0.0405 |
| coupled | context_denoise | relu2 | 4.2859 | 4.3362 | 6 | 0.0438 | 0.0535 | 0.0405 |
| coupled | linear_clean | relu2 | 2.7004 | 2.7424 | 6 | 0.2415 | 0.1293 | 0.0394 |
| coupled | linear_denoise | relu2 | 4.7722 | 4.7656 | 6 | 0.0409 | 0.0538 | 0.0394 |
| coupled | context_clean | silu | 2.0032 | 2.0050 | 6 | 0.3481 | 0.1923 | 0.0405 |
| coupled | context_denoise | silu | 3.8646 | 3.9789 | 6 | 0.0438 | 0.0537 | 0.0405 |
| coupled | linear_clean | silu | 2.0005 | 2.0043 | 6 | 0.2575 | 0.1877 | 0.0394 |
| coupled | linear_denoise | silu | 4.6622 | 4.5940 | 5 | 0.0407 | 0.0536 | 0.0394 |
| historical | L2 | gelu | 1.5933 | 1.3588 | 6 | 0.0024 | 0.0024 | — |
| historical | L4 | gelu | 1.8835 | 2.2043 | 6 | 0.0051 | 0.0012 | — |
| historical | L2 | leaky_relu | 1.9967 | 1.6109 | 6 | 0.0028 | 0.0027 | — |
| historical | L4 | leaky_relu | 1.9174 | 1.9990 | 6 | 0.0053 | 0.0020 | — |
| historical | L2 | relu | 2.0046 | 1.6112 | 6 | 0.0027 | 0.0027 | — |
| historical | L4 | relu | 2.0277 | 1.9990 | 6 | 0.0052 | 0.0020 | — |
| historical | L2 | relu2 | 2.0743 | 1.8998 | 1 | 0.0016 | 0.0016 | — |
| historical | L4 | relu2 | 2.8292 | 3.9965 | 6 | 0.0041 | 0.0007 | — |
| historical | L2 | silu | 1.7105 | 1.5389 | 6 | 0.0024 | 0.0024 | — |
| historical | L4 | silu | 2.5231 | 2.2773 | 6 | 0.0051 | 0.0013 | — |
| orthogonal | context_clean | gelu | 2.0136 | 2.0290 | 6 | 0.3056 | 0.1170 | 0.0335 |
| orthogonal | context_denoise | gelu | 5.2507 | 5.1614 | 0 | 0.0341 | 0.0340 | 0.0335 |
| orthogonal | linear_clean | gelu | 2.0596 | 2.0862 | 6 | 0.1760 | 0.1074 | 0.0326 |
| orthogonal | linear_denoise | gelu | 6.0657 | 5.9161 | 0 | 0.0327 | 0.0326 | 0.0326 |
| orthogonal | context_clean | leaky_relu | 1.9979 | 2.0020 | 6 | 0.2186 | 0.1125 | 0.0335 |
| orthogonal | context_denoise | leaky_relu | 4.7920 | 4.7230 | 6 | 0.0351 | 0.0347 | 0.0335 |
| orthogonal | linear_clean | leaky_relu | 1.9990 | 1.9991 | 6 | 0.1371 | 0.0968 | 0.0326 |
| orthogonal | linear_denoise | leaky_relu | 6.1977 | 6.1124 | 6 | 0.0327 | 0.0327 | 0.0326 |
| orthogonal | context_clean | relu | 1.9977 | 2.0029 | 6 | 0.2069 | 0.1107 | 0.0335 |
| orthogonal | context_denoise | relu | 4.8887 | 4.7977 | 6 | 0.0352 | 0.0348 | 0.0335 |
| orthogonal | linear_clean | relu | 1.9990 | 1.9990 | 6 | 0.1321 | 0.0933 | 0.0326 |
| orthogonal | linear_denoise | relu | 6.2411 | 6.1608 | 6 | 0.0327 | 0.0327 | 0.0326 |
| orthogonal | context_clean | relu2 | 2.3998 | 2.4166 | 6 | 0.3794 | 0.0891 | 0.0335 |
| orthogonal | context_denoise | relu2 | 4.7227 | 4.6631 | 4 | 0.0355 | 0.0350 | 0.0335 |
| orthogonal | linear_clean | relu2 | 2.9888 | 3.0069 | 6 | 0.1532 | 0.0757 | 0.0326 |
| orthogonal | linear_denoise | relu2 | 6.1123 | 5.9704 | 2 | 0.0331 | 0.0331 | 0.0326 |
| orthogonal | context_clean | silu | 2.0039 | 2.0014 | 6 | 0.2798 | 0.1232 | 0.0335 |
| orthogonal | context_denoise | silu | 4.6457 | 4.5392 | 1 | 0.0351 | 0.0348 | 0.0335 |
| orthogonal | linear_clean | silu | 2.0314 | 2.0365 | 6 | 0.1803 | 0.1194 | 0.0326 |
| orthogonal | linear_denoise | silu | 5.5004 | 5.3167 | 0 | 0.0329 | 0.0329 | 0.0326 |

Noise std: 0.35 for new coded-bit tasks, 0.075 for the separate historical signed-sparse task.
The analytic null knows the true feature coordinates and preserves the trained isolated response curves.
See projection-matched diagnostics and per-seed CSV before assigning a mechanistic interpretation.
