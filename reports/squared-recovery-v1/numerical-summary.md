# Primary fixed-target corruption scores

Input noise std 0.075; 2048 clean examples × 4 paired draws.

| Model | All supported MSE | Positive-active MSE | Absent MSE | Positive-active increase from clean |
|---|---:|---:|---:|---:|
| Task on noisy input | 0.0028184 | 0.0056653 | 0.0028166 | 0.0056653 |
| Trained squared ReLU | 0.0004188 | 0.0321375 | 0.0000879 | 0.0111265 |
| Independent control | 0.0003974 | 0.0313008 | 0.0000749 | 0.0107979 |

All supported = the same 50 competent output features. Full 100-output errors, oracle ablations,
zero-noise errors, all noise levels, flags and paired bootstrap differences remain in the saved JSON.
