# qwench

Small experiments on feature-specific perturbation geometry in Qwen3-1.7B and
Qwen3.5-4B, following [Ferreira da Silva & Heimersheim](https://arxiv.org/abs/2606.24964).

The sampler audit recovers the strong 1.7B positive case: p = 2.442 versus
published 2.415. Our first attempt used different anchors because the streaming
shuffle changed between `datasets` versions. The historical-anchor 4B comparison
is still pending; the existing 4B results are exploratory.

- [Sampler audit and findings](reports/implementation-audit-v1/findings.md)
- [Exploratory 4B findings](reports/stress-4b-v1/findings.md)
- [Activations trained from scratch: sharing versus p](reports/toy-scalar-comparison-v1/findings.md)
- [Local toy-CiS pilot](reports/toy-cis-v1/findings.md)
- [Reproduction notes and experiment index](docs/reproduction.md)
- [Original implementation](https://github.com/FranciscoHS/fsec-paper)

These measurements do not establish an error-correction mechanism.
