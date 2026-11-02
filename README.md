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

We also trained tiny networks to ask a question: does a squarer perturbation
boundary actually tell us that neurons share computation? The task has 100 sparse
input features and only 50 hidden neurons. We compared ReLU, GELU, SiLU, Leaky
ReLU, and squared ReLU, each trained with its own weights.

Five activations, two losses, and three initialization seeds gave 30 networks,
each trained for 100,000 updates. Matching runs used identical data, minibatches,
and starting matrices. Under L2, all five families learned 50 features accurately,
with roughly one feature per neuron. Under L4, four families learned all 100
through shared neurons.

The fun part: squared ReLU reached p = 4.04 under L2 while remaining specialized,
compared with 2.15 for matched random directions. GELU and SiLU had higher average
p than ReLU under L4, but similar measured sharing and overlapping seed ranges.
A larger exponent therefore does not reliably mean more sharing.

Squared ReLU also learned the L4 task less accurately; none of its features passed
our declared competence cutoff. That matters when comparing geometry. The reports
retain failed fits, feature errors, controls, checkpoints, and reproduction
instructions. These small toy experiments test interpretations; a general
activation ranking would need broader experiments across tasks.
