# qwench

Small experiments on **what feature geometry tells us about error correction**.

Imagine two knobs inside a neural network. Turn them separately and together,
then draw the boundary where the response changes by a chosen amount. After
rescaling the axes, the shape score **p** is 2 for a circle and larger for
squarer shapes.

![Measured toy boundaries: nearly circular with ReLU, squarer with squared ReLU](docs/images/toy-boundaries.png)

*One measured slice from two trained networks, using the same feature pair.
Both learned roughly one feature per neuron. Positive directions only;
axes normalized to their individual cutoffs. This is an illustration, not an
activation ranking.*

**Our working conclusion:** p describes conditional nonlinear sensitivity.
It can rise when a model learns to denoise, but it does not by itself measure
how much recovery occurs or identify the mechanism. Independent featurewise
computations can also denoise.

## Start here

- [Reading guide](docs/reading-guide.md): geometry, recovery, mechanism, and a
  small mathematical example showing how p > 2 can arise.
- [Training and recovery](reports/toy-characterization-study-v1/findings.md):
  120 fresh toy networks, matched controls, five activations.
- [AND/XOR](reports/toy-logic-v1/findings.md): 60 fresh networks; logical
  interactions, cancellation, and cases where the p fit fails.
- [Weights written from formulas](reports/toy-handwritten-v1/findings.md):
  explicit networks compared with trained models. This study measures task
  performance, not p.

The [experiment index](docs/experiments.md) includes earlier runs and corrections.

## LLM status

For Qwen3-1.7B, correcting an anchor-sampling mismatch recovered p = 2.442 versus
the paper's 2.415 ([audit](reports/implementation-audit-v1/findings.md)).
[Qwen3.5-4B results](reports/stress-4b-v1/findings.md) remain exploratory;
the comparison using historical anchors is unfinished.

## Try it

Python **3.12**, CPU only. From a checkout, redraw the AND/XOR figures:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/figures.txt
.venv/bin/python scripts/plot_logic.py --out work/reproductions/logic-figures
```

[Lightweight checkout](docs/reproduction.md#lightweight-andxor-figure-checkout)
(about 20 MB of figure inputs) · [Full reproduction guide](docs/reproduction.md).
Raw grids, checkpoints, protocols and fit flags are retained in `reports/`.

Starting point: Ferreira da Silva & Heimersheim,
[*Evidence for Feature-Specific Error Correction in LLMs*](https://arxiv.org/abs/2606.24964)
([reference implementation](https://github.com/FranciscoHS/fsec-paper)).
