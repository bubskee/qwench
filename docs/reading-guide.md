# What does a square boundary tell us?

This repo started with [Ferreira da Silva & Heimersheim's FSEC paper](https://arxiv.org/abs/2606.24964).
The geometric finding is interesting on its own: candidate feature directions
can behave differently from mixtures of those directions. Our experiments ask
how far that observation takes us toward an explanation of error correction.

The useful distinction is between **sensitivity**, **recovery**, and
**mechanism**. We measure the first two. The third needs additional evidence.

## Three questions

**Geometry:** how much does the network's output change when we perturb two
candidate feature directions? For a toy map f and a fixed anchor x, we measure

$$
R(a,b)=\|f(x+a u+b v)-f(x)\|_2.
$$

Choose a response threshold, find its contour, normalize by the two axis
intercepts A and B, and fit

$$
\left|\frac{a}{A}\right|^p+\left|\frac{b}{B}\right|^p=1.
$$

Here p=2 is a circle in normalized coordinates; larger p is squarer. The fitted
number depends on the anchor, directions, signs, threshold and probe range.
Some response maps are poorly described by this family. A failed fit is not
evidence that the model has no useful features or cannot denoise.
A tilted quadratic response can also depart from this axis-aligned family;
the detailed fits retain a tilted-quadratic comparison.

**Recovery:** after adding unwanted noise, how close is the answer to the
original clean target? We score

$$
\mathbb E\left[\frac{1}{d}\|f(X+\epsilon)-Y\|_2^2\right],
$$

where d is the output dimension and Y stays fixed while the input is corrupted.
We compare clean error, informed
references and controls, and errors on active/positive versus inactive/negative
examples. Lower average error can hide worse preservation of positive signals.

**Mechanism:** which computation produces that recovery? An independent
threshold on each feature can denoise. A model can also use information about
neighboring features. Distinguishing those explanations requires controls or
interventions that address the proposed mechanism. Neither p nor a recovery
score settles that question by itself.

## A small mathematical example

Suppose positive perturbations change two separate output coordinates as

$$
\Delta f(a,b)=(c_1 a^k,\;c_2 b^k),\qquad a,b\geq0,
$$

with positive c₁, c₂ and k. Then

$$
R(a,b)^2=c_1^2a^{2k}+c_2^2b^{2k}.
$$

After normalizing a fixed-response contour by its axis intercepts, its exponent
is **p=2k**. Linear responses give p=2; independent quadratic responses give
p=4. No interaction between the two feature computations is required for this
construction.

This does not mean that a squarer contour is a false positive for feature
selectivity, or that an independent map cannot correct noise. It shows why the
shape alone does not identify a particular correction mechanism. It also does
not predict p from an activation's name: learned mixtures, biases and nonzero
anchors need not obey this simple power law.

## What the experiments add

1. **[Training and recovery](../reports/toy-characterization-study-v1/findings.md).**
   Across 120 fresh networks, noise training often raises p. A control preserving
   isolated-feature response curves can have similar p while performing worse
   when recovery benefits from neighboring features. That control can itself
   perform scalar denoising. Background activity also changes p with weights
   held fixed.
2. **[AND/XOR](../reports/toy-logic-v1/findings.md).** The exact AND posterior
   denoiser gives p≈0.84 for two input bits feeding one output and p≈4.50 for
   bits feeding separate outputs. XOR can produce cancellation and multiple
   response regions. All final learned-model primary fits for the same-output
   probe are flagged; their nominal numbers are not treated as reliable
   exponents. The underlying input bits need not be independently movable
   logical output features.
3. **[Formula-written weights](../reports/toy-handwritten-v1/findings.md).**
   Explicit weight constructions let us compare known computations with trained
   models on fresh samples. This is a task-performance comparison; it does not
   measure p.

Each report links its protocol, raw results and checks. Geometry uses the
network's own anchor output; recovery uses the clean task target. Where recovery
scores use projection/clipping controls, the reports say so. Projection preserves
the tested feature-plane responses; clipping is not assumed to preserve them.

## Limits and LLM status

These are small, exploratory toy studies with a few initialization seeds and
fixed probe pairs. The newer coded-bit networks have more hidden units than
latent features, so they do not force computation in superposition. The results
do not establish a universal activation ranking or a scalar amount of correction.

Our additive input/output toy probes also differ from the paper's
norm-preserving interventions between LLM residual layers. The AND/XOR examples
are not paper-compatible counterexamples to its LLM result.

For LLM reproduction, start with the [corrected Qwen3-1.7B sampler audit](../reports/implementation-audit-v1/findings.md).
The [4B stress test](../reports/stress-4b-v1/findings.md) is exploratory; its
historical-anchor comparison is unfinished. R/J++ lens observations are a
separate extension, not a completed result of these toy measurements.

[Reproduce the figures](reproduction.md#lightweight-andxor-figure-checkout) ·
[Experiment index](experiments.md)
