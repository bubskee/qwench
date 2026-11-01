# Fresh-model generalization stress test

This run tests the transfer of the paper's early-layer contrastive-direction
geometry procedure to Qwen3.5-4B. It is not another replot of the paper's data.
The candidate prompts and protocol were committed in `aa544d4`; the measured
eligibility and sealed task list were committed in `bfd69e1`, before the first
new response surface was fitted. The capture implementation is `ac3925f`.
Analysis implementation was added after initial surfaces began arriving.

## What was fixed

- One pinned Qwen3.5-4B checkpoint; plain text with no chat template; inference
  mode; native decoder forwards on MPS. No model or lens downloads.
- Last-token residual output of zero-based block 2 perturbed; L2 change at
  block 30 measured in FP32. The primary forward uses BF16.
- Candidate direction is mean(B-A) from matched prompt lists. The 33 original
  contrasts are copied unchanged from the pinned paper registry. Four extra
  contrasts are cold/hot, small/large, slow/fast, and below/above, with 30
  matched prompts each. Their labels are candidate interpretations, not
  demonstrated isolated latent features.
- Raw absolute cosine < 0.1 for features and controls. Rejected controls are
  reported rather than regenerated. Tangent projection at each anchor and
  Gram-Schmidt follow the paper's norm-preserving exponential-map intervention.
- Eight seed-selected English DCLM anchor documents, first five tokens each.
  No token-window or anchor selection by exponent. Remaining 248 documents
  form the disjoint random-activation control pool.
- Cartesian component angles 0..60 degrees in steps of 5; the corner has
  radial angle sqrt(60^2 + 60^2). No small-angle approximation in fitting.
- Pair-specific threshold is half the smaller axis maximum of the median
  response surface. Axis crossings normalize the contour, then the original
  one-parameter exact-geodesic superellipse fitter estimates p.
- Seeded strata samples, fixed challenges, and a seeded direction-coverage
  rule choose the cohort before sweeping. These produce 33 feature pairs
  covering every one of the 37 candidate directions, plus seven controls.
- Three isotropic controls, two accepted mean-of-30 random activation
  difference controls, and two accepted balanced random-label controls using
  the same prompts as Gender/Wealth and Sentiment/Spanish. Random-label
  controls are not guaranteed free of real feature content.
- Three accepted fixed challenge pairs get two Spanish and two code anchors,
  plus three other sign quadrants on the first four English anchors.
- Gender/Wealth and the first isotropic pair get a 2.5-degree grid and a full
  FP32-forward check. These pairs were selected by rule before their p values.

The exact 56-task plan, signs, raw cosines, exclusions, labels, and anchors
are in `plan.json`. Every task is retained. Source snapshots, seals, raw
surfaces, receipts, and analysis records are saved alongside it.

## What differs from the original study

This is one additional model at one layer pair, eight anchors rather than
30, DCLM rather than FineWeb, a 5-degree grid rather than the paper's finer
primary grid, and 248 activation-control documents rather than 10,000.
The model architecture and training differ; no inference about all models
follows from success or failure here. The four new candidate directions and
the direction-coverage additions are exploratory. The cohort is deliberately
a stress cohort, not an unbiased population sample of features or pairs.
The stratified subset is summarized separately. Axis order may differ from
the cached Gemma comparisons; despite the low cosine cutoff, orthogonalization
makes those comparisons descriptive rather than perfectly matched.

There is no output-steering or lens-readout gate for inclusion. Earlier
sentiment/language steering checks were a separate experiment. The present
measurement tests this specific direction-construction recipe without
selectively admitting directions that produce attractive text completions.
R/J++ readouts are deferred and do not enter any p selection or fit.

## Interpretation and checks

p > 2 is a geometric observation. Feature specificity additionally requires
separation from suitable controls; correction as a mechanism requires
additional evidence. Ordinary anisotropy, cross-coupling, and nonlinear
sensitivity remain possible explanations. A single poor fit below 2 is not
by itself a generalization failure. Fit flags, alternative tilted-quadratic
fits, anchor dependence, threshold changes, sign dependence, and numerical
diagnostics are all reported rather than used to curate the primary cohort.

Numerical checks include exact zero-angle response, batch-size-matched clean
baselines, bounded norm error, exact early-stop/full-forward capture parity,
raw-file/source hashes, nonnegative finite responses, disjoint anchor/control
documents, and synthetic known-p recovery at both grid spacings. They do not
establish semantic purity of the candidate directions.

Anchor-bootstrap intervals describe the fixed cohort's sensitivity to the
eight sampled anchors. Shared direction-cluster resampling is conditional
on the tested pair graph, not a population-of-models confidence interval.
Pair counts do not imply independent observations; controls are too few to
justify strong equivalence claims. Low-residual fits are shown alongside all
fits, not used to discard failures from the headline result.

The final anchor bootstrap uses 500 shared resamples. Additional analysis of
the already fixed diagnostic surfaces also holds the matched primary L2
threshold constant across conditions. This separates threshold recalibration
from sign, context, grid, and precision changes. It was added after inspecting
the first sign results; the underlying model runs were fixed beforehand.
