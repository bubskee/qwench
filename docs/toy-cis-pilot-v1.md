# Learned CiS and perturbation geometry: small pilot

Question: does the FSEC contour statistic distinguish these released learned
superposition candidates from simpler nonlinear and linear controls?

We use the authors' released 100→50 L2/L4 checkpoints and an exact dedicated
100→100 ReLU map, all on CPU. Checkpoint labels identify training objectives;
we verify their per-feature behavior instead of assuming L2 means no CiS.
The target is ReLU(x), with Bernoulli(0.02) signed-uniform input features.

The primary intervention is additive at the pre-ReLU 50-neuron bottleneck.
Directions are actual encoder columns. Adding a column is exactly equivalent
to adding the corresponding input feature: this is a sensitivity assay, not a
corruption/recovery experiment. We preserve overlapping codewords; matched
random directions share their norms and cosine. Four signs, three pair activity
states, and four sparse backgrounds make context dependence visible.

A linear decoder produces a quadratic *squared* response, but its ellipse need
not align with the selected axes. Therefore a fitted axis-aligned p need not
be 2 even for a linear map. We fit a tilted quadratic on the same contour and
include post-ReLU decoder-only responses plus explicit tilted-linear controls.
A separable square map is a known-p=4 geometry control without superposition.

The original fitter expects angular chart units and has a one-unit edge
exclusion. We rescale the additive amplitude axis to 0..60 and turn off its
geodesic correction. This leaves normalized additive contours unchanged and
avoids treating a hardcoded one-degree exclusion as one full feature unit.

Preparation selects six common-eligible pairs using only one-feature task
behavior and a fixed seed. Eligibility/exclusions and all 100 feature responses
are saved before measuring p. No ignored features are silently treated as
quadratic plateaus. Both protocol and prepared selection are committed first.

Separately, Gaussian hidden corruption keeps the original clean target fixed.
We report positive-active error and inactive false positives under natural
ReLU, clean-input frozen gates, and a decoder-only control. These share clean
outputs and noise realizations. Gate ablations may operate off the natural
hidden manifold; they test this intervention, not a universal mechanism.

The existing freshly recomputed baked-FSEC toy result, p≈2.907, remains a
separate procedural reference in `reports/paper-audit/toy-boundary`. We do not
retrain it or equate its intervention space with the learned toys'.

Interpretation: a clean p difference is an association in these two checkpoints.
High p in a dedicated or tilted-linear control refutes specificity of the scalar
statistic in that setting. Failure to see high p in a learned toy would show
that this assay is not a universal detector of that learned representation; it
would not disprove CiS. Geometry and corruption accuracy must be reported even
when they disagree. Six pairs and one checkpoint per objective are exploratory.

Stopping rule and exact settings: `configs/toy-cis-v1.json`. Nothing new is
published remotely as part of this local pilot.
