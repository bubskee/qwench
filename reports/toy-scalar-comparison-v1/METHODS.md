# Five activations trained separately on the same sparse-feature task

The active comparison is ReLU, exact GELU, SiLU, Leaky ReLU (negative slope 0.1),
and squared ReLU. Every network has 100 inputs/outputs, 50 hidden neurons,
10,000 weights and no biases. Each activation is independently trained under L2
and L4 loss with initialization seeds 0, 1 and 2: **30 scalar networks**.
No trained ReLU weights are used as frozen weights for another activation.

## Training and provenance

The recipe follows the [released CiS toy training code](https://github.com/FranciscoHS/toy-model-cis-code/blob/8c827b8c07bcf39097ee21707c62641a9201a216/train_from_scratch.py):
800,000 fixed sparse examples; each of 100 signed-uniform features is present
independently with probability 0.02; target ReLU(x); 100,000 Adam updates,
batch 8,192, learning rate 0.01 with cosine decay to zero, default betas and
epsilon, and no weight decay. Encoder weights start uniform ±0.1 and decoder
weights ±0.15. Matching seeds share the exact initial matrices across activation
functions and objectives. A separate 65,536-example set measures task behavior.
All networks see the same training pool and minibatch sequence. Seeds vary
initialization, not data. Training uses native Mac MPS float32.

Two tracked cohorts supply the comparison:

- `toy-scalars-core-v1`: 18 ReLU/GELU/SiLU networks. They began in the original
  mixed-architecture cohort; after the user abandoned SwiGLU, the retained
  networks continued from their saved 30,000-step checkpoint with weights,
  Adam moments, schedule, and RNG state preserved. Overlapping continued losses
  match the original execution bit for bit (`continuation-parity.json`).
- `toy-scalar-additions-v1`: 12 fresh Leaky-ReLU/squared-ReLU networks, selected
  by the user and trained from initialization for the same full budget.

The original unfinished SwiGLU cohort is stopped and excluded. Its partial
records remain as provenance. The two scalar cohorts have separate immutable
protocols and source hashes. Assembly copies their final weights and saved
evaluations without new inference; each network retains its full training
history. Batched execution sums per-network mean losses, preserving standalone
Adam gradient scale. Gradient and parameter-count checks are recorded.

Squared ReLU is a declared **curvature control**: a bias-free squared-ReLU
network is positively homogeneous of degree two, so it cannot exactly reproduce
the degree-one ReLU target across all input amplitudes. All five functions use
the same task; this does not rank their performance on arbitrary tasks or test
grokking in modular arithmetic.

## Task competence and functional neuron sharing

A competent feature has held-out positive-active MSE <0.05, a unit-probe
self-gain ≥0.25, and the largest probe output at its own feature. We retain all
100 feature errors and gains, including failures; the cutoff is not treated as
a binary distinction between learning and learning nothing. Inactive leakage,
false positives >0.1, negative-present error and isolated/multi-input active
errors are also saved.

For isolated unit feature j, define the hidden-unit contribution to output j as
q[j,n] = W_out[j,n] * h_n(e_j). All selected activations have h(0)=0. Contribution
participation is `(sum(abs(q)))² / sum(q²)`: over neurons for each feature, and
over competent features for each neuron. We save the vectors and show both
coverage and sharing. Distributed contributions alone can include cancellation
or redundancy and are not a universal amount of CiS. Calibration on the released
specialized L2 model gives about one feature per neuron; the released L4 codeword
model gives about ten. The exact independent-neuron control gives one.

## Geometry

A deterministic ordering of all feature pairs and four sparse backgrounds was
frozen before the relevant feature/p outcomes. Six fixed index pairs are tried
in every network. After final task evaluation and before measuring p, each
network additionally gets the first three competent pairs in that ordering.
We retain shortfalls and distinguish fixed, competent-fixed, and competence-
conditioned cohorts. The latter's feature indices can differ across networks.

Perturbations are additive encoder-column directions before the nonlinearity;
random controls match both norms and cosine. Four backgrounds, pair inputs
0/+0.5/−0.5, all four signs, and a 41×41 amplitude grid from 0 to 1.5 follow the
previous toy pilot. Responses are L2 distances to each network's own clean
output, computed in CPU float64. Encoder-column additions equal corresponding
input-feature changes; this is sensitivity, not fixed-target error recovery.
Some settings extrapolate beyond the training support. The negative-anchor ++
case stays within input amplitudes −0.5..1.

The median background surface uses threshold fractions 0.25/0.5/0.75 of the
smaller maximum axis response, with 0.5 primary. The original additive
superellipse estimator and a PSD tilted quadratic fit identical contour
points. Coordinates are rescaled to chart units 0..60 for the original fitter's
edge exclusion, with no geodesic sine correction. Per-background fits, flat
axes, weak responses, nonmonotonicity, fit bounds and residual flags remain in
the records. The first competent pair per network gets prespecified 81×81
checks at the negative anchor, ++ sign, for encoder and random directions.

The headline geometry plots show the declared ++ slice; all quadrants remain
in the saved fits/CSV. Network points summarize up to three competent pairs.
Three initialization seeds and shared pairs do not justify population-level
significance claims. Accurate models of the same target can have similar input-
feature contours despite different internals, so p is interpreted separately
from competence and functional sharing.

## Reproduction

Dependencies are in `requirements-toy.txt`. The historical local layout places
qwench in `.../outputs/qwench` and scratch data/checkpoints in `.../work/`.
Each source trainer can run from initialization in a fresh output directory;
exact interruption resume additionally uses its scratch optimizer/RNG state.
Completed directories reject silent overwrites. The dataset SHA-256, source
hashes, final weights, saved evaluations and raw response grids are retained.

```bash
python scripts/train_scalar_core.py          # fresh full core training
python scripts/train_scalar_additions.py     # fresh full additions training
python scripts/assemble_scalar_comparison.py
export QWENCH_TRAINED_DRIVER=scalar_comparison_runtime
python scripts/check_trained_geometry.py
python scripts/geometry_trained_toys.py --stage prepare
# Inspect/commit task-conditioned pair selection before measuring p.
python scripts/geometry_trained_toys.py --stage sweep
python scripts/fit_trained_toys.py
python scripts/plot_trained_toys.py
python scripts/audit_trained_toys.py
```

These notes describe the frozen design. Completion markers and the final
findings identify which stages actually finished. The completed training
artifacts and measurement grids are included with this report.
