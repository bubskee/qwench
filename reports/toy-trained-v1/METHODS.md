# Training activation variants from scratch

**Stopped and superseded:** The user abandoned SwiGLU. The scalar networks continue from the step-30,000 checkpoint in `../toy-scalars-core-v1`; two new scalar controls train separately in `../toy-scalar-additions-v1`. This original 30-network cohort was not completed. Its partial records are retained as provenance.

This experiment trains each architecture separately. It replaces the abandoned
fixed-weight activation-swap idea; those scratch results are excluded here.

## Frozen design

`configs/toy-trained-v1.json` was committed as `ae009ef` before training. The
training implementation was committed as `1d9670e` before its first update.
The recipe follows [the released toy training script](https://github.com/FranciscoHS/toy-model-cis-code/blob/8c827b8c07bcf39097ee21707c62641a9201a216/train_from_scratch.py):
100 sparse signed features, target ReLU(x), activation probability 0.02, an
800,000-example fixed training pool, batch 8,192, Adam at 0.01 with cosine
annealing, and exactly 100,000 updates. There are no biases or weight decay.
We adapt execution to native Mac MPS float32 and use a separately seeded
65,536-example held-out set. This is not an exact historical training replay.

The five configurations are ReLU50, exact GELU50, SiLU50, full SwiGLU33 and full
SwiGLU50. ReLU/GELU/SiLU have 10,000 weights. SwiGLU33 has 9,900 and SwiGLU50
has 15,000. SwiGLU is `SiLU(x W_gate.T) * (x W_value.T)` followed by W_out;
gate and value projections are independently initialized and trained. The
33-channel model approximately matches parameter count; the 50-channel model
matches hidden width with 50% more parameters. Neither comparison matches all
aspects of architecture or computation cost.

Each architecture uses three initialization seeds under both L2 and L4 loss:
**30 networks**, trained for the full budget without early stopping or per-model
optimizer tuning. Role-matched encoder/decoder initial values are shared across
architectures/objectives, truncated for width 33; SwiGLU has an additional
independently drawn value matrix. Initialization scales are the source recipe's
uniform ±0.1 for encoders and ±0.15 for decoders, with no activation-specific
rescaling. Every network sees the same fixed data and minibatch order. Seeds
therefore measure initialization variability, not variation in training data.

Runs are batched for speed but losses are **summed per network** after averaging
over examples and features. This preserves standalone gradient scale and Adam's
epsilon behavior. Independent-versus-batched gradient checks pass to float32
roundoff (`preflight-checks.json`). Checkpoints and validation are saved every
10,000 steps; training loss is logged every 1,000. The scratch resume checkpoint
includes optimizer, scheduler and MPS RNG state. The original fixed dataset is
kept in work/, not copied into the report; its SHA-256 is sealed.

## Competence and functional sharing

We report each feature's positive-active MSE, inactive leakage/false positives,
negative-present error, and isolated versus multi-input active error. A
**competent feature** has held-out positive-active MSE <0.05, unit-probe self-gain
≥0.25, and the largest unit-probe output at its own feature. This operational
threshold is fixed before training. Full error distributions are retained so
crossing the threshold is not equated with learning versus learning nothing.

For isolated unit feature j, let q[j,n] be hidden unit n's contribution to
output j: W_out[j,n] * h_n(e_j). Per-feature participation is
`sum_n(abs(q))^2 / sum_n(q^2)`. Per-neuron feature participation uses the same
formula over **competent features** j. We report both directions and their full
vectors. These measure distributed contributions and neuron reuse. They do not
provide a universal scalar amount of CiS, rule out cancellation/redundancy, or
identify an error-correction circuit. Higher coverage than hidden width and
higher functional sharing are interpreted jointly with task error.

The ReLU-derived binary-code-size test is not imposed on smooth or gated
architectures. SwiGLU has two preactivation branches: 66 or 100 latent coordinates
for 33 or 50 hidden output channels. Counts relative to hidden width should not
be mistaken for a matched count of every internal operation or coordinate.

The task itself favors a ReLU-like input-output map. Differences under this
one optimizer/initialization recipe are not a universal activation ranking.
Failures to reach task accuracy are not evidence of an inability to learn in
superposition under other training settings. This experiment does not test
modular-arithmetic grokking.

## Geometry selection and intervention

The pair-selection algorithm was frozen in the original protocol. Its concrete
arrays were materialized and committed as `e4bba2d` while training was at step
2,000, before held-out feature or p results. Training-loss logs existed by then;
no loss or feature outcome enters that deterministic selection.

The same six fixed index pairs are attempted in every network. After final
training, before any final-network p measurement, we additionally select the
first three task-competent pairs in a frozen seeded ordering of all 4,950
unordered feature pairs. Selection shortfalls, duplicates and exclusions are
retained. The geometry plan is written and committed before the sweep. We
report fixed pairs, competent fixed pairs, and competence-conditioned pairs as
separate cohorts: the latter can differ across networks.

Interventions use actual encoder columns before the nonlinearity. For SwiGLU,
the direction concatenates both gate and value columns and perturbs both
branches. This is equivalent to modifying the corresponding input feature;
it measures feature sensitivity, not fixed-target corruption recovery. Random
controls match both latent column norms and cosine. The random-control space
has dimension N for scalar activations and 2N for SwiGLU.

Four frozen sparse backgrounds, pair inputs 0/+0.5/−0.5, all four signs, and a
41×41 additive grid from 0 to 1.5 follow the earlier pilot. Each response is L2
distance to that network's own clean output, computed on CPU in float64 using
trained weights. Some anchor/sign/range combinations extrapolate beyond the
training amplitude range. The negative-anchor ++ case ranges from −0.5 to +1
and remains within the coordinate-wise training support.

We fit median response over four backgrounds, with thresholds 0.25/0.5/0.75
times the smaller maximum axis response; 0.5 is primary. We retain per-background
fits, fit bounds, weak/flat axes, nonmonotonicity and residual flags. The original
additive superellipse fitter and a PSD tilted quadratic use identical normalized
contour points. Additive coordinates are rescaled to chart units 0..60 solely
for the original fitter's edge exclusion; no geodesic sine correction is used.
The first competent pair per network receives fixed 81×81 checks for encoder
and random directions at the negative anchor and ++ sign.

Because encoder-column perturbations equal input-feature changes, equally
accurate models of the same target can have similar feature contours despite
different internal representations. The random controls and functional-sharing
measurements help expose that limitation; p is not treated as a direct CiS score.

## Reproduction and status

Use `requirements-toy.txt`; the training stage requires MPS. Full checkpoints
and geometry run files use hashes and refuse silent replacement. With the
frozen protocol and dataset sealed:

```bash
python scripts/check_toy_training.py
python scripts/train_toy_activations.py
# If interrupted, resume the existing scratch optimizer/RNG checkpoint:
python scripts/train_toy_activations.py --resume
# After all 100,000 updates:
python scripts/check_trained_geometry.py
python scripts/geometry_trained_toys.py --stage prepare
# Inspect and commit the competence-based plan before measuring p.
python scripts/geometry_trained_toys.py --stage sweep
python scripts/fit_trained_toys.py
python scripts/plot_trained_toys.py
```

Task/sharing figures include every seed and objective. Geometry figures report
network-level medians over the declared pair cohort with validity counts and
quality flags in accompanying data. We do not assign population-level p-values
to three initialization seeds or treat shared pairs/signs as independent trials.
No new results are published remotely as part of this local training run.
