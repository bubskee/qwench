# Squared-ReLU geometry and fixed-target recovery

This post-hoc follow-up tests one already-known specialized checkpoint, selected
by identifier, and its first saved competence-conditioned pair. Existing p≈4
results motivated the experiment. This is not a blinded geometry replication.
The protocol was committed as `d17d90b`; model/input matching and paired noise
were sealed as `ab0e79c` before measuring new recovery outcomes.

## Independent-feature control

Let the trained network be `f(x)=D·ReLU(Gx)²`. For each of its 50 competent
features j, set `n_j=||G[:,j]||₂` and `c_j=f(e_j)[j]`. Assign one control unit k
to j, with `G_control[k,j]=n_j` and `D_control[j,k]=c_j/n_j²`; all other weights
are zero. Thus its supported outputs are exactly `c_j·ReLU(x_j)²`.

This matches the trained positive isolated-feature self-output for **every
positive amplitude**, by degree-two homogeneity, and the supported encoder-column
norms. Shapes and width match, but the control has only 50 nonzero encoder and
50 nonzero decoder connections. It is not an equal-free-parameter trained
competitor. Clean multidimensional behavior is not exactly matched. The trained
pair cosine is −0.001780; the independent pair cosine is zero. Random controls
match each model's own 2×2 direction Gram matrix.

The control is an analytic no-cross-feature baseline. Scalar clipping, curvature,
and the sparse task's prior can still lower error on corrupted inputs.

## Measurement

Geometry uses the same four stored backgrounds and pair (10,65), pair inputs
0/+0.5/−0.5, positive perturbations, encoder-column and Gram-matched random
directions, and a 41×41 additive amplitude grid from 0 to 1.5. We measure L2
distance to each model's own clean output. The original fitter uses the median
background surface and threshold fractions 0.25/0.5/0.75 of the smaller maximum
axis response; 0.5 is primary. Per-background fits and all flags are retained.
These are sensitivity measurements, not fixed-target recovery.

Corruption uses 2,048 independent sparse 100-dimensional inputs: per-feature
presence probability 0.02 and signed Uniform(−1,1) values. Each gets four fixed
Gaussian input-noise draws at coordinate std 0/0.025/0.075/0.15. Clean input
ReLU(x) is the fixed target; targets are never recomputed from corruptions.
The same examples, noise, and 50 supported outputs are used for both networks
and the input reference. Unsupported outputs are preserved in separate full-task
scores. The primary std 0.075 corresponds to expected input-noise L2≈0.75 over
100 coordinates. About 0.062% of corrupted coordinates exceed absolute value 1
at that level, so a small fraction extrapolates beyond training support.

The oracle prediction is `f(x)+J_f(x)·epsilon`, with derivative
`D·diag(2·ReLU(Gx))·G`. It has clean-anchor access, shares clean predictions,
and is not an implementable denoiser. Its derivatives are checked against
autograd. Error records separate positive, absent and negative supported
coordinates; pooled conditional scores use sums/counts, not equal-weight averages
of examples with different active-feature counts.

Bootstrap resamples 256 sets of clean examples, keeping each example's four noise
draws together. All methods share resamples. Intervals do not capture training,
protocol or noise-distribution uncertainty. Clean-to-noisy error differences are
descriptive and can include interactions with a model's existing clean bias.

## Reproduction

Python 3.12, CPU; full dependencies from `requirements/toy.txt` (MPS is not used
for this probe). From the repository root, use a **new** directory:

```bash
.venv/bin/python -m pip install -r requirements/toy.txt
.venv/bin/python scripts/squared_recovery.py --stage prepare --out work/reproductions/squared-recovery
.venv/bin/python scripts/check_squared_recovery.py --out work/reproductions/squared-recovery
# Inputs/control are fixed before measuring; inspect or commit the preparation.
.venv/bin/python scripts/squared_recovery.py --stage measure --out work/reproductions/squared-recovery
.venv/bin/python scripts/plot_squared_recovery.py --out work/reproductions/squared-recovery
.venv/bin/python scripts/audit_squared_recovery.py --out work/reproductions/squared-recovery
```

There is no training or model download. The measurement took under one second
on the original M4 Pro; runtime varies. Plotting and the saved-data audit need
only `requirements/figures.txt`. Omit `--out` from the plotting/audit commands
to regenerate the committed report's derived plots/checks.

The original measurement's first attempt failed while saving fitter NumPy arrays.
`source-amendment.json` records the serialization-only repair, with the prior
preparation preserved. A new run with the repaired script needs no amendment.
The audit checks that the retry's raw geometry exactly matches the first attempt.

`inputs.npz` preserves clean examples and standard noise draws. `models.npz`
preserves both networks. `geometry.npz/json` hold all 12 surfaces and fits.
`corruption-errors.npz` stores each clean example's error sums/counts averaged
over its four noise draws; `corruption.json` retains all methods and paired
intervals. The saved target convention, seeds, source hashes, and complete
protocol are retained for inspection.
