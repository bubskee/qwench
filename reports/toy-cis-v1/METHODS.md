# Toy pilot methods and reproduction

The protocol was committed as `b87b06d` before task checks; checkpoint checks,
eligibility and exact pairs were committed as `ed8bf84` before geometry or
corruption measurement. The source is frozen by the SHA-256 in `prepared.json`.
All runs are local CPU float64, six Torch threads. No training was performed.

## Models, inputs and selection

Source: [toy-model-cis-code](https://github.com/FranciscoHS/toy-model-cis-code),
revision `8c827b8c07bcf39097ee21707c62641a9201a216`. We use the released
`codeword_100f_50n_L2_100000steps.pt` and `...L4...` state dictionaries, loaded
with `weights_only=True`. Hashes are in the preparation record. Checkpoints
remain in the external reference checkout; we do not redistribute their files.
Our independently written forward implements `ReLU(x W_in.T) W_out.T` with no
bias. We check it against the upstream `SimpleMLP`: exact forward agreement.
The dedicated control has W_in=W_out=I_100 and computes the target exactly.

The target is ReLU(x). Each input coordinate is independently present with
probability 0.02 and has a uniform signed value in (−1,1). NumPy PCG64 seed
20261102 generates 32,768 clean evaluation samples. These are fresh validation
inputs, not the original training/evaluation set. A feature is positive-active
when x>0; inactive means x=0; negative-present means x<0. These masks are kept
separate in the saved metrics.

Eligibility uses one-feature unit probes only: encoder norm >1e−8, self-output
gain ≥0.25, and largest output at the intended feature. Fifty features pass in
L2 and all 100 in L4/dedicated. We preserve all 100 gains/errors and select the
first 12 entries of a seeded permutation of the common eligible set, pairing
in order. No selection uses fitted p or corruption accuracy. Exact pairs:
(82,70), (97,0), (2,46), (61,27), (24,51), (18,19).

Four background inputs use seed+1. For each pair, its two coordinates are
replaced with (0,0), (+0.5,+0.5), or (−0.5,−0.5). All other coordinates and
backgrounds are shared across models. Directions are the actual encoder
columns c_i,c_j, with no orthogonalization. Random controls match their 2×2
Gram matrix, hence both norms and cosine, to <4e−16 absolute error. Seeds are
fixed independently by model and pair before measurement.

## Geometry

We evaluate δ=s_1 a c_i+s_2 b c_j before ReLU, for all four signs and
(a,b)∈[0,1.5]^2 on a 41×41 grid. Each response is L2 distance from the clean
output of that same anchor. No output task target enters this sensitivity
measurement. Adding codewords is algebraically equivalent to modifying the
corresponding input coordinates; this is explicitly not fixed-target noise.

For each model/pair/kind/sign/state, take the median response across four
backgrounds. Threshold fractions are 0.25/0.5/0.75 of the smaller maximum of
the two axis curves; the primary fraction is 0.5. We also fit each background
separately at its own primary threshold. Flat axes, bound fits, nonmonotone
axes, weak responses and radial residuals >2% are retained with flags.

We use the original contour extractor, axis-intercept interpolation and
`fit_superellipse`. Additive amplitudes are scaled to a 0..60 chart to respect
the original fitter's hardcoded one-unit edge exclusion; **no sine/geodesic
correction is used**. Each contour is normalized by its own two axis crossings.
On the identical usable points, fit x²+2ρxy+y²=1 by least squares, clipping ρ to
[−1,1] for a positive-semidefinite quadratic and retaining unconstrained ρ and
rank-boundary flags. A small radial residual is goodness of fit, not mechanism
identification. No significance tests are assigned to repeated pairs/signs.

There are 432 primary aggregate surfaces. The after-ReLU control repeats each
model/pair/kind/sign through its linear decoder: 144 surfaces with responses
independent of anchor. Six fixed 81×81 checks use the first pair, inactive
state, ++ sign, both direction kinds, each model. The total is 3,304,296
forward response points. Four analytic controls × four signs check known p=2,
known p=4, tilted-linear response and dedicated ReLU, including flat axes.

The previously recomputed baked-FSEC denoiser p≈2.907 is a separate historical
procedural reference, not fresh evidence from this pilot or a matched model
comparison. Its construction/intervention space differs from these models.

## Fixed-target corruption

A separate 4,096-input sample uses seed+2. Gaussian noise uses seed+3 and is
normalized per example to levels {0,0.1,0.25,0.5,1} times each model's median
encoder-column norm. Models with equal hidden dimension share noise directions;
the 100-dimensional control uses the same source draws extended to 100 entries.
The target ReLU(clean x) remains fixed at every level.

Compare natural gates `ReLU(z+δ)`, frozen gates `ReLU(z)+1[z>0]δ`, and the
decoder-only intervention `ReLU(z)+δ`. All three have identical clean output.
Frozen gates use the clean representation; they are a mechanism ablation with
privileged clean information, not a deployable denoiser. Measure positive-active
MSE, inactive RMSE, inactive output >0.1, negative-present RMSE, and per-feature
active MSE. The diagnostic plot's active panel averages per-feature MSE over the
same frozen common-eligible set across models; full-task errors are also saved.
The checkpoints were not trained on this Gaussian denoising task.

## Reproduction

Run from qwench with `requirements-toy.txt`. The original local layout has this
repo at `.../outputs/qwench` and the external checkout at
`.../work/toy-model-cis-reference`. `run_toy_cis.py` verifies its pinned revision.
The preparation and measurement stages refuse to overwrite existing results;
use a fresh copy of the run folder for reruns, preserving the completed records.

```bash
git clone https://github.com/FranciscoHS/toy-model-cis-code.git \
  ../../work/toy-model-cis-reference
git -C ../../work/toy-model-cis-reference checkout \
  8c827b8c07bcf39097ee21707c62641a9201a216
python scripts/run_toy_cis.py --stage prepare
# Inspect and freeze the prepared pair list before continuing.
python scripts/run_toy_cis.py --stage geometry
python scripts/run_toy_cis.py --stage corruption
python scripts/analyze_toy_cis.py
python scripts/check_toy_cis.py
```

For this completed run, `analyze_toy_cis.py` needs only saved arrays/JSON and
NumPy/SciPy/Matplotlib; no checkpoint access or new forward passes. Quality
checks need the source checkpoint checkout and Torch. Raw source/reference
hashes, full fit flags, all sign/threshold/background results, and exclusions
are saved. The protocol's stopping rule is met; nothing has been pushed remotely.
