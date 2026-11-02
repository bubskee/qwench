# Small AND/XOR pilot

This exploratory follow-up changes the output task from reconstructing input
features to computing logical combinations. It is an additive input-space toy
experiment, not a replication of the paper's norm-preserving LLM intervention.

## Tasks and fresh training

Sixteen independent Bernoulli(0.5) bits are embedded by the same fixed 32×16
orthonormal encoder used previously. Eight outputs compute AND or XOR on fixed
disjoint adjacent pairs. Balanced input bits give every two-bit input state
probability 1/4; output positives occur with probability 1/4 for AND and 1/2 for
XOR. The prior differs from the earlier sparse-bit study, deliberately avoiding
an extremely rare-positive AND task. Compare activations within each task.

The factorial grid is AND/XOR × clean/noisy training × five activations × three
initialization seeds: **60 networks**. All are biased 32→64→8 networks with 2,632
parameters. ReLU, exact GELU, SiLU, Leaky ReLU (negative slope 0.1) and squared
ReLU each learn fresh weights. Initial matrices match across functions and
tasks, with gate std 1/√32 and output std 1/√64; biases start at zero.

Clean examples receive no added noise. Noisy training adds Gaussian observation
noise with coordinate std 0.35, while the clean logical target stays fixed.
Corresponding networks share the 32,768-example latent pool, minibatches and
standard noise draws. Seeds vary initialization. Training uses 30,000 Adam
updates, batch 512, initial learning rate 0.001 decayed to zero by a cosine
schedule, no weight decay, native MPS float32. Per-network batch/output means
are summed, preserving standalone optimizer scale. Checkpoints are 0/5k/30k.
There is no task-specific tuning or outcome-based extension.

The frozen primary protocol, implementation, inputs and preflight were committed
as `1849afd` before actual training. A separate 100-step timing pass discards
its parameters; actual training restarts initialization and RNG. The
[protocol](../../configs/toy-logic-v1.json) is authoritative.

## Geometry and genuine recovery

Geometry probes (0,1), two bits feeding one output, and (0,2), bits feeding
different outputs. All four binary probe states are retained. From each corner,
directions point toward the opposite bit value; amplitudes range 0..1 on a
31×31 grid. These are continuous interpolations between discrete codewords;
there is no extrapolation past the opposite corner.

Four matched backgrounds share nuisance bits, with partners (1,3) set to
00/01/10/11 before the probed bits are overwritten. Every background and the
background-median response are saved. The prespecified comparison uses state
00/background 3 for both pairs; this gives the across-pair AND probe two active
partners, hence two meaningful single-bit axes. Other states and failures are
not filtered out.

The response is the full eight-output L2 change from each map's own anchor
output. The unchanged superellipse fitter uses 0.25/0.5/0.75 times the smaller
maximum axis response, with 0.5 primary. Its chart is 0..60; its returned
intercept units are rescaled for this pilot's amplitude maximum of 1. Original
fit flags are preserved. Additional declared diagnostics flag multiple
below-threshold components of at least three cells, diagonal threshold
touch/crossing followed by return below, and an interior single-grid-step drop
exceeding 1% of grid maximum. A tangent diagonal touch counts for exact XOR.

The full-vector additive axis null matches both single-axis output changes;
its interaction residual is measured separately. It is a geometry diagnostic,
not a global corruption-recovery control. The within-pair output-0-only response
is retained to expose leakage into unrelated outputs. Final seed-0 models
repeat both 00 probes on 61×61 grids, yielding 160 background comparisons.

Recovery uses 1,024 independent clean latent examples, two paired noise draws,
and noise std 0/0.2/0.35/0.6. Targets remain the original clean logical labels.
Report total, positive, negative and class-balanced MSE, positive misses and
false positives. Natural predictions are primary; clipped predictions are also
saved. Compact float32 recovery predictions are scored in float64 from the
exact stored values. Geometry remains float64.

The known-code references are clipped linear bit decoding followed by logical
multilinear interpolation, an exact posterior-mean logical denoiser, and the
constant output prior. For independent bits, posterior AND is `p1*p2` and
posterior XOR is `p1+p2-2*p1*p2`. These formulas were checked against direct
likelihood enumeration over all 65,536 clean latent states. Geometry uses the
oracle at assumed noise std 0.35. These analytic references know the encoder,
prior and noise; they are not equal-budget trained competitors.

A final-only secondary projection/clipping check was logged as `9d82ecb` after
the 5k scores, before final recovery/geometry inspection. It gives the natural
networks the analytic references' known-subspace information and matches output
bounds. It checks feature-plane invariance and independently reconstructs
primary scores with Boolean AND/XOR labels and sums/counts. It adds no training.
After inspecting the coarse oracle contours, this secondary check was extended
to repeat the four canonical oracle cases on 61/121 grids. Main model selection,
training and primary fitting settings were unchanged. Figure recovery panels
show the projected/clipped secondary comparison explicitly; raw primary scores
remain in the tables and stored predictions.

## Reproduction

From the repository root, with Python 3.12 and the full toy dependencies:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/toy.txt
.venv/bin/python scripts/run_logic.py --prepare-only
.venv/bin/python scripts/run_logic.py --resume
```

Outputs go to `work/reproductions/toy-logic-v1/`; use a fresh `--out work/...`
to repeat. Resume verifies frozen source/input hashes. Training requires Apple
MPS; no private scratch files, model downloads or credentials are required.
The full stage scripts were executed for the canonical study; the portable
wrapper was preparation-tested rather than retraining all 60 networks twice.

Saved-data figure regeneration needs only `requirements/figures.txt`:

```bash
.venv/bin/python scripts/plot_logic.py --out work/reproductions/logic-figures
.venv/bin/python scripts/verify_logic.py
```

To rerun saved-weight numerical checks, set `QWENCH_LOGIC_OUT` to a reproduced
output if needed, then run `scripts/check_logic.py --stage audit` and
`scripts/logic_controls.py` with the full dependencies. Primary fit flags and
task failures remain in the raw results. Repeated checkpoints/backgrounds are
not independent training trials; three seeds do not establish a population
activation ranking. The 64-hidden/16-latent setup does not force CiS.

The integrity command checks hashes and saved audit status, without repeating
numerical checks. Its sealing path additionally verifies the 180 derived rows
and 60 seed summaries against their source records. The primary audit's
geometry-recalculation field is a deterministic rerun of the measurement
function; the separate controls audit constructs canonical grids independently.
