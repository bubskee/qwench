# Characterization design, chronology and reproduction

This is an exploratory follow-up to known earlier toy results, with protocols
and controls logged before their corresponding new outcomes. It is not a
blinded replication of the LLM paper.

## Factorial training

The new grid is 2 encoder geometries × 2 decoder mappings × 2 training-noise
conditions × 5 activations × 3 initialization seeds = **120 networks**.
Each is `D·phi(Gx+b)+c`, with 33 inputs, 64 hidden units, 16 outputs and 3,216
parameters. The five functions are ReLU, exact GELU, SiLU, Leaky ReLU with slope
0.1, and squared ReLU. No trained weights are reused under a different function.

Sixteen clean latent bits are independent Bernoulli(0.125). An observed context
is uniformly ±1 and is never corrupted. A fixed 32×16 orthonormal-column matrix
A embeds the bits. For overlap experiments, columns become
`B_even=A_even`, `B_odd=0.6*A_even+0.8*A_odd`, independently within eight pairs.
Each column keeps unit norm; adjacent-column cosine is 0.6 and all other
column cosines are zero. In context tasks, negative context rolls the encoder
columns by eight; positive context retains their order. Linear tasks receive
the same cue but do not require it. The clean target is always the latent bits.

Clean training adds no observation noise. Denoising training adds independent
Gaussian noise with coordinate std 0.35 to the 32 observations. Labels remain
the clean bits. The context coordinate is exact in both conditions.

All corresponding models share latent pools (131,072 examples), contexts,
minibatches, standard noise draws, and initial matrices. Initializations use
normal encoder std 1/√33, decoder std 1/√64, and zero biases. Seeds vary the
weights only. Every activation has its own independently updated parameters.
The overlap adapter calls the unchanged parent trainer; it changes the encoder,
posterior oracle, dual-basis null and provenance paths, not the update loop.

Each model gets 100,000 Adam steps, batch 512, learning rate 0.001 with cosine
decay to zero, default betas/epsilon, no weight decay, native MPS float32.
Summing per-network mean losses preserves standalone Adam scale; independent
gradient and update tests verify this. A 100-step timing-only benchmark preceded
the fresh full run. Saved checkpoints are 0/100/1k/5k/20k/100k. Resume states save
weights, optimizer, scheduler and RNG every 1,000 steps.

The separate historical track reuses five scalar activations × L2/L4 × three
seeds at 10k/30k/100k: 90 frames, with no new training. Its 100→50→100 bias-free
signed-sparse ReLU-target task, optimizer recipe, and competence criteria are
preserved. It is not pooled with the new binary task as an activation ranking.

## Exact reference denoisers

For orthogonal observations, latent posterior means factorize into independent
Bernoulli-Gaussian logistic responses. For the overlapping encoder, the exact
posterior factorizes into eight **two-bit** groups. Its likelihood contains the
Gram cross term, so neighboring decoded estimates can affect the optimal
prediction. Both formulas are checked against direct likelihood enumeration
over all 65,536 bit vectors. The posterior means minimize expected squared error
for the stated generative distribution; they do not minimize positive-class
error separately.

The linear reference uses the known context-dependent dual encoder and clips
decoded estimates to [0,1]. These references have ground-truth generative
information that the networks must learn; they are benchmarks, not equal-budget
trained competitors.

## Primary geometry and recovery

Input perturbations are additive. Geometry measures L2 distance from each
map's own clean output; it does not score closeness to a fixed target.
Two fixed low-overlap feature pairs are used per family: (0,1)/(4,9) in the
orthogonal study and (0,2)/(4,9) in the overlap study. Pair (4,9) is shared for
the matched illustration and comparison table. Each pair is tried at latent
values 0 and 1, with ++ and +- perturbations, four frozen backgrounds (two per
context), and a 31×31 amplitude grid from 0 to 1.5. One seeded isotropic
observation-direction pair is also tried. Context is never perturbed.

The original additive fitter takes the median response over backgrounds and a
threshold of 0.25/0.5/0.75 times the smaller maximum axis response; 0.5 is primary.
Chart coordinates are rescaled to 0..60 for the original edge exclusion, with
no sine correction. A PSD tilted-quadratic fit, per-background fits, and all
failure flags are saved. Fine 61×61 checks cover the first feature pair, absent
++ slice, final checkpoint, seed 0, both maps and all 20 activation/task cells
in each family: 80 checks total.

All fixed pairs are retained, including unlearned features. Competence is an
annotation: clean active MSE<0.1, inactive MSE<0.05, and isolated positive probe
gain>0.5 with correct argmax in both contexts. It is not used to select high-p
cases. Primary fitting has 25,920 surfaces and 100,827,200 response points including refinements.
Repeated pairs/checkpoints/backgrounds are not independent training trials.

The separable restriction is `C_j(u_j,r)=f_j(B_r*e_j*u_j,r)`, with
`u=(B_rᵀB_r)⁻¹B_rᵀ observation`. It preserves the full isolated feature curve,
including biases, for either sign and amplitude. It removes conditioning on
other latent estimates. It has known coordinates and potentially more effective
univariate capacity; it is neither an equal-capacity trained model nor a map
that is incapable of scalar denoising.

Fixed-target recovery uses 1,024 independent clean examples, two paired noise
draws, and observation noise std 0/0.15/0.35/0.6. The target never changes with
noise. Scores separate all outputs, positive bits, inactive bits, misses and
false positives. Primary network predictions are not clipped; bounds are
examined separately. The historical recovery set uses its original signed
sparse distribution and std 0/0.075/0.15.

## Logged secondary checks

| Addition | Why it was added | Protocol |
|---|---|---|
| Overlapping-code training | Orthogonal-code optimal recovery factorizes; add a problem where neighboring estimates help | `toy-characterization-coupled-v1.json` |
| Projection and output clipping | Match the separable control's known-subspace advantage and quantify target-bound effects | `toy-characterization-diagnostics-v1.json` |
| Axis-norm null | Test whether measured one-dimensional nonlinear responses already explain p | `toy-characterization-axis-null-v1.json` |
| Directly coupled pair / full-vector additive null | Primary low-overlap pairs cross independent code blocks; inspect the interaction where it lives | `toy-characterization-pair-diagnostic-v1.json` |
| Exact-oracle geometry | Check whether directly coupled learned-model contours reflect the true posterior or approximation artifacts | `toy-characterization-oracle-geometry-v1.json` |
| Fresh noise and partner-state flips | Validate recovery on new observations and expose conditional behavior hidden by medians | `toy-characterization-validation-v1.json` |

These are explicitly secondary diagnostics, motivated by earlier outcomes or
design inspection. They are not presented as prespecified primary confirmations.
Their settings were logged before each corresponding computation. The original
primary protocol was committed as `7e42ded`, the coupled extension as `97746af`,
and fresh validation inputs as `725f21f`.

The full-vector null predicts `f(x+a*d1)+f(x+b*d2)-f(x)`, matching both complete
axis-response vectors. The directly linked pair (0,1) has input cosine 0.6 in
the overlap code and is outside the paper's pair filter; its p is not a
paper-compatible test. Nonlinear interaction alone is not identified with a
correction module.

Fresh validation uses 4,096 examples, two new noise draws at std 0.35, all final
networks, and both raw and projection-matched/clipped comparisons. A paired
partner-state experiment sets only latent neighbors (5,8) to 0 or 1 around
probe pair (4,9), leaving other background and context fixed. All original
class-conditional scores and fit flags remain available.

## Reproduction

From a clone with Python 3.12, rebuild the six main figures without PyTorch:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/figures.txt
.venv/bin/python scripts/reproduce.py study-verify
.venv/bin/python scripts/reproduce.py study-figures
```

New figures and their source-data receipt go to
`work/reproductions/study-figures/`. Existing outputs are protected; choose a
new `--out work/reproductions/another-name` to repeat. `study-verify` checks
artifact hashes and saved audit status; it does not rerun scientific checks.

To reproduce the full new training/analysis on Apple Silicon MPS:

```bash
.venv/bin/python -m pip install -r requirements/toy.txt
.venv/bin/python scripts/reproduce.py study-run --prepare-only
.venv/bin/python scripts/reproduce.py study-run --resume
```

Preparation copies the frozen sources, protocols, vendored fitter and required
historical checkpoints into an isolated layout under
`work/reproductions/study-run/outputs/qwench/`. It needs no private scratch
files, HF credentials, LLM downloads, or original workstation paths. The actual
update loop still requires MPS. The pipeline runs stages in dependency order,
preserves immutable source hashes, resumes saved training/analysis, and makes
no git commits or remote changes. Before the first 1,000-step resume checkpoint,
an interrupted run needs a new output. Other corruption/partial-file failures
are reported rather than silently accepted.

The rerun is a scientific reproduction, not a promise of bit-identical new
training on different hardware/software. Original training used PyTorch 2.14.1
and NumPy 2.5.4. Full raw grids/diagnostics occupy over 1 GB. Figure redraws
create only small new outputs. The new full pipeline itself was staged and its
wrappers tested; the full 120-network grid was executed through the same frozen
stage scripts during this study, rather than trained a second time to test the
wrapper.

A fresh staged copy regenerated both asset archives, analysis-input archives
and measurement plans byte for byte. Both task preflights passed there. After
copying a saved final checkpoint into that test copy, the independent null and
geometry-calibration checks passed too. These checks do not claim a second
independent training run.

Saved-data numerical checks can be rerun with full dependencies:

```bash
.venv/bin/python scripts/audit_characterization.py --family orthogonal
.venv/bin/python scripts/audit_characterization.py --family coupled
.venv/bin/python scripts/audit_characterization_study.py
.venv/bin/python -m unittest discover -s tests -v
```

The combined saved-data audit and plotting need only the figure dependencies;
the first two audits also use PyTorch to reconstruct initial weights. All
primary row sums/counts, table values, source hashes, exact-posterior checks,
conditional-control identities, and numerical failures are retained. One early
coupled analysis reached the final checkpoint before training had written it;
it resumed after training completed, without changing inputs or finished rows.
The coupled preparation source note records an added pre-training equality
assertion, with no change to asset generation or scientific settings.
