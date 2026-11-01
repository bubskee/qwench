# The small experiment

## 1. Understand one cached boundary

Reproduce Figure 2, Wealth × Gender in Gemma-2-9B at layer 2, using the
reference repo at `a9355d4001500708a40a255bacfe559e6dc269c0`.
Use its existing script before adapting any analysis:

```bash
python scripts/plotting/plot_fig3_boundary.py \
  --target gemma --layer 2 --d1 Wealth --d2 Gender --exact
```

This command runs inside a separately obtained reference checkout with its
dependencies and cached sweep installed. It has now been run here: the
reproduced exponent is 2.40415 and the radial residual is 1.219%.
Use `scripts/reproduce_paper.py` from this repository for the complete
fetch, verification, plotting, and provenance workflow.
Only obtain the needed cache and code initially; the full sweep cache is large.
Record the input checksum, revision, threshold, axis intercepts, exponent,
number of usable contour points, and radial fit residual. Check against the
paper's reported `p ≈ 2.40`. This is the fitting sanity check, not a Qwen result.

## 2. Build one candidate feature plane

Start with positive versus negative sentiment and Spanish versus English.
Use a small crossed set of matched content: each item has both sentiments
in both languages. Average sentiment differences across languages and
language differences across sentiments. Begin with 16 content items for
extraction and 8 separate held-out items for validation.

Extract directions at the same residual location and token convention used
for perturbation. Match the task wrapper and use a common final marker;
record tokenization and sequence lengths. A shared marker reduces an obvious
lexical confound but does not eliminate positional or translation effects.
Check held-out separation and simple axis steering before doing the full sweep.

Fix signs up front: positive sentiment and Spanish are the positive poles.
Require raw `abs(cos(d1, d2)) < 0.1`, following the paper's overlap criterion.
Log anchor-relative tangent projections and how much orthogonalization moves
each direction. If the pair fails, report that and revise it explicitly;
do not quietly orthogonalize a strongly overlapping pair into new features.

## 3. Measure the plateau

Use the exact checkpoint compatible with the existing J/R/J++ lenses. Resolve
the model revision, dtype, chat template, thinking mode, and precise hook
before running. Qwen3.5-4B has hybrid DeltaNet/attention blocks: begin with
full forward passes and caching disabled. Verify that a zero-angle intervention
reproduces an ordinary forward pass. Do not assume a generic transformer
suffix-forward implementation preserves the hybrid model's behavior.

Start at paper layer label 2 and measure at the penultimate residual layer,
before final normalization. Record module paths, zero-based block indices,
pre/post-block placement, and correspondence to the paper's convention.
Perturb and measure the last non-padding prompt token, with no generation.

For each anchor activation `a`, project the two directions into its tangent
plane and Gram–Schmidt orthonormalize to `u1, u2`. Apply

```text
a'(α, φ) = cos(α) a + sin(α) ||a|| [cos(φ) u1 + sin(φ) u2]
R(α, φ)  = ||h_penultimate(a') - h_penultimate(a)||₂
```

This preserves the input activation norm. Use unsquared L2 distance.
Start with four fixed generic anchor prompts, disjoint from direction
extraction and validation. Sweep nine mixing angles `φ = 0°, 11.25°, …, 90°`
and 25 perturbation angles `α = 0°, 2.5°, …, 60°`. This polar sampling is a
pilot adaptation of the reference Cartesian grid. Refine threshold crossings
if the initial resolution materially changes the fitted exponent.

For each direction pair, take the median response across anchors, then define

```text
T_pair = 0.5 × min(max_α R_median(α, 0), max_α R_median(α, 90°))
```

Use one threshold for the whole plane, shared by its pure and mixed sweeps.
Controls get the same threshold-selection rule on their own surfaces.
Extract the first threshold crossing on each ray with interpolation. Flag
missing crossings, nonmonotonicity, disconnected boundaries, and clipped
contours rather than treating them as evidence for a large exponent.

With axis crossings `α1, α2`, fit `x^p + y^p = 1` using

```text
x = sin(α_cross(φ)) cos(φ) / sin(α1)
y = sin(α_cross(φ)) sin(φ) / sin(α2)
```

Use radians inside trigonometric functions. Fit with bounds that include
values below 2; show measured points, the fitted curve, and the `p = 2`
reference in these same coordinates. Report residuals and boundary fits
that hit parameter limits. A poor superellipse fit is a result too.

As a cheap diagnostic, also fit a centered quadratic boundary with a cross
term (`x² + 2bxy + y² = 1`). An axis-aligned `p = 2` comparison does not rule
out a rotated anisotropic quadratic. Compare residuals on held-out mixing
rays; this check needs no additional model passes. Reversed poles would
help distinguish a cross term from a symmetric superellipse if we expand.

## 4. Keep controls matched

Use two predeclared seeds for each of two control families: independent
isotropic Gaussian directions, and differences between unrelated activation
samples at the same layer and token convention. For the latter, match the
number of averaged differences to the candidate extraction procedure.
Apply the same overlap filter, tangent projection, sweep, and fitting rules.

One candidate pair plus four control pairs gives an upper budget of
`5 × 4 × 9 × 25 = 4,500` perturbed examples, before batching and sharing
zero-angle evaluations. Benchmark one batch before estimating wall time.

Keep per-anchor curves. Show leave-one-anchor-out fits and threshold
sensitivity at 0.5× and 1.5× the headline threshold where crossings exist.
With four anchors and one feature pair, these are stability checks, not a
population-level replication claim. Freeze the primary prompts and seeds.
If time remains, add two feature-active anchors as a labelled secondary
check, and test reversed feature poles. Do not pool these into the primary fit.

## 5. Peek through R/J++

Use R/J++ for the current pilot, at blocks 8, 16, 26, and 30. The
[local asset audit](local-assets.md) also found O adapters at block outputs
26 and 30; O and the unlocated standalone J checkpoint are deferred.
R/J++ return token scores and rankings. They do not generate descriptions.
Track a small fixed sentiment/language token set alongside raw and
vocabulary-filtered top tokens, with a baseline from the same anchor/layer.
Future O-lens descriptions would require separate runtime validation.

After fitting, select pure directions and the equal mixture on the same
anchors. Record lens outputs at approximately `0.5×`, `1×`, and `1.25×`
each ray's measured crossing angle, plus the unperturbed baseline. Mark any
requested point outside the measured range. Crossing-based magnitudes are
descriptive selections made after the primary measurement.

Use a few available intermediate lens layers; resolve their exact hook
conventions first. Save activations from the perturbed forward pass and
decode afterward where the lens implementation permits it. Use fixed
readout settings and a short rubric: sentiment, language, both, neither,
or unclear. Preserve raw outputs alongside labels.

Compare with controls at both equal perturbation angle and comparable
downstream response when available. Lens calibration and amplification can
produce apparent disagreement. A striking decoded change is an observation
about the lens, not automatically a large semantic change in the model.

## Stop here

Produce one Qwen boundary plot with matched controls, machine-readable
responses and fit metadata, and a short findings note. The cached paper
reproduction is a separate sanity-check artifact. Add a small lens panel
only if the adapters are readily available.

Interpret `p > 2` as privileged geometry. Nonlinear coordinate sensitivity
or independent feature thresholds can also produce a bulging boundary
without correcting corrupted feature values. A mechanism claim would need
additional evidence of recovery toward a clean feature state. Expand only
if a feature/control difference or lens disagreement survives the checks.
