# Findings

Status: cached paper reproduction and local timing complete, 2026-11-01.
The Qwen feature sweep has not run. R/J++ are active; J and O are deferred.

## Run provenance

The [paper receipt](../reports/paper/reproduction.json) records the pinned
reference commit, cache and source hashes, packages, and original command.
The [timing receipt](../reports/timing/timing.json) records model revision,
script hash, packages, prompt token IDs, checks, and timings. The
[asset inventory](../configs/local-assets.json) records the R/J++ weight
hashes and locations. The timing model was BF16 on Apple M4 Pro MPS;
intervention arithmetic, response subtraction, and lens readouts were FP32.

## Cached figure sanity check

The original pinned Figure 2 script reproduced the Wealth × Gender
boundary from 30 cached Gemma-2-9B anchors on a 31 × 31 grid:

| Quantity | Result |
| --- | ---: |
| Per-pair L2 threshold | 138.60028 |
| Wealth axis crossing | 46.98552° |
| Gender axis crossing | 38.47389° |
| Exact-geodesic exponent | 2.404153 |
| Mean fractional radial fit residual | 1.21948% |
| Usable contour points | 41 |

This matches the paper's rounded exponent 2.40. The cache Git blob identity
was checked before loading; its SHA256 is saved in the receipt. The
[PNG](../reports/paper/paper-figure-2.png), PDF, contour CSV, and compressed
numerical response arrays are saved beside the receipt. This checks our
understanding of the published fitting pipeline; it is not a new replication
on Qwen.

## Local timing and intervention checks

Full text-decoder passes, no vocabulary projection, with a random 15°
norm-preserving intervention at zero-based block output 2. Response was
measured at block output 30 before final normalization. Four layers of
last-token activations were captured. Each timing has two warmed repeats.

| Tokens | Batch | Median seconds/batch | Perturbations/second |
| ---: | ---: | ---: | ---: |
| 5 | 1 | 0.110 | 9.09 |
| 5 | 4 | 0.206 | 19.38 |
| 5 | 8 | 0.304 | 26.32 |
| 32 | 1 | 0.150 | 6.69 |
| 32 | 4 | 0.336 | 11.90 |
| 32 | 8 | 0.604 | 13.25 |

The model loaded in 2.82 seconds. MPS driver memory during forwards was
about 8.9 GB. The zero-angle intervention had exactly zero downstream L2
difference for both prompt lengths. The FP32 rotation norm error was below
printed precision; casting back to BF16 changed the norm by 0.0155% and
0.00791%, respectively. A random 15° perturbation produced finite nonzero
downstream differences. These checks validate basic plumbing, not feature
semantics or error correction.

R/J++ readout also executed successfully at blocks 8, 16, 26, and 30. Summed
median compute time across those eight readouts was 0.159 seconds for a
batch of eight, excluding model/head/map setup and saving. The receipt's
example top tokens come from a random perturbation and have no feature
interpretation attached.

At the measured batch-eight rates, 4,500 examples imply approximately
2.85 minutes (5 tokens) or 5.66 minutes (32 tokens) of forward compute.
Allow extra time for extraction, orchestration, saving, crossing refinement,
and readouts. This was one repeated anchor and two timing repeats; independent
anchors and sustained load may differ. The implementation uses PyTorch's
fallback hybrid-model kernels, not CUDA's optimized DeltaNet path.

## Qwen geometry

Not run. Direction validation and cosine overlap. Candidate and control exponents,
fit residuals, usable crossings, per-anchor behavior, and stability checks.
Link the plot and machine-readable raw responses.

## Lens observations

Not run for feature directions. Baseline versus pure and mixed perturbations, below/near/above the boundary.
Raw readout examples and fixed-rubric labels. Note ambiguity and disagreements
with controls or downstream response. Keep this separate from the geometry fit.

## What this supports

The cached fitting pipeline is reproduced, and the local machine appears
fast enough for the small pilot. We have not measured Qwen's feature
geometry. No conclusion about Qwen error correction or semantic features
follows from this timing run.

## Stop or continue?

Stop after reproduction and timing for now, as requested. The next proposed
step is held-out validation of the sentiment/language directions, followed
by the small feature/control sweep if we proceed.
