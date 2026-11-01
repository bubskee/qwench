# Small fresh positive control and matched transfer

This is six fresh response surfaces: Gender × Age, Wealth × Gender, and one
fixed isotropic pair on Qwen3-1.7B, then the same three on Qwen3.5-4B. Each
surface has 30 anchors and a 31×31 grid (2-degree component spacing, 0–60°).
The nominal total is 172,980 perturbation cases; two reference distances are
recorded from each case. This is a procedural positive control and a small
matched model-transfer check, not an estimate over all feature directions.

The protocol was committed in `c0d0b5a`, implementation and regenerated anchors
in `20022f5`, and both models' fresh directions, fixed prefixes, overlaps, and
parity receipts in `993c063`, before any fresh full-surface exponent was fitted.
No pair or layer changes are allowed in response to the new exponents.
Gender × Age is a known positive case in the released 1.7B data; Wealth ×
Gender is the familiar paper pair. Axis order matches the released raw files.
The random pair is mechanically the first released seed-42 pair, rather than
one selected for a favorable exponent. It need not be representative of the
original random-control distribution.

The source is pinned at `a9355d4001500708a40a255bacfe559e6dc269c0`.
The exact model and dataset revisions are in `configs/positive-control-v1.json`.
The original loader does not pin a model or dataset revision, and its original
token cache is not released. We therefore regenerate its sampling recipe and
cannot verify the identity of its historical 30 snippets. This limitation is
retained in every interpretation of cached-versus-fresh differences.

The actual dataset is `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`.
We use the original streaming shuffle seed 42, buffer of 10,000 documents,
1000-document tokenization batch, five-token truncation, default special tokens,
and first-30 accepted input rule. Both tokenizers accept the same document
indices. Their decoded five-token windows agree in 29/30 cases; one differs
at a newline boundary. Full source texts, token IDs, source identifiers,
acceptance indices, package versions, and hashes are saved in `anchors.json`.
Thus the two fresh models use the same source documents and each model's own
five-token encoding. We retain the tokenizer difference instead of forcing
non-native token IDs between models.

Fresh contrastive directions use the released exact prompts, one unpadded
prompt per forward, and unit-normalized mean(B−A) at zero-based block output 2.
Prompt hashes are checked against the released direction cache. Fresh 1.7B
vectors are compared with those released vectors as an extraction diagnostic.
The isotropic directions are generated on CPU with seed 42, 33 vectors of the
appropriate residual dimension, then normalized; the first two reproduce the
released 1.7B random directions. The same absolute-cosine <0.1 eligibility rule
applies to both models and every proposed pair. All six model/pair attempts
pass; no replacement is needed.

We use BF16 model inference with native SDPA and cache disabled, and FP32
perturbation arithmetic and L2 distances. Perturbations are the original
exponential map in a projected, Gram–Schmidt-orthonormalized tangent plane.
The last-token activation is perturbed at block output 2; the measurement
is the penultimate block output (26 for 1.7B, 30 for 4B).

The original code fixes each single-anchor activation and its preceding-token
context before running suffix batches. We preserve this by replaying the
single-anchor block-2 output at every token, then letting the native decoder
continue. Earlier blocks execute but their block-2 output is replaced. This
retains native masks, position handling, and hybrid architecture behavior.
The 1.7B result is checked for exact agreement with the original manually
iterated suffix on clean and perturbed inputs, at batches 1 and 8. Fixed-prefix
replay also matches the ordinary clean full forward at batch 1, and stopped
captures match full forwards at the perturbation and measurement blocks.

The primary distance uses the original single-clean-anchor reference. As a
numerical sensitivity check, the same perturbed outputs also give distance
to a clean reference with identical batch size and fixed prefix. These modes
are both declared before fresh exponents; we do not select whichever result
looks better. The matched mode must have exactly zero response at zero angle.
The original-reference mode can have a BF16 batch-shape floor, which is reported.
Batch size is 64, including the original grid's one-example final remainder.
Baseline tensors use the same contiguous input layout as perturbed tensors.

For each model, pair, and reference mode, response grids are aggregated by the
median over all 30 anchors. The threshold is 0.5 times the smaller of the two
single-axis maxima. We use the unmodified vendored exact-sine superellipse
fitter, show radial residuals and fit flags, and retain all attempts. Threshold
factors 0.5/1/2 and individual-anchor fits are reported. Anchor intervals use
500 shared resamples, recalibrating the threshold in each draw. Cross-model
intervals describe paired source-document sensitivity for these fixed pairs,
not uncertainty over model families or latent features.

Raw results checkpoint after every anchor. The input seal, source snapshot,
prepared vector/prefix hashes, model pin, and package receipt protect resume
consistency. Preparation timing and engineering preflight forwards are not
counted among the full-grid response measurements. R/J++ readouts and additional
feature searches remain outside this run.
