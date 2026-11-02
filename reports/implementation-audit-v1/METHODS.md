# Implementation and anchor audit

This audit follows the failed point-estimate reproduction in `positive-control-v1`.
It does not launch additional 4B experiments, change layers, or search for better
feature pairs. It reuses Gender×Age, Wealth×Gender, and random_000×random_001.

## Reference inputs and search

Reference revision: `a9355d4001500708a40a255bacfe559e6dc269c0`.
The released source expects `cache/fineweb_cache_qwen/fineweb_n30_seq5_seed42.pt`
and `results/activations/acts_qwen_L2_fineweb.pkl`. Neither input cache was found
in the public release. Streaming activation metadata has `prompt_set_hash=None`.

We searched paths in all 26 reachable commits across the three public branches,
GitHub releases/tags/issues, the arXiv source bundle, and the public HF author
and dataset searches recorded in `anchor-search-public.json`. No original token
file was located. This is a bounded public-artifact search, not a claim about
private files. The public source contains the deterministic generation recipe.

The 14 `sample/10BT` parquet files have identical paths, sizes, and LFS content
IDs in FineWeb-Edu v1.0.0, v1.2.0, v1.3.0, and v1.4.0. This rules out changes to
those data objects between these tagged versions, not every possible change to
loader behavior. The model revision is unchanged from our frozen v1 run.

## Sampling audit

Our v1 environment uses `datasets==5.1.0`. The repository does not pin its
historical dependency versions. We chose **one** pre-release implementation,
`datasets==4.8.5` (April 27, 2026), to test the algorithmic difference, before
looking at the corrected p values. We did not search versions for a favorable
exponent. This does not establish which exact version the authors installed.

In 4.8.5, buffered streaming shuffle yields one randomly selected buffered
example and replaces it with the next input example. In 5.1.0, it permutes the
buffer, emits half, and refills; `.shuffle()` also introduces default interleaving
of up to ten input shards. Thus identical seeds and buffer sizes do not define
identical samples across these versions. The source extracts and diff are saved
here, with upstream links below. These are Hugging Face Apache-2.0 source excerpts.

We isolated 4.8.5 in `work/datasets485-env`, reusing the existing native runtime
through a `.pth` path without modifying it. We streamed the same pinned
FineWeb-Edu revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, sample-10BT,
seed 42, buffer 10,000. We tokenized the first 1,000 returned texts in a batch,
with truncation to five tokens and default special tokens, and retained the
first 30 eligible rows. Both tokenizers are saved for transparency; only 1.7B
is executed in this audit. Full text, source IDs, hashes, and token IDs are in
`legacy-anchors.json`; `legacy-anchor-windows.tsv` is a readable index.

Before fitting corrected p, we measured mean last-token block-2 activation
norm against the fingerprint in the original raw-sweep metadata. This provides
an independent check on input recovery. After inference, we also compared every
anchor's entire 31×31 response grid with the corresponding original grid and
checked whether it is the nearest match among all 30 recovered grids.

## A/A execution audit

`prepare_audit_runtime.py` exports unmodified original `src/model.py` and sweep,
parametrization, activation, and fit modules using `git show` at the pinned
revision. `reference-export.json` records their hashes. `audit_reference_execution.py`
runs the actual reference functions on our frozen v1 inputs. It checks:

- Original `hidden_states[3]` versus our saved block-2 prefixes for all 30 anchors.
- Original final-token `hidden_states[27]` versus our block-26 measurement.
- Original suffix versus our native replay on identical inputs, batches 1/8/64.
- All 961 perturbation points on one anchor, including BF16 casting differences.
- A complete Gender×Age surface with our fresh vectors, then released vectors.

An initial audit-only metadata expression broadcast a final-token measurement
against all five token positions. We corrected it to compare final tokens and
reran the capture/component checks; all comparisons are zero. This did not
affect any sweep or fitted exponent. `execution-audit.json` contains corrected
metadata. `audit_capture_modes.py` independently checks cache and mask settings.

The legacy replay uses the **original released direction vectors and unmodified
reference sweep/suffix functions**. It perturbs after block 2, measures after
block 26, uses BF16 SDPA on MPS, the 0:2:60-degree grid, and batch size 64. The
reference is the original clean single-example output. The runtime remains
our current Torch/Transformers environment; this is not a CUDA or historical
runtime reconstruction. See `environment.json`.

Fitting is unchanged: median response across 30 anchors; per-pair threshold
half the smaller maximum axis response; exact-geodesic contour fitting with
the original fitter. No threshold or selection is tuned to the new outcomes.
Small residual differences from the cached grids are reported, not discarded.

## Reproduction

Run from the qwench repository with the reference git checkout in
`../../work/fsec-reference` and pinned Qwen3-1.7B weights cached locally.
The model scripts require a Mac with MPS and the native packages in
`environment.json`. Do not overwrite the saved inputs to change a protocol.

```bash
python scripts/prepare_audit_runtime.py
# Use an isolated environment with datasets==4.8.5 and pinned tokenizers:
HF_HUB_DISABLE_XET=1 /path/to/legacy-env/bin/python scripts/recover_legacy_anchors.py
# Then use the native inference environment:
python scripts/check_legacy_anchor_fingerprint.py
python scripts/audit_capture_modes.py
python scripts/audit_reference_execution.py
python scripts/replay_legacy_anchors.py
```

Recovery/replay scripts refuse to overwrite completed inputs/results. To examine
this completed audit, use the stored arrays rather than rerunning inference.
The following requires only the reproduction/plot dependencies plus safetensors:

```bash
uv run --no-project --with-requirements requirements-reproduction.txt \
  --with safetensors==0.8.0 python scripts/analyze_implementation_audit.py
```

The original v1 report's frozen inputs and response arrays remain intact. Its
interpretation now needs the sampler correction; it should not be described as
a historical-anchor reproduction or used to claim a 4B-specific failure.

## Primary sources

- [Paper repository data loader](https://github.com/FranciscoHS/fsec-paper/blob/a9355d4001500708a40a255bacfe559e6dc269c0/src/data.py)
- [Original activation extraction](https://github.com/FranciscoHS/fsec-paper/blob/a9355d4001500708a40a255bacfe559e6dc269c0/scripts/lib/activations.py)
- [datasets 4.8.5 streaming implementation](https://github.com/huggingface/datasets/blob/4.8.5/src/datasets/iterable_dataset.py)
- [datasets 5.1.0 streaming implementation](https://github.com/huggingface/datasets/blob/5.1.0/src/datasets/iterable_dataset.py)
