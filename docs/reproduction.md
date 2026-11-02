# Reproduction notes and experiment index

The sampler audit supersedes the initial historical-reproduction interpretation.
Commands below run from the repository root. Machine-specific paths describe the
original local runs; saved plots can be regenerated without model inference.

*Poke Qwen. Plot the plateau. Peek through the lenses.*

A small pilot on feature-specific perturbation geometry in **Qwen3.5-4B**,
with J/R/J++ readouts alongside the activation measurement.

The question: do mixtures of two candidate feature directions stay inside
an activation plateau longer than the pure directions? If so, what do the
lenses see while that happens?

`p > 2` would support privileged feature geometry. It would not, by itself,
establish an error-correction mechanism.

## Start here

- [Experiment plan](../docs/pilot.md): measurement, controls, lens extension, and stopping rule.
- [Draft configuration](../configs/pilot.toml): proposed settings, with unresolved integration details left explicit.
- [Local assets](../docs/local-assets.md): located weights, layer conventions, and remaining gaps.
- [Findings template](../notes/findings.md): record the result, including a null result.

Status: cached paper Figure 2 and a broader cached audit are complete. A fresh
Qwen3.5-4B generalization stress test is complete with a sealed cohort of 37
candidate directions, 33 tested feature pairs, and seven accepted controls.
This geometry experiment has no output-steering gate; the earlier separate
steering checks did not pass. R/J++ readouts remain a separate extension.

## Completed checks

- [Cached Figure 2](../reports/paper/paper-figure-2.png): `p = 2.40415`, radial
  residual `1.219%`, matching the paper's rounded `p = 2.40`.
- [Timing receipt](../reports/timing/timing.json): about 26 examples/s for
  5-token prompts and 13 examples/s for 32-token prompts, batch size 8,
  on the local M4 Pro. Random directions only; no feature exponent measured.
- [Findings](../notes/findings.md): interpretation, sanity checks, and limits.
- [Direction validation](../reports/directions/findings.md): block-2 and block-8
  candidate vectors, held-out recognition, random controls, and steering checks.
  Block 8 passes recognition/overlap, but neither pair is promoted for a sweep.
- [Paper audit](../reports/paper-audit/findings.md): eight original plotting
  scripts, 18 sampled raw-grid refits, the recomputed toy boundary,
  lower-tail counterexamples, and the appendices' limitations. Main cached
  numerical patterns reproduce; the residual figure's exact caption counts differ.

## Reproduce these checks

Run from this repository. The paper script uses git to obtain a sparse
pinned reference checkout containing one 116 KB cached sweep. It runs the
original plotting code and saves provenance, the figure, contour points,
and numerical response data in `reports/paper/`.

```bash
uv run --no-project --with-requirements requirements-reproduction.txt \
  python scripts/reproduce_paper.py
```

For local timing, use an environment matching `requirements-timing.txt` and
the already cached model/lens paths in `configs/local-assets.json`:

```bash
/Users/bubskee/workspace/qwen35-4b-lenses/.venv/bin/python \
  scripts/time_local.py
```

That script performs full BF16 decoder forwards with a norm-preserving
random intervention at block output 2, measures block output 30, verifies
the zero-angle intervention, and times R/J++ decoding separately. It does
not launch the proposed feature sweep or download model weights.

## Direction validation

Using the same local Python environment:

```bash
python scripts/validate_directions.py --stage extract
python scripts/validate_directions.py --stage causal
python scripts/validate_directions.py \
  --protocol configs/direction-validation-v2.json \
  --out reports/directions/v2 --stage all
python scripts/validate_directions.py \
  --protocol configs/direction-validation-v2-large-angle.json \
  --out reports/directions/v2-large-angle --stage causal
python scripts/validate_directions.py \
  --protocol configs/direction-validation-v2-large-angle.json \
  --out reports/directions/v2-large-angle --stage generate
```

The larger-angle follow-up reuses the exact v2 directions. Existing run
directories reject changed data/protocol hashes; use a new version and
directory for substantive revisions. Source snapshots preserve the exact
implementation used by each completed stage. The current script also saves
downstream L2 for causal scoring; earlier v1/v2 snapshots predate that field.

## Broader cached-paper audit

In an environment with `requirements-paper-audit.txt`, first obtain the
reference checkout with `scripts/reproduce_paper.py`, then run:

```bash
python scripts/audit_paper.py --reference-dir /path/to/fsec-reference
python scripts/audit_conic.py --reference-dir /path/to/fsec-reference
python scripts/plot_paper_audit.py
```

The audit fetches fit/direction caches and a small selected set of raw
sweeps with git. It does not fetch LLM weights or the full sweep archive.
Locally, we reused the existing PyTorch environment without modifying it:

```bash
PYTHONPATH=/Users/bubskee/workspace/qwen35-4b-lenses/.venv/lib/python3.12/site-packages \
uv run --no-project \
  --python /Users/bubskee/workspace/qwen35-4b-lenses/.venv/bin/python \
  --with-requirements requirements-reproduction.txt \
  python scripts/audit_paper.py --reference-dir ../../work/fsec-reference
```

Use the same prefix for `scripts/audit_conic.py`. The plotting-only command
needs the reproduction dependencies. To recompute the toy boundary, run
inside the reference checkout's `toy_model/`, with PyTorch and the plotting
dependencies available:

```bash
OMP_NUM_THREADS=6 python -m scripts.plot_toy_boundary \
  --phase2_npz runs/misalignment_sweep_H1024/phase2_d8192_H1024.npz \
  --out_dir /path/to/qwench/reports/paper-audit/toy-boundary
```

The negative-case boundary uses the original figure script with `--d1 Era
--d2 Tense --exact`, after adding its one raw sweep to the sparse checkout.
See `reports/paper-audit/negative-case.json` for the exact command and selection note.

## Starting points

- Ferreira da Silva & Heimersheim, [Evidence for Feature-Specific Error Correction in LLMs](https://arxiv.org/abs/2606.24964).
- [Reference implementation](https://github.com/FranciscoHS/fsec-paper), inspected at
  `a9355d4001500708a40a255bacfe559e6dc269c0`.
- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B).

The reference repo also contains a J-lens **direction** study on Qwen3.6-27B.
Our lens extension examines **readouts of perturbed activations**; the primary
directions come from matched contrastive prompts.


## Fresh Qwen3.5-4B generalization stress test

[Findings](../reports/stress-4b-v1/findings.md): 25/33 feature pairs and 5/7 controls
have p > 2, with medians 2.187 and 2.152. The clean feature/control split does
not emerge in this pilot. Context and sign changes are material; some strong
individual feature geometries survive. All 56 surfaces and 74,896 response
measurements are complete, including fixed grid and precision checks.

The protocol and candidate prompts were committed before inspecting the new
exponents. The prepared pair list was also committed before the sweep.

- Protocol: `configs/stress-4b-v1.json`.
- Exact paper contrasts plus four new candidate directions: `data/stress-features-v1.json`.
- Sealed pair selection, exclusions, anchors, and task list: `reports/stress-4b-v1/plan.json`.
- Every attempted raw response surface and its quality receipt: `reports/stress-4b-v1/surfaces/`.
- All finite fits, fit flags, per-anchor fits, and sensitivity checks: `reports/stress-4b-v1/analysis/`.

With the pinned model cached locally and dependencies in `requirements-stress.txt`:

```bash
python scripts/stress_4b.py --stage prepare
python scripts/stress_4b.py --stage sweep
python scripts/analyze_stress_4b.py --bootstrap 500
python scripts/plot_stress_4b.py
```

The sweep resumes complete surfaces with matching seals. Do not overwrite
this run to change the protocol: use a new version and output directory.
Analysis and plots can run directly from the saved raw surfaces without
loading model weights. The local DCLM corpus is an explicit adaptation from
the paper's FineWeb anchors; its provenance and hashes are sealed.


## Small-model reference comparison

[Qwen3-1.7B comparison](../reports/qwen17-comparison/findings.md) puts the 4B
stress test beside the paper’s tested small model. The original feature median
is 2.226 and the 17 available original isotropic-control raw refits have
median 1.999. Our feature distribution is similar; both models have weaker
geometry on the same 26-pair stress-test intersection. This is cached original
data versus fresh 4B data, not fresh 1.7B inference. The [fresh positive-control attempt](../reports/positive-control-v1/findings.md)
is now complete and qualifies that comparison further.


## Fresh positive control (complete; sampler corrected in follow-up audit)

The [frozen small protocol](../configs/positive-control-v1.json) runs two exact
contrastive pairs and the first isotropic pair on fresh Qwen3-1.7B, followed
by Qwen3.5-4B. Both use 30 regenerated seed-42 FineWeb-Edu anchors and the
original 2-degree grid. [Methods and fidelity checks](../reports/positive-control-v1/METHODS.md)
document the original-reference measurement and batch-matched numerical check.
Both models passed preparation and eligibility checks before the sweeps.


The [completed findings](../reports/positive-control-v1/findings.md) show fresh
1.7B p values 1.997 / 1.767 / 1.735 and matched-setup 4B values
2.049 / 1.929 / 1.900 (Gender×Age / Wealth×Gender / fixed random).
The saved positive feature estimates were not cleanly recovered, and no
paired cross-model interval excludes zero. This small positive-control
attempt is inconclusive; it does not isolate a 4B-specific reproduction
failure. All six surfaces and 172,980 perturbation cases passed artifact
and numerical checks.

To regenerate the plots from the saved response arrays without model weights:

```bash
uv run --no-project --with-requirements requirements-reproduction.txt \
  --with safetensors==0.8.0 python scripts/analyze_positive_control.py --bootstrap 500
```

For native execution, use `requirements-positive.txt` and the pinned model
weights. With the stored anchors and the reference checkout from the earlier
checks, the preparation and sweep stages are:

```bash
python scripts/run_positive_control.py --stage prepare
python scripts/run_positive_control.py --stage sweep
python scripts/audit_positive_control.py
```

The sweep resumes complete anchors only with matching seals and source.
`prepare_positive_anchors.py` documents regeneration of the FineWeb sampling
recipe; it refuses to overwrite the stored input file. The historical
authors’ token cache is unavailable, so historical input identity is not
verified. Both fresh models use the same saved source documents.


## Why the first 1.7B positive control missed

The [implementation audit](../reports/implementation-audit-v1/findings.md) found
a data-sampling mismatch: `datasets` 5.1 changes both the streaming buffer
shuffle and default shard interleaving. Seed 42 therefore selected 30 entirely
different documents. A single pre-paper sampler, `datasets==4.8.5`, reconstructs
anchors whose per-anchor response patterns match the published grids.

Using the original sweep functions and released vectors, the recovered-anchor
Gender×Age estimate is **2.442**, versus published **2.415** and first-run
**1.997**. Wealth×Gender returns to approximately 2. The exact original token
file remains unavailable; ordered full-response fingerprints strongly support
input recovery. See the [audit methods](../reports/implementation-audit-v1/METHODS.md),
[recovered token windows](../reports/implementation-audit-v1/legacy-anchor-windows.tsv),
and [comparison plot](../reports/implementation-audit-v1/sampler-comparison.png).
The frozen v1 inputs/results are preserved. No further 4B inference was run
during this audit.
