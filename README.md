# qwench

*Poke Qwen. Plot the plateau. Peek through the lenses.*

A small pilot on feature-specific perturbation geometry in **Qwen3.5-4B**,
with J/R/J++ readouts alongside the activation measurement.

The question: do mixtures of two candidate feature directions stay inside
an activation plateau longer than the pure directions? If so, what do the
lenses see while that happens?

`p > 2` would support privileged feature geometry. It would not, by itself,
establish an error-correction mechanism.

## Start here

- [Experiment plan](docs/pilot.md): measurement, controls, lens extension, and stopping rule.
- [Draft configuration](configs/pilot.toml): proposed settings, with unresolved integration details left explicit.
- [Local assets](docs/local-assets.md): located weights, layer conventions, and remaining gaps.
- [Findings template](notes/findings.md): record the result, including a null result.

Status: cached paper Figure 2 reproduced, and a small local Qwen timing run
completed. Sentiment/language candidates are built, but have not passed the
output-steering checks. The feature sweep has not started. Active readouts are R/J++;
J and O are deferred. The TOML file records the proposed pilot; the two
scripts below have their own small, fixed workloads.

## Completed checks

- [Cached Figure 2](reports/paper/paper-figure-2.png): `p = 2.40415`, radial
  residual `1.219%`, matching the paper's rounded `p = 2.40`.
- [Timing receipt](reports/timing/timing.json): about 26 examples/s for
  5-token prompts and 13 examples/s for 32-token prompts, batch size 8,
  on the local M4 Pro. Random directions only; no feature exponent measured.
- [Findings](notes/findings.md): interpretation, sanity checks, and limits.
- [Direction validation](reports/directions/findings.md): block-2 and block-8
  candidate vectors, held-out recognition, random controls, and steering checks.
  Block 8 passes recognition/overlap, but neither pair is promoted for a sweep.

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

## Starting points

- Ferreira da Silva & Heimersheim, [Evidence for Feature-Specific Error Correction in LLMs](https://arxiv.org/abs/2606.24964).
- [Reference implementation](https://github.com/FranciscoHS/fsec-paper), inspected at
  `a9355d4001500708a40a255bacfe559e6dc269c0`.
- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B).

The reference repo also contains a J-lens **direction** study on Qwen3.6-27B.
Our lens extension examines **readouts of perturbed activations**; the primary
directions come from matched contrastive prompts.
