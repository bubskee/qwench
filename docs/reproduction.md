# Reproducing qwench

Run all commands from the repository root. Python **3.12** is the tested version.
The portable entry point is `scripts/reproduce.py`; `--help` lists its commands.
Saved results live in `reports/`. New runs go below `work/`, which git ignores.

## Choose a reproduction level

| Command | What it actually does | Dependencies / hardware |
|---|---|---|
| `verify` | Hash-checks the four toy-run manifests and reads the saved audit status | Python standard library; CPU |
| `toy-figures` | Rebuilds five figures and per-network summaries from saved evaluations and fits | `requirements/figures.txt`; CPU |
| `toy-refit` | Recomputes all toy contour fits from saved response grids, then rebuilds figures | `requirements/figures.txt`; CPU |
| `paper` | Downloads one pinned upstream cache and reruns the original paper Figure 2 plotting code | `requirements/figures.txt`; CPU, git, network |
| `toy-train` | Trains all 30 networks from initialization, selects pairs, measures responses, fits, plots, and audits | `requirements/toy.txt`; Apple Silicon MPS |

Figure regeneration uses saved fits. Refitting uses saved inference. Full training
is a separate, substantially more expensive run. None of these commands loads an
LLM. Fresh LLM inference has additional local assets and setup described in the
[experiment index](experiments.md).

## Install

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/figures.txt
```

Or, with uv:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements/figures.txt
```

The analysis dependencies are pinned NumPy, SciPy, and Matplotlib. The toy
dependency set additionally installs PyTorch and safetensors. We do not require
an editable package install, HF login, or model downloads for saved toy analyses.

## Verify and rebuild saved toy figures

```bash
.venv/bin/python scripts/reproduce.py verify
.venv/bin/python scripts/reproduce.py toy-figures
```

Expected outputs in `work/reproductions/toy-figures/`:

- `training-and-sharing.png`: training trajectories and competent neuron sharing.
- `geometry-comparison.png`: encoder versus matched random directions by activity.
- `sharing-versus-p.png`: the visual embedded in the README.
- `feature-errors.png`: all feature errors, including failures.
- `feature-sharing-and-error.png`: contribution spread alongside task error.
- `models.csv`, `model-summary.json`, and `reproduction-check.json`.

The receipt reports both exact equality and a numerical comparison of the
regenerated model summary against the saved JSON. Floating values allow relative
tolerance 1e-6 and absolute tolerance 1e-8 for backend roundoff; schemas, counts,
flags, and other values must match exactly. PNG bytes can depend on fonts/rendering
metadata. The command fails if its numerical comparison exceeds tolerance.
`verify` checks 225 manifest
entries, including source/protocol files; it validates their integrity, not the
scientific interpretation or a new training run.

To refit the saved grids:

```bash
.venv/bin/python scripts/reproduce.py toy-refit
```

This copies roughly 330 MB of raw grids and produces new `analysis/` fits and
the five plots in `work/reproductions/toy-refit/`. It uses the original fitter,
thresholds, cohorts, and flags. It compares the regenerated contour-cohort and
model-summary JSON against the committed versions. The original run took about
109 seconds to fit on the M4 Pro; runtime varies with the machine.

Every analysis starts in a **new** output directory. To run it again:

```bash
.venv/bin/python scripts/reproduce.py toy-figures --out work/reproductions/figures-2
```

The wrapper restricts output to this clone's `work/` and refuses an existing
directory. It copies inputs and leaves committed reports intact. No environment
variables or knowledge of the old local directory layout are required.

## Reproduce the cached paper figure

```bash
.venv/bin/python scripts/reproduce.py paper
```

This fetches a sparse reference checkout at
`a9355d4001500708a40a255bacfe559e6dc269c0` into `work/fsec-reference/`, verifies
the 116 KB sweep's Git object, and runs the original plotting source. It does
not download LLM weights. The output folder is `work/reproductions/paper/`.
Expected fit: **p ≈ 2.40415**, with about **1.219%** radial residual.

This reproduces a cached paper result. It is separate from the fresh 1.7B
sampler audit and from the 4B generalization experiments. The upstream cache is
a pickle loaded only after its pinned Git blob has been checked.

## Train the toy networks from scratch

The full training recipe uses the original **Apple Silicon MPS float32** backend.
CPU-only machines can plot and refit saved measurements; training on other
backends is not exposed as an equivalent reproduction. Install the full set:

```bash
.venv/bin/python -m pip install -r requirements/toy.txt
.venv/bin/python scripts/reproduce.py toy-train --prepare-only
```

Preparation does not train. It copies byte-identical original sources, frozen
protocols, and the stored background/pair ordering into:

```text
work/reproductions/toy-train/
  run.json                    # hashes of staged sources and inputs
  outputs/qwench/              # self-contained original script layout
    scripts/
    configs/
    reports/                  # new training and measurement results
  work/                       # generated dataset and optimizer/RNG checkpoints
```

Then execute the prepared run:

```bash
.venv/bin/python scripts/reproduce.py toy-train --resume
```

Or prepare and execute in one command:

```bash
.venv/bin/python scripts/reproduce.py toy-train --out work/reproductions/full-run-2
```

Both source cohorts start from **initial weights** for a full 100,000 updates.
The published run continued the core cohort through an interruption at step
30,000; this route does not need that private optimizer checkpoint. Training
seeds vary initialization while data/minibatches are paired. The wrapper runs
the 18-network core and 12-network additions sequentially, then assembles them.

After training it checks latent/input intervention equivalence and random-control
Gram matching, writes competence-based pair selection **before** measuring p,
runs the 41×41 sweeps and 81×81 checks, fits all contours, plots, summarizes, and
reruns the numerical audit. It makes no git commits or remote changes. Fresh
results can differ numerically across PyTorch versions or hardware; preserve
the recorded environment and compare coverage, error, sharing, and flagged fits.

Each source trainer saves weights and Adam/scheduler/RNG state every 10,000
updates. Reuse `--resume` on the same output to continue from the last saved
checkpoint; complete cohorts are skipped. Staged source/input hashes must still
match. Log rows after that checkpoint are archived in `resume-discarded-*.jsonl`
before replay, keeping the training audit consistent. If interrupted before the
first checkpoint, choose a new output folder.
An interruption between writing a cohort's final checkpoint and completion marker
can be resumed too. Available disk should accommodate generated datasets,
optimizer checkpoints, and raw response grids (several hundred MB each).

## Earlier LLM experiments

See the [experiment index](experiments.md) for every result and its original
entry points. The [legacy command notebook](legacy-reproduction.md) preserves
the original execution history, including machine-specific paths. Those paths
are provenance, not prerequisites for the portable toy commands above.

The historical-anchor 4B comparison is still pending. Do not describe the
existing exploratory 4B run as that completed matched-anchor comparison.

## Maintainer checks

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/reproduce.py verify
git diff --check
```

The small wrapper tests cover output protection, staged-source integrity, and
resume preflight. Scientific validation is recorded in the run audits and the
reproduction receipts; the wrapper tests do not replace it.
