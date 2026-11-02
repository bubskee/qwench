# Experiment index and repository map

For the quickest runnable path, use the [reproduction guide](reproduction.md).
The original scripts and protocols retain their names because saved results
seal source hashes. This index groups them by experiment without breaking those
links or rewriting historical evidence.

## Completed experiments

| Experiment / status | Evidence | Original entry points / inputs |
|---|---|---|
| Cached paper Figure 2 | [`reports/paper/`](../reports/paper/reproduction.json) | `reproduce_paper.py`; pinned upstream sweep fetched with git |
| Broader cached-paper audit | [Findings](../reports/paper-audit/findings.md) | `audit_paper.py`, `audit_conic.py`, `plot_paper_audit.py`; `requirements-paper-audit.txt` |
| Local timing and lens timing | [Receipt](../reports/timing/timing.json) | `time_local.py`; `configs/local-assets.json`; `requirements-timing.txt` |
| Contrastive-direction checks | [Findings](../reports/directions/findings.md) | `validate_directions.py`, `plot_direction_validation.py`; `data/direction-prompts-*.json` |
| Exploratory fresh 4B stress test | [Findings](../reports/stress-4b-v1/findings.md) | `stress_4b.py`, `analyze_stress_4b.py`, `plot_stress_4b.py`; `configs/stress-4b-v1.json`, `data/stress-features-v1.json` |
| Cached 1.7B versus fresh 4B comparison | [Findings](../reports/qwen17-comparison/findings.md) | `compare_qwen17_reference.py`; cached upstream results plus saved 4B surfaces |
| Fresh matched-setup 1.7B/4B positive-control attempt | [Findings](../reports/positive-control-v1/findings.md) | `run_positive_control.py`, `analyze_positive_control.py`; `configs/positive-control-v1.json`; **superseded historical-reproduction interpretation** |
| Sampler correction and implementation audit | [Findings](../reports/implementation-audit-v1/findings.md), [methods](../reports/implementation-audit-v1/METHODS.md) | `recover_legacy_anchors.py`, `replay_legacy_anchors.py`, `analyze_implementation_audit.py`, audit/check scripts; recovered anchors stored in the report |
| Released-checkpoint toy-CiS pilot | [Findings](../reports/toy-cis-v1/findings.md), [protocol](toy-cis-pilot-v1.md) | `run_toy_cis.py`, `analyze_toy_cis.py`, `check_toy_cis.py`; `configs/toy-cis-v1.json` |
| Five activations trained independently | [Findings](../reports/toy-scalar-comparison-v1/findings.md), [methods](../reports/toy-scalar-comparison-v1/METHODS.md) | `reproduce.py toy-figures/toy-refit/toy-train`; `configs/toy-scalar-comparison-v1.json` |
| Squared-ReLU matched control and fixed-target recovery | [Findings](../reports/squared-recovery-v1/findings.md), [methods](../reports/squared-recovery-v1/METHODS.md) | `squared_recovery.py`, `plot_squared_recovery.py`, `audit_squared_recovery.py`; `configs/squared-recovery-v1.json`; one checkpoint, no new training |

Script names in the table are relative to `scripts/`. Commands for the older
runs are retained in [legacy reproduction notes](legacy-reproduction.md).

## Trained-toy evidence

- `reports/toy-scalars-core-v1/`: the 18 completed ReLU/GELU/SiLU training histories.
- `reports/toy-scalar-additions-v1/`: the 12 completed Leaky-ReLU/squared-ReLU histories.
- `reports/toy-scalar-comparison-v1/`: aggregate of those 30 networks, final
  weights, per-feature evaluations, raw `surfaces/`, contour `analysis/`, figures,
  audits, numerical summaries, and findings.
- `reports/toy-trained-v1/`: **stopped historical mixed-architecture run**. Its
  partial SwiGLU results are excluded from the active comparison. The core models
  continued from its step-30,000 checkpoint. This is provenance, not a completed
  extra cohort or an alternative recommended training entry point.

The frozen `train_scalar_core.py` and `train_scalar_additions.py` contain the
actual training implementation. `reproduce.py` stages them in a separate workspace
and supplies their expected layout. `reproduction_runtime.py` lets the original
plotting/fitting scripts read saved data without importing a training backend.
The historical `finish_scalar_comparison.py` contains original workstation
paths and automatic local git operations; use the portable runner instead.

## Repository layout

| Path | Purpose |
|---|---|
| `README.md` | Main results, toy visual, and minimal setup |
| `docs/` | Current reproduction guide, experiment index, and original plans/history |
| `scripts/` | Portable entry point plus original training, measurement, fit, and audit sources |
| `configs/` | Frozen protocols and historical asset inventories |
| `data/` | Small checked-in contrastive prompt sets |
| `reports/` | Committed inputs, checkpoints, response grids, fit results, provenance, and plots |
| `requirements/` | Current dependency entry points for analysis and full toy runs |
| `requirements-*.txt` | Original dependency sets, retained for historical commands/hashes |
| `vendor/` | Pinned upstream fitter/plotting code and dependency license notices |
| `tests/` | Lightweight checks for the reproduction wrapper |
| `work/` | Ignored generated environments, cached reference checkout, and reproduction outputs |

## LLM setup and remaining work

The saved 4B plots and fits are included. Fresh LLM sweeps use pinned HF model
snapshots, prompt/anchor records, and the original asset inventory in
`configs/local-assets.json`. That inventory records the original workstation's
absolute paths; it is not a portable download/configuration layer. The historical
sampler also requires the pinned `datasets==4.8.5` environment. Follow each run's
methods before attempting inference on a different machine.

R/J++ lens readouts are a separate extension, not a completed primary result in
these toy measurements. The historical-anchor 4B comparison has not been run.
The initial [pilot plan](pilot.md) and [asset notes](local-assets.md) record that
earlier scope; the completed reports take precedence for current status.
