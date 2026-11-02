# Portable reproduction checks

Checked on **2026-11-01**, using Python 3.12.14 on the original Apple Silicon
workstation. The CPU analysis environment was created separately and installed
only `requirements/figures.txt`; PyTorch was confirmed absent.

| Check | Result |
|---|---|
| `reproduce.py verify` | All 225 saved manifest entries match; original experiment sources and evidence retain their hashes |
| `reproduce.py toy-figures` | Five plots regenerated; model-summary JSON exactly equals the committed version |
| `reproduce.py toy-refit` | All 30 networks / 6,264 aggregate surfaces refit; contour-cohort and model-summary JSON exactly equal the committed versions |
| `reproduce.py paper` | Pinned upstream cache fetched and verified; p ≈ 2.40415, radial residual ≈ 1.219% |
| Fresh local clone | Artifact verification, `make figures`, and training preparation succeed without the original sibling work folders |
| Wrapper tests | Seven pass: output protection, symlink containment, staged-source hashes, resume-log rollback, and numerical comparison |
| Markdown links/images | All local targets in the README, docs, and requirements guide resolve |

The fresh clone used a local shared Git object store and the same clean analysis
environment. It had its own working tree and no copied original scratch/model
assets. This checks file/layout independence; it is not a test on another OS or
CPU architecture. Numerical receipts also allow small floating-point differences
on other backends, as documented in the [guide](reproduction.md).

Full retraining was **not repeated** for this repository reorganization. Its
sources are the unchanged, previously executed 100,000-update trainers. The new
wrapper's training preparation, source-integrity checks, and resume-log behavior
were checked; a new complete training run remains a separate use of `toy-train`.
The original numerical audits and checkpoints remain the evidence for the
published trained networks.
