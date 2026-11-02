# Dependency entry points

Use Python 3.12. From the repository root:

| File | Use |
|---|---|
| `requirements/figures.txt` | CPU toy figures/refits and the cached paper figure; no PyTorch |
| `requirements/toy.txt` | Full toy training and numerical measurement; adds PyTorch and safetensors |

Install with `.venv/bin/python -m pip install -r requirements/figures.txt`, or
substitute `requirements/toy.txt` for full training. Training requires Apple
Silicon MPS; saved-data analysis works on CPU. Direct numerical dependencies
are pinned; transitive rendering dependencies and system fonts may affect PNG
bytes, so the runner compares numerical JSON products.

The root `requirements-*.txt` files are the original experiment dependency sets.
They remain in place for existing commands and recorded hashes. LLM-specific
sets and model prerequisites are described in the [experiment index](../docs/experiments.md).
