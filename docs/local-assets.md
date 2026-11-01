# Existing local assets

Checked 2026-11-01. Exact machine-specific paths and checksums are recorded in
[local-assets.json](../configs/local-assets.json). No weights were downloaded,
copied, moved, or deleted during this audit. The only HF downloads were metadata
and the small O-lens checksum manifest.

| Asset | Located | Verification |
| --- | --- | --- |
| Qwen3.5-4B | HF cache | Both shards referenced by the cached index exist; weights not rehashed |
| R | Original release output | Checksum matches release manifest; maps inspected |
| J++ | Original release output | Checksum matches release manifest; maps inspected |
| O, block 26 expanded | Original release output | Adapter and contract match pinned public HF checksums |
| O, block 30 expanded | Original release output | Adapter and contract match pinned public HF checksums |
| J | Not located | No standalone Qwen3.5-4B J file found in searched locations or matching public bubskee HF releases |

The pinned base revision is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
R and J++ each have FP32 2560 × 2560 maps for zero-based block outputs 0–30,
transported to block 31 before final RMSNorm. The reader uses Qwen3.5's
`1 + weight` RMSNorm gain and the native tied LM head in FP32. Preserve this
readout convention when adapting the code.

O-lens L26/L30 refer to zero-based block outputs 26 and 30 before final norm.
Use each adapter's saved tokenizer, prompt IDs, injection slot, and
`8000 * h / ||h||` input transform. The historical O-lens base revision is
null in both contracts; matching release checksums does not resolve that gap.

The machine is an Apple M4 Pro with 48 GB unified memory. An existing Python
environment has PyTorch 2.14.1 and available MPS support. The public O reader
requires CUDA; an exploratory local L26 MPS port exists. L30 still needs an
equivalent adaptation and smoke check. No inference or speed benchmark was
run as part of this inventory.

A prior euphoria-axis bootstrap references an external
`workspace-lenses/qwen3.5-4b/j-lens/lens.pt`. That checkout was not located.
The public R/J++ release does not contain J, and R is not a substitute for J.
Leave this unresolved until the actual checkpoint and provenance are found.

For the pilot, collect R/J++ at blocks 8, 16, 26, and 30, and O at 26 and 30.
Decode O after the sweep, using a small selected panel. Its generated text
must remain separate from the primary residual-distance measurement.
