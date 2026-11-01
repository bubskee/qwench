# Direction validation: built, but not ready for the sweep

2026-11-01. No FSEC surface or exponent was measured in these checks.
Only R/J++ remain enabled for the proposed lens extension; validation here
uses native model probabilities and continuations, independently of lenses.

## What we built

The extraction set has 16 contents, each written positively and negatively
in English and Spanish: 64 prompts. Each sentence ends with the shared
`\nNote:` suffix. We read the last `:` token, average positive-minus-negative
differences across languages, and Spanish-minus-English differences across
sentiments. Both directions are unit vectors. Classification uses a
threshold determined only from the training centroid.

The initial block-2 construction failed the preset recognition/overlap
checks. We made one exploratory revision at block 8, keeping the extraction
set fixed and defining eight fresh test contents and eight fresh causal
anchors before running that revision. The test sets differ between blocks;
this is not a controlled accuracy comparison between layers.

| Check | Block 2, v1 | Block 8, v2 |
| --- | ---: | ---: |
| Sentiment accuracy on 32 prompts from 8 held-out contents | 62.50% | 81.25% |
| Sentiment matched-pair ordering | 12/16 | 16/16 |
| Language accuracy on 32 prompts | 100% | 100% |
| Language matched-pair ordering | 16/16 | 16/16 |
| Signed raw cosine between directions | −0.16234 | −0.00764 |
| Preset absolute-cosine limit | <0.1: failed | <0.1: passed |

At block 8, sentiment accuracy is 81.25% within each language; language
accuracy is 100% within each sentiment. The candidate arrows are good small
test-set detectors, not established context-independent semantic features.
See [recognition and 30° checks](validation.png).

## What pushing the arrows did

For each of eight neutral anchors (four contents in both languages), we scored four fixed continuations:
positive/negative × English/Spanish. Scores are summed native conditional
token log probabilities. To score one feature, we sum probabilities over
the other feature. This measures preferences within a tiny fixed menu;
it is not a general semantic accuracy metric.

We rotate the activation while preserving its norm, at the original prompt's
last token. Block-8 language interventions use the same tangent-plane
Gram–Schmidt order proposed for FSEC: sentiment first, then language.
Two seeded isotropic directions receive the same intervention. Matched
continuation tokenization is checked explicitly. No lens decides success.

At ±30°, block-8 sentiment changes the target preference in the intended
direction on only 2/8 anchors; language does so on only 1/8. Neither passes.
Recognition and steering are different measurements.

Because 30° could still be inside a plateau, we added one bounded exploratory
check at ±45° and ±60°, using the exact same frozen block-8 arrows. We did
not rebuild, select, or reverse them. This follow-up reuses the v2 anchors
and is not independent confirmation.

| Block-8 intervention, comparing +60° with −60° | Median change in praise log-odds | Median change in Spanish log-odds |
| --- | ---: | ---: |
| Sentiment candidate | +0.533 | −0.564 |
| Language candidate | +0.042 | −1.919 |
| Random direction 101 | +0.128 | +1.260 |
| Random direction 202 | +1.691 | −1.129 |

At 60°, the sentiment candidate favors praise on 8/8 anchors, but random
direction 202 does too, with a larger median effect. Language shifts toward
English on 8/8 anchors, opposite to the sign fixed by Spanish-minus-English
extraction. This is not evidence that the language axis is meaningless;
it fails our intended output-steering interpretation. We did not flip its
sign after seeing the result. See [the larger-angle comparison](large-angle-validation.png).

The median downstream L2 response at 60°, pooling signs and anchors, is
20.12 for sentiment and 20.25 for language, versus 18.60 and 18.53 for the
random directions. Maximum observed BF16 norm error is 0.096%. The
interventions do change activations; these measurements alone do not locate
an FSEC boundary or establish a plateau.

## Free writing and limits

We also saved deterministic 20-token continuations for two anchors, at both
signs of each candidate direction. At 30° and 60°, outputs often repeat,
switch into boilerplate, or discuss grammar rather than cleanly changing
review sentiment. For example, on the Spanish botanical-exhibition anchor,
the +30° language intervention starts “The answer should be in English,”
while −30° continues in Spanish. These are qualitative examples, not a
blind-rated evaluation. Full outputs are in `v2/generations.json` and
`v2-large-angle/generations.json`.

The common English `Note:` suffix and plain-text use of a post-trained model
may influence the behavior. Even the baseline continuations are often
boilerplate. Spanish examples are generally longer, and paired sentiment
lengths are not perfectly matched. We have only eight contents/anchors per
test and two random directions. These results do not establish the absence
of semantic features or refute FSEC.

## Decision and files

**Do not start the sentiment × language sweep with these labels yet.**
Both blocks' direction tensors are saved, but neither pair is promoted as
validated. A cleaner review-continuation task with fresh tests is the next
reasonable validation change; fitting an exponent now would get ahead of
our semantic interpretation.

- `v1/`: block-2 directions, raw activations, projections, and native steering scores.
- `v2/`: block-8 directions, fresh held-out tests, anchor geometry, native steering scores, and generations.
- `v2-large-angle/`: same block-8 arrows tested at larger angles, with downstream L2 and generations.
- Each run records input hashes and source snapshots. The large-angle receipts also hash the reused direction tensor.

The vectors are small saved artifacts, not copies of model weights.

Later paper-audit clarification: the output-steering gates above were local
requirements for our intended semantic interpretation, not selection rules
for every candidate pair in the paper. These failed checks do not predict
the directions' FSEC exponents; no such Qwen geometry fit has been run.
See the [paper audit](../paper-audit/findings.md).
