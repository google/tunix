# P38s21 analysis receipt — 2026-08-18

## Run-level status

`ANALYSIS_GRADE_PARTIAL_2_OF_3`.

The committed evidence directory verifies its own 12-file SHA manifest and
contains sealed `ROUND_COMPLETE` receipts for rounds 0 and 1. Round 2 exceeded
the configured 4-GiB local terminal-evidence bound. The run therefore lacks
round 2, controlled exit, root `COLLECTED`, and root `COMPLETE` and cannot be
called a complete three-round target gate.

## Admitted numerical evidence

- Round 0: 45,276 actions; A-B 47 elements / 76 bytes;
  `max_abs=0.162704...`; B-C exact.
- Round 1: 44,695 actions; A-B 7 elements / 9 bytes;
  `max_abs=0.230720...`; B-C exact.
- The committed classifier joins 54/54 selected red points.
- All 54 captured complete final-hidden rows are byte-exact between A and B.
- All 54 first measured red intervals are classified `lm_head_logits` from
  diagnostic multiword logit evidence.

## Claim boundary

This is an interval localization, not a mechanism proof. The evidence does
not contain full-vocabulary byte equality for every output and cannot prove a
specific K-reduction order, XLA tiling, cast/fusion choice, or collective.
Vocabulary is an output dimension of `TD,DV->TV`; the dot reduction is over
hidden K=4096. The exact-hidden claim applies to the selected 54 red points,
not every token in the run.

The next registered phase is P38.2w, which first screens the real checkpoint
lm_head at current M=16/M=256 on one v5p host, then permits one slim at-scale
dot-algorithm discriminator. No additional terminal corpus is required.
