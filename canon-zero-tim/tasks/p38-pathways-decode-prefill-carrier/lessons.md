# Lessons

## 2026-08-10 — A plausible numerical signature is not a gate

A clean mechanism story must be registered as a decision table, not phrased as
the answer an experiment is expected to confirm. Sparse differing bytes did
not imply one-ULP drift: GSM8K and FrozenLake later exposed materially
different maximum errors. Measure exact bits, amplitude, position, shape, and
program context before selecting a numerical repair.

## 2026-08-11 — Evaluate the schedule before judging parameter mutation

An optimizer transaction can be valid while model parameters remain exactly
unchanged. At warmup update 0 the effective LR was exactly zero, so requiring a
model hash change converted correct Adam-state progress into a false failure.
Record the effective schedule value and device-side update evidence before
interpreting parameter immutability.

## 2026-08-11 — Sequence coordinates must use the model's logical prefix

Completion-relative positions can hide a boundary after a long prompt. The
FrozenLake onset appeared only after adding prompt length and expressing each
action in logical KV-prefix and sequence-chunk coordinates. Register and test
the coordinate system itself before inferring a page, tile, or turn boundary.
