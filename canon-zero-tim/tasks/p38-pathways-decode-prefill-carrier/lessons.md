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

## 2026-08-11 — A historical negative arm is not a lost repair

Phase 13's KV-unified branch printed its execution marker and left the measured
values unchanged. Reusing the idea in a new version/topology/workload domain is
valid as a fresh causal test, but describing the old arm as a successful fix
reverses the evidence. Read the terminal verdict, not only the original plan or
mechanism hypothesis.

## 2026-08-11 — Token capsules do not reconstruct scheduler provenance

A capsule containing exact tokens, masks, and logprobs still cannot prove the
original serving call boundaries, page tables, or request distributions. A
schedule inferred from action and validity masks is useful as a controlled
counterfactual, but it must carry an explicit claim ceiling and a local
reproduction prerequisite. Do not rename derived metadata as captured truth.

## 2026-08-13 — Divisibility is not diagnostic coverage

A four-prompt unit solved the DP16 divisibility failure, but stopping after its
32 trajectories silently changed a 256-trajectory experiment into a subset
experiment. The subset was exactly aligned while prior full batches carried
their sparse mismatches in later rows. Batch contracts must attest both local
divisibility and complete source coverage; a partial tail is inconclusive and
must be rejected rather than reported as a workload-wide PASS.
