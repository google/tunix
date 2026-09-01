# P58.32 — Zero-HP finite decode/prefill warning admission

Status: `LOCAL CONSTRUCTION PASS / TARGET NOT RUN`

## Decision

The exact P58 Qwen3-4B-Instruct-2507 Zero-HP production full profile may
continue through a finite `S_decode_vs_S_prefill` difference. The difference
and its directly derived `w`, `w*r`, clip, and TIS observations are emitted as
warnings with full boundary evidence and W&B dose metrics.

This is a convergence experiment, not a Zero-TIM certification. The policy ID
is `deepswe-zero-hp-ab-warning-v1` and the maximum claim is
`convergence-only / alignment-degraded`.

## Closed admission

Warning admission requires all of the following:

- P58 admitted `zero` arm, stage `full`, exactly 1,000 updates;
- profile `qwen3-4b-dp8-tp8-deepswe-v1-hp`;
- Qwen3-4B production geometry: rollout DP8xTP8 plus trainer DP8xTP8 and 128
  global trajectories;
- a real production run, not precheck, checked-VMA, seam, one-host, or other
  diagnostic mode.

Ordinary Zero and every diagnostic profile remain strict. No new flag was
introduced; the existing `CANON_DEEPSWE_ALIGNMENT_WARN_ONLY` is derived by
the renderer and independently revalidated by shell and Python contracts.

## Hard stops preserved

`S_prefill_vs_T_old`, `T_old_vs_T_current`, and derived `r` remain exact.
Any nonfinite value, invalid shape, weight/replica drift, gradient failure,
optimizer transaction failure, OOM, or corrupt evidence remains fatal. A
finite A-B warning never suppresses those failures.

## Validation and target ceiling

Host regressions cover the positive finite A-B path and negative B-C,
trainer-repeat, nonfinite, wrong-profile, ordinary-Zero, and diagnostic
controls. P34 static passes ten suites and the flag registry passes 409/409.
The full digest-pinned image gate passes with `alignment_policy=1`,
`zero_hp_full=1`, `checked_vma_diagnostic=1`, `coarse_seam=1`, and
`P58_EXACT_IMAGE_CPU_PASS`.

No 128-device target has run under this policy. The first separately approved
target must retain every K23 accumulation receipt, complete all 16 backward
groups, keep B-C/current exact, record A-B warning dose, make the intended
first TPU-resident optimizer commit, and write a durable checkpoint. It may
only be reported as an alignment-degraded convergence run. A later strict
target is still required for Zero-TIM promotion.
