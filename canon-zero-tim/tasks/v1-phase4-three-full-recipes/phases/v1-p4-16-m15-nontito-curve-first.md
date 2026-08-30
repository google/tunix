# V1.P4.16 — M15 non-TITO curve first

Status: active source candidate; focused 43/43, P57 181/181, V1 92/92, and
flag audit 409/409 pass. Immutable-image, exact remote readback, render, and
DP8xTP8 target remain outstanding.

## Decision

The user withdrew exact TITO as the M15 production default before its first
one-host or DP8xTP8 target. The immediate goal is a usable M15 training curve
on the historical rendered-text multi-turn input path. Exact TITO remains an
experimental implementation and will be debugged in a separate carrier.

This is a delivery/default rollback, not deletion of the experiment. Preserve
V1.P4.15, commit `3fc7ef8b`, all successful construction evidence, and all
future failures. Do not claim that non-TITO is numerically superior, or that
TITO is wrong, from this decision alone.

## Production contract

- workload: exact P45 and M15/main optimized FrozenLake full recipes;
- horizon: 300 updates, evaluation off, checkpoints off;
- APC: off;
- M15 alignment policy: finite A-B warning lane remains; B-C, nonfinite,
  backward-health, replica, and optimizer failures remain fatal;
- raw manifest: `CANON_M15_TOKEN_CONTINUITY` absent;
- resolved environment: `CANON_M15_TOKEN_CONTINUITY` absent;
- empty and `0` values count as presence and fail;
- runtime: zero `[CANON_M15_TOKEN_CONTINUITY]` receipts;
- neighboring workloads: selector remains absent.

The generic observer/exact implementation in `token_continuity.py` and the
trajectory engine remains untouched so a later dedicated debug carrier can
exercise it. No production renderer may opt into it in this phase.

## Gates

1. Focused renderer/classifier positives prove P45 and M15 raw/resolved
   absence; negatives inject `exact`, empty, wrong-profile, wrong-shape, and a
   runtime token receipt.
2. Full P57, V1, flag-audit, syntax, and diff-hygiene host gates pass.
3. After separate approval and an immutable source SHA, the complete pinned-
   image gate ends with `m15_tito_impl=1 m15_tito_default=off`.
4. After separate commit/push/readback and render/launch approval, the M15 full
   target returns 300 optimizer commits plus all registered warning/fatal,
   backward-health, timing, XProf, Perfetto, and SHA evidence.

The current launch set is M15 only. The render-only wrapper may produce the
paired P45 manifest for mechanical comparison, but its P45 apply command must
not be executed in this wave.

## Claim ceiling

The next M15 run is `convergence-only / alignment-degraded` while its finite
A-B warning lane is active. It is not a TITO experiment and cannot produce a
TITO verdict. A later TITO phase must use a separate run identity and add
real-tokenizer role-boundary and nonterminal-EOS coverage before production
admission is reconsidered.

## Result log

- 2026-08-30: profile, environment admission, both production renderers, full
  classifier, and focused tests changed to require selector absence for M15 as
  well as P45. The experimental runtime implementation was intentionally left
  intact. Focused renderer/classifier suite passed 43/43; Python/Bash syntax
  and diff hygiene passed. Full host then passed P57 181/181 and V1 92/92;
  flag audit passed 409/409. Pinned-image, commit, push, render, launch,
  Kubernetes mutation, TPU use, and target remain unrun.
- 2026-08-30: user clarified the post-publication action: launch only the
  non-TITO M15 full training curve. P45 may be rendered for comparison but is
  explicitly excluded from this wave's apply set.
