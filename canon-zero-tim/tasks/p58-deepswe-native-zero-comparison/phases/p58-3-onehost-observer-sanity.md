# P58.3 — One-host observer and artifact sanity

Status: `WAIVED` by explicit user decision on 2026-08-21. This is not a PASS.

## Purpose

Use a bounded direct-attached v5p host, if separately approved and available,
to exercise the real Qwen/R2E trajectory-to-trainer path before spending a
128-chip allocation. This phase checks observer neutrality and artifact
durability; it cannot reproduce the paired topology or establish treatment
effect.

## Entry state

- P58.1 and P58.2 exact-image gates pass.
- No one-host model/R2E launch has run in this phase.
- No model download, image publication, secret access, or TPU action is
  authorized by the phase document itself.

## Proposed bounded gate

Use existing local Qwen3-4B-Instruct-2507 weights only. Run one real whitelist
task with two generations, prefix cache off, full P58-compatible trajectory
journaling, forward, backward-no-commit, and optional one update only when the
no-commit receipt and HBM margin pass. Verify that enabling the observer does
not alter token selection or reward for a fixed seed when the runtime permits
that comparison.

Required evidence includes source SHA, direct-device inventory, real R2E task
identity, complete trajectory/status/reward, finite logprobs and gradient,
unchanged state for no-commit, device-resident optimizer evidence for any
optional update, artifact digests, cleanup, and an explicit claim ceiling.

## Exit choices

- `PASS`: real one-host observer/artifact sanity passed; activate P58.4 only
  after separate user approval.
- `WAIVED`: the user explicitly chooses to go directly to paired canaries;
  record the waiver without claiming a one-host PASS.
- `BLOCKED_REAL_ENVIRONMENT`: R2E or local weights are unavailable; do not use
  a fake environment and call it end-to-end.

## Recorded exit

The user chose to skip this optional risk-reduction gate and proceed directly
to a native-only 128-chip three-update canary. No one-host model/R2E rollout
was run, so none may be inferred. P58.4N is active; zero remains deferred.
