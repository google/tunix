# T9f — Record-full pre-response termination identity

- Status: source frozen as `89a58e24d02ed42b2bc39126eb592ca0a1426bd3`;
  host and complete pinned-image PASS; publication approved 2026-09-07;
  source base `2833977c1daae9971330e9be9bf16e546b9f0f4f`.
- Scope: fix the M15 r09 policy-step-40 collector-to-evidence failure, also
  covering the same P45 record-full path. Source/evidence publication only;
  no target launch.

## Evidence and first failing boundary

`debug_logs/canon_p57_fl_zero_m15_r09_crash_20260906/` (repository root)
contains the immutable report, raw tail, traceback, and SHA256SUMS; all three
hashes verified on 2026-09-06. The first-turn MODEL_TIMEOUT returns zero
completed responses and an empty request list. `append_full_record_batch_map`
incorrectly rejects that legitimate unexercised row. The sidecar writer and
terminal classifier repeat the same nonempty-list requirement.

## Change contract

Preserve the existing training row, its order, tokens, masks, rewards and
policy version. Record a scoped, explicit empty-response receipt only for
record-full: terminal reason, independently counted completed model calls,
trajectory steps, completion-token count and action-token count. Empty request
lists require zero completed calls/steps/tokens/actions, zero later-turn
comparisons, and no token difference. A timed-out request may have been
submitted; this receipt must not claim that no request reached the engine.
All other missing/duplicate/foreign request identity remains fatal.

Validate the receipt at collector, row map, sidecar and classifier. Bind it
into the immutable sidecar metadata and cross-check actual completion-valid
and action masks. Count these rows separately as no-response/unexercised,
never as token-equality comparisons. Do not change defaults, timeout policy,
numerical gates, training algorithms, JobSet/YAML or launch configuration.
The previously reviewed sorting and Orbax concerns are separate follow-ups.

## Gates

1. Reproduce the old rejection with a zero-response timeout fixture.
2. Host positive/negative controls for row-map and complete sidecar/classifier
   joins, including forged no-response receipts and nonzero action masks.
3. CPU execution of the actual collector timeout path and learner batch
   preparation; verify that normal/legacy rows remain unchanged.
4. P57/V1, flag audit, diff/syntax checks and pinned-image regressions when
   available. Real GCS, one-host and DP8xTP8 remain separate, unrun gates.

## Result log

- 2026-09-06 preregistration: latest pull adds only the r09 error evidence.
  Worktree clean and canonical runtime preflight PASS before edits. The crash
  is host evidence validation after rollout, not a backward exception. The
  supplied head tail is partial and does not certify a full 300-update run.
- 2026-09-06 implementation/host verification: reproduced the old
  `record-full row identity is malformed` at `token_continuity.py:1913` using
  the step-40 first-response-timeout shape. Added the shared receipt validator
  and independently counted completed calls, passed the receipt through the
  learner row map, and bound it into the NPZ sidecar. The full classifier now
  requires `echo_count = trajectories - witnessed_no_response + later_turns`,
  plus exact request-list coverage and uniqueness. Negative controls reject
  missing receipts, nonzero completed-call/step/token/action counters, illegal
  terminal states, a differing token row, a conflicting nonempty request list,
  nonempty sidecar masks and mismatched sidecar receipt identity.
- Verified by P57 238/238, V1 102/102, flags 422/422 and diff/syntax checks.
  The initial host run failed because the standalone alignment test imported
  the training package's optional dependencies; the test now loads the real
  shared validator under its existing lightweight host import harness, not a
  mocked verdict. Failed and successful host outputs are preserved in
  `evidence/t9f-host-20260906/`.
- Verified by pinned CPU execution of the actual collector timeout path,
  sidecar writer, and learner `_process_results`: off/on training tensor
  leaves match exactly and the empty row keeps a zero action mask. One
  focused attempt had missing source/image fields in its fixture; another
  invocation used the wrong test-class name. Both were corrected without
  weakening runtime checks. These tests do not execute a TPU optimizer.
- Pending: complete pinned-image gate is running. Patched one-host/DP8xTP8,
  actual storage recovery and 300-update completion remain unverified. This
  repair does not diagnose the generation service's 1,800-second latency.
- 2026-09-06 final local gate: complete digest-pinned image exits 0 with
  `V1_HP_EXACT_IMAGE_PASS`; both 37-file overlays, TP4/TP8 shim tests, TiTO
  transport/record/capsule/classifier tests, the actual new collector timeout,
  learner off/on tensor equality and sidecar controls pass. Runtime/gate
  hashes are in `evidence/t9f-host-20260906/receipt.json` (SHA256
  `88d28ed4fa4aea68ad1ec330903431b91e5f5e934a0fc7636ae2dab8fbbc5fe0`).
  The console capture is explicitly partial because a tool response was
  truncated; no complete raw-log or signed target claim is made. CPU fixtures
  exercise actual methods but mock sampler/storage/cluster boundaries.
  No deployment configuration, overlay source, precision, backward or
  optimizer code changed. Final runtime hashes match the checked files.
  Next: review local diff/receipt and obtain separate publication approval.
- 2026-09-07 release verification: user approved commit/push; source CL is
  `89a58e24d02ed42b2bc39126eb592ca0a1426bd3`. Pre-publication fetch
  matched the source baseline exactly, so no rebase was needed. P57 238/238,
  V1 102/102, flags 422/422 and diff checks pass again. All six runtime/gate
  SHA256 values match the original pinned-image receipt; no runtime edits
  were made during closeout. The following CL contains only evidence and
  documentation. Rollback is the isolated source CL, preserving evidence.
  Timeout latency/cancellation acknowledgement, patched TPU, real storage
  and full-horizon training remain unverified; no launch is authorized.
