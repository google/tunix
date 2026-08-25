# State

- Status: active
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: a deterministic clean-run reproducer reaches `FIRST_RED_LOCALIZED`; the smallest localized repair passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and a separately approved DP8xTP8 target run with A-B=0 bytes and B-C=0 bytes.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`
- Current baseline: commit `d6629c8c9406c64e578aa84d22a68ed925d2156b`
- Release state: Attempt 1 geometry mismatch preserved; bounded entrypoint repair is host-pass and uncommitted.
- Current phase: Phase B ATTEMPT-1 GEOMETRY REPAIR HOST PASS / EXACT-IMAGE AND TARGET CONTROL NOT RUN, [freeze a replay carrier](phases/phase-b-replay-carrier.md)
- Last verified fact: the entrypoint now distinguishes legacy P38 (`DP16`, 8 x 4-prompt units, token IS) from the exact M15 APC target carrier (`DP8`, 1 x 32-prompt unit, no IS); positives and adjacent negatives pass without changing any numerical path.
- Next action after user approval: commit/push the bounded repair, then separately approve the pinned exact-image gate; only after that may a new APC-off control be rendered from the full source SHA.
- Blockers: post-fix exact-image and fresh DP8xTP8 APC-off target control have not run.
- Key artifacts: [Attempt-0 receipt](evidence/v1_apc_m15_attempt0_20260825/receipt.json), [Attempt-1 receipt](evidence/v1_apc_m15_attempt1_20260825/receipt.json), [Phase3 state](../v1-phase3-prefix-cache/state.md)
- Updated: 2026-08-25T03:14:09Z
