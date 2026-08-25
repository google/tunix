# State

- Status: active
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: a deterministic clean-run reproducer reaches `FIRST_RED_LOCALIZED`; the smallest localized repair passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and a separately approved DP8xTP8 target run with A-B=0 bytes and B-C=0 bytes.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`
- Current baseline: operator commit `20826194620db893ab7ac1f005d86247578abb33`.
- Release state: Attempt 2 errors preserved; mixed-program tail and ledger-capacity repair is host PASS and approved for publication; exact-image/target not run.
- Current phase: Phase B ATTEMPT-2 OBSERVER REPAIR HOST PASS / EXACT-IMAGE AND TARGET CONTROL NOT RUN, [freeze a replay carrier](phases/phase-b-replay-carrier.md)
- Last verified fact: Attempt 2 completed more than 1,800 standard serving calls and all four standard tensor-capture strata, then exposed two observer defects: the incident ledger saturated at 268,192,266 bytes on call 326 and the production-congruent drain tail entered `continue_decode`, which the single-path capture assertion rejected. Neither event is an A-B/B-C numerical verdict.
- Next action after publication: request separate approval for the exact-image gate; if green, render from the published immutable SHA and separately approve the APC-off control. Keep `CANON_CONTINUE_DECODE=8`.
- Blockers: exact-image gate and fresh DP8xTP8 APC-off target control have not run.
- Key artifacts: [Attempt-0 receipt](evidence/v1_apc_m15_attempt0_20260825/receipt.json), [Attempt-1 receipt](evidence/v1_apc_m15_attempt1_20260825/receipt.json), [Attempt-2 receipt](evidence/v1_apc_m15_attempt2_20260825/receipt.json), [Phase3 state](../v1-phase3-prefix-cache/state.md)
- Updated: 2026-08-25T06:30:00Z
