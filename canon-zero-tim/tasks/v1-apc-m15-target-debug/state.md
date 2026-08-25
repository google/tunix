# State

- Status: active
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: a deterministic clean-run reproducer reaches `FIRST_RED_LOCALIZED`; the smallest localized repair passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and a separately approved DP8xTP8 target run with A-B=0 bytes and B-C=0 bytes.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`
- Current baseline: commit `d6629c8c9406c64e578aa84d22a68ed925d2156b`
- Release state: Attempt 1 geometry mismatch preserved; bounded entrypoint repair is host-pass and uncommitted.
- Current phase: Phase B ATTEMPT-1 GEOMETRY REPAIR HOST PASS / EXACT-IMAGE AND TARGET CONTROL NOT RUN, [freeze a replay carrier](phases/phase-b-replay-carrier.md)
- Last verified fact: Attempt 2 ran on 64 TPU (DP8xTP8) with commit `41a2043ca612eeb8dcf77ae1262d18471c26b479` and completed >95% of 15-turn FrozenLake rollout (1800+ calls, 760+ requests, 256 trajectories) before P38 serving capture halted due to `CANON_CONTINUE_DECODE=8` triggering `_execute_continue_decode` while asserting `EXPECTED_PATH="standard"`.
- Next action after user approval: remove `CANON_CONTINUE_DECODE=8` from profile, re-render, and relaunch APC-off control.
- Blockers: fresh DP8xTP8 APC-off target control with continue-decode disabled has not run.
- Key artifacts: [Attempt-0 receipt](evidence/v1_apc_m15_attempt0_20260825/receipt.json), [Attempt-1 receipt](evidence/v1_apc_m15_attempt1_20260825/receipt.json), [Attempt-2 receipt](evidence/v1_apc_m15_attempt2_20260825/receipt.json), [Phase3 state](../v1-phase3-prefix-cache/state.md)
- Updated: 2026-08-25T04:19:00Z

