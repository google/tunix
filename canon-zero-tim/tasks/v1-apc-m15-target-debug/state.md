# State

- Status: active
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: a deterministic clean-run reproducer reaches `FIRST_RED_LOCALIZED`; the smallest localized repair passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and a separately approved DP8xTP8 target run with A-B=0 bytes and B-C=0 bytes.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`
- Current baseline: operator tip `ff913a84`; the intervening raw-log and P58 seed-registry commits were reviewed before fast-forward, with no conflicting numerical hunk
- Release state: the isolated release candidate passed host and pinned exact-image admission; publication is recorded by Git, and no runtime launch is implied by publication
- Current phase: Phase B EXACT-IMAGE PASS / TARGET NOT RUN, [freeze a replay carrier](phases/phase-b-replay-carrier.md)
- Last verified fact: Attempt 0 of APC-off control (`canon-v1-apc-m15-off-d3-eb58954f`) completed cluster boot and overlay verification, but exited with code 1 at Step 90 Python initialization. Failure receipt archived in `evidence/v1_apc_m15_attempt0_20260825/`.
- Next action after publication: fix Step 90 entrypoint invocation for M15 APC debug, re-render, and launch Attempt 1 APC-off target control.
- Blockers: Python launcher exit 1 during Step 90 startup under `qwen3-8b-dp8-tp8-frozenlake-apc-debug.env`.
- Key artifacts: [Attempt-0 receipt](evidence/v1_apc_m15_attempt0_20260825/receipt.json), [Attempt-2 receipt](../v1-phase4-three-full-recipes/evidence/v1_hp_three_full_attempt2_20260824/receipt.json), [M15 raw log](../v1-phase4-three-full-recipes/evidence/v1_hp_three_full_attempt2_20260824/m15_m15i_error.log), [Phase3 state](../v1-phase3-prefix-cache/state.md)
- Updated: 2026-08-25

