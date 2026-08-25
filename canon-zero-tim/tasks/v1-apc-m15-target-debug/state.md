# State

- Status: active
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: a deterministic clean-run reproducer reaches `FIRST_RED_LOCALIZED`; the smallest localized repair passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and a separately approved DP8xTP8 target run with A-B=0 bytes and B-C=0 bytes.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`
- Current baseline: operator tip `16db308b`; the intervening P62 diagnostic commits were reviewed and do not overlap the APC numerical path
- Release state: the isolated release candidate passed host and pinned exact-image admission; publication is recorded by Git, and no runtime launch is implied by publication
- Current phase: Phase B ATTEMPT-0 BOOTSTRAP REPAIR / TARGET CONTROL NOT RUN, [freeze a replay carrier](phases/phase-b-replay-carrier.md)
- Last verified fact: Attempt 0 supplied `--p57_workload_candidate=m15 --p57_data_split=main` only on the CLI while omitting the matching signed environment fields. The FrozenLake entrypoint rejects that identity split before learner construction. The renderer/profile/Step-00 contract now carries and checks exact `m15/main`; host positives and CLI/env/entrypoint negatives pass. No numerical code changed.
- Next action after publication: rerun the pinned exact-image admission for the repaired committed tree, then render a new unique Attempt 1 and launch only the APC-off target control after separate approval.
- Blockers: post-fix exact-image and DP8xTP8 Attempt 1 have not run; Attempt 0 is permanently `INCONCLUSIVE` and cannot be reused.
- Key artifacts: [Attempt-0 receipt](evidence/v1_apc_m15_attempt0_20260825/receipt.json), [Attempt-2 receipt](../v1-phase4-three-full-recipes/evidence/v1_hp_three_full_attempt2_20260824/receipt.json), [M15 raw log](../v1-phase4-three-full-recipes/evidence/v1_hp_three_full_attempt2_20260824/m15_m15i_error.log), [Phase3 state](../v1-phase3-prefix-cache/state.md)
- Updated: 2026-08-25
