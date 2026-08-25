# State

- Status: active
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: a deterministic clean-run reproducer reaches `FIRST_RED_LOCALIZED`; the smallest localized repair passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and a separately approved DP8xTP8 target run with A-B=0 bytes and B-C=0 bytes.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`
- Current baseline: operator tip `ff913a84`; the intervening raw-log and P58 seed-registry commits were reviewed before fast-forward, with no conflicting numerical hunk
- Release state: the isolated release candidate passed host and pinned exact-image admission; publication is recorded by Git, and no runtime launch is implied by publication
- Current phase: Phase B EXACT-IMAGE PASS / TARGET NOT RUN, [freeze a replay carrier](phases/phase-b-replay-carrier.md)
- Last verified fact: Phase B captures all 256 producer rows and a host-only envelope for every A/B serving call, then mechanically joins both to the first red. The large carrier is included in the existing GCS serving archive, the replay envelope is included in live snapshots, and the GCS-side wrapper was integration-tested against a fake immutable bucket: download, root/nested SHA verification, small-receipt upload, manifest-last completion, and overwrite rejection all passed. No numerical red has been freshly reproduced and no replay has run.
- Next action after publication: request separate approval for the APC-off target control. APC-on target treatment remains a later independent approval; the remote agent follows `RUNBOOK.md` without editing code/YAML.
- Blockers: the exact historical `m15i` request/token/cache chronology was not archived and cannot be reconstructed from hashes. A fresh, fully captured red must become the strict replay source.
- Key artifacts: [Attempt-2 receipt](../v1-phase4-three-full-recipes/evidence/v1_hp_three_full_attempt2_20260824/receipt.json), [M15 raw log](../v1-phase4-three-full-recipes/evidence/v1_hp_three_full_attempt2_20260824/m15_m15i_error.log), [Phase3 state](../v1-phase3-prefix-cache/state.md)
- Updated: 2026-08-25
