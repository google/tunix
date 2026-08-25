# State

- Status: active
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: a deterministic clean-run reproducer reaches `FIRST_RED_LOCALIZED`; the smallest localized repair passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and a separately approved DP8xTP8 target run with A-B=0 bytes and B-C=0 bytes.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`
- Current baseline: commit `283cb67e184239530ac68e3d1c66edf8d37a3c09`
- Release state: Attempt 1 geometry mismatch recorded; repair needed in train_frozenlake_qwen3.py
- Current phase: Phase B ATTEMPT-1 GEOMETRY REPAIR / TARGET CONTROL NOT RUN, [freeze a replay carrier](phases/phase-b-replay-carrier.md)
- Last verified fact: Attempt 1 (`canon-v1-apc-m15-off-d4-283cb67e`) booted and passed all 6 overlay components and GCS preflight, but failed Step 90 due to `train_frozenlake_qwen3.py` geometry validations rejecting M15 DP8 target parameters (`mini_batch_size: 32 vs 4`, `sampler_is: none vs token`).
- Next action after publication: fix geometry checks in `train_frozenlake_qwen3.py` for M15 APC DP8 target debug, commit and relaunch APC-off control.
- Blockers: `train_frozenlake_qwen3.py` geometry checks reject DP8 target parameters.
- Key artifacts: [Attempt-0 receipt](evidence/v1_apc_m15_attempt0_20260825/receipt.json), [Attempt-1 receipt](evidence/v1_apc_m15_attempt1_20260825/receipt.json), [Phase3 state](../v1-phase3-prefix-cache/state.md)
- Updated: 2026-08-25T03:05:00Z
