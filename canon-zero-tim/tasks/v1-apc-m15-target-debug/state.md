# State

- Status: active
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: a deterministic clean-run reproducer reaches `FIRST_RED_LOCALIZED`; the smallest localized repair passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and a separately approved DP8xTP8 target run with A-B=0 bytes and B-C=0 bytes.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`
- Current baseline: operator commit `95e290b02421e4213589430ff0c745cc91f4f648`.
- Release state: Attempt 3 error preserved; append-only patch 28 removes the invalid "four standard strata first" precondition from the M15-only `continue_decode` replay path. Host gates, the targeted installed-runner P33 exact-image gate, and the aggregate V1 exact-image gate pass; the target has not run.
- Current phase: Phase B ATTEMPT-3 OBSERVER REPAIR AGGREGATE-EXACT-IMAGE PASS / TARGET CONTROL NOT RUN, [freeze a replay carrier](phases/phase-b-replay-carrier.md)
- Last verified fact: Attempt 3 used the patched runner and still failed before A/B/C classification because APC-on entered `continue_decode` before all four standard capture strata existed. Patch 27 admitted that path only after capture completion, so the observer rejected a valid production program transition. This is an observer control-flow failure, not an evaluation result or numerical verdict.
- Next action after an approved publication: render from the immutable SHA and separately approve a fresh DP8xTP8 APC-off control. Keep `CANON_CONTINUE_DECODE=8`; only a green control may unlock the APC-on treatment.
- Blockers: the observer repair is not committed/published and a fresh DP8xTP8 APC-off target control has not run.
- Key artifacts: [Attempt-0 receipt](evidence/v1_apc_m15_attempt0_20260825/receipt.json), [Attempt-1 receipt](evidence/v1_apc_m15_attempt1_20260825/receipt.json), [Attempt-2 receipt](evidence/v1_apc_m15_attempt2_20260825/receipt.json), [Attempt-3 receipt](evidence/v1_apc_m15_attempt3_20260825/receipt.json), [Phase3 state](../v1-phase3-prefix-cache/state.md)
- Validation: APC target-carrier 44/44; P38 classifier 37/37; Phase3 12/12;
  V1 Phase4 CPU 67/67; flag audit 378/378; Python/shell syntax and
  `git diff --check` PASS. The targeted installed-runner exact-image gate runs
  35/35 tests on each of the Qwen3-1.7B and Qwen3-8B overlays. The expanded
  aggregate exact-image gate exits 0 with `apc_m15_carrier=44`,
  `p64_numeric=4`, and `p64_capsule=3`.
- Limitation: exact-image admission is complete, but the DP8xTP8 target has
  not run. The next target attempt must use a newly committed source SHA,
  label, and GCS attempt.
- Updated: 2026-08-25T09:41:00Z
