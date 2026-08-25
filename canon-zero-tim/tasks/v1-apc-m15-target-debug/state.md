# State

- Status: active
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: a deterministic clean-run reproducer reaches `FIRST_RED_LOCALIZED`; the smallest localized repair passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and a separately approved DP8xTP8 target run with A-B=0 bytes and B-C=0 bytes.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`
- Current baseline: operator commit `9f79cc562b2032f3fe02297ce5608023d907361e`; the task release is the commit containing this state file.
- Release state: Attempt 4 error is preserved. Patch 28 worked far enough to complete all 2,560 APC-on rollout requests, but the learner then rejected the carrier's signed no-IS recipe before A/B/C. The bounded admission repair is host- and aggregate-exact-image green; the post-fix target has not run.
- Current phase: Phase B ATTEMPT-4 SAMPLER ADMISSION REPAIR AGGREGATE-EXACT-IMAGE PASS / TARGET CONTROL NOT RUN, [freeze a replay carrier](phases/phase-b-replay-carrier.md)
- Last verified fact: Attempt 4 reached alignment with 92.5% prefix-cache hit rate after a complete rollout, then failed because the generic FrozenLake gate admitted `sampler_is=None` only for GSM8K/P34/P57. The exact M15 DP8xTP8 zero-commit carrier intentionally uses rollout logprobs with no token-IS weights. This was an admission omission, not an A/B/C numerical verdict.
- Next action after an approved publication: render both arms from one immutable SHA and immediately submit the fresh DP8xTP8 APC-off control and APC-on treatment under one paired-launch approval. Keep `CANON_CONTINUE_DECODE=8` and `--sampler_is=none`; do not wait between submissions. Classify off first, and use on for an APC-specific claim only if off is `CONTROL_GREEN`.
- Blockers: the fresh matched DP8xTP8 off/on pair has not run.
- Key artifacts: [Attempt-0 receipt](evidence/v1_apc_m15_attempt0_20260825/receipt.json), [Attempt-1 receipt](evidence/v1_apc_m15_attempt1_20260825/receipt.json), [Attempt-2 receipt](evidence/v1_apc_m15_attempt2_20260825/receipt.json), [Attempt-3 receipt](evidence/v1_apc_m15_attempt3_20260825/receipt.json), [Attempt-4 receipt](evidence/v1_apc_m15_attempt4_20260825/receipt.json), [Phase3 state](../v1-phase3-prefix-cache/state.md)
- Validation: APC target-carrier 46/46; P38 classifier 37/37; Phase3 12/12;
  P57 146/146; V1 Phase4 CPU 67/67; flag audit 378/378; Python/shell syntax
  and `git diff --check` PASS. The aggregate pinned-image gate exits 0 with
  `apc_m15_carrier=46`, `p64_numeric=4`, and `p64_capsule=3`.
- Limitation: exact-image admission is complete, but the DP8xTP8 target has
  not run. The next target attempt must use a newly committed source SHA,
  label, and GCS attempt.
- Updated: 2026-08-25T18:26:00Z
