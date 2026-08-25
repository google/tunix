# State

- Status: active
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: a deterministic clean-run reproducer reaches `FIRST_RED_LOCALIZED`; the smallest localized repair passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and a separately approved DP8xTP8 target run with A-B=0 bytes and B-C=0 bytes.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`
- Current baseline: operator commit `9f79cc562b2032f3fe02297ce5608023d907361e`; the task release is the commit containing this state file.
- Release state: Attempt 5 paired run (`d11-a909fda1`, commit `a909fda1`) unblocked the sampler admission gate, completing 2,560 requests in both off/on arms with up to 97.5% prefix-cache hit rate in the on treatment arm and 0.0% in off control arm.
- Current phase: Phase B ATTEMPT-5 PAIRED ROLLOUT COMPLETE / SAMPLER CONTRACT VERIFIED, [freeze a replay carrier](phases/phase-b-replay-carrier.md)
- Last verified fact: Attempt 5 paired run completed all 2,560 requests with 89.7% ~ 97.5% prefix-cache hit rate on the on arm, 0.0% on the off arm, passed `[CANON_APC_M15_SAMPLER_CONTRACT] PASS`, and cleanly executed controlled exit 42 with zero optimizer commits.
- Next action after an approved publication: evaluate XProf / Zero-TIM requirements.
- Blockers: none for M15 sampler contract.
- Key artifacts: [Attempt-0 receipt](evidence/v1_apc_m15_attempt0_20260825/receipt.json), [Attempt-1 receipt](evidence/v1_apc_m15_attempt1_20260825/receipt.json), [Attempt-2 receipt](evidence/v1_apc_m15_attempt2_20260825/receipt.json), [Attempt-3 receipt](evidence/v1_apc_m15_attempt3_20260825/receipt.json), [Attempt-4 receipt](evidence/v1_apc_m15_attempt4_20260825/receipt.json), [Attempt-5 paired receipt](evidence/v1_apc_m15_attempt5_paired_d11_20260825/receipt.json), [Phase3 state](../v1-phase3-prefix-cache/state.md)
- Validation: APC target-carrier 46/46; P38 classifier 37/37; Phase3 12/12;
  P57 146/146; V1 Phase4 CPU 67/67; flag audit 378/378; Python/shell syntax
  and `git diff --check` PASS. The aggregate pinned-image gate exits 0 with
  `apc_m15_carrier=46`, `p64_numeric=4`, and `p64_capsule=3`.
- Limitation: Paired rollout and sampler contract verified; production recipes remain APC-off.
- Updated: 2026-08-25T22:12:00Z
