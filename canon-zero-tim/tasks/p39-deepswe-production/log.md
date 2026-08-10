# P39 log

## 2026-08-10

- Created an isolated worktree from
  `697a29ab4b27015297af8e3dbb37c49db3560445`; the original dirty P36/P37
  worktree was not modified.
- Compared the P34 command with `yuxzhang/deepswe-quality-fix` at
  `023978b976dd6d94e7a42948c3f3a68e34d73744`.
- Found one real launch blocker: `sampler_is=None` was rejected by the generic
  alignment guard before backward.
- Initially suspected a P33 workload-name/TP4 preflight conflict, then withdrew
  it after the renderer-to-`00_env.sh` test passed. That code is gated by
  `CANON_P32_DP_ADMISSION=1` and is not on the P34 path. The unrelated edit was
  reverted.
- Added a renderer-to-`00_env.sh` positive test and two one-fault negative
  controls. The first execution exposed a test-fixture-only missing state
  directory; the fixture was corrected and the gate passed.
- Static gate result: `P34_STATIC_PASS suites=7`.
- Pinned exact-image result: `P34_EXACT_IMAGE_CPU_PASS unit_cases=45
  pallas_cases=1 contract_cases=5 scheduler_cases=1 overlay=qwen32b`.
- The adjacent P33 sampler contract passed 5/5 in the pinned image. The host
  invocation is `INCONCLUSIVE` because the host lacks `metrax`; no dependency
  was installed or changed.
- Before publication, fetched `yuxzhang/canon-zero-tim` at
  `0fe5f6609df06895d93cbf2e54cada22ad7f2697`. Its only change since the P39
  starting point was `cluster/jobset-64chip.yaml`, so it does not overlap the
  P39 change set. The final commit is rebased onto that revision.

No cloud action, commit, push, PR, credential change or production-default
change was performed.
