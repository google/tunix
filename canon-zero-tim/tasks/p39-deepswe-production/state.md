# P39 state

Updated: 2026-08-12 UTC

Current phase: **P39.4 target Attempt `p34r02` FAILED during rollout-engine
mesh initialization; the local repair is PASS and the target retry is NOT
RUN.** The complete archived log proves Qwen3-32B trainer-side model loading
and identifies a stale four-device mesh-ID assertion before the first rollout.
No optimizer-capacity or convergence claim exists.

| Ledger | Status | Detail |
|---|---|---|
| Implementation | REPAIR PUBLISHED / LOCAL PASS | mesh-admission repair commit `562f55b077bdadbcfa160177715b0d8ca903f457`; direct full stage plus an explicit P34 ban on inheriting the one-host model-mesh ID assertion |
| Static validation | STATIC PASS | `bash canon-zero-tim/tests/p34_deepswe/run_static.sh` |
| Exact image | CPU PASS | pinned image; 55 unit, 3 alignment, 2 Pallas, 5 contract and 1 scheduler cases |
| Adjacent P33 regression | PASS | workload CPU gate and both pinned-image overlays remain green |
| Toxic SHA round trip | LOCAL PASS | `022893e2` remains an explicitly quoted string after render and parse |
| Direct TPU | TARGET PARTIAL | 256 devices, 64 hosts, both DP16xTP8 roles and trainer-side 32B loading passed; recorded trainer-role HBM was 30.5 GiB/device before rollout initialization |
| Pathways/GKE | TARGET FAILED | rollout-engine `_init_mesh()` compared the healthy 128-device role with stale one-host IDs `[0,2,1,3]` and raised before rollout |
| Real DeepSWE recipe | MODEL/DATA PASS / ROLLOUT NOT ENTERED | pinned R2E-Gym and clean filter passed 4578 -> 1851; Qwen3-32B actor/reference initialization reached vLLM construction, but no rollout, trajectory, backward or optimizer record exists |
| P39 64-chip pilot implementation | PUBLISHED / LOCAL PASS | published at `7328cde7`; separate DP4xTP8 profile and renderer, one/three-update stages, resident-only optimizer contract, dedicated classifier and 15-test CPU gate |
| P39 64-chip pilot target | DEFERRED / NOT RUN | the available 4x8x8 slice makes this capacity pilot optional; no resident-optimizer evidence exists |
| P34 256-chip launch selection | ATTEMPTED / FAILED | `p34r02` stopped at `CANON_EXPECT_MODEL_MESH_IDS`; the local profile/renderer/preflight repair passes all affected gates, but has not run on the target |

Next action: pull `yuxzhang/canon-zero-tim` into a clean worktree, record the
exact 40-character HEAD, render a fresh manifest and retry the same full
32B/data/topology/device-optimizer configuration.  The retry must get past
`Creating new model mesh` without a mesh-ID mismatch before rollout evidence
can begin.

## First hard boundary

Before P39, the real learner would reject P34 because the recipe pins
`sampler_is=None` while the generic alignment guard admitted that policy only
for GSM8K. Existing static tests did not enter that runtime branch. P39 admits
the exception only when all three P34 facts are true: DeepSWE mode, sampler IS
disabled, and TIS disabled.

P39.4 changes the convergence policy: every finite A-B or B-C residual is
flushed, fsynced, printed and allowed to continue.  Nonfinite or structurally
invalid records still stop before backward.  A warning is never a zero-TIM
promotion.

The current hardening runs the existing device-side exact mapped-trainer versus
live-engine leaf comparison before every P34 rescore. One record per update is
fsynced to `weight_attestation.jsonl`; the classifier rejects a missing,
duplicate or mismatching record. The 4x8x8 target has not executed this gate.

## Claim ceiling

This change makes P34 a locally testable production candidate. It does not
prove that Qwen3-32B DP16xTP8 is bitwise exact or trainable on Pathways.
