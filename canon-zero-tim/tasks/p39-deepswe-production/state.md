# P39 state

Updated: 2026-08-13 UTC

Current phase: **P39.5 bounded DeepSWE lifecycle and dual debug/production
defaults implemented; local frozen-image gates PASS; target retry NOT RUN.**
Attempt `p34r03` reached real Qwen3-32B rollout but never completed update zero:
60 trajectories logged `ENV_TIMEOUT` with negative remaining time and four
vLLM requests were still running at the end of the returned log. No trajectory
batch artifact, backward, optimizer commit or convergence claim exists.

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
| P34 target Attempt `p34r03` | TARGET INCONCLUSIVE / UNBOUNDED ROLLOUT | source `65b0cd0a84807f2308409d1867022407ae53f8fb`; topology/model/clean data passed; update-zero rollout ran over four hours, emitted 60 negative-remaining-time `ENV_TIMEOUT` records, ended with four active vLLM requests and no completed batch |
| P39.5 lifecycle repair | LOCAL PASS / UNPUBLISHED | true vLLM request abort; shared trajectory/batch deadlines; bounded reset/step/reward/cleanup; confirmed R2E pod deletion; locked Qwen3-4B three-update and Qwen3-32B full YAML defaults |
| P39.5 exact image | CPU PASS | P34 55 unit cases plus P44 41 cases and new request-abort/trajectory-cleanup controls in pinned image `sha256:418dc632...d5e53a` |

Next action: after explicit commit/push approval, publish only to
`yuxzhang/canon-zero-tim`, read back its exact 40-character HEAD, and have the
launch agent render the Qwen3-4B `three-update` debug JobSet for whichever
64/256 allocation is available. Inspect its durable trajectories and deadline
markers. Then render a fresh Qwen3-32B `full` manifest; update zero must either
complete or terminate within the signed 5400-second rollout-batch boundary.

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
