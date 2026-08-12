# P39 state

Updated: 2026-08-12 UTC

Current phase: **P39.4 locally complete — publication to the operator branch
is approved.** No target run has
been launched and no Qwen3-32B optimizer-capacity or convergence claim exists.

| Ledger | Status | Detail |
|---|---|---|
| Implementation | LOCAL PASS / PUBLICATION APPROVED | direct full stage, pinned clean data, device optimizer CLI, finite alignment warning policy and durable production trajectory capture |
| Static validation | STATIC PASS | `bash canon-zero-tim/tests/p34_deepswe/run_static.sh` |
| Exact image | CPU PASS | pinned image; 55 unit, 3 alignment, 2 Pallas, 5 contract and 1 scheduler cases |
| Adjacent P33 regression | PASS | workload CPU gate and both pinned-image overlays remain green |
| Toxic SHA round trip | LOCAL PASS | `022893e2` remains an explicitly quoted string after render and parse |
| Direct TPU | TARGET NOT RUN | no operator or full-model claim from this change |
| Pathways/GKE | TARGET NOT RUN | 4x8x8 P34 has never run |
| Real DeepSWE recipe | TARGET NOT RUN | no backward, optimizer commit or convergence claim |
| P39 64-chip pilot implementation | PUBLISHED / LOCAL PASS | published at `7328cde7`; separate DP4xTP8 profile and renderer, one/three-update stages, resident-only optimizer contract, dedicated classifier and 15-test CPU gate |
| P39 64-chip pilot target | DEFERRED / NOT RUN | the available 4x8x8 slice makes this capacity pilot optional; no resident-optimizer evidence exists |
| P34 256-chip launch selection | SELECTED / NOT RUN | direct 4x8x8 DP16xTP8 full training with device-resident optimizer, clean-data pin, durable trajectory capture and finite alignment warning-only policy |

Next action: pull the published operator branch into a clean worktree, record
its exact 40-character HEAD, rerun the local gates, then render and review the
signed `full` manifest before any apply.  Never substitute the pre-publication
base SHA for the publication HEAD.

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
