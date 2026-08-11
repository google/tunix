# P39 state

Updated: 2026-08-11 UTC

Current phase: **P39.2 locally complete — the 64-chip separated DP4xTP8
device-resident optimizer pilot is implemented, but target NOT RUN.** No target
run has been launched and no resident-optimizer claim exists for Qwen3-32B.

| Ledger | Status | Detail |
|---|---|---|
| Implementation | IMPLEMENTED | sampler contract, pre-backward report, exact cross-role weight attestation, toxic-SHA-safe renderer, classifier wiring and explicit production CLI |
| Static validation | STATIC PASS | `bash canon-zero-tim/tests/p34_deepswe/run_static.sh` |
| Exact image | CPU PASS | pinned image; 54 unit, 1 Pallas, 5 contract and 1 scheduler cases |
| Adjacent P33 regression | PASS | workload CPU gate and both pinned-image overlays remain green |
| Toxic SHA round trip | LOCAL PASS | `022893e2` remains an explicitly quoted string after render and parse |
| Direct TPU | TARGET NOT RUN | no operator or full-model claim from this change |
| Pathways/GKE | TARGET NOT RUN | 4x8x8 P34 has never run |
| Real DeepSWE recipe | TARGET NOT RUN | no backward, optimizer commit or convergence claim |
| P39 64-chip pilot implementation | LOCAL PASS | separate DP4xTP8 profile and renderer, one/three-update stages, resident-only optimizer contract, dedicated classifier and 15-test CPU gate |
| P39 64-chip pilot target | TARGET NOT RUN | no 4x4x4 rollout, backward, commit, HBM, IFRT or W&B evidence |

Next action: publish only after approval, rerun the P39 and P34 local gates at
the exact publication SHA, and execute the one-update 4x4x4 pilot through
`../../cluster/P39_DEEPSWE_64CHIP_PILOT_RUNBOOK.md`. The three-update stage is
admitted only after the one-update classifier passes.

## First hard boundary

Before P39, the real learner would reject P34 because the recipe pins
`sampler_is=None` while the generic alignment guard admitted that policy only
for GSM8K. Existing static tests did not enter that runtime branch. P39 admits
the exception only when all three P34 facts are true: DeepSWE mode, sampler IS
disabled, and TIS disabled.

P39 also enables the existing pre-backward gate. A nonzero A-B or B-C record is
flushed, fsynced and printed before backward. It is not converted to a
tolerance and cannot reach an optimizer commit.

The current hardening runs the existing device-side exact mapped-trainer versus
live-engine leaf comparison before every P34 rescore. One record per update is
fsynced to `weight_attestation.jsonl`; the classifier rejects a missing,
duplicate or mismatching record. The 4x8x8 target has not executed this gate.

## Claim ceiling

This change makes P34 a locally testable production candidate. It does not
prove that Qwen3-32B DP16xTP8 is bitwise exact or trainable on Pathways.
