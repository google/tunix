# P39 state

Updated: 2026-08-10 UTC

| Ledger | Status | Detail |
|---|---|---|
| Implementation | IMPLEMENTED | sampler contract, pre-backward report, classifier wiring, explicit production CLI and env preflight test |
| Static validation | STATIC PASS | `bash canon-zero-tim/tests/p34_deepswe/run_static.sh` |
| Exact image | CPU PASS | pinned image; 45 unit, 1 Pallas, 5 contract and 1 scheduler cases |
| Direct TPU | TARGET NOT RUN | no operator or full-model claim from this change |
| Pathways/GKE | TARGET NOT RUN | 4x8x8 P34 has never run |
| Real DeepSWE recipe | TARGET NOT RUN | no backward, optimizer commit or convergence claim |

## First hard boundary

Before P39, the real learner would reject P34 because the recipe pins
`sampler_is=None` while the generic alignment guard admitted that policy only
for GSM8K. Existing static tests did not enter that runtime branch. P39 admits
the exception only when all three P34 facts are true: DeepSWE mode, sampler IS
disabled, and TIS disabled.

P39 also enables the existing pre-backward gate. A nonzero A-B or B-C record is
flushed, fsynced and printed before backward. It is not converted to a
tolerance and cannot reach an optimizer commit.

## Claim ceiling

This change makes P34 a locally testable production candidate. It does not
prove that Qwen3-32B DP16xTP8 is bitwise exact or trainable on Pathways.
