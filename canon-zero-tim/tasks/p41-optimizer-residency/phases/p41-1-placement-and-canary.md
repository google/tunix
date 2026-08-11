# P41.1 — Optimizer placement and GSM8K canary

- Status: passed

## Finding

- Confirmed: offload stores optimizer state in `pinned_host` between updates,
  moves it to `device` for AdamW, and moves it back after commit.
- Confirmed: `update_accumulation_pending` precedes optimizer H2D; residency can
  improve commit time but cannot remove the sixteen gradient microbatches.
- Hypothesis: removing the 1.7B optimizer round trip is memory-safe on TP4 and
  improves the one-update transaction wall time.

## Execution

1. Add `CANON_OPT_STATE_RESIDENT=0|1` to both P33 workload profiles.
2. Require actual placement to be exactly one of offload or resident.  Preserve
   offload as the default and reject ambiguous configuration.
3. Wire both recipes to the selected placement and attest `device` before and
   after commit in resident mode.
4. Update P33 classification and negative controls for both placements.
5. Run the complete P33 CPU gate.
6. Run two bounded DP1xTP4 GSM8K one-update arms with identical workload inputs:
   offload baseline and resident candidate.  Record wall time, HBM snapshots,
   commit evidence, and final state fingerprints.

## Exit gate

- CPU pass: both legal placements classify; unset/default remains offload;
  `resident=1 + offload=1`, missing attestation, and wrong before/after memory
  kinds are rejected.
- Hardware pass: both arms complete exactly one nonzero update with finite
  gradients, unchanged reference state, reset accumulator, and equal final
  parameter/optimizer fingerprints; resident has no OOM.
- Fail: any OOM, evidence mismatch, state drift, or missing measurement keeps
  resident mode experimental and leaves default offload unchanged.

## Result

The exact-image P33 CPU gate passed 73 tests.  Focused gates passed for both
profiles, 43 workload tests, 12 renderer tests, 13 classifier tests, two P41
classifier controls, and the existing two-commit offload-vs-device bitwise
equivalence test.  The final source adds one P41 scheduling regression, so the
exact-image P33 gate now passes 74 tests.

The `p41a15` DP1xTP4 hardware pair passed with one real update per arm.  Both
arms used the same engine-seeded serial temperature-1 rollout and produced the
same token hash, exact A/B/C log probabilities, the same gradient norm, and
bitwise-identical final model, optimizer, reference, and accumulator evidence.
The resident arm did not OOM.  Its optimizer transaction was 39.1262 seconds
versus 46.9757 seconds offloaded (1.2006x); the full measured reverse-plus-commit
window was 159.1951 versus 168.0918 seconds (5.29 percent lower).  Peak HBM rose
from 33,093,301,760 to 34,676,120,576 bytes per chip.  This result admits the
GSM8K candidate only; FrozenLake device residency remains unmeasured.
