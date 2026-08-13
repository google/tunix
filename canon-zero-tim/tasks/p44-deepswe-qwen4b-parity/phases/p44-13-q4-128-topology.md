# P44.13 — Q4 128-chip topology migration

- Status: active; implementation published, local CPU contracts pass, target not run

## Decision

The unavailable Q4 256-chip variant is replaced by a 128-chip `4x4x8`
variant. Q4 now admits exactly 64 or 128 chips. This change does not alter the
Qwen3-32B P46 contract, which remains exactly 64 or 256 chips.

## Signed geometry

| Allocation | Physical slice | Role split | Mesh per role | Local trajectories | Global M | Per-DP scheduler slots |
|---|---|---|---|---:|---:|---:|
| 64 | `4x4x4` | 32 rollout + 32 trainer | DP4 x TP8 | 4 | 1024 | 4 |
| 128 | `4x4x8` | 64 rollout + 64 trainer | DP8 x TP8 | 2 | 2048 | 2 |

The 128-chip role partition is host-complete: 32 four-device hosts split into
16 rollout hosts and 16 trainer hosts without crossing a host boundary. The
evaluation lane has no role split and uses all 128 devices as DP16 x TP8.

## Implementation

- Added `split_4x4x8_role_devices` and routed training by exact devices per
  role: 32 -> `4x4x4`, 64 -> `4x4x8`, 128 -> `4x8x8`.
- Replaced the P44 256 workload, profile, artifact and classifier entries with
  the 128-chip DP8 contract.
- Added workload-specific P46 topology allowlists: Q4 is 64/128; Q32 is
  64/256. Q4-256 and Q32-128 are negative controls.
- Updated the P46 and P44 runbook/handoff sources of truth. Historical 256-chip
  evidence was preserved and is not reclassified or resumed.

## Evidence and claim ceiling

`P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS` passes 41 cases and
`P46_DEEPSWE_PROFILES_CPU_PASS cases=40` passes. These prove contract,
renderer, topology arithmetic and fail-closed wiring only. No 128-chip TPU,
Pathways, R2E, HBM, rollout, backward or optimizer-update behavior has run.

## Next action

Read back the exact operator-branch SHA and require implementation commit
`267a35ef41198dab55fd892a681c3a34b9331a78` in its ancestry. Use an admitted
64/128 Q4 topology to rerun the repaired evaluation l0/p0 under a new run id.
Only after real tool observations and zero adapter-invalid records may the
operator run Q4 three-update and continue the washing campaign.
