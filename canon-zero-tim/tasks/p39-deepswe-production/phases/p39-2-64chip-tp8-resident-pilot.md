# P39.2 — 64-chip separated TP8 resident-optimizer pilot

- Status: local complete; target NOT RUN

## Finding

- Confirmed: the current P34 contract admits one 4x8x8 slice split into two
  128-device roles, each DP16xTP8, with pinned-host optimizer offload.
- Confirmed: a 64-chip 4x4x4 pilot split equally gives 32 rollout devices and
  32 trainer devices. Each role must therefore be DP4xTP8, not DP16xTP8.
- Confirmed: the existing preflight hard-codes DP16xTP8, role-local 128 devices,
  global M4096, four requests per DP rank, and optimizer offload. The pilot is a
  new bounded contract and cannot inherit P34 target-green status.
- Hypothesis: DP4xTP8 is a conservative capacity probe for TP8 device-resident
  optimizer state because each trainer DP rank handles more local trajectories
  than the eventual DP16xTP8 deployment, while the TP8 optimizer shard size is
  unchanged.

## Pilot geometry

| Quantity | 64-chip pilot |
|---|---:|
| Physical topology | one 4x4x4 slice, 64 chips |
| Rollout role | 32 chips = DP4xTP8 |
| Trainer role | 32 chips = DP4xTP8 |
| Parameters | replicated over DP4, sharded only over TP8 |
| Global workload | 8 prompts x 8 generations = 64 trajectories |
| Trajectories per DP rank | 16 |
| Gradient trajectory microbatch | 4 global = 1 per DP rank |
| Gradient accumulation | 16 fixed-order groups |
| Local canonical M | 256 |
| Global padded M | 1024 = DP4 x M256 |
| Scheduler request capacity | 16 per rank, 64 global |
| Scheduler token capacity | 256 per rank, 1024 global |
| Prefix cache | disabled |
| Optimizer | device-resident; offload disabled |
| Run length | one bounded JobSet, up to three committed updates |
| W&B | online, pilot-specific project/group/run name |

The pilot may bound response length and turns to control rollout cost. Such a
bound limits the claim to integration and optimizer capacity; it does not prove
the production 32768-token, 50-turn memory envelope.

## Execution

1. Add a separate DP4xTP8 pilot profile and renderer. Do not loosen the existing
   DP16xTP8 production contract and do not add FSDP.
2. Pin the 4x4x4 physical device inventory and two host-complete 32-device role
   meshes. Record both flattened device orders.
3. Register the DP4 arithmetic: role devices 32, local M256, global M1024,
   16 requests per DP rank, and 64 global trajectories.
4. Extend the deterministic reducer only if its existing generic path proves
   DP4 fixed-order reduction and post-reduction replica equality. Never infer
   DP4 from a DP16 artifact.
5. Select device residency explicitly with
   `CANON_OPT_STATE_RESIDENT=1`, `CANON_P30_OPT_STATE_OFFLOAD=0`, and
   `--no-optimizer-offload`. Require zero optimizer H2D/D2H transfer.
6. In one JobSet, initialize both TP8 engines, attest cross-role weights, produce
   a real rollout, run the forward and pre-backward reports, execute one real
   backward and commit, then continue through at most three updates to detect
   immediate memory growth. Do not queue three separate target jobs.
7. Keep finite alignment differences report-only for this integration pilot.
   NaN/Inf, topology, weight, metadata, optimizer transaction, gradient,
   replica, OOM, IFRT, and W&B failures remain hard errors.

## Exit gate

The pilot is admitted for 256-chip promotion only if all of the following hold:

1. exactly 64 physical devices are visible and both role meshes are exactly
   DP4xTP8 with no DP-sharded parameter leaf;
2. rollout and trainer TP8 initialize from the same bitwise weight state;
3. at least one real environment rollout and one finite, nonzero backward
   complete;
4. device-resident optimizer state is observed before and after every commit,
   optimizer H2D/D2H bytes are zero, and parameters change after commit;
5. DP4 reduced gradients and post-update parameters are replica-exact;
6. no OOM or IFRT disconnect occurs through three updates;
7. peak trainer HBM leaves at least 8 GiB per chip after the highest observed
   update; and
8. all planned measurement records and online W&B markers are present.

If the optimizer fits but leaves less than 8 GiB, record capacity as measured
but do not promote resident mode. If resident mode OOMs, the 256-chip fallback
is pinned-host offload. A numerical warning does not promote zero-TIM and must
remain visible in the classification.

## Promotion to 256 chips

After a passing pilot, the production candidate returns to one 4x8x8 slice
split into two 128-device roles, each DP16xTP8. The per-chip TP8 optimizer shard
is unchanged while local trajectories fall from 16 to 4, so the pilot provides
a conservative capacity signal. It does not validate DP16 collective behavior;
the 256-chip run must re-run mesh, reducer, replica, cross-role weight, W&B, and
Pathways health gates.

The 256-chip first run may use resident optimizer only if the 64-chip resident
gate passes. Otherwise it uses the existing offload path. Full production
training remains a separate user-approved launch.

## Rollback

Do not select the DP4 pilot profile. The existing DP16xTP8/offload P34 profile
and renderer remain unchanged. A resident failure rolls back only optimizer
placement to pinned-host offload; it does not change precision, loss, sampling,
TP8, or the DeepSWE workload recipe.

## Result

Implemented locally. The separate profile and renderer admit only one 4x4x4
slice split into two disjoint 32-device roles, each DP4xTP8. They register
global M1024, 64 requests, 16 fixed-order accumulation groups, TP8-only
parameter sharding, online W&B, and device-resident optimizer state. The
postflight selects the dedicated P39 classifier rather than the P34 production
classifier and accepts Pathways HBM telemetry only when it covers at least 32
devices per update and leaves 8 GiB free.

Local results:

- `bash canon-zero-tim/tests/p39_deepswe_pilot/run_cpu.sh`:
  15 tests, `P39_DEEPSWE_PILOT_CPU_PASS`.
- `bash canon-zero-tim/tests/p34_deepswe/run_static.sh`:
  `P34_STATIC_PASS suites=10`.

No target device initialized, no rollout/backward/optimizer commit occurred,
and no capacity or zero-TIM claim was promoted.
