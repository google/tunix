# P44.11 — real one-host Qwen3-4B DeepSWE integration

- Status: rollout integration PASS; backward executed but
  `INCONCLUSIVE_NO_SIGNAL`; implementation commit
  `29cea119259f1f7fe583a3e3dd1cb190acc0bf63` published; clean-source repeat pending

## Scope and provenance

- Hardware: one direct-attached v5p host exposing exactly four TPU v5
  devices; no Pathways and no role separation.
- Model: existing local
  `Qwen/Qwen3-4B-Instruct-2507` snapshot, DP1 x TP4, bf16 parameters and
  activations.
- Workload: one reviewed DeepSWE task, two generations, response length 512,
  two turns, shared rollout/trainer mesh, Docker `SWEEnv`, prefix cache off.
- R2E-Gym source:
  `0d94c4eb9431cd195c55a7ea3abd54006c9a1735`.
- Selected image:
  `namanjain12/orange3_final:2d9617bd0cb1f0ba61771258410ab8fae8e7e24d`.
- The development runs recorded base commit
  `3ec5fd7c3074844c62d3a9ff2c95179449a66129` plus the then-uncommitted
  P44.10/P44.11 diff. They are development evidence, not clean publication
  provenance. The later operator head
  `d8184123448d0add72b72f09d0a6faf5d326c26e` contains P38-specific
  capture/precheck hardening. Its shared learner change is guarded by the P38
  alignment-precheck mode and does not alter the one-host mode when those
  flags are absent; it is covered by the post-reconciliation regressions but
  was not part of the real v5p development run.

## Implementation and failure repairs

1. Added a default-off, mutually exclusive
   `CANON_DEEPSWE_ONEHOST_SMOKE=1` profile. It admits only Qwen3-4B,
   DP1 x TP4 on the same four devices, a one-prompt/two-generation batch,
   real Docker R2E, and `rollout-only` or `backward-no-commit`.
2. Added local trajectory, batch-metric, run-manifest, and no-commit report
   schemas without changing P34/P39/P43/P44 production schemas.
3. Kept the dataset library's removed `trust_remote_code` argument absent,
   selected cached `R2E-Gym/R2E-Gym-V1`, and required exactly one reviewed
   whitelist row.
4. Exposed CPU only as vLLM's staging backend with `JAX_PLATFORMS=tpu,cpu`;
   TPU remains the default execution backend.
5. Named the colocated replicated data axis `dp`, while retaining the
   existing production `fsdp` default outside one-host mode.
6. Fixed rollout-only to skip the unnecessary trainer-logprob forward.
7. Set prompt 3584 plus response 512 so the exact trainer sequence is 4096,
   satisfying the Splash-attention block divisibility gate.
8. Implemented fail-closed backward-no-commit: compute the real gradient,
   fingerprint model/reference/optimizer/accumulator state, then skip every
   optimizer, accumulator, train-step, and checkpoint mutation.
9. The repeatable runner now defaults artifacts to the persistent v5p data
   disk and rejects a dirty tracked worktree unless development evidence is
   explicitly requested with `DEEPSWE_ONEHOST_ALLOW_DIRTY=1`.

## Real execution evidence

### Rollout-only

- Persistent directory:
  `/mnt/disks/tunix-data/deepswe-onehost-evidence/20260812-p44-local-dev/rollout-only`.
- Terminal marker: `DEEPSWE_ONEHOST_ROLLOUT_PASS`.
- Both trajectories loaded the real task and executed one real `search` tool
  action in Docker; recorded environment step latencies were approximately
  0.194 s and 0.155 s.
- Both then ended `MAX_CONTEXT_LIMIT_REACHED`, reward `0`, so this proves a
  real environment action and durable trajectory wiring, not a completed
  episode or solve-quality result.
- SHA-256:
  - trajectory: `5ae030e50962be65aed58966b087c7f026da53b443866801a20226168b6f1d2d`;
  - metrics: `c79b822cd64e86e65f726aab9f21d84a8c5d42b14ba4ada29f91739337fa5c32`;
  - manifest: `5fa7073e6cabc2d69992b631c1af856f40c135d67ecc0658c83299ae92de4a16`.

### Backward-no-commit

- Persistent directory:
  `/mnt/disks/tunix-data/deepswe-onehost-evidence/20260812-p44-local-dev/backward-no-commit`.
- The run completed model load, two real rollouts, finite trainer forward,
  exact sampler/trainer logprob comparison, and a real backward invocation.
- Report verdict: `INCONCLUSIVE_NO_SIGNAL`, deliberately returned as exit 3.
  Both rewards and RLOO advantages were zero, so the finite gradient norm was
  exactly `0.0`; nonzero backward signal is not proved.
- No-commit boundary: `commits=0`, train step `0 -> 0`, and no changed model,
  reference, optimizer, or accumulator paths.
- Optimizer placement: `device`; no optimizer offload was used.
- Peak HBM: device 0 `38,568,704,000` bytes; devices 1-3
  `38,566,601,728` bytes each; device limit `102,803,437,568` bytes
  (about 35.92 GiB of 95.74 GiB per device).
- SHA-256:
  - backward report: `d70ac3b1056d6c7e7c40b629ce630e935b659c512ecbef29ebf1a94f6c8871d2`;
  - trajectory: `aaaaa6445fac47275815653a56edc1cac1d18f12684a9c88d127a29add46b67e`;
  - metrics: `7b25b727ccfc5bd52e77db1b385642103a779482d80a1dddb848f261dd311f3e`;
  - manifest: `4a55d653c3e1b4a704647dbc8de5ebce3eabb2a30f7ae180110079e4ef093088`.

### Latest-head reconciliation rerun

After fast-forwarding to operator head
`d8184123448d0add72b72f09d0a6faf5d326c26e`, both stages were repeated on
the same real v5p host against that head plus the uncommitted P44 diff. The
runner recorded `tracked_dirty=1`, so these remain explicitly labeled
development evidence. P38's newly shared precheck change did not alter the
one-host path:

- rollout terminal marker: `DEEPSWE_ONEHOST_ROLLOUT_PASS`;
- backward terminal marker:
  `DEEPSWE_ONEHOST_BACKWARD_INCONCLUSIVE_NO_SIGNAL`, exit 3;
- both stages again executed two real Docker tool actions, with environment
  step latencies between about 0.148 s and 0.164 s;
- sampler/trainer logp and probability differences were exactly zero;
- no-commit, optimizer placement, gradient, and HBM report is byte-identical
  to the earlier backward report.

Persistent directories and SHA-256:

- rollout:
  `/mnt/disks/tunix-data/deepswe-onehost-evidence/onehost-latest-d818-rollout-20260812T0433Z/`;
  trajectory `0561a53a2c852adb1c568eec06681f0592de9c585a77f91be7677a751ef2e1a5`,
  metrics `0e5cae06e6a80376b9e285c8a45d8eced92e8d7aca923675b660457399aaa804`,
  manifest `df8017fea75b7675ce50a78fb60b2150384e95bb41be613d11cbe0ed799c1186`;
- backward:
  `/mnt/disks/tunix-data/deepswe-onehost-evidence/onehost-latest-d818-backward-20260812T0441Z/`;
  report `d70ac3b1056d6c7e7c40b629ce630e935b659c512ecbef29ebf1a94f6c8871d2`,
  trajectory `8bebc3fa389e2536971bca62843b4d61eb93e6ddbcb09b4cbdfee911e9a68e8a`,
  metrics `20f57db6ad41ee1becf7ffabcb2ad7cbde24f2d2c5fbcbcc495d663b8226a83c`,
  manifest `6f6629be29fb256c66d12a53da6d60dd1b02080c5f8e0cf657cd69a2685ec135`.

## Regression evidence

- P44 CPU: PASS, 40 tests.
- P44 Qwen3-4B exact image: PASS, overlay 29/29, contract 5/5,
  SwiGLU/matmul exact forward and VJP probes, and two learner tests.
- P43: PASS, 22 tests. P39: PASS, 15 tests.
- P34: static 10 suites, trajectory, update, and exact-image gates PASS;
  exact-image terminal marker reports 55 unit cases, two Pallas cases, five
  Qwen3-32B contract cases, and the scheduler gate.
- Syntax/compile/diff checks pass after the final repair.

## Claim ceiling and next gate

This proves only one-host Qwen3-4B frontend, cached dataset, reviewed task
selection, vLLM rollout, real Docker environment action, durable trajectory
and solve-metric artifacts, trainer forward/backward execution, device-resident
optimizer placement, HBM headroom, and a mutation-free no-commit boundary.

It does not prove a completed episode, nonzero learning signal, one optimizer
update, Qwen3-32B, TP8 kernels, Pathways, separated roles, DP4/DP16 reduction,
64/256-chip behavior, model quality, zero-TIM, or production readiness. Do not
promote a one-update local stage from this zero-signal batch. After explicit
publication, the operator should first repeat `rollout-only` from a clean
operator-branch checkout, then use the independent 64/256 rollout-only ladder
for topology evidence.
