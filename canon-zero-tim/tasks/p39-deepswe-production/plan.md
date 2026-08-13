# P39 DeepSWE production contract

## Objective

Turn the additive P34 Qwen3-32B DeepSWE package into a reviewable production
candidate without changing precision, loss, sampling, gradient, optimizer, or
credential semantics. The first target remains an attempt-zero full training
run. Finite alignment residuals are convergence telemetry; every structural,
nonfinite, optimizer, replica and infrastructure contract remains fail-closed.

## Sources

- Current P39.5 bounded-lifecycle development base after synchronization:
  `yuxzhang/canon-zero-tim` at
  `6905ca7c8551eeb8be772c40213e57e91bcfb0a7`. Implementation started from
  `4e4ca2891a01448f09428affd1eb2434bbd61657`; the intervening operator commits
  changed only FrozenLake/P38 files outside the P39.4 change set.

- Production-candidate hardening base: `yuxzhang/canon-zero-tim` at
  `5ee6dbfb5601cf1d1f864ccf6859764ba1f321fe`. P39 was developed from
  `697a29ab4b27015297af8e3dbb37c49db3560445`; the intervening remote change
  touched only `cluster/jobset-64chip.yaml`.
- Workload implementation reference: `yuxzhang/deepswe-quality-fix` at
  `023978b976dd6d94e7a42948c3f3a68e34d73744`.
- The quality-fix branch supplies the DeepSWE dataset, agent, environment,
  reward, GRPO defaults and runtime behavior. It is not a zero-TIM topology or
  numerical contract and is not merged wholesale.

## Production configuration ledger

| Boundary | Signed value |
|---|---|
| Model | Qwen3-32B, full-parameter training |
| Role topology | one 4x8x8 slice split into two host-complete 128-device roles |
| Mesh per role | DP16xTP8; parameters replicated over DP, sharded only over TP |
| Workload | 8 prompts x 8 generations = 64 global trajectories |
| Per-DP trajectories | 4 |
| Trajectory mini batch | 64 global trajectories |
| Gradient trajectory micro batch | 4 global trajectories |
| Prompt / response / turns | 4096 / 32768 / 50 |
| Sampling | temperature 1.0, top-k disabled, top-p 1.0 |
| GRPO | one iteration, beta 0, epsilon 0.2/0.28, off-policy steps 0 |
| Loss | `sequence-mean-token-scale`, RLOO advantages |
| Optimizer | AdamW, lr 1e-6, b1 0.9, b2 0.99, wd 0.01, grad norm 1.0 |
| Run length | full = exactly 1000 updates |
| Checkpoint | every 8 updates, retain 8 |
| Evaluation cadence | 10; no eval dataset is supplied by this recipe |
| Importance paths | rollout logprobs enabled; sampler IS and TIS disabled |
| Prefix cache | disabled |
| Optimizer state | device-resident; host offload is an explicit relaunch fallback |
| W&B | online and monotonic metrics required |
| Dataset | `R2E-Gym/R2E-Gym-Subset` train at `2e8108ff942f24fcb5686badfaf7f9a8808566d5` |
| Clean whitelist | 1851 unique images; SHA-256 `2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7` |
| Trajectory capture | every batch, durable and fail-closed, with solve/group metrics |
| Finite alignment residuals | warning-only for full convergence; nonfinite remains fatal |
| Zero-signal batch | record and continue normal commit; never resample or inject signal |
| Rollout lifecycle | turn 300 s; trajectory 4800 s; step/reward 1800 s; cleanup 300 s; R2E pod 5100 s; whole batch 5400 s |

The older quality-fix launcher used FSDP-named axes and smaller defaults in
some entry points. P39 does not copy those topology defaults. FSDP would add a
trainer-only parameter all-gather program and is outside the signed zero-TIM
forward contract. The P39 command pins every algorithm field instead of
depending on Python defaults.

## Shape ledger

| Quantity | Compact rollout | Padded prefill |
|---|---:|---:|
| caller-global M | 256 | 4096 |
| shard-local M after DP16 | 16 | 256 |
| canonical-kernel M | 256 | 256 |
| semantic valid rows per rank | 16 | 256 |
| per-rank scheduler token capacity | 256 | 256 |

Scheduler request capacity is 4 per rank and 64 globally. The only admitted
global token padding is 4096. Global M512 and every additional precompile
bucket remain rejected.

## Phases

1. Repair the signed `sampler_is=None` P34 admission and add adjacent negative
   controls.
2. Require and persist the pre-backward A/B/C report; make the P34 classifier
   consume exactly one pre-alignment record per update.
3. Run renderer output through the real `00_env.sh` preflight in CPU tests.
4. Pin the quality-fix-derived algorithm fields in the renderer and validate
   them in the real DeepSWE entry point.
5. Pass static and pinned exact-image gates.
6. After the shared P38 changes, harden ambiguous SHA serialization, require
   one exact rollout/trainer weight attestation per update, and rerun the
   complete local gate from the new base.
7. Target `backward-no-commit`, then separately approve promotion. Target
   status remains `TARGET NOT RUN` until a raw 4x8x8 Attempt 0 artifact passes.
8. Before the 4x8x8 launch, add a separate 64-chip integration and capacity
   pilot: split one 4x4x4 slice into 32 rollout and 32 trainer devices, use
   DP4xTP8 per role, and exercise device-resident optimizer state for up to
   three updates. The detailed contract is in
   `phases/p39-2-64chip-tp8-resident-pilot.md`.
9. Promote to the existing 4x8x8 DP16xTP8 production geometry only after the
   pilot classifies optimizer capacity. Resident mode requires the pilot's HBM
   margin gate; otherwise the 256-chip run retains pinned-host offload. The
   256-chip run must independently revalidate DP16 collective and replica
   behavior.
10. Operator supersession on 2026-08-12: defer the optional capacity pilot and
    launch one reviewed `full` production-topology manifest directly.  Default
    to device-resident optimizer state, pin the clean dataset/whitelist, persist
    every trajectory batch, and keep finite A-B/B-C residuals warning-only.
    The short stages remain available but are not launch prerequisites.
11. Attempt `p34r02` failed before rollout because P34 inherited the
    direct-attached four-device `CANON_EXPECT_MODEL_MESH_IDS=0,2,1,3`
    assertion into a 128-device Pathways role.  Explicitly clear the
    allocation-specific assertion in the P34 profile/renderer and reject any
    nonempty P34 value during preflight.  After publication, retry the same
    signed 32B/data/topology/device-optimizer configuration; do not hard-code
    device IDs observed on a prior allocation.
12. Attempt `p34r03` entered real 32B rollout but update zero remained active
    for more than four hours. The old trajectory engine allowed negative
    remaining time and lacked request, reward, cleanup and shared batch
    deadlines. Add true vLLM abort, one trajectory wall clock, bounded reward
    and cleanup, confirmed R2E pod deletion, and one 5400-second batch
    watchdog. Lock a matching Qwen3-4B three-update debug recipe to a
    3600-second batch budget on both 64 and 256 devices before the next 32B
    target attempt.

## Rollback

Do not render or apply P34, or leave all P34 admission variables at zero. The
changes are additive and default-off. Existing P33 recipes remain unchanged.
