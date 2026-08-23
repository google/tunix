# P58.5N — Native 128-chip full campaign

Status: active.

## Purpose

Run the untreated native DeepSWE-derived Qwen3-4B-Instruct path for exactly
1,000 committed optimizer updates on the frozen P58 recipe. This is the user's
direct-full decision: updates 1–3 are live monitoring milestones within one
continuous job, not a separate canary and not an early-stop condition.

This phase proves native integration and training only. It does not run or
validate the deferred zero arm and cannot establish a paired treatment effect.

## Immutable contract

- source: exact post-push readback of `yuxzhang/canon-zero-tim`;
- image: registry digest, never a mutable tag;
- model/data: Qwen3-4B-Instruct-2507 and the frozen 1,012-task clean list;
- topology: one 128-chip `4x4x8` slice split into rollout DP8 x TP8 and trainer
  DP8 x TP8;
- Kueue affinity: worker sentinel `tpu-v5p-slice` delegates the concrete pool
  to ResourceFlavor and must not appear as a literal node-pool selector;
- network: the CPU head and TPU workers retain the proven Pathways host
  network; required hostname-level anti-affinity prevents any two JobSet
  `pathways-head` Pods from sharing fixed ports 29000/29001, and workers reach
  that exact head's RM through generated JobSet Pod DNS on port 29001;
- recipe: B8 x G16, response 16,384, 50 turns, RLOO, fixed-context
  `sequence-mean-token-scale`, TPU-resident optimizer, optional interventions
  off, prefix cache off; all 128 trajectories run as one concurrency wave,
  equal to rollout DP8 x max-seqs16 capacity;
- stage/run: `full`, next fresh run-id `p58f13`, exactly 1,000 optimizer commits;
- arm: `native` only. Rendering or applying `zero` is outside this phase.

## Admission gate

The rendered worker must retain `google.com/tpu: 128`, TPU accelerator
`tpu-v5p-slice`, and exact topology `4x4x8`, while omitting literal
`cloud.google.com/gke-nodepool: tpu-v5p-slice`. Server-side dry-run must pass,
then Kueue must report `QuotaReserved=True` before runtime diagnosis begins.
Failure here is admission `INCONCLUSIVE`, not training evidence.

The rendered CPU head must keep `hostNetwork:true` and
`dnsPolicy: ClusterFirstWithHostNet`, use `cpu-np`, and carry required Pod
anti-affinity selecting `jobset.sigs.k8s.io/replicatedjob-name=pathways-head`
at `kubernetes.io/hostname`. The JobSet must publish not-ready Pod DNS names.
The TPU worker also remains `hostNetwork:true`; both its ResourceManager
argument and `PATHWAYS_HEAD` must name the generated P58 head Pod DNS, never
localhost, a node IP, or another JobSet. This makes the scheduler/autoscaler
reserve a separate CPU node for every fixed-port head without changing the
proven Pathways transport.

## Online monitoring

- admission: quota reservation, concrete flavor, 32 four-chip worker pods, Pathways
  device count 128;
- first completed batch: 128 journal rows, sandbox-start/environment/model
  timeout split, cleanup receipts, solve/signal group metrics;
- commits 1–3: finite forward/backward, finite nonzero Native mismatch on A-B
  or B-C, shape-valid finite trainer-program observations, device-resident optimizer, and
  monotonic transaction and journal state;
- later milestones: checkpoint 8, then updates 32, 100, and every 100 updates;
- evaluation remains at the signed recipe cadence; checkpoint cadence remains
  every 8 committed updates.

A healthy job continues after update 3. Stop only for a signed hard failure,
not because the former canary horizon has been crossed.

## Exit gate

The native full classifier must report `PASS` from complete, digest-verified
artifacts and exactly 1,000 commits. It must prove a finite nonzero serving-path
treatment dose across A-B or B-C, shape-valid finite stock-program boundaries, finite
training values, TPU-resident optimizer state, complete 128-row trajectory
batches, journal continuity, sandbox cleanup, evaluation/checkpoint cadence,
and transaction integrity.

An ordinary model/context/runtime all-filtered batch may advance `batch_index`
without an optimizer commit; it must preserve unchanged optimizer state and
may make the number of consumed batches exceed 1,000. A full sandbox-start
outage must journal and stop without consuming a later prompt. Partial or
tampered evidence, no Native serving-path mismatch (`NO_TREATMENT`),
nonfinite/invalid Native drift, or any Zero-arm drift cannot be promoted.

## Attempt boundary

P58c05 and p58f01 through p58f10 are immutable `INCONCLUSIVE` evidence.
P58f01 exposed sandbox LocalQueue and reset-time provenance faults. P58f02
exposed a CPU-flavor/node-pool mismatch; moving the head and sandboxes to
`cpu-np` was the correct repair. P58f03 then completed 128 real trajectories
in 616.3 seconds and durably journaled them, proving that rollout and sandbox
throughput are no longer the first failure. It stopped before trainer forward,
backward, or update because native was routed to a canonical-adapter-only
weight gate.

The repaired weight gate uses an exact read-only live-weight observer for signed P58
native and keeps the canonical registered-adapter path for zero. Native still
has no numerical hook. Any mismatch, invalid mesh, missing signature, or
leaked adapter remains fatal. P58f03 has no optimizer checkpoint and is not a
resumable training root; preserve its trajectory journal as diagnostic
evidence only.

P58f04 completed 128 real trajectories in 557.2 seconds, durably journaled
them, and passed exact live-weight attestation for 398 leaves and
4,022,468,096 elements. It then failed before trainer forward/backward/update
because processed `S_prefill` was wired only to the canonical processed engine,
which native correctly disables. The repair adds an independent, observer-only
stock B overlay gated solely by the signed P58 native tuple. Native retains
`CANON_PROMPT_PROCESSED_LOGPROBS=0`, `CANON_ENGINE_MODULE_C=0`, and every other
zero-TIM disable/absence. Zero retains the canonical engine and sets the stock
observer to zero.

P58f05 proved that observer repair. It completed 128 trajectories in 486.4
seconds, durably journaled 126 `SUCCEEDED` plus two
`MAX_CONTEXT_LIMIT_REACHED` rows, observed six solved trajectories across two
mixed/effective groups, passed exact live weights over 398 leaves, and emitted
one processed-B marker covering all 2,048 prompt rows. The alignment sidecar
was attached, then `gsm8k_ab_report_policy()` rejected the run before trainer
forward/backward/update: P58 was present in the alternative-workload count but
that branch admitted only `one-update/three-update`, not the signed full stage.

The p58f05 repair did not add a flag. It admits P58 Native only when
`CANON_P58_TIM_ADMITTED=1`, no P39/P43/P44 mode competes, and the stage/horizon
is exactly `three-update/3` or `full/1000`. P58f06 proved that admission repair:
its 492.7-second rollout wrote 128 durable rows, observed three solved
trajectories, passed exact 398-leaf weights, processed all 2,048 Native B rows,
and executed alignment over 405,827 action tokens. Both A-B and B-C were valid
and finite, but the policy's P58-specific boundary tuple still allowed only
A-B and therefore blocked finite B-C before trainer forward/backward/update.

P58f07 proved that correction: 128 real RepoEnv trajectories completed with
`N_action=436,464`; pre-backward alignment passed with A-B/B-C warnings; and
the trainer entered a real value-and-grad/backward call. It then stopped on
post-backward `T_old_vs_T_current` and derived `r`. The durable P58 logps marker
from the same frozen launcher family shows that standalone `T_old` was scored
as one 128-trajectory program, while the signed trainer geometry slices the
same ordered rows into eight 16-trajectory value-and-grad programs. That is a
program-shape mismatch, not a valid same-program trainer repeat.

The corrected Native contract does not replace the quality-fix program. It
keeps the stock standalone 128-trajectory `T_old` observer and treats finite
`T_old_vs_T_current` plus finite derived ratios as measurement. With
`use_rollout_logps=true` and sampler-IS disabled, rollout A—not observer
`T_old`—is the old logprob input to the loss. The classifier requires every
observed Native boundary to be present, shape-valid, and finite; Zero still
requires all boundaries exact.

P58f08 stopped before rollout: six concurrent host-network Pathways heads
already occupied all six `cpu-np` nodes, and the seventh was packed onto an
occupied node. Its worker reached foreign CL/42 on port 29001 rather than its
own CL/956357083 RM. A `deepswe-cpu-pool` trial also failed because the worker
could not maintain the scheduler pipe across the node-pool subnet boundary.
The correct placement repair is required hostname anti-affinity while keeping
the head on `cpu-np` and preserving host networking.

P58f09 proved correct head attachment and completed 128 Step-0 rollout slots
in 1,699.1 seconds. Reset-deadline trajectories that ended before their first
observation retained no `agent.trajectory.task`; learner preprocessing then
passed `None` to `merge_micro_batches()` and crashed before the P58 journal,
alignment, forward, backward, update, or checkpoint. The collector repair
uses `env.task` as the original-input fallback only for this pre-observation
case and fails closed if no dictionary is available. Compact rows retain their
existing status and zero policy mask; there is no filtering or resampling
change. P58f08 and p58f09 are not resumable training roots.

P58f10 entered Step-0 rollout but retained concurrency 64 for the B8 x G16
batch, so 128 trajectories ran in two sequential waves. At the unchanged
3,600-second batch deadline only 5/8 prompt groups had completed and the
orchestrator failed closed before journal, trainer, or optimizer state. The
repair makes concurrency 128, exactly one wave and exactly the provisioned
rollout capacity DP8 x max-seqs16. It does not extend the 3,000-second episode,
300-second cleanup, or 3,600-second batch deadlines. Per-trajectory compact
timeouts continue as zero-mask rows; a whole batch that cannot drain remains a
hard failure. P58f10 has no resumable training state. After fetching and
reading back the final operator tip, the next attempt is fresh Native `p58f11`. Zero remains
deferred.

P58f11 proved that concurrency 128 drains the full B8 x G16 batch in one wave:
all 8 prompt groups and 128 trajectories completed in 1,209.2 seconds. One
`env.reset` timeout then exercised the p58f09 fallback. `SWEEnv` had preserved
the dataset row in `self.entry` but had initialized inherited `self.task` as an
empty mapping; pre-reset provenance added only `policy_version`. The fallback
was therefore a mapping but lacked `prompts`, and learner processing stopped
with `KeyError: 'prompts'` before journal, alignment, trainer, optimizer, or
checkpoint state.

The published repair seeds `SWEEnv.task` with the normalized prompt before reset, keeps
the singleton batch shape required by learner merge, and makes the
policy-seeded environment task authoritative for both successful and
pre-observation termination paths. A policy-seeded mapping without `prompts`
is now rejected at collection. The timeout row remains compact-filtered with
its zero policy mask; no row is dropped or resampled. P58f11 is immutable
`INCONCLUSIVE` and not resumable. After final operator-tip readback, use fresh Native
`p58f12`; Zero remains deferred.

P58f12 wrote a durable 128-row Step-0 journal and thereby target-proved the
p58f11 normalized-prompt schema repair. It did not reach model rollout:
128/128 sandboxes remained Kueue `scheduling_gated` until start timeout, so
all rows were compact-filtered `ENV_TIMEOUT`, completion/action token counts
were zero, and `generate()` had no sampling provenance to expose. Processed-B
rescore nevertheless ran and stopped before alignment/backward/update with
`processed S_prefill must follow generate()`. P58f12 is immutable
`INCONCLUSIVE`; its journal is diagnostic evidence, not resumable trainer
state.

The local phase repair makes ordinary all-filtered model/context/runtime
outcomes a complete no-commit transaction. Empty completion targets skip the
engine with signed `engine_called=false` provenance; any nonempty target still
requires real post-generation sampling provenance. Alignment admits zero
actions only for the signed durable all-compact P58 batch. Trainer and outer
progress remain unchanged: no optimizer commit, weight sync, policy-version
increment, or trainer/RL global-step increment. Only `batch_index` advances,
and the next prompt batch is consumed without resampling.

The p58f12-shaped full sandbox-start outage is intentionally stricter. After
the 128-row journal is durable, it emits `[P58.SANDBOX_CAPACITY] BLOCKED` and
raises `BLOCKED_SANDBOX_CAPACITY` before rescore/trainer or any later prompt
consumption. Fresh Native `p58f13` is next only after publication/readback,
one production-shaped sandbox reaches Running through Kueue, and the operator
confirms capacity/quota for the 128-Pod request. All geometry, deadlines,
optimizer placement, Native/Zero flags, and the Zero deferment remain
unchanged. Main's SandboxFleet implementation remains deferred and off.
