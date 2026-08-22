# P58 DeepSWE native-first training handoff

## Current checkpoint

P58 was developed in isolated worktree
`/home/yuxuan/code_rl_repro/worktrees/p58_deepswe_native_zero_0821` and approved
for publication to `yuxzhang/canon-zero-tim`. The implementation commit is
`c5bdc9d993dfaf1a6956335609fbf259f9ed95f7`; its first post-push readback
matched exactly with ahead/behind `0/0`. Always obtain the runnable
40-character source SHA by fetching and reading back that branch after push;
the latest branch may contain this later publication-evidence-only commit. Do
not infer the runnable SHA from the historical base or a mutable tag.

The latest P58 admission repair and direct-full phase implementation commit is
`abbc76008e0a7fcb63562c27d5cf4608fb4f4e90`. Its first post-push readback
matched exactly with ahead/behind `0/0`. This documentation checkpoint advances
the branch once more, so the executor must still fetch and use the final
operator-branch SHA rather than pinning the implementation commit directly.

Latest source intake fast-forwarded this isolated worktree through immutable
p58f09 evidence to operator tip
`3edf480072126145acc2df259419e12dd2737c69`. P58f07 proved the published finite
Native B-C warning repair, then exposed an over-strict Native trainer-program
observer gate. The correction was published as
`81622977bf15393798c671e578ee059d1268e78b`; its first readback matched local
HEAD, `FETCH_HEAD`, and `origin/yuxzhang/canon-zero-tim` with ahead/behind
`0/0`. This documentation checkpoint advances the branch once more, so the
executor must fetch and exactly read back the final operator tip. The p58f09
repair was published as `678bc5cfbcec386fd655e6685365c937e826d547`; its
first readback matched local HEAD, `FETCH_HEAD`, and the remote-tracking branch
with ahead/behind `0/0`. Source intake then fast-forwarded to exact operator tip
`28817bfb3a14c95f42b3950f03380d1c6c03d336`, which contains immutable p58f10
timeout evidence. P58f10 reached Step-0 rollout but the B8 x G16 batch was
throttled into two waves by concurrency 64; only 5/8 prompt groups completed
before the 3,600-second hard batch deadline. The local repair makes all 128
trajectories one wave, matching rollout DP8 x max-seqs16 capacity. It was
published as implementation commit
`44b6fb4527a8a05bf649b5140d12142e2abef83f`; its first remote readback matched
local HEAD, `FETCH_HEAD`, and the remote-tracking branch with ahead/behind
`0/0`. This documentation checkpoint advances the branch once more, so fetch
the final operator tip. The next run id is fresh `p58f11`.

The user previously waived P58.3 and the separate three-update stop, then chose
the native 128-chip full 1,000-update stage. That historical phase remains
waived rather than promoted. For the p58f05 repair, the user later requested a
new bounded direct-attached one-host gate before publication. Its runner is
implemented, but this container exposes no `/dev/vfio` and returned
`P58_ONEHOST_ALIGNMENT_BLOCKED`; it is not a TPU PASS. Updates 1–3 remain live
monitoring milestones in the same full job, not an early-stop condition. Zero
is not optimized enough for launch and is explicitly deferred. No Kubernetes
apply or TPU launch is authorized by this handoff alone.

Native attempts `p58c01`, `p58c02`, and `p58c03` are bootstrap
`INCONCLUSIVE` results.
P58c01 failed in `00_env.sh`; its published fix preserves native
`CANON_P32_DP_REDUCTION_ADMITTED=0`, exports three unrelated FrozenLake zeros,
and passes the renderer-to-real-`00_env.sh` regression. The fix implementation commit
`acd3136267214b367a6755d0ba28d80e883d6753` was pushed and its first remote
readback matched exactly with ahead/behind `0/0`. Fetch again and use the
final operator-branch SHA because this publication note is a later docs commit.

P58c02 then initialized Pathways and stopped before importing the model: direct
file execution of `/app/examples/deepswe/canonical_entrypoint.py` did not put
`/app` on `sys.path`, so its package-qualified `examples.deepswe` target could
not be found. The local fix derives the repository root from `__file__`, adds
it before the package import, and changes native stock preflight to exercise
the identical direct-file entrypoint. The exact command now exits zero from
`/tmp` in the pinned image, and the complete exact-image gate passes. These
changes were published as `82d82f72a7220d945737d95f6266b5b7e2cfe706`;
the first post-push readback matched exactly with ahead/behind `0/0`. Fetch the
final operator tip because this publication checkpoint advances it once more.

P58c03 proved that the preceding admission, install, stock-engine, Pathways,
and direct-entrypoint fixes work, then stopped before model initialization.
`00_env.sh` correctly removed native-only presence-sensitive zero-TIM switches
inside its child shell, but its generated `env.sh` contained exports only.
When the parent entrypoint sourced it, the raw renderer value
`CANON_LOGPROB_M=256` remained present and the DeepSWE Python contract
correctly rejected the native environment. The W&B-run fatal printed after
that exit is derivative, not the first failure.

The fix turns the generated `env.sh` into an authoritative snapshot of
all managed non-secret namespaces: it clears the caller's managed values,
then exports the exact resolved set. Secret injection variables and token
values are neither cleared nor serialized. The exact regression seeds the
raw parent with `CANON_LOGPROB_M=256`, executes real `00_env.sh`, sources its
snapshot, verifies native absences, and passes the Python contract. Focused
P58/P34 tests, the P57 81-test adjacent suite, and the full pinned-image gate
pass. It was published as `c0ca41805bd65a4fdede4825ed2835cdce6e13ed`;
the first post-push remote readback matched exactly with ahead/behind `0/0`.
Fetch the final operator tip because this publication-evidence checkpoint
advances the branch once more.

P58c04 proved the complete bootstrap and initialization chain through real
128-chip Pathways discovery, Qwen3-4B/vLLM initialization, W&B initialization,
and entry into `run_producers_from_stream`. It then requested all 128 RepoEnv
sandboxes concurrently. No sandbox was logged Running before the 1,200-second
start deadline, and the interleaved log retains at least 121 explicit timeout
records. The pinned R2E `start_container` swallowed the start `TimeoutError`,
deleted the pod, and returned with `container=None`; later
setup attempted a websocket exec into that deleted pod. Kubernetes' real 404
was then obscured by the client library's `None.decode` AttributeError. The
websocket payload decoder is not the root cause and must not be patched or
made permissive.

The local repair bypasses the upstream exception-swallowing wrapper only for
the Kubernetes backend, propagates the original timeout after confirmed pod
deletion, and proves that a reset-time start failure becomes the existing
signed `ENV_TIMEOUT` trajectory status. Docker behavior remains delegated to
upstream. A bounded timeout marker preserves pod phase and scheduler
conditions without inspecting the pod spec/environment. At the p58c04 repair
checkpoint, the P58 renderer used reference sandbox concurrency 64, so the
unchanged B8 x G16 batch was created in two waves. That historical choice is
superseded by the p58f10 one-wave repair below. This changes neither
data, sampling, RLOO/loss, meshes, optimizer, nor update horizon. Two newly
shared stock-contract booleans are explicitly zeroed in the native profile;
that is compatibility hardening, not a new treatment. Focused tests and the
full pinned-image gate pass. The trajectory journal and W&B now retain bounded
timeout provenance: status; sandbox/model/environment/reward/deadline stage;
unschedulable; and insufficient CPU/memory counts and ratios. Raw scheduler
messages stay in the run log. These changes were published as
`174fcf3a42af3e9cd465307843a1c19a08098c99`; its first remote readback matched
with ahead/behind `0/0`. Fetch the final operator tip after the publication
checkpoint rather than pinning this implementation commit directly.

P58c05 never reached the runtime. Its Workload remained
`QuotaReserved=False`; Kueue reported that flavor `0xv5p-8` did not match the
worker node affinity. The rendered worker combined exact `4x4x8` topology with
literal node-pool selector `tpu-v5p-slice`. That value is a Kueue sentinel, not
a concrete node pool, so it contradicted ResourceFlavor admission. No JobSet
pod, Pathways process, model, sandbox, trajectory, optimizer action, or
checkpoint started. The evidence under `evidence/p58c05_admission/` is
immutable and there is no resumable state.

The local repair makes all registered Kueue sentinels delegate concrete pool
affinity to ResourceFlavor while retaining the TPU accelerator and exact
topology. Explicit real node-pool names remain exact. The next run is fresh
native full-stage `p58f01`, not a retry or resume of p58c05.

P58f01 proved that repair: it reached 128 Pathways devices, the exact 64/64
role split, Qwen3-4B/vLLM and online W&B initialization, and the rollout
producer. It did not produce a usable R2E trajectory. All 128 environment
resets timed out, and at least 127 bounded Pod markers say
`PodScheduled=False`, reason `SchedulingGated`. The runtime-created standalone
Pods lacked `kueue.x-k8s.io/queue-name`; on this cluster Kueue therefore added
an admission gate but had no LocalQueue through which to admit them. After the
all-timeout batch completed, a second local bug raised
`policy_version is missing from trajectory task`: environment reset had failed
before `_model_call` assigned that provenance, so the batch crashed before
the P58 journal boundary. P58f01 is `INCONCLUSIVE`, immutable, and not
resumable. Its raw log SHA-256 is
`16c513c773ac2bfb1542178b4e42b03098bb9114564106b03f83c0195a0d542f`.

The repair derives `R2E_K8S_QUEUE_NAME` from the parent JobSet queue label,
persists it through the authoritative environment snapshot, validates it
without normalization, and applies it to every sandbox Pod. It also assigns
the current `policy_version` when the environment is constructed, before
reset, while retaining the strict downstream missing-provenance check.
`SchedulingGated` is now a separate bounded trajectory/W&B dimension. The next
fresh native full attempt is `p58f02`; do not reuse the p58f01 root.

The p58f01 repair was published as
`c67e9d5bfa3f1b3b592a2440075eb165e073e6ac`; its first remote readback matched
exactly with ahead/behind `0/0`. This publication checkpoint advances the
branch once more, so the executor must fetch and use the final operator tip
rather than pinning the implementation commit directly.

P58f02 then reached Step 0 but the sandboxes stayed `SchedulingGated`: the
cluster's `cpu-user` flavor requires `nodeSelector: cpu-np`, while the job was
requesting `deepswe-cpu-pool`. The user confirmed that moving the CPU head and
sandboxes to `cpu-np` resolves this; a general in-process CPU fallback is not
part of the solution. That routing repair was published in source
`7208d7b330759ac7dc31493ece65d32a6c355308`.

P58f03 used that source and completed the first real rollout batch in 616.3
seconds. Its durable journal has 128 rows: 126 `SUCCEEDED`, two
`MAX_CONTEXT_LIMIT_REACHED`, three solved trajectories, two mixed/effective
groups, and 32 nonzero advantages. No sandbox-start timeouts occurred. The raw
log is `evidence/p58f03/run.log`, SHA-256
`fdb958d5e1db8bafa25b6df8c3223a3c6a642d00c6a1915bb34a8e17b5bcf600`.
The journal is
`/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-native-full-p58f03/debug/batch-000000.trajectories.jsonl.gz`,
SHA-256
`26c92d2153865cc14296303fcb97afd98f857744e50574032b6eba8631f23a9e`.

P58f03 then stopped before trainer forward/backward/update. The generic P34
weight gate called `attest_canonical_engine_weights`, which intentionally
requires a registered canonical adapter, while native correctly runs with
`CANON_ENGINE_MODULE_C=0`. The first failure was therefore a routing/contract
bug, not rollout throughput or weight drift. The local repair exposes a shared
exact-live-weight interface: zero still delegates to its registered canonical
adapter; signed P58 native performs only the same pure leaf mapping and
bitwise live-weight comparison. It neither registers an adapter nor changes
serving math. Missing/mismatched weights, invalid DP8 x TP8 mesh, unsigned
native routing, and a leaked native adapter remain fatal. Focused routes, all
15 rollout canonical tests, and the full pinned P58 exact-image gate pass.
The implementation was published as
`234eaddb8e3543083927aa10effe101abef18a91`; its first remote readback matched
exactly with ahead/behind `0/0`. This publication-evidence checkpoint advances
the branch once more, so fetch and pin the final remote tip rather than the
implementation commit directly. That repair was exercised by fresh native
`p58f04` below rather than by resuming p58f03. Zero remains deferred.

P58f04 completed the next real rollout batch in 557.2 seconds and durably
journaled 128 rows: 125 `SUCCEEDED`, three `MAX_CONTEXT_LIMIT_REACHED`, six
solved trajectories, five all-failed groups, one mixed/effective group, two
incomplete groups, and 16 nonzero advantages. It proved the preceding repair
with `[P34.WEIGHTS] EXACT` over 398 leaves and 4,022,468,096 elements. The raw
log is `evidence/p58f04/run.log`, SHA-256
`a7b0cda5e7d359c7e320b29f8af197db0dd6c46dc34850aa55ffb350fb766fdd`.
The trajectory journal is
`/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-native-full-p58f04/debug/batch-000000.trajectories.jsonl.gz`,
SHA-256
`e39caf5df63ba54406a36427a413dea562e5771f4c52b30c840229d3178c1f3b`.

P58f04 then failed before trainer forward/backward/update. The shared
processed-`S_prefill` interface required the canonical
`CANON_PROMPT_PROCESSED_LOGPROBS=1` engine path, while native correctly keeps
that flag and `CANON_ENGINE_MODULE_C` at zero. Reusing the stock raw helper
would be wrong because it rolls targets across a DP-packed buffer and can cross
request/padding boundaries. Enabling the canonical flag would contaminate the
native treatment.

The local repair adds a separately signed, observer-only P58 native stock-B
overlay. It is installed only after the six stock files verify; it changes one
runner call site plus one helper under an exact two-file manifest. It applies
decode-equivalent temperature/top-k/top-p transforms and derives targets from
absolute request history. It does not enter generation, trainer forward, loss,
backward, optimizer math, or commits. Native still has
`CANON_PROMPT_PROCESSED_LOGPROBS=0`, `CANON_ENGINE_MODULE_C=0`, and every other
zero-TIM numerical switch disabled/absent. Zero sets the new P58 observer flag
to zero and retains the complete canonical engine. Mixed tuples fail closed.
P58f05 proved the observer repair. It completed the next 128-row batch in
486.4 seconds: 126 `SUCCEEDED`, two `MAX_CONTEXT_LIMIT_REACHED`, six solved,
two mixed/effective groups, and 32 nonzero advantages. All timeout dimensions
were zero. Exact live weights passed over 398 leaves and 4,022,468,096
elements, and one observer marker covered all 2,048 prompt rows. The raw log
is `evidence/p58f05/run.log`, SHA-256
`73def19531ca1a9ef083a30d11ceb89696afcbe4125bd128f7ff0e7152ec06a6`.
The trajectory journal is
`/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-native-full-p58f05/debug/batch-000000.trajectories.jsonl.gz`,
SHA-256
`90c179d799bb97416f1a4e6cf944a15326cef56360da179c771fad79fa02bcac`.

P58f05 then attached the alignment sidecar and stopped before trainer
forward/backward/update. `gsm8k_ab_report_policy()` already recognized the P58
arm and enforced Native-warning/Zero-strict, but its workload admission placed
P58 in a branch that accepted only `one-update/three-update`. The signed
`CANON_P34_RUN_STAGE=full` plus `CANON_P58_EXPECTED_UPDATES=1000` tuple was
therefore incorrectly rejected. This is a stale stage enumeration, not an
alignment red or missing treatment dose.

The published p58f05 repair separates P58 from the P39/P43/P44 debug-update branch and
admits only its signed Native tuple: `CANON_P58_TIM_ADMITTED=1`, no competing
DeepSWE mode, and an exact `three-update/3` or `full/1000` stage/horizon. It
does not add a flag. P58f06 proves it: the 492.7-second rollout durably wrote
128 rows (126 `SUCCEEDED`, two `MAX_CONTEXT_LIMIT_REACHED`, three solved),
with five all-failed groups, one mixed/effective group, two incomplete groups,
and 31 effective nonzero advantages. All timeout dimensions were zero. Exact
live weights passed over 398 leaves/4,022,468,096 elements and the Native
processed-B observer covered all 2,048 prompt rows. The raw log is
`evidence/p58f06/run.log`, SHA-256
`34c6830d5b4179cf8ccdd697a0b03d9764fc75ffefa9313d5a1910914e774fd9`.
The trajectory journal is
`/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-native-full-p58f06/debug/batch-000000.trajectories.jsonl.gz`,
SHA-256
`ddaefb3c0efc8eb7f29724c80b5aa88ab38e8b49e7bd3cf7134c4916afe2e6f3`.

Alignment then executed over 405,827 action tokens. Both
`S_decode_vs_S_prefill` and `S_prefill_vs_T_old` were shape-valid and finite;
the former was already a warning, but the P58-specific tuple still treated the
latter as blocking. That contradicted the untreated Native treatment and the
user's earlier decision that finite B-C must not stop Native training. The
local correction makes both finite serving-path boundaries warnings and
updates the classifier to accept a finite nonzero dose on either. Nonfinite,
shape, weight, replica, transaction, and optimizer errors remain hard. Zero
remains strict at all boundaries. P58f06 has no optimizer checkpoint and is
not resumable training state.

P58f07 completed all 128 real SWE RepoEnv trajectories (`N_action=436,464`),
passed pre-backward with finite A-B/B-C warnings, completed Rescore B in 26.9
seconds, and entered real value-and-grad/backward. It then stopped at the
first post-backward gate on `T_old_vs_T_current` and derived
`r_all_exactly_1`. The durable launcher marker from this attempt family shows
`T_old` was computed by one standalone 128-trajectory trainer program, while
the frozen update structure computes `T_current` in eight ordered
16-trajectory value-and-grad programs. The arrays therefore came from
different batch programs; exactness is therefore not a valid admission
requirement for an untreated Native arm.

The corrected contract preserves the complete stock quality-fix program,
including the standalone 128-trajectory `T_old` rescore. With
`use_rollout_logps=true` and sampler-IS disabled, the loss uses rollout A as
`old_per_token_logps`; `T_old` is observer-only. Signed Native now records any
shape-valid finite `T_old_vs_T_current` and finite derived ratio drift as a
warning, while the classifier requires that boundary to be present and
finite. Zero remains exact. B8 x G16, all 128 training rows, rollout logps,
loss, eight-step gradient accumulation, optimizer placement and math, commit
cadence, and every Native/Zero numerical flag remain unchanged.

P58f07 has no durable optimizer receipt or checkpoint and is not resumable
training state. P58f08 then stopped before rollout: six concurrent Pathways
heads already occupied all six `cpu-np` nodes, so Kubernetes packed the next
host-network head onto an occupied node. Port 29001 connected its CL/956357083
worker to a foreign CL/42 ResourceManager. A follow-up placement on
`deepswe-cpu-pool` started the head but could not maintain the worker scheduler
pipe across the node-pool subnet boundary. The correct infrastructure repair
is therefore not a CPU-pool or Pod-network change: retain `cpu-np` and
`hostNetwork:true`, and require hostname anti-affinity between every JobSet
`pathways-head` Pod.

P58f09 proved correct Pathways attachment and completed all 128 Step-0 rollout
slots in 1,699.1 seconds. Reset-deadline rows that terminated before first
observation had `agent.trajectory.task=None`, even though `env.task` still
contained the original input. Learner `merge_micro_batches()` dereferenced
that value and crashed before the P58 journal, alignment, forward, backward,
optimizer receipt, or checkpoint. The local repair preserves the agent task
when present, otherwise falls back to `env.task`, and fails closed if neither
is a dictionary. Compact timeout/context rows retain the existing zero policy
mask and are neither dropped nor resampled. Renderer validation requires the
exact hostname anti-affinity plus retained head/worker host networking,
JobSet DNS, and RM/PATHWAYS_HEAD route. P58f08 and p58f09 are not resumable
training state.

P58f10 ran the source containing the prior placement/input repairs and entered
real Step-0 rollout. The batch deadline prevented post-rollout merge, so the
original-input fallback remains target-unproven despite exact-image coverage.
Its 128 trajectories were still admitted with
`max_concurrency=64`, creating two sequential waves. At 3,600 seconds only
5/8 prompt groups were complete, so the batch orchestrator correctly failed
closed before durable journal, trainer, optimizer receipt, or checkpoint. The
published repair sets concurrency to 128, exactly the raw batch and exactly rollout
DP8 x max-seqs16 capacity. Episode 3,000 s, cleanup 300 s, and batch 3,600 s
remain unchanged. Individual timeout/context outcomes still become compact
zero-mask rows; only a whole one-wave batch that cannot drain is fatal. P58f10
is not resumable. After separate publication/readback, use fresh Native
`p58f11`. Zero remains deferred.

Never modify or push `main`. The publication target is exclusively
`yuxzhang/canon-zero-tim`; the p58f09 repair is published there as
`678bc5cfbcec386fd655e6685365c937e826d547`, and the p58f10 one-wave repair as
`44b6fb4527a8a05bf649b5140d12142e2abef83f`. Always fetch the later final
documentation tip before rendering.

## What was implemented

- additive P58 DP8 x TP8 per-role workload/profile and a `4x4x8` paired
  renderer for `native|zero` and `three-update|full`;
- frozen Qwen3-4B-Instruct-2507 B8 x G16 recipe on the 1,012-task clean list;
- explicit 16,384 `sequence-mean-token-scale` norm and effective-row
  denominator matching the pinned DeepSWE quality-fix compact-filter path;
- denominator-weighted eight-way gradient accumulation for the stock trainer;
- matching global denominator behavior in the canonical segmented path;
- all-filtered no-commit for both paths, with no resampling;
- durable full-trajectory P58 journal, separate `batch_index` and
  `optimizer_step`, restart continuity/digest verification, per-batch solve and
  signal metrics, and W&B forwarding;
- native stock-engine verification and absence checks for the complete
  canonical numerical bundle;
- independent native-only processed-B observer with absolute request-history
  targets, exact two-file manifest, and mutually exclusive Native/Zero flags;
- native finite A-B/B-C/T_old-T_current warning boundaries with finite ratio
  diagnostics, and zero all-boundary strictness;
- native stock optimizer transaction receipts plus zero explicit fixed-tree
  transaction receipts;
- P58 fail-closed postflight classifier and automatic invocation from
  `90_run.sh`; and
- negative/regression controls for P34/P44 and the shared trainer/loss paths.
- authoritative resolved-environment reload semantics so child-shell unsets
  remain absent in all later entrypoint steps.
- required hostname anti-affinity for fixed-port Pathways heads while
  preserving host-network transport; and
- pre-observation reset-timeout original-input recovery from the environment,
  with a hard error when no mapping exists; and
- exact one-wave rollout admission: B8 x G16 = concurrency 128 = rollout DP8 x
  max-seqs16, without extending the signed timeout hierarchy.

The exact run instructions and artifact interpretation are in
`canon-zero-tim/cluster/P58_DEEPSWE_TIM_RUNBOOK.md`.

## Validated locally

Pinned local image ID:

```text
sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

Terminal marker:

```text
P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 stock_observer=1 regressions=1
```

The gate covers the P58 loss oracle, unequal-effective-row gradients, real
trainer accumulation, stock/canonical all-filter discard, journal resume,
native-dose/zero-exact classifier negatives, both renderer arms/stages,
environment resolution, the full alignment suite, and relevant P34/P44
regressions.

Host validation passes profile 2/2, renderer 15/15, alignment policy 9/9,
environment 5/5, P34 static 10 suites, and current P57 adjacency
105/105. In the pinned image, classifier 5/5 and the shared alignment
regression 42/42 pass; the targeted trajectory batch passes 6/6, including
reset-timeout fallback and missing-input fail-closed controls. Python compilation,
the 320/320 flag-registry audit, and `git diff --check` pass. The complete
pinned-image gate emits the terminal marker above.

P58f10 is the latest target execution. It proves that the p58f09 placement and
original-input fixes reach real Step-0 rollout, but fails at the two-wave batch
deadline before a durable journal or trainer call; p58f07 remains the latest
attempt to enter real value-and-grad/backward, also without an optimizer
receipt/checkpoint. The training venv loads JAX/libtpu,
but this container exposes no `/dev/vfio` and reports zero chips; the bounded
runner emitted `P58_ONEHOST_ALIGNMENT_BLOCKED reason=device_inventory_timeout`
instead of PASS. The repair claim is implementation plus CPU/exact-image
validation; target execution remains limited to the pre-forward boundaries
named above.

## Next executor sequence — native only

1. Read `state.md`, `plan.md`, this handoff, the superseded P58.4N phase file,
   the active `phases/p58-5-native-full.md`, and
   `cluster/P58_DEEPSWE_TIM_RUNBOOK.md` completely.
2. Fetch `yuxzhang/canon-zero-tim`, detach at its exact remote-tracking SHA,
   prove a second remote readback matches, and require a clean tree. Never use
   `main`.
3. Rerun syntax, `git diff --check`, the P58 renderer/profile/environment
   tests, and the pinned exact-image gate. On a real direct-attached four-chip
   v5p host, also run
   `tests/p58_deepswe_native_zero/run_onehost_alignment_v5p.sh`; require its
   renderer-profile-policy PASS marker without treating it as a Qwen/R2E or
   DP8 x TP8 training result.
4. Publish or select a client image by immutable registry digest and verify the
   mounted Qwen3-4B-Instruct-2507 weights and frozen clean-list digest without
   printing credentials.
5. Render only `arm=native, stage=full` with fresh run-id `p58f11` and worker
   sentinel `tpu-v5p-slice`. Require exact `4x4x8` topology and no literal
   `cloud.google.com/gke-nodepool: tpu-v5p-slice`; require B8 x G16 =
   concurrency 128 = rollout DP8 x max-seqs16; require head pool `cpu-np`,
   head and worker host networking, exact required hostname anti-affinity over
   the JobSet `pathways-head` label, both JobSet DNS-publication settings, and
   the exact generated head DNS in both worker RM fields. Preserve the
   YAML/digest and run server-side dry-run before the separately approved apply.
6. Require stock preflight, one P58 stock-observer processed-B marker, exact
   live weights, shape-valid finite Native boundaries/ratios, finite
   forward/backward, and the first optimizer commit.
   Then monitor commits 1–3 without stopping a healthy job. Continue through
   checkpoint 8, updates 32 and 100, then every 100 updates.
7. Require the full native classifier JSON to say `PASS`, including a finite
   nonzero serving-path dose on A-B or B-C, finite trainer-program observation,
   exactly 1,000 commits, device optimizer, complete journal, cleanup,
   evaluation, checkpoint, and transaction receipts.
8. Do not render or apply zero.

Do not reuse any failed `p58c01` through `p58c05` or `p58f01` through `p58f10`
YAML/run root. P58f03 through p58f07 have diagnostic trajectory/alignment
evidence but no durable trainer update or optimizer checkpoint, so none is
resumable training state. The attempts remain immutable failure evidence.
P58f08 has no trajectory at all; p58f09 completed rollout processing but
crashed before the durable journal; p58f10 timed out at the batch orchestrator
before the journal. None has resumable state.
if a CL mismatch recurs, collect all three head-container logs plus one worker
log and verify its resolved RM address before deleting the failed JobSet.
Earlier evidence remains under `evidence/p58c01/`, `evidence/p58c02/`,
`evidence/p58c03/`, and
`evidence/p58c04/`. The
p58c03 hashes are `15aa9968200c55a02ef47c72c5e209277397835e1752a4dbd9699fce3b2c42b4`
for `run.log` and
`d5e8b5b1941aa5632fa6267cfdac445727c175bf8d2bbcc79c1ece7cf7aba1e2`
for `head_container.log`.
P58c04 hashes are
`f5caf2efb70bfec083a4454e441ce7f4b5b0632abbd206439ba9497bca5a6a40`
for `run.log` and
`a311eb64ee30b1fa0a168b68d9f17661756ed9cb3b272dd19d9bdddbc7f34666`
for `env.sh`.
P58c05 admission hashes are
`d0845e3da4fc106afa3e0f8aa4af387cf44335f21ba696713fd382bbc32b4cf5`
for `workload.yaml` and
`cbcf60c467c758601f42221ce050f5dac329ab1f696ba735c60ac809b33fec05`
for `workload_describe.txt`.

## Important operational semantics

- `use_rollout_logps=true` is shared. Sampler-IS, TIS correction, group
  clip/filter, degenerate-group masking, and flat-group resampling are off.
- All-zero/all-one reward groups remain. They naturally produce zero RLOO
  advantage and are logged.
- Compact-filter statuses are not malformed trajectories. They remain in the
  full journal but have zero policy mask. Structural missing/duplicate/parser
  failures remain fatal.
- A Kubernetes sandbox start exception must propagate after deletion is
  confirmed. `ENV_TIMEOUT` is an admitted compact-filter status; a
  half-created RepoEnv with `container=None` is forbidden. If an entire
  p58f11 batch has zero confirmed Running pods, classify infrastructure
  capacity/scheduling before another launch instead of patching websocket
  decode or inventing a successful trajectory.
- Read `deepswe/all_sandbox_start_timeout_batch` first. Value `1` means the
  effective R2E environment throughput was zero and the model was not the
  first bottleneck. A zero sandbox-start ratio plus a nonzero
  `deepswe/status/model_timeout_ratio` instead points to model-serving
  throughput. W&B dimensions are fixed and low-cardinality; detailed
  scheduler text is available only in the bounded raw marker.
- If an entire batch is compact-filtered, `batch_index` advances but
  `optimizer_step` and commit count do not. Relaunching into a complete
  journal continues at the next batch index; a partial/digest-mismatched
  journal stops fail-closed.
- The native arm is stock numerical training plus observation. It must not
  inherit `CANON_FIXED_AR`, `CANON_LOGPROB_M`, the canonical module, VJP2, or
  the excess-precision pin. The zero arm retains the complete bundle.
- Native processed B must come only from the P58 stock observer while
  `CANON_PROMPT_PROCESSED_LOGPROBS=0`; require exactly one
  `[P58.STOCK_OBSERVER] PROCESSED_PROMPT_LOGPROBS_PASS` marker. Zero must keep
  the stock observer off and use its canonical processed engine. Never enable
  both to make a run pass.
- Exact live-weight attestation is shared evidence, not a native numerical
  treatment. Native may use only the observer interface and must keep the
  canonical adapter absent; zero uses the registered adapter. Require exact
  leaf equality and public DP8 x TP8 mesh provenance before A/B/C.
- `env.sh` is an authoritative managed-environment snapshot, not a layered
  override. If the parent retains a renderer variable that the profile made
  absent, the p58c03 regression must fail before publication.
- `CANON_P34_TRAJECTORY_CAPTURE=0` is intentional. P58 uses its own full
  trajectory journal and does not enable the older large P34 alignment-tensor
  capture mode.
- Optimizer state is TPU device-resident in both arms. Host offload is a hard
  configuration error.

## Claim ceiling

A native 128-chip PASS proves only that the untreated Qwen3-4B clean-data
training path completed the signed 1,000-update full campaign. It does not
estimate a native-versus-zero effect, prove zero-TIM, isolate one kernel,
reproduce DeepSWE-32B, prove packing, or establish 256-chip production
behavior. No finite Native serving-path mismatch on either A-B or B-C is
`NO_TREATMENT`; missing evidence or interrupted execution is inconclusive.
