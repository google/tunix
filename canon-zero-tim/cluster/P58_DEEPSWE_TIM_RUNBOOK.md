# P58 Qwen3-4B DeepSWE native-first runbook

P58 retains a paired 128-chip study design. Each arm uses one `4x4x8` v5p
slice, synchronously split into a 64-device rollout role and a 64-device
trainer role. Both roles are DP8 x TP8. The two arms share data, seeds,
sampling, loss, optimizer, deadlines, artifacts, and update horizon.

- `native` preserves the stock serving/trainer numerical programs from the
  pinned DeepSWE quality-fix lineage. Finite A-B and B-C differences are the
  measured serving-path treatment dose; finite T_old-T_current and derived
  ratio differences are additional stock-program observations. Nonfinite
  values, invalid shapes, replica/transaction failures, and corrupt evidence
  remain fatal.
- `zero` enables the complete canonical numerical bundle. A, B, and C must be
  exact at every admitted boundary.

P58 does not modify `main`. Rendering and local validation do not authorize a
Kubernetes apply. An operator must separately approve image publication and
each launch.

Current execution decision: run only the native full 1,000-update campaign.
P58.3 was explicitly waived, not passed, and the user superseded the separate
three-update stop. The later p58f05 repair has a separate bounded one-host
admission gate; it does not retroactively promote P58.3. The first local
attempt at that gate was blocked because the container exposed no `/dev/vfio`,
so it is not a TPU PASS. Updates 1–3 are monitored inside the same full job and
do not terminate a healthy run. Zero is deferred while its optimization work
continues and must not be rendered or applied. A native result cannot be
reported as a paired comparison.

Attempt history: native `p58c01` failed in `00_env.sh` before any TPU program;
that admission fix was published as
`acd3136267214b367a6755d0ba28d80e883d6753`. Native `p58c02` initialized
Pathways but failed before model import because direct execution of the wrapper
did not make the repository root importable. Native `p58c03` passed those
boundaries, then stopped before model initialization because the parent
entrypoint retained the renderer's stale `CANON_LOGPROB_M=256` after the
native profile had unset it in child-shell `00_env.sh`. All three roots are
immutable and have no resumable trajectory state. Use a final operator-branch
readback SHA containing the authoritative environment-snapshot fix. Native
`p58c04` passed all bootstrap gates and initialized Pathways, Qwen3-4B, vLLM,
W&B, and the rollout loop, but all 128 concurrently requested RepoEnv pods
remained unconfirmed Running until their 1,200-second start deadline. Pinned
R2E swallowed those timeouts, then attempted setup against deleted pods; the
real Kubernetes 404 was obscured by the client's `None.decode` error. P58c04
is also immutable and has no resumable trajectory state. Use a final
operator-branch readback SHA containing the fail-closed start repair. Native
`p58c05` then failed even earlier: its Workload remained
`QuotaReserved=False` because the rendered worker treated the Kueue sentinel
`tpu-v5p-slice` as literal node-pool affinity, so flavor `0xv5p-8` could not
match. No workload pod or training process started. The renderer repair
delegates concrete node-pool selection to Kueue for registered sentinels while
retaining exact `4x4x8` topology. Use fresh full-stage run-id `p58f01`. Never
reuse a p58c01 through p58c05 YAML/root.

Native `p58f01` then passed JobSet admission and the complete 128-device
Pathways/Qwen3-4B/vLLM initialization chain, but every runtime-created R2E Pod
remained `SchedulingGated`. Those standalone Pods lacked the parent JobSet's
Kueue queue label, so all 128 resets timed out. The resulting all-timeout
batch exposed a second bug: `policy_version` was assigned only on the first
model call, which reset-time failures never reached, and strict processing
crashed before journaling. The repaired path derives the sandbox queue from
the parent JobSet, writes it to every Pod, seeds policy provenance before
reset, and records `scheduling_gated` separately. P58f01 is immutable,
`INCONCLUSIVE`, and has no resumable state.

Native `p58f02` passed initialization and started Step 0, but sandboxes remained
`SchedulingGated` because `multislice-queue` CPU flavor `cpu-user` requires
`nodeSelector: cpu-np`, whereas sandboxes defaulted to `deepswe-cpu-pool`.
The fix routes sandboxes and head pod to `cpu-np` (`NODE_SELECTOR_VAL=cpu-np`).
P58f02 is immutable, `INCONCLUSIVE`, and has no resumable state. Use fresh
full-stage run-id `p58f03`; never reuse its YAML/root.

Native `p58f03` proved that CPU routing repair. It completed the first real
rollout batch in 616.3 seconds and wrote 128 durable rows: 126 `SUCCEEDED`, two
`MAX_CONTEXT_LIMIT_REACHED`, three solved trajectories, two mixed/effective
prompt groups, and 32 nonzero advantages. Sandbox-start timeout count was
zero. The run then stopped before trainer forward, backward, or update because
the shared P34 weight gate called a canonical-adapter-only method even though
native correctly had `CANON_ENGINE_MODULE_C=0`. P58f03 is immutable and
`INCONCLUSIVE`; it has a useful trajectory journal but no optimizer checkpoint
and is not a resumable training root.

The repaired gate is arm-aware and remains fail-closed. Zero delegates exact
live-weight comparison to its registered canonical adapter. Signed P58 native
uses the same pure trainer-to-engine leaf mapping and bitwise comparison as a
read-only observer; it does not construct/register that adapter or replace a
serving function. Missing or unequal leaves, DP8 x TP8 mesh drift, an unsigned
route, or a leaked canonical adapter in native is fatal. The receipt exposes
contract axis names `dp/tp` even though vLLM internally names them `data/model`.

Native `p58f04` proved that weight repair: after a 557.2-second rollout it
wrote 128 rows (125 `SUCCEEDED`, three `MAX_CONTEXT_LIMIT_REACHED`, six solved,
one mixed/effective group, 16 nonzero advantages) and emitted
`[P34.WEIGHTS] EXACT` for 398 leaves and 4,022,468,096 elements. It then failed
before trainer forward/backward/update because the shared processed
`S_prefill` call accepted only `CANON_PROMPT_PROCESSED_LOGPROBS=1`. Native
correctly keeps that canonical flag at zero, so the fail-closed rejection was
correct but the arm routing was incomplete. P58f04 is immutable,
`INCONCLUSIVE`, and has no resumable optimizer state.

The repair uses two mutually exclusive processed-B implementations. Native
keeps the complete zero-TIM bundle disabled/absent and alone sets
`CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER=1`. After all six stock files are
verified, a two-file, digest-pinned observer overlay computes only the
post-rollout B values with decode-equivalent temperature/top-k/top-p transforms
and absolute request-history targets. It never participates in generation,
trainer forward, loss, backward, optimizer math, or commit accounting. Zero
sets the P58 observer to zero and retains
`CANON_PROMPT_PROCESSED_LOGPROBS=1`, `CANON_ENGINE_MODULE_C=1`, and the complete
canonical engine. Environment and runtime contracts reject any mixed tuple.
Native `p58f05` proved that observer repair. Its 486.4-second rollout wrote 128
durable rows: 126 `SUCCEEDED`, two `MAX_CONTEXT_LIMIT_REACHED`, six solved,
two mixed/effective groups, and 32 nonzero advantages. All timeout dimensions
were zero. Exact live weights passed over 398 leaves and the processed-B
observer covered all 2,048 prompt rows. The alignment sidecar then attached,
but the run stopped before trainer forward/backward/update because the warning
policy admitted P58 only in an obsolete short-update branch and rejected the
signed `stage=full, expected_updates=1000` tuple.

The p58f05 repair admits P58 Native only with `CANON_P58_TIM_ADMITTED=1`, no competing
P39/P43/P44 mode, and an exact `three-update/3` or `full/1000` pair. It does
not enable a zero-TIM numerical flag. P58f06 proved that admission and reached
alignment after 128 durable trajectories, exact weights, and a complete Native
processed-B observation. Both A-B and B-C were shape-valid and finite across
405,827 action tokens, but the P58-specific warning tuple still blocked B-C
before trainer forward. The correction makes both finite serving-path
boundaries warnings while keeping nonfinite/shape, weights, replica,
transaction, and optimizer errors hard. Zero remains strict.
P58f07 proved the warning repair: all 128 real RepoEnv trajectories completed,
pre-backward passed with finite A-B/B-C warnings, and the trainer entered real
value-and-grad/backward. It then stopped on `T_old_vs_T_current` because the
Native gate incorrectly required an observer-only stock rescore to match the
value-and-grad primal exactly. The correction preserves the stock
quality-fix 128-trajectory observer and makes every shape-valid finite Native
program boundary and derived ratio observational. With rollout logprobs on and
sampler-IS off, the loss uses rollout A rather than observer `T_old`. Zero
remains exact, and no training data, accumulation, loss, optimizer, or
numerical treatment flag changes.
P58f08 never reached rollout. Its CPU head inherited `hostNetwork:true` and
shared node port 29001 with another Pathways ResourceManager. Six concurrent
heads already occupied the six available `cpu-np` nodes, so the scheduler
packed this seventh head onto an occupied host. The P58 CL/956357083 worker
resolved that foreign CL/42 server and failed strict compatibility. Moving the
head to `deepswe-cpu-pool` was tested and rejected because the worker could not
maintain its scheduler pipe across the node-pool subnet boundary. The correct
repair keeps `cpu-np` and the proven host-network transport, but adds required
hostname anti-affinity between every JobSet `pathways-head` Pod so Kubernetes
or the autoscaler must supply an unused CPU node.

P58f09 proved correct Pathways attachment and completed all 128 Step-0 rollout
slots in 1,699.1 seconds. Some environment resets ended at the admitted
3,000-second trajectory deadline before first observation. Those rows had
`agent.trajectory.task=None`, although the original input remained in
`env.task`; learner `merge_micro_batches()` then crashed on the `None` before
the P58 journal, alignment, forward, backward, update, or checkpoint. The
collector repair preserves the agent task when present, otherwise uses the
environment task, and fails closed if neither is a dictionary. Timeout/context
rows retain their compact status and zero policy mask; they are not removed or
resampled. No training or numerical parameter changed.

Native `p58f10` ran the source containing that repair and entered Step-0
rollout, then exposed a separate scheduling-geometry error. The hard batch
timeout prevented post-rollout merge, so the original-input fallback remains
target-unproven despite its exact-image coverage. B8 x G16 produces 128 trajectories, but
`max_concurrency=64` split them into two sequential waves. The unchanged
3,600-second rollout-batch deadline expired with only 5/8 prompt groups
complete. No journal, trainer program, optimizer receipt, or checkpoint was
created. The repair sets concurrency to 128, matching both the raw batch and
the provisioned rollout capacity DP8 x max-seqs16. Episode 3,000 s, cleanup
300 s, and batch 3,600 s remain unchanged. Use fresh full-stage run-id
`p58f11` after fetching and exactly reading back the final operator tip; never reuse p58f08,
p58f09, or p58f10 YAML/root. None has optimizer state to resume.

Native `p58f11` proves that repair: all 128 trajectories and all 8 prompt
groups completed in one wave in 1,209.2 seconds. One generation terminated in
`env.reset` and exercised the compact fallback. It then exposed a task-schema
bug before journaling: `SWEEnv.entry` had the normalized prompt, but inherited
`SWEEnv.task` had only `policy_version`, so learner processing raised
`KeyError: 'prompts'`. The repaired source seeds `SWEEnv.task` with a
singleton-batched normalized prompt before reset and uses that policy-seeded
task for both normal and pre-observation termination rows. Missing `prompts`
now fails at collection. Use fresh `p58f12`; never reuse p58f11 YAML/root.

The direct-entrypoint implementation commit is
`82d82f72a7220d945737d95f6266b5b7e2cfe706`. Resolve the final runnable SHA by
fetching the operator branch after the later publication checkpoint; do not
launch from the historical p58c02 source.

The authoritative resolved-environment snapshot implementation commit is
`c0ca41805bd65a4fdede4825ed2835cdce6e13ed`. Its first post-push readback
matched exactly with ahead/behind `0/0`; still fetch the final branch tip after
the publication-evidence checkpoint rather than pinning this historical
implementation commit directly.

## 1. Frozen recipe

| Field | Value |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| Clean data | 1,012 promoted P46 tasks |
| Clean SHA-256 | `ec297c9cbc39cd67db15b0b9db6a229b15671b848df5ec3101de9ef8df7c9973` |
| Prompt batch / generations | B8 x G16 = 128 raw trajectories |
| Sandbox concurrency | 128; exactly one wave matching both the raw batch and rollout DP8 x max-seqs16 capacity |
| Prompt / response / turns | 4,096 / 16,384 / 50 |
| Sampling | temperature 1.0, top-p 1.0, top-k 0 |
| Roles | rollout DP8 x TP8 + trainer DP8 x TP8 |
| Objective | RLOO; `sequence-mean-token-scale`; fixed norm 16,384 |
| Trainer observer | Stock quality-fix `T_old`: one prompt-counted 128-trajectory rescore for B8 x G16; observer-only under `use_rollout_logps=true` |
| PPO | epsilon 0.20, epsilon-high 0.28, beta 0 |
| Optimizer | Adam 1e-6, betas 0.9/0.99, weight decay 0.01, grad clip 1.0 |
| Optimizer placement | TPU device-resident; host offload forbidden |
| Update geometry | prompt mini-batch 8; 128 trajectory mini-batch; trajectory micro-batch 16; accumulation depth 8 |
| Optional interventions | sampler-IS off; group clip/filter off; degenerate masking off; flat-group resampling off |
| Prefix cache | off |
| Active horizon | full campaign, exactly 1,000 commits; commits 1–3 are monitoring milestones |

Compact filtering is part of the shared recipe. These terminal statuses are
journaled but get an all-zero policy mask:

```text
MAX_STEPS_REACHED
MAX_CONTEXT_LIMIT_REACHED
TIMEOUT
ENV_TIMEOUT
MODEL_TIMEOUT
REWARD_TIMEOUT
```

Partial filtering uses
`sum(mask * token_loss) / (B_eff * 16384)`. If all 128 rows are filtered,
the transaction is discarded without an optimizer commit and the next data
batch is consumed. It is not resampled. `batch_index` still advances while
`optimizer_step` does not; this separation makes the journal resumable and
prevents an all-filtered batch from overwriting the preceding artifact.

Timeout nesting is fixed: turn 300 s, step/reward 600 s, trajectory 3,000 s,
sandbox 3,300 s, cleanup 300 s, and the shared rollout-batch deadline 3,600 s.
The renderer requires B x G = `max_concurrency` = rollout DP x
`rollout_vllm_max_num_seqs` = 128. A per-trajectory timeout returns through the
normal compact zero-mask path and does not abort sibling trajectories. The
rollout-batch deadline remains a hard orchestration bound: if the complete
one-wave batch cannot drain by 3,600 seconds, fail closed rather than fabricating
missing rows or extending the signed one-hour batch.

Kubernetes sandbox start is fail-closed. A pod that does not become Running
within 1,200 seconds is deleted and confirmed absent, and its original
`TimeoutError` must reach the trajectory collector as signed `ENV_TIMEOUT`.
R2E must never return a RepoEnv with `container=None` or continue with a
websocket exec into a deleted pod. If a target run again reports zero Running
pods across a batch, preserve pod scheduling/events evidence and treat it as
CPU-pool capacity/admission work; do not patch the websocket decoder.
The bounded marker `[P34.R2E] KUBERNETES_START_TIMEOUT` records only pod name,
phase, and scheduler condition/reason/message; it never serializes the pod
spec or environment.

## 2. Fetch, pin, and validate the published source

Only launch a clean, freshly read-back publication on
`yuxzhang/canon-zero-tim`. The publication SHA is deliberately resolved at
execution time so this versioned document never contains a stale or
self-referential commit ID.

```bash
git fetch origin yuxzhang/canon-zero-tim
git switch --detach origin/yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git ls-remote origin refs/heads/yuxzhang/canon-zero-tim | awk '{print $1}')"
test "$SOURCE_SHA" = "$REMOTE_SHA"
test "$(printf '%s' "$SOURCE_SHA" | wc -c)" -eq 40
test -z "$(git status --porcelain)"
```

Run the pinned-image gate with the exact launch image. A registry digest is
required by the renderer; a local Docker image ID is only suitable for the
local gate.

```bash
bash canon-zero-tim/tests/p58_deepswe_native_zero/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

Required terminal marker:

```text
P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 stock_observer=1 regressions=1
```

That marker proves CPU/image wiring, not TPU execution, HBM, real R2E rollout,
native mismatch dose, or zero exactness.

For the p58f05 admission repair, run the bounded direct-attached v5p gate when
an actual four-device host is available:

```bash
DEEPSWE_TRAIN_PYTHON=/mnt/disks/tunix-data/venvs/train/bin/python \
  bash canon-zero-tim/tests/p58_deepswe_native_zero/run_onehost_alignment_v5p.sh
```

Required terminal marker:

```text
P58_ONEHOST_ALIGNMENT_ADMISSION_PASS ... devices=4 scope=renderer-profile-policy
```

The gate requires four real v5p devices, executes a TPU matmul, then checks the
P58 alignment policy and the renderer-derived full-stage environment. It does
not run Qwen/R2E rollout, trainer forward/backward, DP8 x TP8, Pathways, or an
optimizer update. A missing TPU metadata path or device inventory is
`BLOCKED_DIRECT_TPU_METADATA`, not PASS; do not inject fake topology metadata.
This repair-only check does not retroactively promote the waived P58.3 phase.

Before rendering, verify read-only that the mounted PVC contains
`Qwen3-4B-Instruct-2507`, the R2E dependency imports, the clean JSONL has 1,012
lines, and its digest matches the frozen value. Never print secret values.

## 3N. Render and launch the native full campaign

Use the exact source SHA, image digest, CPU pool, Kueue worker sentinel, PVC,
and a unique run id. Never hand-edit rendered YAML. This phase permits only
`native`.

```bash
CLIENT_IMAGE_DIGEST='registry.example/tunix@sha256:<64-hex-digest>'
CPU_NODEPOOL='cpu-np'
TPU_NODEPOOL='tpu-v5p-slice'
MODEL_PVC='haoyugao-cpu-np-pvc'
RUN_STEM='p58f12'
STAGE='full'

ARM='native'
OUTPUT="/tmp/p58-${ARM}-${STAGE}-${RUN_STEM}.yaml"
python3 canon-zero-tim/cluster/render_p58_deepswe_tim.py \
  --base canon-zero-tim/cluster/jobset-64chip.yaml \
  --output "$OUTPUT" \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_STEM" \
  --stage "$STAGE" \
  --arm "$ARM" \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC"
sha256sum "$OUTPUT"
kubectl apply --server-side --dry-run=server -f "$OUTPUT"
```

The renderer must emit
`P58_DEEPSWE_TIM_RENDER_PASS arm=native stage=full`.

`tpu-v5p-slice` is a Kueue-managed sentinel, not a concrete node-pool name.
Before server-side dry-run, inspect the rendered worker and require all of the
following:

```text
google.com/tpu: 128
cloud.google.com/gke-tpu-accelerator: tpu-v5p-slice
cloud.google.com/gke-tpu-topology: 4x4x8
no cloud.google.com/gke-nodepool: tpu-v5p-slice
JobSet label kueue.x-k8s.io/queue-name: multislice-queue
jax-tpu env R2E_K8S_QUEUE_NAME: multislice-queue
head nodeSelector cloud.google.com/gke-nodepool: cpu-np
head hostNetwork: true
head dnsPolicy: ClusterFirstWithHostNet
head required podAntiAffinity selector: jobset.sigs.k8s.io/replicatedjob-name In [pathways-head]
head required podAntiAffinity topologyKey: kubernetes.io/hostname
JobSet spec.network.enableDNSHostnames: true
JobSet spec.network.publishNotReadyAddresses: true
worker hostNetwork: true
worker dnsPolicy: ClusterFirstWithHostNet
worker --resource_manager_address: <jobset>-pathways-head-0-0.<jobset>:29001
worker PATHWAYS_HEAD: <jobset>-pathways-head-0-0.<jobset>
```

Kueue's selected ResourceFlavor supplies the concrete pool affinity. If the
literal sentinel appears as node-pool affinity, stop before apply; that is the
p58c05 admission bug.

If the head loses host networking, uses `deepswe-cpu-pool`, or lacks the exact
required hostname anti-affinity, stop before apply. The anti-affinity is the
fixed-port isolation; changing the proven transport is not. After apply,
confirm each Pathways head is on a distinct CPU hostname. If a strict-CL
mismatch recurs despite these invariants, preserve logs from
`pathways-proxy`, `pathways-rm`, `jax-tpu`, and one `pathways-worker` plus the
worker's resolved RM address before deletion. Do not wait for rollout: this is
an immediate bootstrap failure.

After apply, inspect the first sandbox before waiting for a whole batch. Its
metadata must contain `kueue.x-k8s.io/queue-name=multislice-queue`; its Kueue
Workload must become admitted, and the `kueue.x-k8s.io/admission` scheduling
gate must disappear before the Pod can be called healthy. If it remains gated,
preserve the Pod conditions and matching Workload/LocalQueue status and stop;
do not tune model concurrency or wait another 1,200 seconds first.

Before apply, preserve the resolved-environment regression result. It must
prove that a parent process seeded with the renderer's
`CANON_LOGPROB_M=256` loses that variable after sourcing the native
`env.sh`, while the zero arm still resolves it to `256`. Do not work around a
failure by relaxing `deepswe_contract.validate_environment`; absence is part
of the native treatment definition.

The explicit launch boundary, only after operator approval, is:

```bash
kubectl apply -f /tmp/p58-native-full-${RUN_STEM}.yaml
```

Do not produce or apply a zero YAML in this phase. Preserve the exact native
YAML and digest with the returned run.

## 4. Evidence and full-campaign interpretation

Each run root is:

```text
/mnt/disks/linchai_data/deepswe_zero_tim/<jobset-name>/
```

Important artifacts:

```text
run.log
weight_attestation.jsonl
pre_alignment.jsonl
alignment.jsonl
updates.jsonl
p58_deepswe_<arm>_<stage>.classification.json
debug/run_manifest.json
debug/batch_metrics.jsonl
debug/batch-<batch_index>.trajectories.jsonl.gz
```

Before interpreting A/B/C, require one exact weight receipt for the active
arm. Zero obtains it through the registered canonical adapter. Native obtains
the same bitwise trainer-to-live-engine proof through an observer-only route;
the adapter must remain absent. The log must contain `[P34.WEIGHTS] EXACT`.
Missing/mismatched leaves, invalid mesh, or native adapter leakage is a hard
failure and must not be converted to warning-only.

For native, also require exactly one engine marker before B is accepted:

```text
[P58.STOCK_OBSERVER] PROCESSED_PROMPT_LOGPROBS_PASS ... targets=absolute-request-history treatment=observer-only
```

The preceding install log must say `canonical_bundle=off`, while the native
environment retains `CANON_PROMPT_PROCESSED_LOGPROBS=0` and
`CANON_ENGINE_MODULE_C=0`. Seeing a canonical engine marker or either flag at
one is treatment contamination and a hard stop. Conversely, the zero arm must
set `CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER=0`; it must never install or emit
the stock-observer marker.

The gzip files contain the complete redacted conversation/tool trajectory,
raw final reward, training reward, advantage, status, task identity,
`batch_index`, and `optimizer_step`. Inspect without editing:

W&B receives both counts and ratios for solved trajectories, all-solved,
all-failed, mixed, incomplete, and effective prompt groups, plus compact-filter
counts/ratios. `effective` and `nonzero_advantage` mean usable policy signal:
compact-filtered rows retain their raw advantage in the journal but do not
inflate these metrics. A separate `raw_nonzero_advantage_ratio` is retained for
audit.

Timeout telemetry is deliberately low-cardinality. W&B receives counts and
ratios for all timeout statuses, `ENV_TIMEOUT`, sandbox-start timeouts,
`scheduling_gated` and unschedulable sandboxes, and insufficient CPU/memory,
plus the batch booleans
`deepswe/all_env_timeout_batch` and
`deepswe/all_sandbox_start_timeout_batch`. Full scheduler messages remain only
in the raw `[P34.R2E] KUBERNETES_START_TIMEOUT` log marker; they are never used
as W&B keys or values. Interpret the first completed batch as follows:

| Observation | First boundary | Action |
|---|---|---|
| `all_sandbox_start_timeout_batch=1` | no R2E pod became Running; model rollout was not the bottleneck | preserve pod events and inspect CPU-nodepool scheduling/capacity |
| sandbox-start ratio is nonzero but below one | partial sandbox admission/throughput | inspect scheduler reasons before tuning model concurrency |
| sandbox-start ratio is zero and `status/model_timeout_ratio` is nonzero | sandbox ran; model generation exceeded its deadline | investigate serving throughput/model limits |
| `timeout_stage_histogram.environment_step` is nonzero | sandbox started; repository command execution timed out | inspect R2E task/runtime behavior |

The W&B batch metrics are emitted only after a 128-row trajectory batch has
been journaled. If the process dies before that boundary, use the bounded raw
timeout markers and Kubernetes events; absence of W&B metrics is not evidence
that the sandboxes ran.

```bash
RUN_ROOT='/mnt/disks/linchai_data/deepswe_zero_tim/<jobset-name>'
jq . "$RUN_ROOT/debug/run_manifest.json"
jq -c '{step,optimizer_step,trajectory_solve_ratio,all_solved_prompt_groups,all_failed_prompt_groups,mixed_prompt_groups,incomplete_prompt_groups,effective_prompt_groups,compact_filtered_trajectories,status_histogram,timeout_stage_histogram,timeout_scheduler_reason_histogram,timeout_resource_histogram,all_env_timeout_batch,all_sandbox_start_timeout_batch}' \
  "$RUN_ROOT/debug/batch_metrics.jsonl"
gzip -cd "$RUN_ROOT/debug/batch-000000.trajectories.jsonl.gz" \
  | head -n 1 | jq .
jq . "$RUN_ROOT/p58_deepswe_<arm>_<stage>.classification.json"
```

Full-stage PASS requires exactly 1,000 committed update records. There may be
more than 1,000 trajectory batches if an entire batch was compact-filtered;
every such extra batch must have a zero-commit receipt, unchanged state, and
the same optimizer step as its successor. Any partial journal, missing digest,
duplicate/missing trajectory, wrong task identity, or non-signed filtered
status is fatal.

Monitor without stopping the healthy job:

| Milestone | Required evidence |
|---|---|
| Kueue admission | `QuotaReserved=True`, selected TPU flavor, 32 four-chip worker pods, 128 Pathways devices |
| first completed batch | 128 journal rows; timeout split; cleanup; solve, all-zero/all-one/mixed/effective-group metrics |
| commits 1–3 | finite forward/backward; finite nonzero A-B or B-C dose; all Native program boundaries and ratios finite; TPU optimizer; monotonic transaction/journal state |
| commit 8 | first expected checkpoint artifact and digest |
| commits 32, 100, then each 100 | continued finite training, checkpoint/evaluation cadence, no journal or cleanup drift |

Crossing commit 3 is not a stop condition. The classifier cannot say full
`PASS` until commit 1,000 and complete postflight evidence exist.

The native classifier requires at least one finite, nonzero mismatch across
`S_decode_vs_S_prefill` or `S_prefill_vs_T_old`. Exactness on both Native
serving boundaries is `NO_TREATMENT`, not a successful comparison. Native
`T_old_vs_T_current` may differ only when the boundary and derived ratios are
shape-valid and finite. The zero classifier requires all boundaries exact.
Both require device-resident optimizer evidence and no blocking reds.

Useful scan:

```bash
grep -aE '\[P58\.|CANON_ALIGN_PRE_JSON|CANON_ALIGN\]|COMPACT_FILTER|update_step_committed|optimizer_transaction|ONLINE_RUN_PASS|Traceback|OOM|RESOURCE_EXHAUSTED|CANCELLED|IFRT' \
  "$RUN_ROOT/run.log"
```

The run may not be promoted merely because Python exits zero. Require the
classification JSON verdict `PASS` and preserve its digest with the rendered
manifest and raw log.

## 5. Completion and follow-up

At update 1,000, preserve the full native classifier, raw log, run manifest,
all journal/checkpoint/evaluation digests, rendered YAML, source SHA, and image
digest before declaring the campaign complete. A later zero canary or paired
campaign still requires a separate user decision and must restore the paired
invariants. P58 does not claim Qwen3-32B or 256-chip production readiness.

## 6. Stop and escalation rules

Stop rather than retrying the same manifest if any of these occurs:

- source/image/data digest drift;
- native/zero processed-B treatment mixing, a missing or duplicate native
  stock-observer marker, or a canonical engine marker in native;
- native has no observed finite serving-path mismatch dose;
- any Native boundary or derived ratio is nonfinite/invalid, or any Zero
  boundary differs;
- NaN/Inf, invalid shape, replica drift, optimizer/weight attestation failure;
- host optimizer offload, prefix cache, sampler-IS, group filtering, or flat
  resampling appears;
- fewer/more than 128 raw trajectory records in any batch;
- journal continuity or digest failure;
- sandbox cleanup failure, OOM, IFRT/CANCELLED, or deadline nesting drift; or
- classifier verdict is not `PASS`.

Archive the exact YAML, its SHA-256, source SHA, image digest, raw log,
artifacts, and classification. A failed prerequisite or interrupted target run
is `INCONCLUSIVE`, never PASS.
