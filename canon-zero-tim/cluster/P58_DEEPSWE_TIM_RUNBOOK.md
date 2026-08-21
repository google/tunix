# P58 Qwen3-4B DeepSWE native-first runbook

P58 retains a paired 128-chip study design. Each arm uses one `4x4x8` v5p
slice, synchronously split into a 64-device rollout role and a 64-device
trainer role. Both roles are DP8 x TP8. The two arms share data, seeds,
sampling, loss, optimizer, deadlines, artifacts, and update horizon.

- `native` preserves the stock serving/trainer numerical programs from the
  pinned DeepSWE quality-fix lineage. Finite A-B mismatch is the measured
  treatment dose. B-C, nonfinite values, invalid shapes, replica/transaction
  failures, and corrupt evidence remain fatal.
- `zero` enables the complete canonical numerical bundle. A, B, and C must be
  exact at every admitted boundary.

P58 does not modify `main`. Rendering and local validation do not authorize a
Kubernetes apply. An operator must separately approve image publication and
each launch.

Current execution decision (2026-08-21): run only the native full 1,000-update
campaign. The optional one-host phase was explicitly waived, not passed, and
the user superseded the separate three-update stop. Updates 1–3 are monitored
inside the same full job and do not terminate a healthy run. Zero is deferred
while its optimization work continues and must not be rendered or applied. A
native result cannot be reported as a paired comparison.

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
| Sandbox concurrency | 64; two waves per unchanged 128-trajectory batch |
| Prompt / response / turns | 4,096 / 16,384 / 50 |
| Sampling | temperature 1.0, top-p 1.0, top-k 0 |
| Roles | rollout DP8 x TP8 + trainer DP8 x TP8 |
| Objective | RLOO; `sequence-mean-token-scale`; fixed norm 16,384 |
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
P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1
```

That marker proves CPU/image wiring, not TPU execution, HBM, real R2E rollout,
native mismatch dose, or zero exactness.

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
RUN_STEM='p58f03'
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
```

Kueue's selected ResourceFlavor supplies the concrete pool affinity. If the
literal sentinel appears as node-pool affinity, stop before apply; that is the
p58c05 admission bug.

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
| commits 1–3 | finite forward/backward; finite nonzero A-B; exact B-C; TPU optimizer; monotonic transaction/journal state |
| commit 8 | first expected checkpoint artifact and digest |
| commits 32, 100, then each 100 | continued finite training, checkpoint/evaluation cadence, no journal or cleanup drift |

Crossing commit 3 is not a stop condition. The classifier cannot say full
`PASS` until commit 1,000 and complete postflight evidence exist.

The native classifier also requires at least one finite, nonzero
`S_decode_vs_S_prefill` mismatch. Exact native A-B is `NO_TREATMENT`, not a
successful comparison. Native `S_prefill_vs_T_old` and
`T_old_vs_T_current` remain exact. The zero classifier requires all boundaries
exact. Both require device-resident optimizer evidence and no blocking reds.

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
- native has no observed mismatch dose;
- native B-C or any zero boundary differs;
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
