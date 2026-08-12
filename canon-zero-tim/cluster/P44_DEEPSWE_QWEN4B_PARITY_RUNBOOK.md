# P44 Qwen3-4B DeepSWE 64/256 parity-debug runbook

P44 is a fast DeepSWE systems-debug lane for either one 64-device `4x4x4`
slice or one 256-device `4x8x8` slice. Both variants use Qwen3-4B, TP8, four
prompts, four generations per prompt, the same rollout bounds, GRPO logic,
resident optimizer policy, durable trajectory schema, solve metrics, and
`rollout-only` -> `one-update` -> `three-update` stage ladder.

The only admitted differences are physical topology, worker count, DP size,
per-rank trajectory partitioning, and the DP-derived global carrier geometry:

| Allocation | Role split | Mesh per role | Local trajectories | Global M | Workers |
|---|---|---|---:|---:|---:|
| 64 devices | 32 rollout + 32 trainer | DP4 x TP8 | 4 | 1024 | 16 |
| 256 devices | 128 rollout + 128 trainer | DP16 x TP8 | 1 | 4096 | 64 |

This is functional systems parity, not bitwise equivalence across topologies,
performance parity, a model-quality comparison, zero-TIM proof, or admission
of the Qwen3-32B production recipe. A nominal 257-device allocation still has
an exact 256-device P44 target: the extra device is not part of the `4x8x8`
Pathways mesh and must not change the rendered topology.

The implementation agent does not apply these JobSets. Rendering, server-side
dry-run, apply, and promotion are operator-owned actions that require launch
approval.

## 1. Fetch and pin the publication

The execution source is the final exact commit published to
`origin/yuxzhang/canon-zero-tim`. Do not launch an uncommitted worktree, the
local development branch, or the moving branch name without also pinning its
read-back 40-character SHA in the JobSet.

From a clean remote checkout:

```bash
git fetch origin yuxzhang/canon-zero-tim
git switch --detach origin/yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git ls-remote origin refs/heads/yuxzhang/canon-zero-tim | awk '{print $1}')"
test "$SOURCE_SHA" = "$REMOTE_SHA"
test "$(git status --porcelain | wc -l)" -eq 0
```

Compare `SOURCE_SHA` with the publication SHA delivered in
`tasks/p44-deepswe-qwen4b-parity/HANDOFF.md`. Stop if any SHA differs.

Run the local release gates before consuming TPU time:

```bash
bash canon-zero-tim/tests/p44_deepswe_qwen4b_parity/run_cpu.sh
bash canon-zero-tim/tests/p43_deepswe_debug/run_cpu.sh
bash canon-zero-tim/tests/p39_deepswe_pilot/run_cpu.sh
bash canon-zero-tim/tests/p34_deepswe/run_static.sh
```

Required terminal markers are `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS`,
`P43_DEEPSWE_DEBUG_CPU_PASS`, `P39_DEEPSWE_PILOT_CPU_PASS`, and
`P34_STATIC_PASS suites=10`.

## 2. Verify immutable runtime inputs

Required inputs:

- a client image in registry-digest form `repository@sha256:<64 lowercase
  hex>`; a local Docker image ID is not a registry publication digest;
- the Qwen3-4B checkpoint at
  `/mnt/disks/linchai_data/models/Qwen3-4B` on the mounted model PVC;
- the real DeepSWE gold whitelist and its lowercase SHA-256, computed from
  the file visible through that same PVC;
- topology-matching CPU and TPU node pools;
- existing `HF_TOKEN` and `WANDB_API_KEY` Kubernetes secret references; and
- a `very-high` PriorityClass with value `1000` and policy
  `PreemptLowerPriority`.

Run the exact-image gate with the actual registry image selected for the
remote launch:

```bash
bash canon-zero-tim/tests/p44_deepswe_qwen4b_parity/run_exact_image.sh \
  "$CLIENT_IMAGE_DIGEST"
```

The required marker is `P44_EXACT_IMAGE_CPU_PASS overlay=qwen4b`. Verify the
checkpoint, whitelist, and whitelist digest from a read-only pod mounting the
same PVC. Also check the PriorityClass without changing cluster state:

```bash
kubectl get priorityclass very-high \
  -o jsonpath='{.metadata.name}{" value="}{.value}{" policy="}{.preemptionPolicy}{"\n"}'
```

The exact output must be
`very-high value=1000 policy=PreemptLowerPriority`. Stop before render/apply if
R2E-Gym cannot import, the gold set is empty, dataset construction still
passes the removed `trust_remote_code` argument, or any immutable input does
not match its recorded digest.

## 3. Select one topology and render one stage

Run the ladders independently. Do not use a PASS from one allocation to skip a
stage on the other. Set `TOPOLOGY=64` for one `4x4x4` slice or `TOPOLOGY=256`
for one `4x8x8` slice, then use the node pool that supplies exactly that slice.

```bash
TOPOLOGY=64
STAGE=rollout-only
RUN_ID=p44-t64-rollout-01
CLIENT_IMAGE_DIGEST=registry.example/tunix@sha256:replace-with-real-digest
CPU_NODEPOOL=replace-with-cpu-nodepool
TPU_NODEPOOL=replace-with-topology-matching-tpu-nodepool
MODEL_PVC=haoyugao-cpu-np-pvc
WHITELIST=/mnt/disks/linchai_data/deepswe/gold.jsonl
WHITELIST_SHA256=replace-with-real-lowercase-sha256
OUTPUT="/tmp/p44-${TOPOLOGY}-${STAGE}-${RUN_ID}.yaml"

python3 canon-zero-tim/cluster/render_p44_deepswe_parity.py \
  --base canon-zero-tim/cluster/jobset-64chip.yaml \
  --output "$OUTPUT" \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_ID" \
  --stage "$STAGE" \
  --topology "$TOPOLOGY" \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC" \
  --whitelist "$WHITELIST" \
  --whitelist-sha256 "$WHITELIST_SHA256"
sha256sum "$OUTPUT"
kubectl apply --server-side --dry-run=server -f "$OUTPUT"
```

The renderer must end with
`P44_PARITY_JOBSET_RENDER_PASS topology=<64|256>`. It rejects floating images,
unsupported topology/stage, Qwen3-4B or TP8 drift, mutable evidence paths,
optimizer offload, and recipe mismatch. Never hand-edit the rendered YAML.

Applying is the explicit launch boundary:

```bash
kubectl apply -f "$OUTPUT"
```

Use a fresh `RUN_ID` and output file for every attempt and stage. Expected
names and evidence counts are:

| Stage | JobSet prefix | Batches | Optimizer commits |
|---|---|---:|---:|
| `rollout-only` | `canon-p44-ds4b-t${TOPOLOGY}-rollout-` | 1 | 0 |
| `one-update` | `canon-p44-ds4b-t${TOPOLOGY}-one-` | 1 | 1 |
| `three-update` | `canon-p44-ds4b-t${TOPOLOGY}-three-` | 3 | 3 |

Promote only after the current stage's classifier says `PASS`. A failed or
inconclusive stage requires a fresh Attempt 0; do not reuse its run id or
artifacts.

## 4. Inspect trajectories, solve metrics, and classification

Every completed batch writes under
`/mnt/disks/linchai_data/deepswe_zero_tim/<jobset-name>/debug`:

- `run_manifest.json` with source, model, topology, stage, and schemas;
- `batch-<step>.trajectories.jsonl.gz` with 16 real trajectories grouped as
  four prompts x four generations; and
- `batch_metrics.jsonl` with trajectory solve ratio and all-solved,
  all-failed, mixed, incomplete, and effective-prompt group counts.

The solve definition is `r2egym_final_reward_eq_1`: a trajectory is solved
only when status is `SUCCEEDED` and its finite raw final reward is exactly
`1.0`. These metrics describe the batch; they do not establish general model
quality.

```bash
RUN_ROOT=/mnt/disks/linchai_data/deepswe_zero_tim/<jobset-name>
ls -lah "$RUN_ROOT" "$RUN_ROOT/debug"
jq . "$RUN_ROOT/debug/run_manifest.json"
jq -c '{step,trajectory_solve_ratio,all_solved_prompt_groups,all_failed_prompt_groups,mixed_prompt_groups,incomplete_prompt_groups,effective_prompt_groups,status_histogram}' \
  "$RUN_ROOT/debug/batch_metrics.jsonl"
gzip -cd "$RUN_ROOT/debug/batch-000000.trajectories.jsonl.gz" \
  | head -n 1 | jq .
jq . "$RUN_ROOT/p44_deepswe_${TOPOLOGY}_${STAGE}.classification.json"
```

Useful log scan:

```bash
grep -aE '\[P34.CLI\]|\[P34.TOPOLOGY\]|\[P44\.|CANON_ALIGN_PRE_JSON|update_step_committed|Traceback|OOM|RESOURCE_EXHAUSTED|CANCELLED|IFRT' \
  "$RUN_ROOT/run.log"
```

For update stages, finite A/B/C alignment differences are warning-only and
the claim remains systems-debug functional parity. Missing/non-finite values,
invalid B/C structure, weight or replica mismatch, non-finite gradients,
missing trajectory artifacts, R2E failure, OOM, IFRT failure, an unexpected
retry, or a classifier `FAIL` remain hard stops.

## 5. Failure return package

On failure, preserve and return:

- topology, stage, run id, JobSet name, exact source SHA and branch, image
  digest, whitelist path/digest, rendered YAML, and YAML SHA-256;
- `kubectl describe` for the JobSet and all pods, plus namespace events;
- complete logs for `jax-tpu`, `pathways-proxy`, `pathways-rm`, and failed
  `pathways-worker` containers;
- the complete persistent run directory, including classifier, debug
  artifacts, and weight/pre-alignment/alignment/update reports; and
- the first fatal traceback and the last 300 log lines before termination.

Use `grep -a` because progress-control bytes can make ordinary grep silently
treat logs as binary. Never return token values or a process-environment dump.
Do not alter TP8, precision, reward/advantage logic, gold filtering, alignment
policy, or an existing failed manifest to force a green result.

## Rollback

Do not render/apply P44, or leave `CANON_P44_DEEPSWE_PARITY=0`. P34, P39, and
P43 remain separately selected. A P44 failure does not change any production
admission flag.
