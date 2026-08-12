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

The P44 gate must report 40 cases and includes negative controls for the
Pathways `logical_task` host mapping, exact 4-device host cardinality,
single-conversation generation batching, and trajectory-counted logprob
microbatching. It also rejects missing Qwen3-4B `1216->1280` SwiGLU runtime
evidence or either required K/N matmul-padding runtime trace. An older 34-case
marker does not contain the r05 repair.

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

Require all three markers:

```text
SWIGLU_FEATURE_PADDING_INTERPRET_PASS model=qwen3-4b-tp8 shape=129x1216 padded=256x1280 forward_exact=1 vjp_exact=1 negative=1
MATMUL_DIM_PADDING_PASS mode=interpret cases=2/2 forward_exact=1 vjp_exact=1 negatives=2/2 devices=1
P44_EXACT_IMAGE_CPU_PASS overlay=qwen4b
```

The first two markers prove exact forward and custom-VJP behavior against the
canonical SwiGLU/matmul implementations in Pallas interpret mode and reject
adjacent unregistered widths. They are not TPU target evidence. Verify the checkpoint,
whitelist, and whitelist digest from a read-only pod mounting the same PVC.
Also check the PriorityClass without changing cluster state:

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
grep -aE '\[P34.CLI\]|\[P34.TOPOLOGY\]|\[P44\.|\[PATHTRACE\]|CANON_ALIGN_PRE_JSON|update_step_committed|Traceback|OOM|RESOURCE_EXHAUSTED|CANCELLED|IFRT' \
  "$RUN_ROOT/run.log"
```

Before accepting any stage, require these exact runtime lines once per run:

```text
# TOPOLOGY=64
[P34.DEVICE_INVENTORY] PASS devices=64 host_source=logical_task hosts=16 devices_per_host=4 rollout_hosts=8 trainer_hosts=8

# TOPOLOGY=256
[P34.DEVICE_INVENTORY] PASS devices=256 host_source=logical_task hosts=64 devices_per_host=4 rollout_hosts=32 trainer_hosts=32
```

Require this line exactly once per completed batch (`1` for rollout-only and
one-update, `3` for three-update):

```text
[P44.LOGPS_BATCH] configured_prompts=4 generations=4 execution_trajectories=16 observed_trajectories=16
```

Require at least one Qwen3-4B SwiGLU runtime line before accepting any stage:

```text
[PATHTRACE] CANON_PALLAS_SWIGLU_MPAD=1 M=<positive> Mp=<BM128-aligned> F=1216 Fp=1280 row_padded=<0|1> feature_padded=1
```

The classifier validates the numeric fields and fails closed if the marker is
missing. Do not accept a generic MPAD line from the older row-only wrapper.

Require both Qwen3-4B matmul directions before accepting any stage:

```text
[PATHTRACE] CANON_PALLAS_MPAD=1 M=<positive> Mp=<BM128-aligned> padded=<0|1> K=2560 Kp=2560 N=1216 Np=1280 contract_padded=0 output_padded=1
[PATHTRACE] CANON_PALLAS_MPAD=1 M=<positive> Mp=<BM128-aligned> padded=<0|1> K=1216 Kp=1280 N=2560 Np=2560 contract_padded=1 output_padded=0
```

The first proves model-pinned gate/up output padding and semantic output
slicing; the second proves down-projection contracted-K padding. A generic
BN/BK128 trace or only one direction is insufficient.

`p44r02` is a known pre-repair 256-device failure: it found all 256 Pathways
devices but grouped them under degenerate `process_index=0` and stopped before
mesh construction. The standalone CPU IFRT diagnostic showing one CPU device
is not the failure root cause. Do not rerun from `p44r02`'s source SHA, and do
not bypass the new inventory check.

`p44r03` proved the repaired host-complete 256-device split and then failed an
inherited one-host model-mesh-id assertion. `p44r04` proved the dynamic mesh,
checkpoint load, W&B session, and execution into the MLP, then failed at
TP8-local SwiGLU feature width `1216`. `p44r05` then proved the SwiGLU repair
across all 36 layers and failed Mosaic
lowering on BN64/BK64 matmul block specs. Current remote head
`d8184123448d0add72b72f09d0a6faf5d326c26e` archives r05 plus P38-specific
capture/precheck hardening but does not contain the locally validated P44.10
BN/BK128 plus K/N-padding or P44.11 one-host integration repair. Do not launch
again until the P44 handoff records a newer publication commit containing
both repairs.

## 4a. Optional repeatable one-host v5p gates

On an authorized direct-attached v5p host with the immutable image present:

```bash
bash canon-zero-tim/tests/p44_deepswe_qwen4b_parity/run_onehost_v5p.sh \
  "$PINNED_LOCAL_IMAGE_ID"
```

Require `MATMUL_DIM_PADDING_PASS mode=tpu cases=5/5 ... devices=4` and
`P44_ONEHOST_V5P_MATMUL_PASS model=qwen4b devices=4`. The probe uses the r05
target M=4096 and executes real Pallas forward plus the promoted custom VJP in
all five unique Qwen3-4B TP8-local projection shapes, including both K/N
padding directions. This is only a direct-attached matmul construction gate;
it does not prove a model load, R2E trajectory, TP8, Pathways, 64/256 topology,
backward across the model, optimizer state, or a completed P44 stage.

For the real local DeepSWE integration smoke, use a clean checkout of the
exact published operator SHA. The runner rejects tracked worktree changes by
default, pins R2E-Gym, requires exactly four direct-attached TPU devices and a
working Docker daemon, remains offline, disables W&B and checkpoints, and
writes complete artifacts under the persistent data disk by default:

```bash
git status --short --branch
git rev-parse HEAD
bash canon-zero-tim/tests/p44_deepswe_qwen4b_parity/run_onehost_deepswe_v5p.sh \
  rollout-only
```

Require:

```text
[DEEPSWE.ONEHOST.DEVICES] PASS count=4 ...
[DEEPSWE.ONEHOST.R2E] PASS docker=1 import=1
[DEEPSWE.ONEHOST.DATASET] PASS rows=1 ...
DEEPSWE_ONEHOST_ROLLOUT_PASS model=qwen3-4b-instruct-2507 devices=4 trajectories=2
```

The inventory line prints the exact source SHA/branch, tracked-dirty bit,
R2E-Gym SHA, stage, and artifact directory. Return that entire directory,
including `run_manifest.json`, `batch-000000.trajectories.jsonl.gz`, and
`batch_metrics.jsonl`, plus SHA-256 for every file. Inspect the batch metrics;
the terminal rollout marker proves artifact-complete integration, not that the
two trajectories solved or completed their episodes.

Only after rollout-only reaches its terminal marker may the operator run:

```bash
bash canon-zero-tim/tests/p44_deepswe_qwen4b_parity/run_onehost_deepswe_v5p.sh \
  backward-no-commit
```

The runner must return one of these outcomes:

- `DEEPSWE_ONEHOST_BACKWARD_NO_COMMIT_PASS` with exit 0: gradient finite and
  nonzero, zero commits, train step unchanged, and no changed
  model/reference/optimizer/accumulator paths.
- `DEEPSWE_ONEHOST_BACKWARD_INCONCLUSIVE_NO_SIGNAL` with exit 3: the real
  backward ran but the finite gradient was zero. Preserve the report and do
  not promote a one-update run.
- any other nonzero exit: FAIL or blocked; return full logs and artifacts.

Inspect `backward_no_commit.json` for gradient norms, state fingerprints,
optimizer memory kinds, and per-device HBM before/after/peak. The signed local
geometry is Qwen3-4B, DP1 x TP4 colocated, one prompt x two generations,
response 512, two turns, exact trainer sequence 4096, Docker R2E, prefix cache
off, and device-resident optimizer state. Never use this local runner as proof
of TP8, Pathways, role separation, DP4/DP16, 64/256 behavior, Qwen3-32B, or
zero-TIM. `DEEPSWE_ONEHOST_ALLOW_DIRTY=1` is for explicitly labeled local
development evidence only and is forbidden for operator acceptance evidence.

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
