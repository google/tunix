# P43 DeepSWE 64-chip Qwen3-8B debug runbook

Status: implementation and CPU gates are local until the publication SHA is
recorded in `tasks/p43-deepswe-64-debug/state.md`. The remote operator must use
the final exact SHA on `yuxzhang/canon-zero-tim`, never a local development
branch or an unpushed commit.

P43 is a systems bring-up ladder for one v5p `4x4x4` slice. It splits the 64
devices into host-complete 32-device rollout and trainer roles, each DP4xTP8.
It uses Qwen3-8B, four real gold-filtered prompts, four generations per prompt,
at most five turns, and a 4096-token response bound. It does not change or
promote the P34/P39 Qwen3-32B recipes.

The stages are `rollout-only`, `one-update`, and `three-update`, in that order.
Do not skip a stage after a failed or inconclusive predecessor. A P43 PASS has
claim level `systems-debug-convergence-only`; it is not a zero-TIM or model
quality claim.

## What is persisted

Each completed rollout batch writes under
`/mnt/disks/linchai_data/deepswe_zero_tim/<jobset-name>/debug`:

- `run_manifest.json`: source, model, topology, batch, stage, seed, schemas,
  and solve definition;
- `batch-<step>.trajectories.jsonl.gz`: 16 real post-environment
  conversations with group/sample identity, status, raw final reward,
  training reward, advantage, tokens, timing, original input, and policy
  version; and
- `batch_metrics.jsonl`: solve ratios; all-solved, all-failed, mixed, and
  incomplete prompt-group counts; nonzero-advantage/effective-prompt counts;
  status histograms; and raw reward histograms.

The same scalar ratios are sent through the normal metric logger under the
`deepswe/` prefix. Artifact writes finish and fsync before alignment or any
optimizer update.

R2E-Gym exposes a scalar final reward but no independent boolean verdict in
this code path. P43 therefore records the definition
`r2egym_final_reward_eq_1`: a trajectory is solved only if it completed with
status `SUCCEEDED` and its finite raw final reward is exactly `1.0`. A positive
non-binary reward is retained and counted but is not called solved.

## Required inputs and read-only preflight

- The final 40-character SHA published at
  `origin/yuxzhang/canon-zero-tim`.
- A client image pinned as `...@sha256:<64 lowercase hex>`; tags are rejected.
- A fresh lowercase run id for every stage.
- CPU and `4x4x4` TPU node-pool names.
- The output/model PVC, containing the Qwen3-8B checkpoint at
  `/mnt/disks/linchai_data/models/Qwen3-8B`.
- The real DeepSWE gold whitelist and its lowercase SHA-256.
- Existing Kubernetes `HF_TOKEN` and `WANDB_API_KEY` secret references. Never
  put secret values in commands, rendered YAML, logs, or handoff notes.
- A cluster `very-high` PriorityClass whose value is `1000` and policy is
  `PreemptLowerPriority`.

From the remote agent's clean checkout:

```bash
git fetch origin yuxzhang/canon-zero-tim
git switch --detach origin/yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git ls-remote origin refs/heads/yuxzhang/canon-zero-tim | awk '{print $1}')"
test "$SOURCE_SHA" = "$REMOTE_SHA"
test "$(git status --porcelain | wc -l)" -eq 0
bash canon-zero-tim/tests/p43_deepswe_debug/run_cpu.sh
bash canon-zero-tim/tests/p39_deepswe_pilot/run_cpu.sh
bash canon-zero-tim/tests/p34_deepswe/run_static.sh
```

Required terminal markers are `P43_DEEPSWE_DEBUG_CPU_PASS`,
`P39_DEEPSWE_PILOT_CPU_PASS`, and `P34_STATIC_PASS suites=10`. Stop if the
recorded publication SHA, remote SHA, or checkout SHA differ.

After setting the real `CLIENT_IMAGE_DIGEST`, run the qwen8b overlay and P43
gate inside that exact image:

```bash
bash canon-zero-tim/tests/p43_deepswe_debug/run_exact_image.sh \
  "$CLIENT_IMAGE_DIGEST"
```

The required marker is `P43_EXACT_IMAGE_CPU_PASS overlay=qwen8b`.

Before apply, verify without mutating cluster state:

```bash
kubectl get priorityclass very-high \
  -o jsonpath='{.metadata.name}{" value="}{.value}{" policy="}{.preemptionPolicy}{"\n"}'
```

The exact output must be
`very-high value=1000 policy=PreemptLowerPriority`. Also verify the checkpoint
directory, whitelist file, and whitelist digest from a pod that mounts the
same PVC. Do not rely on a similarly named local file.

## Render one stage

Set these operator-owned values. `CLIENT_IMAGE_DIGEST` must be a real digest,
not the placeholder below.

```bash
STAGE=rollout-only
RUN_ID=ds8b-rollout-01
CLIENT_IMAGE_DIGEST=registry.example/tunix@sha256:replace-with-real-digest
CPU_NODEPOOL=deepswe-cpu-pool
TPU_NODEPOOL=replace-with-4x4x4-nodepool
MODEL_PVC=haoyugao-cpu-np-pvc
WHITELIST=/mnt/disks/linchai_data/deepswe/gold.jsonl
WHITELIST_SHA256=replace-with-real-lowercase-sha256
OUTPUT=/tmp/p43-${STAGE}-${RUN_ID}.yaml

python3 canon-zero-tim/cluster/render_p43_deepswe_debug.py \
  --base canon-zero-tim/cluster/jobset-64chip.yaml \
  --output "$OUTPUT" \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_ID" \
  --stage "$STAGE" \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC" \
  --whitelist "$WHITELIST" \
  --whitelist-sha256 "$WHITELIST_SHA256"
sha256sum "$OUTPUT"
kubectl apply --server-side --dry-run=server -f "$OUTPUT"
```

The renderer must end with `P43_DEBUG_JOBSET_RENDER_PASS`. It rejects a
floating image, non-TP8 topology, wrong model/batch, optimizer offload,
unbounded stage, or mutable evidence path. Never hand-edit the rendered YAML.

Applying the manifest is operator-owned and requires launch approval:

```bash
kubectl apply -f "$OUTPUT"
```

Use a new `RUN_ID` and output path for every stage. The JobSet names are:

| Stage | JobSet prefix | Expected batches | Expected commits |
|---|---|---:|---:|
| `rollout-only` | `canon-p43-ds8b-rollout-` | 1 | 0 |
| `one-update` | `canon-p43-ds8b-one-` | 1 | 1 |
| `three-update` | `canon-p43-ds8b-three-` | 3 | 3 |

## Stage promotion gates

For `rollout-only`, require:

- exactly one `[P43.TRAJECTORY_BATCH]`, one
  `[P43.BATCH_METRICS_JSON]`, and one `[P43.ROLLOUT_ONLY] PASS`;
- 16 readable trajectory rows, four groups of four, complete raw reward and
  status fields, and matching artifact SHA-256;
- no backward or optimizer-commit marker; and
- `p43_deepswe_rollout-only.classification.json` with verdict `PASS`.

For `one-update`, require the same artifact checks plus one exact cross-role
weight attestation, four alignment rows, one finite DP4 transaction set, one
device-resident optimizer commit, and HBM telemetry with at least 8 GiB free
per reported device. The required classifier is
`p43_deepswe_one-update.classification.json`.

For `three-update`, require three batch artifacts and metric rows, three exact
weight attestations, twelve alignment rows, and update records whose
`train_steps_before` are `[0,1,2]` and `train_steps_after` are `[1,2,3]`.
The required classifier is
`p43_deepswe_three-update.classification.json`.

Finite alignment differences are warning-only for the two update stages;
their claim remains convergence-only. Nonfinite ratios/gradients, topology or
replica mismatch, weight mismatch, R2E failure, missing artifacts, OOM, IFRT
failure, and optimizer failure remain hard stops.

## Read the artifacts

Set `RUN_ROOT` to the persistent directory printed in the manifest:

```bash
ls -lah "$RUN_ROOT" "$RUN_ROOT/debug"
jq . "$RUN_ROOT/debug/run_manifest.json"
jq -c '{step,trajectory_solve_ratio,all_solved_prompt_groups,all_failed_prompt_groups,mixed_prompt_groups,incomplete_prompt_groups,effective_prompt_groups,status_histogram}' \
  "$RUN_ROOT/debug/batch_metrics.jsonl"
gzip -cd "$RUN_ROOT/debug/batch-000000.trajectories.jsonl.gz" | head -n 1 | jq .
jq . "$RUN_ROOT/p43_deepswe_${STAGE}.classification.json"
```

Useful log markers:

```bash
grep -aE '\[P34.CLI\]|\[P34.TOPOLOGY\]|\[P43\.|update_step_committed|Traceback|OOM|RESOURCE_EXHAUSTED|CANCELLED|IFRT' "$RUN_ROOT/run.log"
```

Do not infer health from a quiet log. The classifier is fail-closed: missing
rows or markers are `FAIL`/inconclusive, never PASS.

## Failure return package

If a stage fails, do not edit its manifest or reuse its run id. Return:

- exact source SHA/branch, image digest, rendered YAML and its SHA-256,
  whitelist path/digest, stage, run id, and JobSet name;
- `kubectl describe` for the JobSet and all its pods, plus namespace events;
- complete logs for `jax-tpu`, `pathways-proxy`, `pathways-rm`, and every
  failed `pathways-worker` container (use `grep -a` for binary-looking logs);
- the whole persistent run directory: `run.log`, classifier, debug artifacts,
  weight/pre-alignment/alignment/update reports, and a recursive file listing;
- the first fatal traceback and the last 300 log lines before termination.

Never include token values or a dump of the process environment. Preserve the
failed JobSet and artifacts until the debugging agent confirms the package is
complete.

## Rollback

Do not render/apply P43, or leave `CANON_P43_DEEPSWE_DEBUG=0`. P34 production
and the P39 Qwen3-32B pilot remain unchanged. A P43 failure does not authorize
changing TP8, precision, reward/advantage logic, gold filtering, or production
admission flags.
