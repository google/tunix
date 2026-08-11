# P42 FrozenLake full-training evaluation runbook

Status: implementation and local gates are available; the 64-chip target has
not run. This runbook selects the separate evaluation-enabled full-training
manifest. The existing evaluation-disabled manifest remains the rollback.

The run is Qwen3-8B DP16xTP4 with 32 prompts x 8 generations, 450 committed
updates, a 2048-token response limit, and a five-step FrozenLake environment.
Held-out evaluation uses 100 prompts x 8 generations at optimizer steps
0, 10, ..., 440. Finite numerical alignment drift is warning-only, while
NaN/Inf, topology, weight, gradient, optimizer, HBM, Pathways, and W&B failures
remain hard errors. A passing run has claim level `convergence-only`; it is not
a bitwise zero-TIM result.

## Required operator inputs

- A clean, pushed `yuxzhang/canon-zero-tim` commit, recorded as an exact
  40-character SHA.
- The reviewed 64-chip cluster with one 4x4x4 v5p slice and a valid
  `very-high` PriorityClass.
- Existing `yuxzhang-secrets` entries for `HF_TOKEN` and `WANDB_API_KEY`.
  Never put either secret value in a command, manifest, log, or handoff.
- A new lowercase run id of at most 16 characters.
- A persistent operator-side evidence directory. The JobSet state directory is
  under host `/tmp`; archive it and the pod log before deleting any Pod or
  JobSet.

## Local gates at the publication SHA

Run from the repository root:

```bash
git status --short --branch
git rev-parse HEAD
bash canon-zero-tim/tests/p33_workloads/run_cpu.sh
```

If the host lacks the pinned dependencies, run the same gate in the reviewed
image:

```bash
sudo docker run --rm \
  -v "$PWD:/workspace:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=cpu \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  bash canon-zero-tim/tests/p33_workloads/run_cpu.sh
```

The required terminal marker is
`[P33.WORKLOAD] CPU_GATE PASS workloads=2 p35_postflight=1 p35_stage_probe=1`.
Any failing negative control stops the launch.

## Render and inspect

Set the exact published SHA and a new run id. Use a persistent location for
the rendered manifests:

```bash
SOURCE_SHA="$(git rev-parse HEAD)"
RUN_ID="fl-eval-01"
OUTPUT_DIR="/mnt/disks/linchai_data/launch_manifests/${RUN_ID}"
mkdir -p "$OUTPUT_DIR"
python3 canon-zero-tim/cluster/render_p33_jobsets.py \
  --source-commit "$SOURCE_SHA" \
  --run-id "$RUN_ID" \
  --output-dir "$OUTPUT_DIR"
FL_YAML="$OUTPUT_DIR/jobset-p33-frozenlake-full-eval.yaml"
```

The selected manifest must contain exactly:

```text
CANON_P33_ENABLE_EVAL=1
CANON_P33_DISABLE_EVAL=0
CANON_P31_ENABLE_EVAL=1
CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY=1
CANON_P33_RUN_STAGE=full
CANON_P33_NO_COMMIT=0
--num_test_batches=4
--eval_every_n_steps=10
```

It must not contain a literal credential. The other five rendered manifests
are not part of this launch.

## Kubernetes admission and launch

Read-only preflight:

```bash
kubectl get priorityclass very-high \
  -o jsonpath='{.metadata.name}{" value="}{.value}{" policy="}{.preemptionPolicy}{"\n"}'
kubectl apply --server-side --dry-run=server -f "$FL_YAML"
```

The priority result must be exactly
`very-high value=1000 policy=PreemptLowerPriority`. After explicit launch
approval, apply the already reviewed file:

```bash
kubectl apply -f "$FL_YAML"
```

Do not edit the rendered YAML in place. If any input is wrong, render a new
manifest with a new run id.

## Monitoring and evidence

Check every five to ten minutes rather than continuously. Obtain the JobSet
name from the rendered manifest, then inspect the JobSet and its Pods:

```bash
JOBSET_NAME="canon-p42-fl-eval-${RUN_ID}-${SOURCE_SHA:0:8}"
kubectl get jobset "$JOBSET_NAME"
kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=$JOBSET_NAME" -o wide
```

The log must contain exactly one enablement marker:

```text
[CANON_P33_EVAL] ENABLED workload=frozenlake cadence=10 held_out_rows=100 generations=8
```

It must then contain exactly 45 passing reward inventories and 45
`[CANON_FROZENLAKE_P42_JSON]` summaries, at steps 0, 10, ..., 440. Each
inventory has 100 prompts, 8 generations, and 800 rewards. W&B must show the
monotonic `frozenlake_eval/eval/{reward,solve,n,wall_seconds,policy_step}`
curves. A warning-only alignment row is expected to remain visible; a health
error must still stop the run.

Before deleting anything, save the complete `jax-tpu` log and copy the entire
`/tmp/canon-state/$JOBSET_NAME` directory into the persistent evidence
directory. Preserve the rendered manifest and compute SHA-256 digests for all
copied files. The postflight-generated
`p33_frozenlake_full.classification.json` is authoritative only when it says
`PASS_WITH_ALIGNMENT_WARNINGS`, `evaluation_enabled=true`, and has no reasons.

## Rollback

For the next run, select
`jobset-p33-frozenlake-full.yaml` instead of the evaluation-enabled manifest.
That manifest sets the evaluation triple to `0/1/0`. Do not alter the learner
or remove the P31 evaluation path. Stopping an already running JobSet requires
separate operator approval and does not erase its evidence.
