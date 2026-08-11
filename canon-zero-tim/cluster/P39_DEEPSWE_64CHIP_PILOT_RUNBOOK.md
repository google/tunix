# P39 DeepSWE 64-chip DP4xTP8 resident-optimizer pilot runbook

Status: CPU contracts pass; the target has not run. This is a bounded systems
pilot for one 4x4x4 v5p slice. It splits 64 devices into a 32-device rollout
role and a 32-device trainer role. Both roles use DP4xTP8. The pilot tests
Qwen3-32B rollout, training forward/backward, deterministic DP4 reduction, and
device-resident optimizer state before any 256-chip launch.

The pilot bounds the response to 4096 tokens, the episode to five turns, and
the run to one or three committed updates. Finite alignment differences are
warning-only. All nonfinite, topology, weight, metadata, gradient, optimizer,
replica, HBM, IFRT, and W&B failures remain hard errors. A PASS has claim level
`systems-pilot-alignment-degraded`; it is not a DeepSWE quality or zero-TIM
result.

## Required operator inputs

- A clean, pushed `yuxzhang/canon-zero-tim` source commit, recorded as an
  exact 40-character SHA.
- A client image pinned by registry SHA-256 digest. A floating tag is rejected.
- CPU and 4x4x4 TPU node-pool names, plus the model/output PVC name.
- A mounted DeepSWE gold whitelist and its lowercase SHA-256 digest.
- Existing Kubernetes secret references for Hugging Face and W&B. Never pass
  secret values to the renderer.
- A new lowercase run id and an operator-side evidence directory.

## Local gates at the publication SHA

```bash
git status --short --branch
git rev-parse HEAD
bash canon-zero-tim/tests/p39_deepswe_pilot/run_cpu.sh
bash canon-zero-tim/tests/p34_deepswe/run_static.sh
```

Required terminal markers:

```text
P39_DEEPSWE_PILOT_CPU_PASS
P34_STATIC_PASS suites=10
```

The P34 gate protects the unchanged DP16xTP8 production path. A P39 pass does
not replace it.

## Render the one-update pilot

```bash
SOURCE_SHA="$(git rev-parse HEAD)"
RUN_ID="ds64-one-01"
CLIENT_IMAGE_DIGEST="registry.example/tunix@sha256:replace-with-real-digest"
WHITELIST="/mnt/disks/linchai_data/deepswe/gold.jsonl"
WHITELIST_SHA256="replace-with-real-lowercase-sha256"
OUTPUT="/mnt/disks/linchai_data/launch_manifests/${RUN_ID}.yaml"
python3 canon-zero-tim/cluster/render_p39_deepswe_pilot.py \
  --base canon-zero-tim/cluster/jobset-64chip.yaml \
  --output "$OUTPUT" \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_ID" \
  --stage one-update \
  --cpu-nodepool deepswe-cpu-pool \
  --worker-nodepool replace-with-4x4x4-nodepool \
  --model-pvc haoyugao-cpu-np-pvc \
  --whitelist "$WHITELIST" \
  --whitelist-sha256 "$WHITELIST_SHA256"
```

The renderer must finish with `P39_PILOT_JOBSET_RENDER_PASS`. It rejects full
training, retries, offload, floating images, DP16 geometry, or a missing
warning-only policy.

## Admission and launch

```bash
kubectl get priorityclass very-high \
  -o jsonpath='{.metadata.name}{" value="}{.value}{" policy="}{.preemptionPolicy}{"\n"}'
kubectl apply --server-side --dry-run=server -f "$OUTPUT"
```

The required priority result is exactly
`very-high value=1000 policy=PreemptLowerPriority`. After explicit launch
approval:

```bash
kubectl apply -f "$OUTPUT"
```

The JobSet is strict attempt zero: `maxRestarts=0`, worker backoff zero, and no
checkpoint resume. An eviction or IFRT disconnect is infrastructure
`INCONCLUSIVE`, not a numerical FAIL and not a reason to relabel a retry as
Attempt 0.

## One-update exit gate

The automatically selected P39 classifier must write
`p39_deepswe_one-update.classification.json`. PASS requires all of the
following:

- exactly 64 visible devices and two disjoint 32-device role meshes;
- DP4xTP8 on both roles and exact cross-role weights;
- one real rollout, 16 alignment groups, finite nonzero gradient, and one
  optimizer commit;
- 16 fixed-order DP4 transactions with four reduction rounds and four rank
  pullbacks per transaction;
- optimizer memory kind `device` before and after commit, with no P30 host
  roundtrip markers;
- HBM telemetry at all three boundaries with at least 8 GiB free on every
  reported device;
- scheduler geometry M1024 and 64 requests; and
- one online W&B attestation and no blocking alignment or health red.

Archive the rendered YAML, complete `jax-tpu`, `pathways-proxy`, and
`pathways-rm` logs, and the complete persistent run directory under
`/mnt/disks/linchai_data/deepswe_zero_tim/<jobset-name>`. Record SHA-256 for all
evidence before deleting resources.

## Three-update confirmation

Only after the one-update classifier passes, render a new manifest with a new
run id and `--stage three-update`. Do not reuse or edit the first manifest.
The same classifier must pass three updates, with three exact weight
attestations, 48 alignment groups, three commits, and no declining HBM margin
below 8 GiB.

## Promotion to 256 chips

A three-update P39 PASS admits only a 256-chip candidate review. It does not
launch production automatically. The 4x8x8 target returns to DP16xTP8 on each
128-device role and must independently re-prove DP16 collectives, role meshes,
replica equality, W&B, and Pathways health.

The existing 256-chip profile remains pinned-host offload by default. Promote
device-resident optimizer state only after the P39 margin is accepted in a
separate review. If the 64-chip pilot OOMs or leaves less than 8 GiB, retain
offload for the 256-chip run. Never compensate by changing precision, FSDP,
loss, sampling, or TP8.

## Rollback

Do not render or apply the P39 pilot. The existing P34 DP16xTP8/offload profile
is unchanged. A resident failure rolls back only optimizer placement; preserve
the failed manifest and all evidence.
