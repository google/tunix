# P58.24 — JobSet-level exclusive-topology repair

Status: implemented locally; host and pinned-image construction pass; target
not run.

## Incident

K03 was admitted by Kueue and started its CPU head, but the indexed TPU worker
Job could not create its follower Pods. The first failing boundary was the
`vpod.kb.io` admission webhook:

```text
follower pod node selector for topology domain not found
missing selector: cloud.google.com/gke-nodepool
```

No rollout, trajectory, trainer program, backward, optimizer commit, or
checkpoint existed. K03 is infrastructure `INCONCLUSIVE`, not a numerical
Zero-TIM result.

## Root cause

The base manifest placed
`alpha.jobset.sigs.k8s.io/exclusive-topology=cloud.google.com/gke-nodepool`
on the worker Pod template. JobSet exclusive placement is a JobSet metadata
contract: the controller needs that top-level context to map the indexed Job
to the Kueue-selected or NAP-created node pool and constrain its followers.
The misplaced Pod annotation activated a follower admission check without the
JobSet-level placement contract, producing K03's rejection.

## Repair contract

- P58 writes the exclusive-topology annotation exactly once at
  `JobSet.metadata.annotations`.
- The same annotation is forbidden on the worker Pod template.
- `auto`, `none`, `any`, and `tpu-v5p-slice` remain Kueue-managed sentinels:
  they omit a literal nodepool selector while retaining accelerator
  `tpu-v5p-slice` and exact topology `4x4x8`.
- An operator-supplied concrete node pool remains legal and is retained
  exactly. It must match the selected ResourceFlavor.
- TPU geometry, 32 four-chip workers, DP8xTP8 roles, B8xG16, model/data, and
  every numerical/system-optimization flag are unchanged.
- Server-side dry-run remains mandatory before a separately approved apply.

The dynamic `CANON_TPU_INFERENCE_PATH` discovery from the K03 evidence intake
is retained for image-layout compatibility. It is independent of the worker
admission failure and does not authorize a mutable or source-mismatched image.

## Construction evidence

- P58 renderer and annotation-scope negatives pass.
- V1 system-optimization workload contract passes.
- A direct full Zero-HP CLI render with the Kueue sentinel emits
  `P58_DEEPSWE_TIM_RENDER_PASS`, JobSet-level exclusive topology, no Pod-level
  copy, 32 workers, `4x4x8`, B8xG16, and DP8xTP8.
- The digest-pinned complete gate emits `P58_EXACT_IMAGE_CPU_PASS` with
  `system_optimization=1`, `trajectory_replay_b2g2=1`, and `regressions=1`.

No image was published and no Kubernetes, Pathways, or TPU work was launched.
