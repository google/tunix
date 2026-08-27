# P58.16 — DeepSWE NNX loader-metadata state contract

Status: LOCAL IMPLEMENTATION + DEPENDENCY-IMAGE CPU GATES PASS; UNCOMMITTED,
UNPUBLISHED, AND 128-TPU RETRY NOT RUN.

## Trigger

Immutable attempt `p58z06` reached the exact Qwen3-4B-Instruct 128-device
DP8xTP8 rollout plus DP8xTP8 trainer geometry, loaded the 1,012-task clean
list, initialized the Pathways dummy model, and completed vLLM warmup. It then
failed during canonical adapter construction, before any rollout:

```text
[CANON_ADAPTER] live engine contract ... state_leaves=398 ...
[CANON_ADAPTER.PLACEMENT] PASS relation=disjoint rollout_devices=64 trainer_devices=64 execution_role=trainer
FunctionalMappingError: canonical trainer execution trainer-mesh reconstruction changed the NNX state tree
```

No rollout, trajectory journal, trainer logprob, alignment, backward, AdamW,
optimizer commit, or checkpoint completed. The later vLLM finalizer
`AttributeError` is shutdown noise, not the first failure. The raw log is
7958 lines / 829400 bytes with SHA-256
`4f271091120a98d11721b8a18422f8aa07bb2be2d33ff842d06bfcf156daf1ee`.
It does not embed an exact source SHA, so source identity is not claimed.

## Root cause

The pinned `PathwaysDummyModelLoader` populates each NNX parameter through
`assign_and_shard_param()`, which adds `_is_loaded=True` as loader provenance.
Flax includes variable metadata in the State PyTree auxiliary data. The live
398-leaf runner State therefore has a different raw treedef from a
weight-free `nnx.eval_shape` reconstruction even when logical paths, variable
types, all other metadata, shapes, and dtypes are identical.

P58.15 compared those raw treedefs. It therefore rejected a loader-only
provenance difference before it could build the trainer-bound nested JITs.
The segmented trainer path contained a second instance of the same overly
strict comparison and would have failed next.

## Repair

- Compare logical NNX State treedefs after removing only `_is_loaded=True`
  from copied Variables. Never mutate the live or reconstructed State.
- Reject `_is_loaded` when its value is anything other than the exact boolean
  `True`.
- Preserve exact comparison of paths, variable types, and every other NNX
  metadata field through the normalized treedef.
- Preserve exact leaf count, shape, and dtype checks against the live runner
  leaves.
- Apply the same contract to the segmented forward/backward reconstruction.
- Emit one signed startup receipt. The P58 full classifier requires exactly
  one fixed-Qwen3-4B receipt:

```text
[CANON_ADAPTER.PLACEMENT] trainer state contract PASS relation=disjoint leaves=398 normalized_loader_metadata=_is_loaded live_markers=398 reconstruction_markers=0
```

No model, data, seed, topology, sampling, compact-filter, loss, alignment,
backward, optimizer, checkpoint, or Native/Zero flag changes.

## Validation

- Python compilation and `git diff --check`: PASS.
- Pinned-image forced four-CPU-device tests: real nested
  `jax.jit(value_and_grad)`, segmented layer pullback, and partial-overlap
  rejection: 3/3 PASS. Both positive runners carry the real loader marker;
  an exact false-valued-marker negative fails closed.
- P58 Zero-HP full classifier: 7/7 PASS, including a missing-state-receipt
  negative.
- Complete dependency-image gate on
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  exits zero with:

```text
P58_EXACT_IMAGE_CPU_PASS ... disaggregated_trainer_mesh=4 ... regressions=1
```

The local image exposes no TPU devices. This is construction evidence, not a
Pathways or 128-chip training PASS.

## Next gate

Do not launch from this dirty worktree. Commit/push, matching-image
publication, Kubernetes apply, and target execution each require their own
user approval. After exact remote source and image readback plus sandbox
admission, use fresh Attempt-0 id `p58z07`; never resume or overwrite
`p58z01` through `p58z06`.

Require exactly one each of trainer placement, trainer state contract,
trainer model-JIT rebuild, and trainer logprob-scorer rebound receipts. Then
require completed trainer old/current logps, strict A=B=C, finite nonzero
16-group backward, and one coherent update-0 transaction before continuing
the same signed 1,000-update job.
