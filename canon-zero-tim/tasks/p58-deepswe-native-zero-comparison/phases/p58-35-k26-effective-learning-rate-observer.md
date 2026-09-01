# P58.35 — K26 effective-learning-rate observer repair

Status: `LOCAL CONSTRUCTION PASS / K27 TARGET NOT RUN`

## Incident

K26 `canon-p58-ds4b-zero-hp-full-k26` ran on 128 TPU v5p devices as
rollout DP8xTP8 plus trainer DP8xTP8. The returned incident tail proves:

- 16/16 post-backward alignment records and 383,383 action tokens with all
  three boundaries exact, `w/r/wr` exact, zero clip hits, and zero TIS hits;
- a finite, nonzero precommit over 398/398 leaves with stable norm
  `0.01175018586218357`;
- a TPU-resident Adam transaction, trainer step `0 -> 1`, and
  3,655,535,873 changed parameter elements.

The first-update gate then failed before outer weight synchronization because
the commit record contained `effective_learning_rate=None`. K26 therefore
proves strict numerical identity, the complete backward, and one
trainer-local optimizer transaction. It does not prove a synchronized Step-0
cycle, checkpoint, resume, continued training, convergence, or a completed
Zero-TIM run.

The immutable analysis-grade incident is
`canon-zero-tim/evidence/p58_k26_effective_learning_rate_incident/`; both
listed checksums verify. Its report names source
`cfa6ccf7d0c8faecaeeb99f666f8e77a28e93245`, but that object is unavailable
in this local object store. Treat that source identity as unverified until a
remote-read receipt or source bundle is returned.

## Root cause

DeepSWE constructs the exact optimizer shape:

```python
optax.chain(
    sft_utils.overflow_safe_clip_by_global_norm(1.0),
    optax.schedules.inject_hyperparams(optax.adamw)(
        learning_rate=1.0e-6, ...
    ),
)
```

Its Optax state is a tuple whose first member is `EmptyState` for gradient
clipping and whose later member owns the injected AdamW hyperparameters.
`PeftTrainer._try_get_learning_rate()` stopped at the first `EmptyState`, so
it never inspected the real optimizer learning-rate state.

This is an observer defect after a real commit, not an optimizer, gradient,
rollout, reward, or alignment failure.

## Repair

The state scanner now skips state-free transforms and continues to the later
hyperparameter state. It does not read a potentially stale training config,
invent a value, change the optimizer, or relax the first-update validator.
If no optimizer state exposes a learning rate, the observer still returns
`None` and the gate remains fail closed.

The pinned P58 image runs two regressions:

1. the exact DeepSWE clip-plus-injected-AdamW chain reports configured
   `1.0e-6` within `float32` representation;
2. a chain without observable hyperparameter state still reports `None`.

No flag, model, loss, reduction, clipping, sampling, topology, checkpoint, or
warning-only policy changed.

## Next target

K26 cannot resume because P58 checkpointing is intentionally disabled and
outer synchronization did not run. After separately approved publication,
image preparation, and cluster launch, render a fresh K27 from one verified
40-character remote-read source SHA and its matching digest-pinned image.

Require the same K26 numerical/backward receipts, then require a commit record
with a finite positive learning-rate receipt matching configured `1e-6`
(`float32` may render `9.999999974752427e-07`), successful first-update gate,
outer weight synchronization, and the next rollout/update boundary. K27 is
the first target that can validate this repair.

## Construction evidence

- incident `SHA256SUMS`: PASS for both files;
- Python compilation and diff hygiene: PASS;
- exact optimizer-state shape reproduced locally as
  `[EmptyState, InjectStatefulHyperparamsState]`;
- complete digest-pinned image gate:
  `P58_EXACT_IMAGE_CPU_PASS`, including both new observer regressions;
- flag registry change: none.

No image publication, Kubernetes mutation, TPU launch, commit, or push is
part of this local repair.
