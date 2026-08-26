# V1.P4.13 — FrozenLake effective-learning-rate receipt recovery

Status: source published by the current CL; host gates and immutable-image
admission PASS; post-fix target not run.

## First red

P45 Wave 02 used source
`bde8f4c6e055ff077b24af716857786ce967f422`. The immutable raw log is
`../../p57-frozenlake-tim-causal-study/evidence/f45w02_head_container.log`,
SHA-256 `1f5455b707599ff7fcff6976b980a441434479c4ee27621744808faa19bdff20`.
It proves:

- Step-0 strict pre-alignment: 45,727 actions, A-B/B-C `0/0`;
- 32/32 post-backward records: A-B/B-C/C-current all `0/0`, zero FAIL;
- accumulator: denominator 32, 399/399 nonzero leaves, all finite,
  `stable_norm=0.6722502708435059`;
- AdamW: 73.546 seconds, trainer step 0 to 1, finite parameter deltas,
  6,950,316,141 changed elements;
- first failure: `effective_learning_rate=None` before outer weight sync and
  checkpoint.

Therefore this is not a Zero-TIM, backward, gradient, or AdamW failure. The
trainer-local transaction did occur, but the run is incomplete and cannot be
resumed as scientific evidence because rollout weights were not synchronized
and no checkpoint was written.

## Cause and repair

FrozenLake passes scalar `LEARNING_RATE` directly to `optax.adamw`. A scalar
Optax transform does not expose a `hyperparams["learning_rate"]` state entry,
so `PeftTrainer._try_get_learning_rate()` correctly returns `None`. GSM8K
already solves this by registering its schedule with the trainer.

The repair registers `optax.constant_schedule(LEARNING_RATE)` with
`actor_trainer.register_learning_rate_schedule()` immediately after
`RLCluster` construction. The AdamW construction remains scalar and unchanged.
The registration is observation-only: it does not wrap the transform, add
optimizer state, or alter loss, gradients, clipping, reduction, update,
checkpoint, or serving behavior.

## Gates

- focused P57 entrypoint contract: 8/8 PASS;
- P57 host: 147/147 PASS;
- Phase4 host: 89/89 PASS. The first invocation stopped only because host
  `/tmp` had 120 MB free and renderer temporaries hit ENOSPC; the same source
  passed with `TMPDIR` on the work disk;
- pinned image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`:
  the new regression and complete P45 gate PASS, terminal
  `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`.

The image transcript was not durably saved. TPU target, outer weight sync,
policy step 1, evaluation, convergence, and final checkpoint remain unverified.

## Downside and rollback

The entrypoint now creates one small constant-schedule callable solely for
receipts, and the pinned-image gate gains one AST regression invocation.
Reverting the entrypoint/test/image-gate changes restores the false red. Do not
replace this repair by accepting `None` in the first-update gate: that would
remove proof that the configured optimizer rate is coherent with the observed
parameter mutation.
