# P45.3a — GCS checkpoint/resume admission

- Status: local implementation complete; target fresh/resume gate pending

## Objective

Add crash-tolerant committed-step continuation to the isolated P45 DP8xTP8
FrozenLake carrier without changing its model, loss, optimizer, topology,
batch, evaluation cadence, or warning-only alignment policy.

## Frozen contract

- root:
  `gs://yuxzhang-tunix-models/canon-zero-tim/checkpoints/frozenlake`;
- one stable lowercase Kubernetes-safe campaign tag supplied at render time;
- mode is exactly `new` or `resume`;
- save interval is exactly 10 committed optimizer updates;
- preserve only the newest complete checkpoint (`LatestN(1)`);
- disable the trainer's forced final save so an off-interval shutdown cannot
  replace the newest 10-step boundary;
- Pathways persistence is required;
- the RL cluster appends `actor/` beneath the campaign root;
- no automatic JobSet restart is admitted in this phase. A failed launch is
  resumed explicitly with a new run ID, the same campaign tag, and mode
  `resume`.

`new` must reject a prefix containing a complete checkpoint. `resume` must
reject an empty prefix, a missing optimizer state, a mismatched checkpoint
contract, or a restored actor/global-step disagreement.

The checkpoint contract includes source commit, profile/workload/model
identity, DP/TP, batch/generation/microbatch geometry, optimizer
hyperparameters and placement, sequence limits, loss settings, dataset seed,
evaluation cadence, maximum steps, checkpoint tag/interval/retention, and a
schema version. It is stored as checkpoint custom metadata and compared
exactly on restore.

## Resume ordering gate

The vLLM sampler is constructed before the actor trainer restores its latest
checkpoint. Therefore a successful restore must run one explicit
`rl_cluster.sync_weights()` after learner construction and before
`GRPOLearner.train()`. The canonical engine weight attestation must then be
bitwise exact. Missing sync or a failed attestation is a hard error.

Required markers:

```text
[P45.CHECKPOINT] PREFLIGHT mode=new|resume ... interval=10 max_to_keep=1
[P45.CHECKPOINT] NEW_PASS latest=none
```

or:

```text
[P45.CHECKPOINT] RESTORE_PASS step=<10*N> optimizer_state=1 contract_match=1
[P45.CHECKPOINT] ROLLOUT_SYNC_PASS step=<10*N> weights_equal=1
```

## Local exit gate

1. Pure contract tests reject partial env, unsafe tags/roots, wrong interval,
   retention other than one, `new` over an existing checkpoint, `resume`
   without a checkpoint, missing optimizer state, and metadata drift.
2. Renderer tests prove full/eval manifests carry the same GCS root/tag,
   explicit mode, interval 10, retention one, and still retain
   `maxRestarts: 0`.
3. A toy local checkpoint roundtrip restores model, optimizer, global step and
   exact custom contract metadata.
4. Source/AST tests prove resume weight sync occurs before `train()`.
5. Existing P45 exact-image and adjacent P33 workload/alignment gates remain
   green.

## Target gate

Run one fresh P45 evaluation carrier through committed step 10. Require one
complete `actor/10` GCS checkpoint, no HBM/OOM regression, and continued step
11 execution. Then terminate only under operator control and render a new
JobSet with a new run ID, the identical source and campaign tag, and mode
`resume`. Resume must restore step 10, sync vLLM, skip the first 10 data
batches, and commit step 11 without resetting optimizer state.

## Claim ceiling

This resumes at a completed optimizer boundary. Up to nine updates after the
latest retained checkpoint may be replayed after failure. In-flight rollout,
environment state, vLLM sampling RNG, and W&B run identity are not restored;
the resumed trajectory stream is not claimed bitwise-identical to an
uninterrupted run.
