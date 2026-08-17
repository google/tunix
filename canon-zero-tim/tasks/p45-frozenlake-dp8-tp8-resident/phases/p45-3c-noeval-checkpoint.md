# P45.3c — no-eval full training and checkpoint continuation

- Status: local implementation and one-host mechanism gate complete; 64-chip
  fresh/resume target pending

## Objective

Run the P45 DP8xTP8 resident carrier as pure full training with no in-training
held-out evaluation.  Preserve checkpoint interval 10 and `LatestN(1)` so a
later infrastructure interruption can resume from the newest committed
boundary.

## Failure established by P45r7

P45r7 did not fail in Adam, checkpoint serialization, HBM, or W&B.  Evaluation
streams completed generation groups into `_batch_to_train_example()`, whose
canonical prefill rescore requests `reset_prefix_cache=True`.  The in-process
driver can perform that reset only after its Python pending queue, submission
queue, and engine request set are all empty.  Streaming evaluation kept work
alive, so the reset timed out after 300 seconds:

```text
TimeoutError: timed out waiting for the in-process driver to become idle before resetting prefix cache
```

`eval_future.result()` merely propagated this exception.  Adding another
outer timeout does not resolve the contract conflict.

## Local changes

1. `_should_run_eval()` treats every nonpositive cadence as disabled before
   evaluating the modulo expression.
2. The P45 FULL manifest explicitly carries `--eval_every_n_steps=0` in
   addition to `CANON_P33_ENABLE_EVAL=0` and
   `CANON_P33_DISABLE_EVAL=1`.
3. The P45 EVAL manifest remains cadence 10 for future isolated evaluation
   repair, but it is quarantined from the current full-training launch.
4. A pinned-image one-host v5p gate saves a sharded model and device-resident
   Adam state at step 10, corrupts both, restores them exactly with contract
   metadata, and proves interval 10 plus `LatestN(1)`.

## Local exit gate

Require:

```text
P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8
P45_ONEHOST_CHECKPOINT_PASS backend=tpu devices=4 topology=DP1xTP4 step=10 model_exact=1 optimizer_exact=1 metadata_exact=1 interval=10 latest_n=1 optimizer_restored=1 memory_kinds=['device'] scope=mechanism-only
```

The one-host gate is mechanism evidence only.  It does not exercise Pathways,
DP8xTP8, or GCS persistence.  The local VM service account currently receives
HTTP 403 when listing the production bucket, so the existence of the reported
P45r7 `actor/10` object is not locally verified.

## Target gate

Use a new campaign tag and the FULL manifest.  Require the resolved command to
contain `--eval_every_n_steps=0`; reject any held-out evaluation marker.  Run
through step 10 and step 11, return a cluster-authorized GCS listing proving
one complete `actor/10`, then interrupt under operator control and launch a
new JobSet from the identical source/tag with mode `resume`.  Before its first
rollout require:

```text
[P45.CHECKPOINT] PREFLIGHT mode=resume ... latest=10
[P45.CHECKPOINT] RESTORE_PASS step=10 optimizer_state=1 contract_match=1
[P45.CHECKPOINT] ROLLOUT_SYNC_PASS step=10 weights_equal=1
```

The next committed update must be step 11.

## P45r7 checkpoint boundary

The old `fl-prod-001` checkpoint, if present, freezes source
`a94d6c0cd0e08b9bed418331974b8694eb49507e` and evaluation cadence 10 in its
exact metadata contract.  The new no-eval source/cadence intentionally differs,
so current fail-closed validation will reject a direct restore.  Do not weaken
the contract or claim that the old checkpoint is resumable under this source.
Reusing it requires a separately reviewed, explicit two-field migration; the
default P45.3c launch starts a new tag.

## Claim ceiling

The local gate proves TPU checkpoint mechanics, not production restore.  A
fresh 64-chip step 10/11 plus a separate identical-source resume is required
before P45 checkpoint continuation is admitted.
