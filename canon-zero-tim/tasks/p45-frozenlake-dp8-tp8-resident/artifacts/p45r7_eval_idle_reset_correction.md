# P45r7 evaluation failure correction

This note supersedes the root-cause and storage-location claims in
`p45r7_step10_eval_deadlock_report.md` without deleting that received report.

## Verified from the archived traceback and source

- The terminal exception is a 300-second timeout from
  `submit_requests_after_idle_prefix_cache_reset`, not an indefinitely blocked
  `eval_future.result()`.
- Evaluation yields generation groups while other requests remain live.  Each
  group enters canonical prefill rescore with `reset_prefix_cache=True`, which
  requires the entire in-process driver to become idle.  These contracts are
  incompatible under the current streaming evaluation path.
- `eval_future.result()` propagates the inner timeout.  Wrapping it in another
  timeout would bound teardown but would not make evaluation correct.
- The P45 source contract writes actor checkpoints beneath
  `gs://yuxzhang-tunix-models/canon-zero-tim/checkpoints/frozenlake/<tag>/actor`.
  The committed evidence does not prove a PVC checkpoint at
  `/mnt/disks/tunix-data/frozenlake/checkpoints/`.

## Still unverified

The local v5p service account cannot list the bucket and receives HTTP 403.
Therefore the reported P45r7 Step 10 checkpoint must be confirmed by a
cluster-authorized GCS listing before any claim that the object is intact.

## Operational decision

Current full training uses the P45 FULL manifest with explicit
`--eval_every_n_steps=0`.  The EVAL manifest is not applied until its streaming
rescore/reset contract is redesigned and independently gated.
