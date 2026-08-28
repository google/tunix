# Phase D3 seal/ACK hardening

## Objective

Make the first completed M15 observer round either advance durably to the next
round or fail with a small, immutable stage receipt.  Attempt 14 proved that
all surviving shards belong to round 0, but it did not preserve the worker log
or the failing sub-stage between shard flush and acknowledgement.

This phase changes durability and diagnostics only.  It must not change APC,
RoPE, attention/RPA, KV contents, LM head, loss, backward, optimizer, the
independent full-reset B arm, or any production APC default.

## Deliverables

1. The learner watches both `round-N.ack` and `round-N.failure.json`; a valid
   failure terminates the wait immediately and fail-closed.
2. The survivor worker publishes an atomic local failure receipt if shard
   flush or round persistence fails.
3. The M15 round publisher writes small remote stage receipts around assemble,
   classify, package, local export, manifest, upload, remote verification, and
   completion publication.
4. A host test proves three sequential requests produce acknowledgements for
   rounds 0, 1, and 2, and a forced persistence failure produces no ACK plus a
   machine-readable failure receipt.
5. Attempt-14 documentation distinguishes receipt/manifest verification from
   an independent archive-payload hash.  No archive-content claim is admitted
   until a bucket-capable executor hashes the actual tar objects.

## Local gates

- shell and Python syntax;
- focused round-worker three-round positive control;
- forced round-persistence failure negative control;
- Attempt-14 flat-shard audit tests;
- P38 persistence suite;
- M15 target-debug suite;
- flag registry audit and `git diff --check`.

## Implementation

The seal protocol is now explicit and monotonic:

```text
learner writes round-N.request
  -> worker flushes bounded shards
  -> publisher emits STARTED/PASS (or FAIL) for each ordered stage
  -> WIDE_ROUND_COMPLETE is uploaded and round-tripped
  -> worker writes round-N.ack
  -> learner advances to N+1
```

If shard flush or persistence fails, the worker writes an atomic
`round-N.failure.json`.  The learner validates its full identity and exits
immediately with stage and exit code; it no longer waits for the 900-second
ACK timeout.  Stale request, ACK, or failure files fail closed.

The small GCS return now retrieves the stage JSONs without downloading token
payloads.  Its additional statuses are:

- `ROUND_STAGE_FAILURE_IDENTIFIED`: a remote `FAIL` receipt names the stage and
  positive exit code;
- `ROUND_STAGE_PROGRESS_ONLY`: ordered receipts exist, but no round sealed;
- `NO_DURABLE_ROUND`: neither a sealed round nor a stage receipt exists.

Stage receipts never carry a numerical equality claim.  Only a `SEALED` round
with the official classifier can do that.

## Local result

`LOCAL PASS / EXACT-IMAGE NOT RUN / TARGET NOT RUN`.

- M15 task discovery: 137/137 PASS;
- multiround return audit and fake-GCS wrapper: 10/10 PASS;
- wide durability learner fail-fast tests: 8/8 PASS;
- P38 persistence suite: PASS, including three ACKs and two forced-failure
  paths;
- Attempt-14 flat-shard audit: 12/12 PASS;
- flag registry: 394/394, `FLAG_AUDIT_PASS`;
- Bash syntax, Python compilation, and `git diff --check`: PASS.

No pinned image, TPU, Kubernetes, GCS mutation, commit, or push was performed.

## Target gate

A separately approved control run must return all three round completions,
official classifier output, terminal receipts, worker log, and an independently
verified manifest.  Only after that durability canary may a matched APC-on arm
be interpreted.  This phase does not authorize a TPU or Kubernetes launch.

## Claim ceiling

`DURABILITY_REPAIR_LOCAL_PASS / EXACT_IMAGE_NOT_RUN /
NUMERICAL_PATH_UNCHANGED / TARGET_NOT_RUN / FIRST_RED_NOT_LOCALIZED /
PHASE_E_CLOSED`.
