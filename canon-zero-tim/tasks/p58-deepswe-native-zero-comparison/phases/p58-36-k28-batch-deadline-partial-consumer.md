# P58.36 — K28 shared batch deadline and partial-consumer repair

Status: `LOCAL CONSTRUCTION PASS / K29 TARGET NOT RUN`

## Incident boundary

K28 completed Step 0 end to end: 128 trajectories, six solved trajectories,
strict A=B=C over 374,516 action tokens, all sixteen backward groups, one
TPU-resident update, outer weight synchronization, and entry into Step 1.
Step 1 later exposed a partial producer result to the consumer. Only two
complete prompt groups (32 trajectories) reached the consumer before the
rollout path became terminal, and the downstream 128-row artifact assertion
masked the producer-side timeout with a shape error.

The committed incident package is immutable and contains an earlier
microbatch interpretation. Do not rewrite it. P58.36 records the corrected
operational contract: a full P58 update is B8 x G16 = 128 trajectories; a
32-row tail is never a trainable or persistable batch.

## Deadline contract

All collectors belonging to one update now share one absolute monotonic batch
start. The three bounds have different ownership:

| Bound | Value | Owner and meaning |
|---|---:|---|
| trajectory | 3,000 s | Complete one trajectory, including late sandbox admission; a late collector receives only the remaining batch budget |
| sandbox Pod active deadline | 3,300 s | Kubernetes hard stop: trajectory budget plus 300 s bounded cleanup margin |
| rollout batch watchdog | 3,600 s | Producer hard stop: Pod/collector deadline plus 300 s result-drain margin |

Normal trajectory/model/environment/reward deadline expiry returns one compact
filtered row with zero policy mask. Cleanup failures remain fatal because a
leaked sandbox is not a valid compact result. Under ordinary bounded cleanup,
the producer must therefore return all eight groups and all 128 rows before
the 3,600-second watchdog.

## Consumer and artifact contract

- The exact P58 full identity requires consumer-side processing, eight prompt
  groups, sixteen generations per group, pair indices 0 through 15 exactly
  once, and 128 total rows.
- A partial queue tail waits for the producer Future and propagates its
  original exception. It cannot enter reward computation, Rescore B, artifact
  persistence, forward, backward, or optimizer update.
- If the producer terminated cleanly but coverage is partial, the consumer
  raises an explicit P58 full-coverage error at the same boundary.
- `persist_batch` remains strict at 128 rows. No padding, resampling, or
  partial-batch relaxation was added.

## Timing evidence

Every P58 trajectory carries finite nonnegative lifecycle fields for sandbox
acquire, sandbox start, environment reset, model generation, environment
steps, final reward, cleanup, collector start skew, trajectory elapsed, and
batch completion. The durable batch artifact records p50/p90/p99/max per
stage and all eight group completion times. The same bounded set is exported
to W&B under `deepswe/timing/*`.

Required runtime markers include:

```text
[P58.36.BATCH] FULL_CONSUMER_PASS prompt_groups=8 generations=16 trajectories=128 partial=reject-before-processing
[P58.36.BATCH] DEADLINE_START ... trajectory_deadline_secs=3000.0 batch_deadline_secs=3600.0
[P58.36.GROUP] COMPLETE ordinal=8/8 trajectories=16 ...
```

## K29 gate

K28 has no resumable checkpoint because P58 checkpointing is intentionally
disabled. K29 must be a fresh full Zero-HP render from the final clean remote
readback SHA. It must retain B8 x G16, max concurrency 128, DP8xTP8 rollout,
DP8xTP8 trainer, TPU-resident optimizer, 1,000 updates, TiTO, warning-only's
narrow finite A-B scope, and all existing hard gates.

K29 validates this repair only after Step 1 produces a complete 128-row
artifact or propagates the true producer error before any downstream work.
Step 0 alone is not sufficient because K28 already passed it.

## Construction evidence

- Python and Bash syntax plus `git diff --check`: PASS;
- focused 128-row artifact, shared-deadline, late-collector, partial-consumer,
  and classifier regressions: PASS;
- exact renderer timeout ladder and 128-concurrency geometry: PASS;
- P34 static: `P34_STATIC_PASS suites=10`;
- registry: `FLAG_AUDIT_PASS`, 409 declared/actual/unique and no new flag;
- complete pinned dependency image: `P58_EXACT_IMAGE_CPU_PASS`.

These are construction results. No K29 target result or Step-1 repair claim is
made by them.
