# Phase E0t — Attempt-19 carrier repair

## Question

Can the existing three-round E0 KV discriminator preserve and classify the
real Attempt-19 first action, then repeat the same D3e-bound prompt inventory
for rounds 1 and 2 without changing any numerical path?

This is a carrier repair. It is not a RoPE, RPA, attention, page-table, KV,
LM-head, loss, backward, optimizer, or production-APC repair.

## Attempt-19 evidence boundary

The incident bundle is
`evidence/m15_e0_kv3_attempt19_incident/`. Its `SHA256SUMS` file has SHA256
`bc824561d39ed4e0bb5df65f56baff68e86ac64b8694a073f13a40bf31ba1636`.

- APC-on round 0 reached the real numerical gate: A-B was 366 bytes / 160
  elements, B-C was zero, the first mismatch was row 131 / completion position
  0 / logical KV prefix 1226, and the alignment array geometry was
  `[256,8192]`. Prefix-cache hit rate was 92.8%.
- APC-off round 0 completed and sealed exact evidence. APC-off round 1 was
  numerically exact but emitted zero targeted KV records. Round 2 is absent.
- APC-on preserved the round-0 classifier-input archive, then classification
  failed before `ROUND_COMPLETE`. Rounds 1 and 2 are absent.
- Therefore Attempt 19 is `INCONCLUSIVE_CARRIER_FAILURE`; it is neither a
  three-round mechanism verdict nor evidence that the APC red disappeared.

The incident report's initial suggestion that the sampled aliases did not
coincide with the red row is superseded by source-and-log review. The selected
snapshot has prefix length 1226 and the first red action also has logical KV
prefix 1226. The old classifier admitted only
`prompt_length + completion_position < snapshot_length`, so it excluded the
causally valid next-token equality boundary.

## Repair contract

### Next-token boundary

A KV snapshot containing tokens `[0,L)` is the state used to score the next
token at logical KV prefix `L`. The source-request binding and red-candidate
join therefore admit both:

```text
red logical prefix <  snapshot length  # red already inside the snapshot
red logical prefix == snapshot length  # next token scored from that snapshot
```

They still reject any red logical prefix greater than the snapshot length.
The report keeps the two position classes separate as
`snapshot_mismatch_positions` and
`next_token_boundary_mismatch_positions`.

### Frozen prompt inventory

The D3e target selector names one immutable prompt prefix. Advancing the
dataset between diagnostic rounds makes rounds 1 and 2 ineligible by
construction. Only the exact signed profile
`CANON_APC_M15_TARGET_DEBUG=off|on`,
`CANON_P38_PRECHECK_ONLY=1`,
`CANON_P38_DURABILITY_PROFILE=m15-e0-kv-v1`, and
`CANON_P38_DIAGNOSTIC_ROUNDS=3` freezes the round-0 batch and requeues deep
copies after rounds 0 and 1.

The learner hashes all 32 prompt identities using `p57_index`, `seed`, and
`map_sha256`, requires unique indices and valid 64-character lowercase hashes,
and emits one frozen marker plus exactly two requeue markers with one common
SHA. `90_run.sh` fails closed on any count or SHA drift. Rollout requests,
request IDs, calls, cache chronology, sampling, A, B, and C are rerun in every
round; only dataset advancement is disabled. Neighboring profiles keep the
old dataset-advance behavior.

## Immutable numerical contract

```text
A = rollout decode (APC off control / APC on treatment)
B = serving prefill rescore of A action IDs, reset_prefix_cache=True
C = trainer old-policy forward

every round: B-C = 0
control:     A-B = 0
```

B must continue to prove all cached-token counts are zero. The three-round
transaction remains `16 KV records -> classifier-input checkpoint/readback ->
classifier -> final upload/readback -> ROUND_COMPLETE -> learner ACK`.

## Gates

1. Host focused and aggregate validation.
2. Clean exact-SHA prepare with a fresh label using
   `prepare_m15_attempt20_e0_kv3_pair.sh`.
3. Separately approved official pinned exact-image aggregate.
4. Separately approved fresh matched DP8xTP8 APC-off/on launch.
5. Separately approved read-only GCS recovery using
   `run_m15_attempt20_e0_kv3_return_recovery.sh`.

No gate inherits approval from an earlier gate. Attempt-19 YAML, labels, and
outputs are never reused.

## Current status

Implementation and aggregate host validation pass. The terminal marker is
`M15_E0_KV3R_HOST_PASS task_discovery=193 return=1 v1_cpu=91
p3_prefix_cache=31 persistence=1 flags=408 manifest=dae6dfa8 syntax=1
diff_check=1 exact_image=0 target=0 gcs=0 kubernetes=0 tpu=0`.

Raw host log:
`/tmp/m15-e0-kv3r-host-gate-postrebase-20260830.log`, SHA256
`ef6992bc55079965759b12395f15378c0ca1d693628ac05e5d60742f4712e811`.
Post-repair pinned exact-image and target are NOT RUN. Phase E remains closed.
