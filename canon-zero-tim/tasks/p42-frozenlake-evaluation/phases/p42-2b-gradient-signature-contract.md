# P42.2b — correct the production gradient-signature contract

- Status: local complete; target retry pending

## Finding

Target attempt `p42e2` completed the 800-trajectory step-0 evaluation and
entered the first real segmented reverse transaction. It reported the admitted
DP16 geometry (`local_M=256`, `global_M=4096`) and created the fixed reducer,
then failed before reduction or optimizer commit because the reducer required
all 16 compact rank-gradient signatures to be pairwise distinct.

That requirement is not part of fixed-order reduction correctness. Production
data may legitimately yield equal contributions. FrozenLake uses binary
rewards and RLOO; if all eight generations for a prompt have the same reward,
their advantages are exactly zero. More generally, the signature contains five
aggregate float32 statistics rather than a bytewise hash of every gradient
element, so signature equality cannot prove that rank isolation failed.

## Execution

1. Keep `FixedDPRankGradientReducer` strict by default so synthetic admission
   probes still reject duplicate signatures.
2. Make the production segmented adapter explicitly select
   `require_distinct_fingerprints=False`.
3. Preserve hard checks for monotonic rank cadence, exactly `dp_size`
   contributions, admitted mesh/sharding, the registered fixed reduction tree,
   finite gradient health, and byte-exact equality of every reduced replica.
4. Record unique and duplicate signature counts in the reducer report and
   print `unique_rank_fingerprints=K/16` for every production group.
5. Add a positive duplicate-zero-gradient test, retain a strict duplicate
   negative control, and attest that the adapter selects only the production
   policy.

## Local evidence

- Pinned image, 64 forced CPU devices: `dp_training_test.py`, 19/19 passed.
- Pinned image, 64 forced CPU devices:
  `canonical_qwen3_adapter_test.py`, 36/36 passed.
- Complete pinned P33 workload gate: passed with terminal marker
  `[P33.WORKLOAD] CPU_GATE PASS workloads=2 p35_postflight=1
  p35_stage_probe=1`.
- `git diff --check`: passed before documentation update and must be rerun
  before publication.

## Target exit gate

The retry must print 16 `reverse_group_done` records with 16 rank pullbacks,
eight reduction rounds, and `replicas_exact=1`; complete one finite nonzero
gradient transaction; commit the optimizer once; and continue to the next
training step. Duplicate signature counts may be nonzero and are diagnostic,
not failures. Missing rank contributions, cadence drift, nonfinite gradients,
unequal replicas, or optimizer transaction failures remain hard failures.

## Claim boundary

Local tests prove the corrected reducer contract, not target training success.
No 64-chip reduction or optimizer commit has yet completed with this change.
