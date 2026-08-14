# P38.2m — Fixed-M single-active discriminator

- Status: active

## Finding

- Confirmed: P38s15 joins all 64 A-B mismatch elements to exact serving calls.
  At least six mismatches occurred with one scheduled request, while the run-wide
  decode tail attested `decode_rows=16`, `canonical_rows=256`, and the adapter
  attested `global_M=4096`, `local_M=256`.
- Confirmed: `scheduled_request_count=1` is scheduler occupancy, not an XLA
  input shape. The existing single-host runner uses DP1, batch size 1, and
  concurrency 1; it is E0-lite and cannot prove the production executable.
- Confirmed: no same-DP simultaneous physical-page alias was found at the exact
  P38s15 mismatch calls. Sequential page reuse remains observable but does not
  prove stale KV content.
- Hypothesis: the carrier is either stale/incorrect live KV content after
  sequential page reuse or a decode-only program seam. A fixed-M natural
  single-active call removes live co-batch ambiguity without changing the
  production aval.

## Execution

1. Rename the existing DP1 mask-derived replay contract to E0-lite everywhere;
   refuse to treat it as strict E0 or production proof.
2. Extend the host-only incident ledger with the production compile geometry:
   DP size, padded rows, canonical logprob M, and shape/dtype/sharding for model
   inputs and attention metadata. Do not materialize device arrays.
3. For natural single-active incident calls only, persist the exact token IDs.
   Multi-request calls retain hashes only so the bounded ledger does not become
   a full workload dump.
4. Validate fixed-M geometry fail-closed. A one-row input aval must be rejected
   even when `scheduled_request_count=1`.
5. After a new exact-call join supplies fixed-M evidence, choose one of two
   mutually exclusive next observers:
   - compare live KV page hashes/small samples with deterministic recomputation,
     after an observer-neutrality gate; or
   - if KV content is exact, run the first-divergence seam walk
     (`q_norm -> post-RoPE -> RPA -> residual -> MLP -> logits -> normalizer`).

## Decision table

| Observation | Meaning | Next action |
|---|---|---|
| Natural single-active mismatch, fixed-M attestation PASS | Large co-batch is not necessary without introducing a new XLA shape | Use this call for the decisive observer |
| Input aval collapses to one row | Different executable; comparison is confounded | Reject as `INCONCLUSIVE_SHAPE_SUBSTITUTION` |
| Live KV differs from deterministic recomputation | Stale/write/lifecycle carrier | Repair page/KV writer lifecycle |
| Live KV exact; first red is post-RoPE | Decode RoPE/position seam | Canonicalize and re-gate RoPE |
| Live KV exact; first red is RPA or later | Decode program seam at the first red stage | Repair only that stage |

## Exit gate

- Commands:
  - `python3 tasks/p38-pathways-decode-prefill-carrier/scripts/test_classify_p38_serving_capture.py`
  - `bash tests/p33_workloads/run_exact_image.sh tunix_frozenlake_image:vllm-tpu0.25.0`
- Pass:
  - a natural single-active record contains exact tokens and fixed-M geometry;
  - input/position aval rows equal the production padded global rows;
  - `CANON_LOGPROB_M` is present;
  - a shape-one substitution fails closed;
  - the observer performs no `jax.device_get`.
- Target pass: one exact mismatch join carries the new fixed-M fields. This
  phase does not claim strict E0, stale KV, a program seam, or a repair until
  the corresponding observer returns evidence.

## Result

Implementation is locally complete. Patch 15 adds host-only compile-geometry
attestation and exact tokens for natural single-active records; the classifier
and exact-image gates reject shape-one substitution and device fetch. The old
DP1 replay is relabeled E0-lite. No cluster launch occurred; publication was
approved only after the local gates passed. No root-cause or repair claim is
made.
