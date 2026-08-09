# P35.2 three-arm producer

Status: in progress

## Implemented

- `VllmRollout.get_grouped_prefill_rescore_logps` submits complete fixed-size request groups
  through the native serving API.
- Every group resets prefix cache and preserves processed temperature/top-k/top-p semantics.
- The RL cluster exposes the diagnostic method without changing the normal rescore path.
- Exact-image controls pass for complete groups and reject a partial final group.
- `classify_envelope.py` mechanically classifies all four A/B/C outcomes and rejects incomplete
  evidence.

## Existing instrumentation that can be reused

The canonical TPU-runner patch already has default-off P18 capture for prompt-logprob requests.
It records input IDs, positions, block tables, sequence lengths, query starts, request
distribution, mesh order, cache sharding and an engine weight fingerprint. A and B can therefore
reuse this capture instead of adding another engine-side callback.

The engine weight fingerprint is a checksum, not a collision-free equality proof. It may support
provenance but cannot satisfy the trainer-anchor versus engine exact-weight gate by itself.

## Remaining producer work

1. In the learner, retain A from the normal native rescore and call grouped native rescore for B
   under a default-off P35 environment switch.
2. Select one complete 16-row DP group and compare A/B/C under the same action mask.
3. Parse and cross-reference the existing P18 A/B metadata records; fail closed unless the
   scheduler actually placed one request on each DP rank with the admitted local M256 program.
4. Add an exact device-side bitwise equality check between trainer-anchor mapped leaves and live
   engine leaves. Do not use byte sums as the release equality gate.
5. Emit one P35 schema row, inject one masked action-value drift in a classifier-only control and
   stop before backward.

## Target admission

NOT ADMITTED. There is no target command until steps 1–5 pass CPU/schema tests and the rendered
manifest has a server-side dry run. An unchanged r18 rerun cannot answer the P35 question.

## Rollback

Leave the P35 switch unset. The grouped method is then unreachable from the workload path, and
normal serving, rescore, training, W&B and optimizer behavior remain unchanged.
