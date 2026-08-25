# Phase A — M15 target evidence decoding

- Status: complete

## Finding

- Confirmed: Attempt-2 is hash-valid. `m15i` is strict A-B red and B-C exact at step 0.
- Confirmed: first red is row 192 / completion 0 / logical prefix 1226 / turn 0 / action-run start, not a 256-token boundary and below the older 1686 diagnostic depth.
- Confirmed: the full 760-record mismatch list is present and untruncated. Every red belongs to prompt group 24. Generation counts are `g0=76`, `g1=46`, `g2=79`, `g3=0`, `g4=134`, `g5=195`, `g6=80`, and `g7=150`.
- Confirmed: red logical prefixes span 1226-6610 and turns 0-14. Only 6 records are exactly on a 256-token boundary, so boundary alignment alone does not explain the incident.
- Confirmed: the amplitude is mixed, not a one-ULP-only floor: 73 records exceed 0.1, 101 are `(0.01,0.1]`, 259 are `(1e-4,0.01]`, and 327 are `(1e-6,1e-4]`.
- Confirmed: the archive has no `.npy`, `.npz`, or safetensors input artifact and no request ID/order, token history, block table, `num_computed_tokens`, per-request cached-token count, or page owner/generation/hash.
- Confirmed: raw log line 15 records runtime `HEAD=71d889a32f4668353c758d5c00df88299e6c0d35`, while `receipt.json` says `7a2a456c...`. The runtime sync gate is authoritative because it aborts on an expected/actual SHA mismatch.
- Hypothesis: the target run includes a cache/read/scheduler/topology degree of freedom absent from the Phase3 one-host carrier. No mechanism has been selected.

## Execution

1. Enumerate all durable Attempt-2 artifacts and all `m15i` paths referenced by the raw log.
2. Parse the full alignment-pre JSON marker into a compact deterministic Phase-A report.
3. Tabulate mismatches by sequence row, turn, prompt length, logical KV prefix, block index/offset, DP rank/TP shard if present, and action-run boundary.
4. Attest source/image/model/policy/tokens/action mask and identify whether complete token arrays are durable or only hashes exist.
5. Search for request IDs/order, APC hit tokens, scheduler occupancy, block tables, page ownership/generation/hash, logical positions, and `num_computed_tokens`.
6. Emit `M15_FIRST_RED_INPUT_CONTRACT` only with fields supported by immutable evidence; list every missing field required by strict replay.

## Exit gate

- Command: `python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/analyze_m15i_evidence.py --evidence-dir canon-zero-tim/tasks/v1-phase4-three-full-recipes/evidence/v1_hp_three_full_attempt2_20260824 --output <scratch-output>`
- Pass: deterministic report includes source hashes, comparator negative/strict status, full mismatch distribution, first-red context, and an exact-replay readiness verdict with no inferred scheduler/page fields.
- Fail: missing raw record or inconsistent counts -> classify evidence unusable; missing tokens/order/cache lineage -> Phase A may pass as an evidence audit but strict replay remains blocked and Phase B must first add a bounded capture carrier.

## Result

`M15_FIRST_RED_INPUT_CONTRACT manifest=PASS mismatches=760 replay=INSUFFICIENT_FOR_STRICT_REPLAY`

The deterministic report SHA in the current uncommitted tree is
`7b0467356aadb7b161dfd38da5df5d384c8e4357968485ed66cc7fb549508887`.
Rerun the documented command whenever the analyzer changes and treat the new
SHA as the only current receipt.

Phase A passes as an evidence audit, not as replay admission. The next phase
must capture a new red with raw arrays and exact request/cache chronology.
