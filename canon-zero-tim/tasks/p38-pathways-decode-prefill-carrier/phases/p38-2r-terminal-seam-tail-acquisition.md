# P38.2r — Single-run terminal seam-and-tail acquisition

Status: P38s18r target attempt is
`INCONCLUSIVE_DURABILITY_SEAL_TIMEOUT`. Its round-0 numerical record is
analysis-grade only because no immutable round bundle or complete three-round
package was returned. A strict round-scope repair passes local CPU and
fake-GCS gates but remains uncommitted/unpublished; no replacement target is
authorized yet. P38.2q remains retired as
`INCONCLUSIVE_NO_ELIGIBLE_SNAPSHOT`; it cannot supply P38.2r data.

## Entering evidence

P38s18l measured two A-B-red/B-C-exact rounds, but its seam payload was split
across non-self-contained live snapshots. Snapshot `000020` has only round 0;
snapshot `000021` reports rounds 0 and 1 but lacks the source manifest and
paired seam NPZ files. Consequently no byte-preserving reduction can join all
47 red points / 94 A/B keys. This is a durability failure, not evidence that
the hidden chain is exact or that the tail is the carrier.

## Deliverable

One production-shape stock diagnostic run must return all data needed to choose
the first divergent region offline. It keeps `DP16xTP4`, concurrency 256,
fixed canonical M, frozen weights, zero backward, and zero optimizer commits.

For each of three rounds, the surviving snapshot worker must atomically seal:

1. the immutable mismatch capsule and pre-alignment record;
2. every A/B layer-seam JSON/NPZ required by that round's red rows;
3. bounded tail checkpoints for the same rows: `final_norm`, raw target logit,
   raw log-normalizer, processed target logit, processed log-normalizer, and
   final target logprob;
4. request/call/program provenance needed for exact row joins;
5. a round-local object inventory and self-excluding SHA manifest; and
6. `ROUND_COMPLETE.json` written last, only after the uploaded round bundle
   downloads and verifies.

Round `n+1` may not start until round `n` is sealed. Final `COLLECTED.json`,
postflight `COMPLETE.json`, and controlled exit remain worker-owned. No
end-of-process shell step is the sole owner of evidence.

## Implemented contract

- `patches/tpu_inference/19-tpu-runner-p38-terminal-tail.patch` observes the
  already-produced A/B logits and production logprob after the unchanged
  sampling/scoring calls. It emits target logits, raw/processed vocabulary
  normalizers, an independent observer subtraction, and the production
  endpoint. It does not wrap `model_fn`, `sample`, or the production scorer in
  a new program.
- The layer observer still supplies every layer input/output plus the final
  norm fingerprint. Tail records share its exact round/token-prefix join key.
- `alignment.py` publishes a round-seal request after every frozen round and
  blocks before the next round. The survivor worker stages, hashes, uploads,
  downloads, and verifies that round before atomically acknowledging it.
- `ROUND_COMPLETE.json` is the final object in each immutable round prefix.
  Missing capsules, journals, incident records, required seam/tail pairs,
  manifest entries, or acknowledgements fail closed.
- The official seam classifier requires every red action to join A and B in
  both observers. It also requires the captured production endpoints to equal
  the mismatch capsule exactly, preventing a tail record from describing a
  different execution.

## Local gate

- combined observer off/on endpoints are bitwise identical for three one-host
  rounds;
- one normal-value fingerprint mutation is detected;
- one tail-value mutation is detected;
- deleting any JSON, NPZ, capsule, round marker, or SHA entry makes packaging
  fail closed;
- duplicate row candidates are admitted only by the P38.2q numerical-alias
  rule; and
- a fake-GCS abrupt exit after rounds 0 and 1 leaves two independently
  auditable round bundles rather than one incomplete latest snapshot.

Observer evidence (2026-08-16): pinned-image install/manifest verification
passes for Qwen3-1.7B and Qwen3-8B; both overlays pass 34 runner tests. Source
`ae63d44e...` passed a local v5p off-versus-seam-tail
comparison across three frozen rounds. Both arms completed with zero backward
and optimizer commits; the combined arm recorded 130 seam and 130 tail rows;
and the complete alignment contracts were equal except timestamps.

The first target source `6b75e3cf...` reached round 0 but failed its durability
seal. Review of the later `fbb4b278` repair found two defects: training `step`
is not the frozen diagnostic-round counter, and admitting arbitrary unscoped
JSONL records is fail-open. The local replacement labels diagnostic
pre-alignment records with `p38_diagnostic_round_index()`, requires that field
for scoped pre-alignment and incident streams, and treats only the schema-
validated request journal as cumulative-unscoped. Focused round-stage,
postflight, seam-neutrality, fake-GCS two-round isolation/abrupt-exit, pinned-
image alignment, and the complete P33 CPU ladder pass. These are host evidence-
control changes; no model executable patch or canonical kernel changed. The
replacement source must be explicitly approved and published before launch.

## Target gate

```text
attempt=0
geometry=DP16xTP4/concurrency256/fixed-M
frozen_rounds=3
backward=0
optimizer_commits=0
round_complete_markers=3
capsules=3
required_arm_keys=2*red_points
matched_arm_keys=required_arm_keys
unmatched_keys=[]
payload_conflicts=[]
bundle_auditor=PASS
official_classifier=reproducible_from_returned_bytes
```

## Decision table

| First measured red checkpoint | Decision |
|---|---|
| hidden layer input/output | Localize the earliest hidden layer; tail values are downstream evidence only |
| final norm exact, raw target logit red | Localize lm_head/projection |
| raw target exact, raw normalizer red | Localize raw vocabulary reduction |
| raw path exact, processed target/normalizer red | Localize logits processing |
| all prior values exact, final logprob red | Localize target gather/subtraction |
| all measured checkpoints exact | Extend the seam only after proving the current observer reached every red key; do not name a cause |
| missing round/key/file or observer perturbation | `INCONCLUSIVE`; do not interpret numerical values |

## Claim ceiling

This phase may identify the earliest measured divergent region and choose one
repair experiment. Fingerprint equality is not full-tensor equality, and the
run is diagnostic rather than training admission. No tail substage is promoted
before all earlier registered checkpoints are exact for every joined red key.

## Rollback

Leave every P38 seam/tail observer variable unset. The observer is default-off
and must never ride the P45 full-training lane.

## Operator card

After the replacement publication, use `P38S18R_RUNBOOK.md`. Do not reconstruct the
launch from historical handoff sections and do not manually add environment
variables to a rendered YAML.
