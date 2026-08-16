# P38.2p — GCP-side seam evidence reduction

Status: complete/inconclusive. V1 verified the selected source but covered only
round 0 and stopped on duplicate A records 319/398. P38.2q supersedes the
operator path with automatic two-round selection and a full ambiguity audit.

## Goal

Recover a locally reclassifiable, byte-preserving subset from the large
P38s18l live snapshot without transferring every seam record and without
manufacturing the missing third diagnostic round or terminal markers.

## Entering evidence

- The committed P38s18l package is SHA-consistent but partial. Its raw log has
  two `PRECHECK_ROUND_COMPLETE` markers, two pre-alignment records, no terminal
  `PRECHECK_COMPLETE`, and ends during the third rollout.
- The committed directory contains no `p38_seam_*.json/.npz`; therefore the
  committed classification cannot be reproduced with the official classifier.
- Its hand-authored classification says 20 of 47 red points joined. That is a
  useful candidate direction, not an admitted tail localization.
- The full live snapshot remains in GCS and is too large/file-dense to commit
  directly.

## Deliverable

Run the reducer beside GCS. It must:

1. verify the source live snapshot's existing `SHA256SUMS`;
2. derive every A/B-red source-token key from immutable round capsules;
3. stream all seam records and require exactly one A and one B match per key;
4. copy only the selected raw JSON/NPZ files, byte-for-byte and without
   renumbering;
5. record source URI, source-manifest SHA, original record indices, file SHAs,
   missing/ambiguous keys, capsule rounds, completed rounds, and terminal
   marker count in `REDUCTION_MANIFEST.json`;
6. re-run the official classifier in manifest-attested sparse-index mode; and
7. seal the compact hierarchy with a new `SHA256SUMS` and upload it under the
   source run's `derived/` prefix.

The derived package is analysis evidence. It never writes `LIVE.json`,
`COLLECTED.json`, or `COMPLETE.json` and never changes the source GCS objects.

## Local gate

Run:

```bash
python3 canon-zero-tim/tests/p38_serving/test_seam_classifier.py
python3 canon-zero-tim/tests/p38_serving/test_reduce_p38_seam_evidence.py
bash -n canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_reduce_p38s18l_on_gcp.sh
```

Required negative controls:

- a source file changed after `SHA256SUMS` is rejected;
- a missing A/B record yields `INCONCLUSIVE_REDUCTION_JOIN` and nonzero exit;
- sparse original record indices are accepted only with a valid reduction
  manifest;
- an observed-hidden exact result classifies as
  `hidden_chain_exact_tail_localization_required`, not as a normalizer result.

## Target gate

The GCP operator returns all of:

- `[P38.REDUCE.GCP] COMPLETE ...` stdout;
- derived GCS URI;
- compact archive SHA-256;
- `REDUCTION_MANIFEST.json` SHA-256;
- `red_points`, `matched_arm_keys`, `unmatched_keys`, and `ambiguous_keys`;
- the unedited `verdict.json` and `classification.json`.

The selection gate is `matched_arm_keys == 2 * red_points`, with zero unmatched
and ambiguous keys. The run-level verdict remains `INCONCLUSIVE_PARTIAL_RUN`
because P38s18l completed only two of three preregistered rounds.

## Decision table

| Reduced result | Next action |
|---|---|
| Every red point joins A/B; observed hidden/final fingerprints exact | Build one bounded tail observer; do not name lm_head or normalizer yet |
| A hidden checkpoint differs | Withdraw the tail-only claim and select the earliest measured layer/checkpoint |
| Missing or ambiguous joins | Keep P38s18l partial; inspect selection metadata before any new target run |
| Source SHA or selected-file SHA fails | Reject the reduction; do not copy or classify corrupted evidence |

## Claim ceiling

- The compact files are byte-identical selected inputs, but the package is a
  derived subset of an interrupted live snapshot.
- Fingerprints are non-cryptographic diagnostics, not full tensor-byte proofs.
- Equality through final norm means only that an unobserved tail must be
  measured next. It does not isolate lm_head, logsumexp, gather, processing, or
  subtraction.

## Rollback

Delete only the uncommitted derived output or revert this diagnostic CL. Do not
delete or overwrite the immutable source live snapshot in GCS.
