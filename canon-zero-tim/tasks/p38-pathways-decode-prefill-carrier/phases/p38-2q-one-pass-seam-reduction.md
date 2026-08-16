# P38.2q — One-pass seam reduction and ambiguity audit

Status: active. Implementation is local; GCP execution is pending publication.

## Entering correction

P38.2p/v1 proved that the source snapshot and reducer transport can be
SHA-consistent while still returning too little evidence to decide the case.
The selected `live/000020` snapshot contained only round 0, not both completed
rounds. One round-0 A key also appeared in records 319 and 398. The v1
manifest retained their file identities but did not say whether the matching
token rows were numerically identical or conflicting.

The old policy “more than one record is ambiguous” is too coarse because seam
records can overlap. Picking the first record would be fail-open. Requiring a
new TPU run would be wasteful. P38.2q therefore makes the reduction itself
return enough byte-preserving evidence to decide either outcome offline.

## Deliverable

One GCP command must produce a compact bundle with:

1. an inventory of every six-digit live snapshot;
2. automatic selection requiring contiguous capsule rounds 0 and 1, paired
   seam JSON/NPZ, run log, pre-alignment log, LIVE marker, and source manifest;
3. source inventory and SHA verification before interpretation;
4. one join entry per `(round, token-prefix SHA, arm)` with every candidate's
   record index, row offset, packed row, position, token, request/call
   provenance, and numerical-payload SHA;
5. deterministic alias selection only when checkpoint metadata plus all
   layer/final fingerprints are identical;
6. fail-closed payload-conflict records containing every candidate;
7. all matching source JSON/NPZ copied byte-for-byte under `records/`;
8. all immutable capsules, source manifests, snapshot selection, ambiguity
   audit, official classification when permitted, verdict, and a complete
   self-excluding SHA manifest; and
9. a separate bundle auditor that verifies inventory/verdict consistency and
   reproduces the official classifier from the compact bundle alone.

The bundle is capped at 90 MB. It is committed under a new
`evidence/p38s18l/reduction-v2/` subdirectory only after its local audit passes.

## Local gate

```bash
python3 canon-zero-tim/tests/p38_serving/test_select_p38_live_snapshot.py
python3 canon-zero-tim/tests/p38_serving/test_reduce_p38_seam_evidence.py
python3 canon-zero-tim/tests/p38_serving/test_reduce_p38_gcp_wrapper.py
python3 canon-zero-tim/tests/p38_serving/test_seam_classifier.py
python3 -m py_compile \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/select_p38_live_snapshot.py \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/reduce_p38_seam_evidence.py \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/audit_p38_seam_reduction.py \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_seam.py
bash -n \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_reduce_p38s18l_on_gcp.sh
```

Required controls:

- a newer one-round snapshot loses to an older two-round snapshot;
- an unpaired seam JSON/NPZ snapshot is rejected;
- a missing arm remains `INCONCLUSIVE_REDUCTION_JOIN`;
- numerically identical duplicate rows become provenance-attested aliases;
- numerically different duplicates retain both raw records and remain
  fail-closed;
- source and returned-bundle byte mutations are rejected; and
- a fake-GCS end-to-end wrapper run selects the two-round snapshot and returns
  a bundle accepted by the standalone auditor.

## Target gate

For P38s18l, require:

```text
capsule_rounds=[0,1]
red_points=47
required_arm_keys=94
matched_arm_keys=94
unmatched_keys=[]
ambiguous_keys=[]
bundle auditor=PASS
scientific verdict=INCONCLUSIVE_PARTIAL_RUN
```

Equivalent aliases are allowed only when fully enumerated in
`AMBIGUITY_AUDIT.json`. The original run remains partial because it lacks a
completed third round and terminal precheck marker.

## Decision table

| Result | Decision |
|---|---|
| No eligible two-round snapshot | Preserve inventory; do not substitute round 0 or launch TPU |
| Payload conflict | Analyze returned candidate records offline; strengthen identity only if the conflict is provenance rather than numerical |
| All keys joined, hidden/final fingerprints exact | Build a bounded tail observer; do not name its substage in advance |
| Any hidden fingerprint red | Select the earliest measured checkpoint and withdraw tail-only localization |

## Claim ceiling

This phase can make P38s18l locally reproducible and choose the next observer.
It cannot convert an interrupted run into signed three-round evidence, prove
full hidden tensor bytes from fingerprints, isolate the normalizer, or admit a
repair/training run.

## Rollback

Revert only the P38.2q diagnostic CL or leave the versioned v2 derived prefix
unused. Never overwrite/delete the v1 reduction or any immutable live snapshot.
