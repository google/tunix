# P38.2s — Round-0 alias-aware seam-and-tail reduction

Status: active; Deliverable A is implemented and locally gated in an isolated
uncommitted review worktree. This is a zero-TPU, GCS-side analysis phase. It
supersedes the direct whole-directory classifier command in
`P38S18R_RUNBOOK.md`; it does not supersede or modify any immutable P38s18r2
source object.

## Entering evidence

P38s18r2/source `10fe951f0186256aa106627c4323de1f5aa168be`
completed one numerical round and then lost the remaining two rounds to the
durability-ACK timeout. The immutable Round 0 directory is:

```text
gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18r2-10fe951f/attempt-0/rounds/000000
```

The returned receipt at commit `a514c3bf` mechanically establishes:

- 3,896 listed source objects;
- 3,894 entries in the source `SHA256SUMS`, whose filename set is exactly the
  object listing after excluding `ROUND_COMPLETE.json` and `SHA256SUMS`;
- 972 paired seam JSON/NPZ records and 972 paired terminal-tail JSON/NPZ
  records;
- a sealed Round 0 manifest SHA of
  `ce7df453259dd070472486e053dbb26b03dad7b6259784cde74da7fe9efe227e`;
- exact B-C and A-B red at 45 bytes / 32 elements with
  `max_abs=0.10101699829101562`; and
- official classifier rc 1 at `duplicate seam token-prefix record`, before
  any of the 32 red points could be classified.

The direct-classifier receipt is a complete failure receipt, not a successful
classification. Its empty `source_round_gcs_uri` is also a provenance defect
that the replacement bundle must reject rather than repeat.

## Objective

Resolve overlapping observations without choosing the first or last record,
then run the unchanged official layer-plus-tail classifier over a compact,
byte-preserving subset. Answer the existing Round 0 question without another
TPU launch:

```text
For each of the 32 red actions, where is the first measured A/B divergence:
the hidden seam, final norm, or one of the bounded terminal-tail checkpoints?
```

## Deliverable A — reviewed reducer support

Implementation inventory (pending review/publication):

- `scripts/reduce_p38_seam_tail_evidence.py`
- `scripts/audit_p38_seam_tail_reduction.py`
- `scripts/p38s18r2_round0_contract.json`
- `scripts/run_reduce_p38s18r2_round0_on_gcp.sh`
- `tests/p38_serving/test_reduce_p38_seam_tail_evidence.py`

Extend the existing v2 reduction/audit path, or add a narrowly scoped sibling,
with all of these properties:

1. accept an immutable round directory plus one mismatch capsule;
2. require `--mode layer --require-tail` for P38s18r2;
3. derive exactly 32 red points and 64 required
   `(round, token-prefix SHA, arm)` keys from the capsule;
4. enumerate every seam candidate and every tail candidate for each required
   key, including original record index and row offset;
5. classify repeated seam candidates as equivalent aliases only when
   position, source token, checkpoint names, layer indices, every layer
   fingerprint, and final-norm fingerprint are byte-identical;
6. classify repeated tail candidates as equivalent aliases only when
   position, source token, target token, logit-row index, checkpoint names,
   and every tail value are byte-identical;
7. select the lowest original record index only after full payload identity is
   proven; never use filesystem order as scientific evidence;
8. retain every conflicting candidate and return
   `INCONCLUSIVE_REDUCTION_JOIN` when either seam or tail payload conflicts;
9. retain every matching source JSON/NPZ byte-for-byte under `candidates/`,
   retain the deterministically selected source records under `records/`, and
   include the capsule plus source provenance;
10. invoke `classify_p38_seam.py` with both the reduction manifest and
    `require_tail=True` only after all 64 keys have unambiguous seam and tail
    selections; and
11. make the standalone bundle auditor rerun the same official classifier
    with `require_tail=True` from the returned compact bundle alone.

The reduction manifest must explicitly record `require_tail: true`, separate
seam and tail alias/conflict counts, all selected source indices, the nonempty
source GCS URI, source round-manifest SHA, and the classifier source SHA.

## Local gates

The implementation CL is not ready for remote use until focused fixtures prove:

- one unique seam/tail pair joins;
- equivalent duplicate seam candidates are enumerated and admitted;
- equivalent duplicate tail candidates are enumerated and admitted;
- numerically different duplicate seam candidates remain fail-closed;
- numerically different duplicate tail candidates remain fail-closed;
- a missing A/B seam or tail candidate remains fail-closed;
- the direct whole-directory classifier still detects the raw duplicate
  fixture, proving the negative control is live;
- a reduced 32-red-point fixture reports 64/64 required keys and 32/32 joined
  red points with `tail_observer_required_and_joined=true`;
- source-byte, selected-record, capsule, manifest, and classifier-output
  mutations are rejected by the standalone auditor; and
- an empty `source_gcs_uri` is rejected.

Run the normal Python compilation, shell syntax, credential scan, focused P38
tests, and `git diff --check`. Stop before commit or push for user review.

## Deliverable B — one immutable GCS reduction

After the implementation CL is reviewed and published, execute it from that
exact clean SHA on a GCS-authorized machine. Use a new destination and never
overwrite the failed v1 receipt:

```text
gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18r2-10fe951f/attempt-0/derived/p38s18r2-round0-seam-tail-reduction-v2
```

Before interpretation, require all of the following:

- source `ROUND_COMPLETE.json` schema and diagnostic round are correct;
- its manifest SHA equals the actual source `SHA256SUMS` SHA;
- every source manifest entry verifies;
- source inventory reports 972 seam and 972 tail records;
- the object listing contains exactly 3,896 objects and equals the 3,894
  manifest names plus the two sealing files; and
- source GCS URI and analysis source commit are nonempty.

The returned compact bundle must contain:

```text
OBJECT_LISTING.txt
SOURCE_ROUND_COMPLETE.json
SOURCE_ROUND_INVENTORY.json
SOURCE_SHA256SUMS
SNAPSHOT_SELECTION.json
REDUCTION_MANIFEST.json
AMBIGUITY_AUDIT.json
capsules/mismatch-capsule.npz
candidates/<every candidate source JSON/NPZ for the required keys>
records/<selected source JSON/NPZ used by the classifier>
classifier.stdout
classifier.stderr
classifier.rc
classification.json                 # only when official classification succeeds
analysis_source_commit.txt
classifier_source.sha256
verdict.json
PACKAGING.txt
SHA256SUMS
```

Return the complete compact `files/` hierarchy, not only summary JSON. Run the
standalone auditor after downloading it, and return its audit JSON alongside
the bundle. Raw unselected records remain in GCS.

## Target gate and decision table

For a successful reduction require:

```text
source seam/tail records = 972 / 972
red_points = 32
required_arm_keys = 64
matched_seam_keys = 64
matched_tail_keys = 64
unmatched_keys = []
payload_conflict_keys = []
joined_red_points = 32
tail_observer_required_and_joined = true
standalone bundle audit = PASS
scientific verdict = INCONCLUSIVE_PARTIAL_RUN
```

Equivalent aliases are permitted only when fully enumerated in
`AMBIGUITY_AUDIT.json`. The overall verdict stays
`INCONCLUSIVE_PARTIAL_RUN` because rounds 1 and 2 and the terminal three-round
contract do not exist.

| Result | Decision |
|---|---|
| All 32 red points join and a hidden/final checkpoint is first red | Select the earliest measured hidden seam; do not claim a tail-only cause |
| Hidden/final checkpoints are exact and a terminal-tail checkpoint is first red | Select that bounded tail substage for the next repair/probe |
| Multiple first-difference signatures | Preserve every signature; do not collapse them to one root cause |
| Equivalent aliases only | Admit the deterministic reduced subset with full alias provenance |
| Any seam/tail payload conflict or missing key | `INCONCLUSIVE_REDUCTION_JOIN`; analyze the returned candidates offline, with no TPU rerun yet |
| Classifier or auditor failure | `INCONCLUSIVE_REMOTE_CLASSIFICATION`; repair the offline tool or package, not the model run |

## Claim ceiling

This phase may produce one analysis-grade first-divergence classification from
an immutable Round 0. It cannot manufacture the missing two rounds, upgrade
P38s18r2 to signed evidence, prove full hidden tensor bytes from integer
fingerprints, admit a numerical repair, or close zero-TIM.

## Rollback

Do not delete or overwrite the source Round 0 or failed v1 receipt. Revert only
the reducer/auditor CL or leave the versioned v2 derived prefix unused. No
training, evaluation, optimizer, checkpoint, or model executable is changed by
this phase.
