# Phase D3e — canonical first-action classifier scope

## Objective

Decide the Attempt-17 completion-position-zero tensor interval from the
immutable sealed treatment bundle without hiding later mixed red signatures
and without launching another TPU pair. This phase changes classifier decision
accounting only. It does not change A/B/C, prefix-cache reads, RoPE,
RPA/attention, KV values, production flags, backward, or optimizer behavior.

## Evidence that forced this phase

The verified D3d return at
`evidence/v1_apc_m15_attempt17_d36_offline_binding_20260829/` proves:

- A-B is 207 differing bytes / 95 elements and B-C is zero;
- source row 217 / completion position 0 uniquely binds to A request
  `79-b8334848` with selected future-prefix proof 1300 at or beyond the
  required elimination horizon 1227;
- the unique first-action candidate is Layer 0 `rpa_output`;
- seven standard-path red points are joinable, with global signatures
  Layer 0 `rpa_output` and `final_norm`; 88 continue-decode red points remain
  explicitly unobserved;
- the old classifier selected completion-position-zero anchors for its public
  result but computed the localization gate across all joinable red points.

The D3d request-identity hypothesis succeeded. Repeating D3d unchanged or
adding another request-identity field is therefore not the next operation.

## Decision contract

When `require_first_action=True`, the localization decision scope is every
exactly joined completion-position-zero candidate. Later standard-path joins
remain in the report as `all_join_*` diagnostics and all continue-decode red
points remain counted as unobserved. They do not inherit the first-action
boundary and do not veto it merely because a later action first differs at a
different checkpoint.

The gate remains fail closed:

- mixed signatures within the completion-position-zero decision scope remain
  `FIRST_RED_CANDIDATE_SET`;
- an exact-through decision candidate still blocks localization;
- same-request conflicting duplicates and multiple B numeric variants remain
  fatal;
- no completion-position-zero join remains fatal;
- B-C nonzero remains fatal;
- a localized offline interval never authorizes a numerical repair by itself.

The report records both `first_difference_signatures` for the decision scope
and `all_join_first_difference_signatures` for every joinable red point. The
claim ceiling explicitly prevents later or unobserved red actions from
inheriting the selected boundary.

## Implementation

1. `classify_m15_apc_wide_seam.py` names the decision scope and computes the
   gate only from that scope while retaining global coverage and signatures.
2. `review_m15_attempt17_d36_candidate.py` records the scope, global mixed
   status, pinned-image debt, and `numerical_repair_authorized=false`.
3. `run_m15_attempt17_d3e_canonical_action.sh` delegates to the existing
   manifest-bound, read-only D36 recovery and then verifies the immutable
   Attempt-17 numerical boundary, source/request binding, fingerprint
   geometry, source anchors, and presence of cache-page coordinates. It writes
   no remote object and performs no Kubernetes or TPU operation.

## Local gate

```bash
python3 -m unittest discover \
  -s canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts \
  -p 'test_*.py'
bash -n \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt17_d3e_canonical_action.sh
python3 -m py_compile \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/classify_m15_apc_wide_seam.py \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/review_m15_attempt17_d36_candidate.py
git diff --check
```

Required focused negatives include later-point mixed signatures, mixed
first-action signatures, exact-through first-action candidates, insufficient
future-prefix horizon, B numeric variants, B-C red, and wrapper mutation
surface checks.

## Pinned exact-image gate

This separate gate passed on the official aggregate, not a hand-selected
subset:

Read-only inspection on the current host resolves the pinned image to
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
The approved command ran without a pipe and preserved the raw log:

```bash
test ! -e /tmp/m15-d3e-exact-image-b74c4ba3-20260829.log
bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a \
  >/tmp/m15-d3e-exact-image-b74c4ba3-20260829.log 2>&1
```

The command resolves the local image reference to and prints an immutable
`sha256:` image ID before running. The complete terminal marker must remain:

```text
V1_HP_EXACT_IMAGE_PASS ... apc_m15_carrier=68 m15_d3e=1 m15_durability=1 m15_round_provenance=1 ... manifests=3
```

The gate uses local Docker/CPU only. It does not access TPU, Kubernetes, or
GCS. Preserve the complete raw log on any nonzero exit or missing marker.

Observed exit was 0. Matching `image_ref` and `image_id` were
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`;
the required marker included `apc_m15_carrier=68`, `m15_d3e=1`,
`m15_durability=1`, `m15_round_provenance=1`, and `manifests=3`. Raw-log
SHA256 is
`59efa6ddc6e0399050cbbbbc5b463fc6b94486d96834f1e8b50f4fd9d3b22d97`.

## Bucket-capable execution gate

Only after host and pinned exact-image PASS, an explicitly approved published
analysis commit, and separate GCS-read approval may another agent create a
clean `local/*` worktree at that exact commit and run:

```bash
RETURN=/mnt/disks/tunix-data/m15-d3e-canonical-action-return-<fresh-label>
test ! -e "$RETURN"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt17_d3e_canonical_action.sh \
  "$RETURN" /mnt/disks/tunix-data
```

The command must run directly without a pipe. Expected final markers are:

```text
M15_D3E_CANONICAL_ACTION_REVIEW_PASS status=<status> decision_scope=completion-position-zero ... numerical_repair_authorized=0
[M15.D3E.OFFLINE] COMPLETE status=<FIRST_RED_LOCALIZED|FIRST_RED_CANDIDATE_SET_PRESERVED> ...
[M15.D3E.OFFLINE] TARGET_NOT_RUN gcs_read=1 gcs_write=0 kubernetes=0 tpu=0
```

Missing markers, nonzero exit, auth/network failure, manifest drift, or a
preserved scratch directory is `INCONCLUSIVE`; preserve the raw log and
scratch path and stop.

## Decision table

| Result | Meaning | Next action |
|---|---|---|
| `FIRST_RED_LOCALIZED`, unique request binding, `k_post_rope -> rpa_output`, verified fingerprint geometry, page receipt, and source anchors | the immutable first-action interval is admitted for review; later mixed actions remain separate | review the complete returned coordinate/shape ledger with the user; only then discuss a minimal Phase-E hypothesis |
| `FIRST_RED_CANDIDATE_SET_PRESERVED` | the decision-scope candidates are still mixed or exact-through | add direct producer/request provenance and original checkpoint-shape metadata, pass host and exact-image gates, then request a fresh matched DP8xTP8 pair |
| identity, manifest, B invariant, source-anchor, geometry, or page-receipt failure | reclassification evidence is invalid | preserve and stop; do not launch TPU |
| infrastructure failure | `INCONCLUSIVE` | retry only the read-only recovery after fixing infrastructure |

No result from this phase is a numerical repair or target rerun. Production
M15 APC remains off and Phase E remains closed until the user reviews a
complete accepted localization ledger.
