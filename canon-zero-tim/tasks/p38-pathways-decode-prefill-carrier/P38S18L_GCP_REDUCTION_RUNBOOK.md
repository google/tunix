# P38s18l one-pass GCP evidence-reduction runbook

This is the only current operator card for P38s18l. It inventories immutable
GCS snapshots, selects the best eligible source automatically, resolves benign
duplicate observations by numerical identity, preserves every conflicting
candidate, and returns a self-contained bundle. It does not launch TPU work.

The first `v1` reduction is retained as evidence. It selected `live/000020`,
which contained only round 0, and stopped on duplicate A records 319/398. Do
not overwrite or reinterpret that result. Run the hardened flow below as `v2`.

## 1. Use a clean reviewed source tree

```bash
set -euo pipefail
git fetch origin yuxzhang/canon-zero-tim
git worktree add --detach /tmp/p38-reducer-v2 origin/yuxzhang/canon-zero-tim
cd /tmp/p38-reducer-v2
python3 canon-zero-tim/.claude/skills/manage-canon-zero-tim-branch/scripts/preflight_runtime.py \
  --repo "$PWD"
```

The checked-out commit must contain all four scripts:

```text
select_p38_live_snapshot.py
reduce_p38_seam_evidence.py
classify_p38_seam.py
audit_p38_seam_reduction.py
```

Do not run from an edited checkout. Do not select a six-digit snapshot by
hand; the wrapper inventories all of them and requires at least capsule rounds
0 and 1.

## 2. Run exactly one GCP-side reduction

```bash
set -euo pipefail
LIVE_ROOT="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18l-9a834574/attempt-0/live"
DEST="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18l-9a834574/attempt-0/derived/p38s18l-seam-reduction-v2"

set +e
bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_reduce_p38s18l_on_gcp.sh \
  "$LIVE_ROOT" "$DEST"
REDUCE_RC=$?
set -e
case "$REDUCE_RC" in
  0|4) ;;
  *) echo "unexpected P38 reducer rc=$REDUCE_RC" >&2; exit "$REDUCE_RC" ;;
esac
```

`0` means a source was admitted and reduced. `4` is accepted only when the
wrapper already uploaded a sealed selection-only bundle and printed
`COMPLETE verdict=INCONCLUSIVE_NO_ELIGIBLE_SNAPSHOT`; the remaining audit steps
are mandatory in both cases.

The wrapper performs all of the following before upload:

1. records every six-digit live snapshot and its capsule/seam inventory;
2. rejects snapshots without `LIVE.json`, `SHA256SUMS`, `run.log`,
   `pre-alignment.jsonl`, paired seam JSON/NPZ, or contiguous rounds 0 and 1;
3. prefers capsule coverage before snapshot number, then chooses the newest
   equally covered snapshot;
4. verifies the selected snapshot's exact file inventory and every source SHA;
5. derives every red A/B key from every immutable capsule in that snapshot;
6. records every candidate record and exact row offset for each key;
7. classifies repeated candidates as:
   - `equivalent_alias` only when position, token, checkpoint metadata, all
     layer fingerprints, and final-norm fingerprints are identical; or
   - `payload_conflict`, which remains fail-closed;
8. retains every matching raw JSON/NPZ byte-for-byte under `records/`;
9. writes `SNAPSHOT_SELECTION.json`, `AMBIGUITY_AUDIT.json`, the complete join
   map, source/capsule provenance, classifier output when permitted, verdict,
   and a self-excluding `SHA256SUMS`; and
10. uploads both the unpacked files and one compressed archive.

If no two-round snapshot exists, the script does not download or reduce a
source. It still seals and uploads a selection-only evidence bundle containing
the raw object listing, selector JSON/stdout/stderr, verdict, packaging note,
archive, and SHA inventory. It exits 4 only after upload. Return and audit that
bundle; do not weaken the minimum to one round.

## 3. Required returned marker and metadata

Return the complete final line beginning with:

```text
[P38.REDUCE.GCP] COMPLETE
```

That line must contain:

```text
verdict reducer_rc snapshot red_points matched_arm_keys aliases conflicts
unmatched manifest_sha256 ambiguity_audit_sha256 archive_sha256 destination
```

Then return these exact files without editing:

```bash
gcloud storage cat "$DEST/files/SNAPSHOT_SELECTION.json"
gcloud storage cat "$DEST/files/verdict.json"
gcloud storage cat "$DEST/files/OBJECT_LISTING.txt" || true
gcloud storage cat "$DEST/files/selector.stdout" || true
gcloud storage cat "$DEST/files/selector.stderr" || true
gcloud storage cat "$DEST/files/AMBIGUITY_AUDIT.json" || true
gcloud storage cat "$DEST/files/REDUCTION_MANIFEST.json" || true
gcloud storage cat "$DEST/files/classification.json" || true
gcloud storage cat "$DEST/files/SHA256SUMS"
gcloud storage ls -l "$DEST/**"
```

For a selection-only `INCONCLUSIVE_NO_ELIGIBLE_SNAPSHOT` bundle,
`OBJECT_LISTING.txt` and selector stdout/stderr are required, while reduction
manifest, ambiguity audit, records, capsules, and classification must be
absent. For `INCONCLUSIVE_REDUCTION_JOIN`, return the reduction manifest,
stderr, and ambiguity audit; classification must be absent.

## 4. Download and mechanically audit the complete compact bundle

Do not return only metadata. The selected raw rows are about megabytes, not the
multi-thousand-file source snapshot. Download the complete `files/` hierarchy:

```bash
set -euo pipefail
RESULT_DIR="/tmp/p38s18l-seam-reduction-v2"
mkdir -p "$RESULT_DIR"
gcloud storage rsync --recursive "$DEST/files" "$RESULT_DIR"
(cd "$RESULT_DIR" && sha256sum -c SHA256SUMS)

python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/audit_p38_seam_reduction.py \
  --bundle-dir "$RESULT_DIR" \
  --output /tmp/p38s18l-seam-reduction-v2.audit.json
cat /tmp/p38s18l-seam-reduction-v2.audit.json
```

The auditor verifies the exact bundle inventory and every file SHA. For a
selection-only bundle it re-runs the selector from `OBJECT_LISTING.txt` and
requires exact JSON equality. For an admitted snapshot it verifies capsule
provenance, join/ambiguity/verdict totals, re-runs the official classifier, and
requires byte-for-byte equivalent JSON content.

## 5. Acceptance and decision table

| Result | Meaning | Next action |
|---|---|---|
| `INCONCLUSIVE_NO_ELIGIBLE_SNAPSHOT` + auditor PASS | GCS never persisted one self-contained two-round source | Commit the inventory bundle, retire P38s18l, and do not infer a tail cause |
| `INCONCLUSIVE_REDUCTION_JOIN` with `payload_conflict` | Same red key has numerically different candidates | Bundle already contains every conflicting record; analyze it offline, no TPU rerun |
| `INCONCLUSIVE_PARTIAL_RUN` and classifier hidden-exact | All available red keys joined, but source run did not finish 3 rounds | Admit only analysis-grade tail-localization direction |
| `INCONCLUSIVE_PARTIAL_RUN` and hidden seam red | All available keys joined and a measured hidden checkpoint differs | Select earliest measured seam; withdraw tail-only hypothesis |
| `PASS` | Selection and the original three-round terminal contract both complete | Signed diagnostic result, subject to normal P38 postflight |

For a successful two-round P38s18l reduction, require:

- `capsule_rounds == [0, 1]`;
- `red_points == 47`;
- `required_arm_keys == matched_arm_keys == 94`;
- empty `unmatched_keys` and `ambiguous_keys`;
- any `equivalent_alias_keys` fully described in `AMBIGUITY_AUDIT.json`;
- local bundle auditor `PASS`; and
- expected scientific verdict `INCONCLUSIVE_PARTIAL_RUN`, because the original
  run has no completed third round or terminal marker.

Hidden/final fingerprint equality does **not** isolate the normalizer. It only
authorizes a bounded tail observer spanning lm_head/raw logits, target gather,
normalizer, processed target, and final subtraction.

For the expected selection-only outcome, require:

- wrapper exit code `4` after a `[P38.REDUCE.GCP] COMPLETE` line;
- verdict `INCONCLUSIVE_NO_ELIGIBLE_SNAPSHOT`;
- `candidate_count == 22` and `qualified_candidate_count == 0`;
- complete `OBJECT_LISTING.txt` plus selector JSON/stdout/stderr;
- local bundle auditor `PASS`; and
- no reduction manifest, ambiguity audit, classifier, capsules, or records.

This outcome exhausts P38s18l. It does not satisfy the prerequisite for a tail
observer. The next target acquisition, if approved separately, follows
`phases/p38-2r-terminal-seam-tail-acquisition.md` and captures hidden seams and
the bounded tail together with per-round atomic persistence.

## 6. Publish the compact evidence, not the raw snapshot

After the bundle audit passes, prepare a new append-only directory:

```bash
EVIDENCE="canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/evidence/p38s18l/reduction-v2"
AUDIT_EVIDENCE="canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/evidence/p38s18l/reduction-v2.audit.json"
test ! -e "$EVIDENCE"
test ! -e "$AUDIT_EVIDENCE"
mkdir -p "$EVIDENCE"
cp -a "$RESULT_DIR/." "$EVIDENCE/"
cp /tmp/p38s18l-seam-reduction-v2.audit.json "$AUDIT_EVIDENCE"
git add -- "$EVIDENCE" "$AUDIT_EVIDENCE"
git diff --cached --check
git status --short
```

Commit this directory as one evidence-only CL. Do not modify or replace the
existing `p38s18l` files. Per repository policy, wait for explicit user
approval before pushing the evidence CL.

## Remote-agent prompt

> Read `P38S18L_GCP_REDUCTION_RUNBOOK.md` completely and execute only its v2
> flow from a clean checkout. Do not launch TPU work, do not choose a snapshot
> manually, do not lower the two-capsule minimum, and do not modify/delete any
> source or v1 derived object. Run the wrapper against the live root and the
> versioned v2 destination. Return the COMPLETE line and every file listed in
> §3. Download the entire compact `files/` hierarchy, run the bundle auditor,
> and prepare the append-only `evidence/p38s18l/reduction-v2/` evidence CL from
> those exact bytes. Exit code 4 is expected when no snapshot qualifies, but it
> is accepted only after the wrapper prints COMPLETE, the selection-only bundle
> exists at DEST, and the standalone auditor passes. In that case return the raw
> object listing and selector artifacts; do not fabricate reduction files and
> do not advance to a tail-only probe. Stop before push unless the user
> explicitly approves that evidence push.
