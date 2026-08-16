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
python3 .claude/skills/manage-canon-zero-tim-branch/scripts/preflight_runtime.py \
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

bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_reduce_p38s18l_on_gcp.sh \
  "$LIVE_ROOT" "$DEST"
```

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

If no two-round snapshot exists, the script prints the complete snapshot
inventory and exits before downloading or reducing anything. Return that JSON;
do not weaken the minimum to one round.

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
gcloud storage cat "$DEST/files/AMBIGUITY_AUDIT.json"
gcloud storage cat "$DEST/files/REDUCTION_MANIFEST.json"
gcloud storage cat "$DEST/files/classification.json" || true
gcloud storage cat "$DEST/files/SHA256SUMS"
gcloud storage ls -l "$DEST/**"
```

Absence of `classification.json` is valid only for
`INCONCLUSIVE_REDUCTION_JOIN`; return stderr and the ambiguity audit instead.

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

The auditor verifies the exact bundle inventory, every file SHA, snapshot and
capsule provenance, join totals, ambiguity totals, verdict consistency, and—if
selection completed—re-runs the official classifier and requires byte-for-byte
equivalent JSON content.

## 5. Acceptance and decision table

| Result | Meaning | Next action |
|---|---|---|
| No eligible two-round snapshot | GCS never persisted both completed rounds | Return snapshot inventory; do not run a one-round substitute |
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
> those exact bytes. If the verdict is inconclusive, still return and preserve
> the complete records plus ambiguity audit; they are designed to permit
> offline diagnosis without another GCS or TPU run. Stop before push unless the
> user explicitly approves that evidence push.
