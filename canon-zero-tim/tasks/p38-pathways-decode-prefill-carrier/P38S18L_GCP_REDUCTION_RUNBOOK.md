# P38s18l GCP evidence-reduction runbook

This is the only current operator card for reducing the oversized P38s18l seam
snapshot. Run it on a GCP VM/pod with Application Default Credentials and enough
ephemeral disk for one live snapshot. It does not launch TPU work.

## 1. Use the reviewed source tree

```bash
set -euo pipefail
git fetch origin yuxzhang/canon-zero-tim
git worktree add --detach /tmp/p38-reducer origin/yuxzhang/canon-zero-tim
cd /tmp/p38-reducer
python3 .claude/skills/manage-canon-zero-tim-branch/scripts/preflight_runtime.py \
  --repo "$PWD"
```

Do not run from an edited checkout. The commit used here must contain
`reduce_p38_seam_evidence.py` and this runbook.

## 2. Identify the source live snapshot

The source must be one immutable six-digit directory under:

```text
gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/
  canon-p38-fl-stock-p38s18l-9a834574/attempt-0/live/NNNNNN
```

List candidates without copying their contents:

```bash
gcloud storage ls \
  gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18l-9a834574/attempt-0/live/
```

Choose the highest snapshot that contains `LIVE.json`, `SHA256SUMS`, both
immutable round capsules, and seam records. Do not use the incomplete committed
directory as the source.

## 3. Run the reducer

Replace only `NNNNNN` below:

```bash
set -euo pipefail
SOURCE="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18l-9a834574/attempt-0/live/NNNNNN"
DEST="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18l-9a834574/attempt-0/derived/p38s18l-seam-reduction-v1"

bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_reduce_p38s18l_on_gcp.sh \
  "$SOURCE" "$DEST"
```

The wrapper downloads within GCP, verifies the source manifest, selects the
minimum raw A/B records required by the red capsules, runs the official
classifier, creates a compressed archive, and uploads only the compact result.
It refuses to overwrite an existing derived result.

## 4. Return these exact outputs

Return the complete final line beginning with:

```text
[P38.REDUCE.GCP] COMPLETE
```

Also return:

```bash
gcloud storage cat "$DEST/files/verdict.json"
gcloud storage cat "$DEST/files/REDUCTION_MANIFEST.json"
gcloud storage cat "$DEST/files/classification.json"
gcloud storage cat "$DEST/files/SHA256SUMS"
gcloud storage ls -l "$DEST/**"
```

If `classification.json` is absent, return `verdict.json` and the reducer's
stderr unchanged. Do not hand-edit a PASS result.

## 5. Acceptance rules

- `selection_complete` must be true.
- `matched_arm_keys` must equal `2 * red_points`.
- `unmatched_keys` and `ambiguous_keys` must be empty.
- Every compact file must pass the derived `SHA256SUMS`.
- For this interrupted source run, the expected run verdict is
  `INCONCLUSIVE_PARTIAL_RUN`; that is not a reducer failure.
- Never create or rename files to `LIVE.json`, `COLLECTED.json`, or
  `COMPLETE.json` in the derived package.

## 6. Failure handling

| Marker | Meaning | Action |
|---|---|---|
| `REFUSING: source manifest SHA failed` | GCS snapshot/download is not byte-consistent | Stop; choose a different immutable snapshot |
| `INCONCLUSIVE_REDUCTION_JOIN` | At least one red action lacks a unique A/B record | Return manifest; do not rerun classifier manually |
| `INCONCLUSIVE_PARTIAL_RUN` | Selection succeeded, but source did not finish 3 rounds | Return compact bundle; use it for analysis only |
| `derived evidence already exists` | Destination is immutable | Inspect existing result; use a new versioned suffix only after review |

## Remote-agent prompt

> Read `P38S18L_GCP_REDUCTION_RUNBOOK.md` completely. Work on a clean detached
> checkout containing the reducer. Do not launch a TPU JobSet, do not change or
> delete the source GCS snapshot, and do not fabricate round 2 or terminal
> markers. Select the latest immutable p38s18l live snapshot with seam files,
> run the exact wrapper command, then return the COMPLETE line, the derived GCS
> URI, archive/manifest SHA values, `verdict.json`, reduction coverage totals,
> and classifier output. Stop on any SHA, missing-join, or ambiguous-join error.
