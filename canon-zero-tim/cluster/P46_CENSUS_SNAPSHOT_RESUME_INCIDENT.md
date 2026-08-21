# P46 128-chip census snapshot-resume incident

Incident: `canon-p46-eval-census-128-p46c128a0`

Observed: 2026-08-21 00:45:51 UTC

Failed destination tag: `p46q4census01`

## Verdict

The run did not start Qwen3 inference or census sampling. All 32 Pathways
worker pods (128 TPU chips) reached Running, but the coordinator failed while
validating the imported trajectory snapshot. No new census trajectory was
produced by this attempt.

The existing raw trajectories are still reusable. They must be copied into a
fresh destination tag and imported with their exact historical sampler SHA.
Do not restart the wash from zero, and do not overwrite or delete either the
old source evidence or the failed destination.

## Exact failure

```text
P46 clean-data gate PASS dataset=4578 whitelist=1851 sha256=2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7
ValueError: legacy trajectory contract mismatch in
/mnt/disks/linchai_data/deepswe_eval/p46q4census01/imports/p46e12806-v6-final/trajectories/q4i16k-n16-128-01b3047f8a076bc3.p0.20260816T001929Z.jsonl:
{'config_fingerprint': '01b3047f8a076bc33209e9d340ad85a443928c26a346a1f4811612ca72b4e0af',
 'run_tag': 'q4i16k-n16-128-01b3047f8a076bc3'}
[P46.EVAL.POSTFLIGHT] rc=1 transport_rc=0 subshard=0 report=0 campaign=0 census=0
```

The launch selected `--legacy-import-id`, and the importer accepted the row's
schema as `canon.p46.deepswe-eval.trajectory.v5`; otherwise `schema` would
also appear in the mismatch. The directory suffix `-v6-final` is therefore a
misleading name, not evidence that the rows are trajectory-v6.

The returned row records
`sampled_by=stock@ac2c31bc7f6f82d33b3a62d62e1c390c8338b60e`, but the
manifest supplied `--source-commit 5f2d016147a55c032ea7b89b156a583d3b4ca7e8`
without `--sampling-source-commit`. The renderer inferred the new harness SHA
as the sampling source, so the derived legacy fingerprint and run tag could
not match the old evidence. The importer correctly failed closed.

## Why `p46q4census01` must not be reused

The old entrypoint wrote `outputs/resume_contract.json` before validating the
snapshot. Therefore `p46q4census01` is now immutably bound to the incorrect
sampling source. A corrected launch against the same tag will correctly fail
with an existing-contract mismatch. Preserve that tag and its log as incident
evidence; use a fresh tag such as `p46q4census02`.

The final repair requires a SHA-sealed `legacy_source_contract.json` before it
validates any legacy-v5 import. That contract binds stable model/data/sampling/
topology facts and enumerates every observed opaque v5
`(logical_shard_index, config_fingerprint, run_tag)` cohort with exact
cardinality. It does not reconstruct historical absolute paths, and it retains
global consecutive-attempt/no-attempt-after-valid checks across cohorts. The
renderer also rejects every import that omits an explicit
`--sampling-source-commit`.

Current operator HEAD `d1646526c37b642ece5c7318a4c39ab3a43d30ac` is not an
execution pin: its partial multi-cohort refactor fails six P46 CPU tests because
observed cardinality uses integer shard keys while expected cardinality uses
`(shard, fingerprint)` keys. It also incorrectly admits trajectory-v6 through
the legacy-v5 path. Repair must keep legacy adoption v5-only and use the tuple
key consistently before any launch.

## Recovery without rerunning completed trajectories

First prove the producer of the source snapshot is terminal and no file is
still growing. Then make a new v5-only sealed copy. Copy, never move. Do not
copy `resume_contract.json` into a legacy-v5 snapshot.

```bash
SOURCE_SNAPSHOT=/mnt/disks/linchai_data/deepswe_eval/p46q4census01/imports/p46e12806-v6-final
NEW_RESUME_TAG=p46q4census02
IMPORT_ID=p46e12806-v5-final
NEW_ROOT=/mnt/disks/linchai_data/deepswe_eval/$NEW_RESUME_TAG
SNAPSHOT=$NEW_ROOT/imports/$IMPORT_ID
SAMPLING_SOURCE_SHA=ac2c31bc7f6f82d33b3a62d62e1c390c8338b60e

test -d "$SOURCE_SNAPSHOT/trajectories"
test ! -e "$SNAPSHOT"
install -d "$SNAPSHOT/trajectories"
cp -a "$SOURCE_SNAPSHOT/trajectories/." "$SNAPSHOT/trajectories/"
test ! -e "$SNAPSHOT/resume_contract.json"
python3 canon-zero-tim/cluster/seal_p46_legacy_v5_snapshot.py \
  --snapshot-dir "$SNAPSHOT" \
  --sampling-source-commit "$SAMPLING_SOURCE_SHA" \
  --topology 128
test -f "$SNAPSHOT/legacy_source_contract.json"
test -f "$SNAPSHOT/SHA256SUMS"
(cd "$SNAPSHOT" && sha256sum -c SHA256SUMS)
chmod -R a-w "$SNAPSHOT"
```

Require `P46_LEGACY_V5_SEAL_PASS`. Never hand-author the source contract or
reuse a partial/failed staging seal. The sealer rejects a mixed fingerprint/
run-tag cohort even when every individual value is syntactically valid.

Inspect at least the schema and sampler provenance before rendering:

```bash
FIRST_JSONL=$(find "$SNAPSHOT/trajectories" -type f -name '*.jsonl' | LC_ALL=C sort | head -n 1)
head -n 1 "$FIRST_JSONL" | jq -r '.schema, .sampled_by, .run_tag, .config_fingerprint'
```

Expected schema and sampler:

```text
canon.p46.deepswe-eval.trajectory.v5
stock@ac2c31bc7f6f82d33b3a62d62e1c390c8338b60e
```

The old `f823bb6a` pin is superseded. Sealed-contract implementation
`9cebe0d1671f6da1748bc53ed0da07a5f970fb37` remains an ancestry floor, but
neither it nor blocked HEAD `d1646526` is the final launch pin. Only after the
minimal repair is approved, published, and read back may the exact repaired
operator-branch HEAD be used as the source SHA. The concrete node-pool and
image digest still come from the current allocation:

```bash
SEALED_CONTRACT_BASE_SHA=9cebe0d1671f6da1748bc53ed0da07a5f970fb37
REPAIR_SHA=REPLACE_WITH_APPROVED_40_CHARACTER_REPAIR_SHA
RUN_ID=p46c128a1
BASE=canon-zero-tim/cluster/jobset-256cluster-64chip.yaml

[[ "$REPAIR_SHA" =~ ^[0-9a-f]{40}$ ]]
git fetch origin yuxzhang/canon-zero-tim
git merge-base --is-ancestor "$SEALED_CONTRACT_BASE_SHA" FETCH_HEAD
git merge-base --is-ancestor "$REPAIR_SHA" FETCH_HEAD
SOURCE_SHA=$(git rev-parse FETCH_HEAD)
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]

python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
  --base "$BASE" \
  --output "/tmp/p46-census-128-${RUN_ID}.yaml" \
  --workload q4-clean-eval \
  --topology 128 \
  --source-commit "$SOURCE_SHA" \
  --sampling-source-commit "$SAMPLING_SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_ID" \
  --resume-tag p46q4census02 \
  --legacy-import-id p46e12806-v5-final \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC" \
  --full-campaign \
  --first-pass-census
```

Do not apply the rendered JobSet without separate launch authority. Before an
apply, inspect the manifest and require the explicit old sampling SHA, fresh
tag, legacy import id, 128-chip `4x4x8` topology, 32 workers, DP16 x TP8,
reward-only mode, N16, 16,384 response tokens, 50 steps, concurrency 64 and a
3,600-second physical-wave deadline.

The first corrected launch must print this marker before runtime setup:

```text
[P46.RESUME] LEGACY_IMPORT_PASS import_id=p46e12806-v5-final records=<actual> valid_records=<actual-valid> manifest_sha256=<sha256> source_contract_sha256=<sha256> ...
```

Imported durable identities are then skipped by census. Only identities with
no durable record are sampled. Later census relaunches keep the same fresh tag,
harness SHA, sampling SHA and workload fields but omit `--legacy-import-id`.
After `P46_EVAL_CENSUS_PASS`, strict repair keeps that same tag and omits both
import and census flags.

## Cardinality and claim ceiling

The incident report observed 510 raw trajectory records in the selected
snapshot. If the sealed copy contains exactly 510, exactly those 510 records
are reused; the campaign fills the remaining identities. If a terminal source
tree contains 6,460 or more raw records, seal that complete tree instead and
the importer will reuse the greater exact count after validation. Report the
actual import marker; never infer cardinality from a directory name or stale
registry entry.

The 22,918-row file under `clean_data/p46_128chip_deepswe_campaign/` contains
five-field outcome summaries, not raw trajectories. It cannot seed resume and
does not prove full washing. Final completion still requires:

```text
P46_EVAL_CENSUS_PASS tasks=1851 scheduled_identities=29616 unattempted=0
P46_EVAL_CAMPAIGN_PASS tasks=1851 n_sample=16 valid_trajectories=29616 logical_shards=58
```
