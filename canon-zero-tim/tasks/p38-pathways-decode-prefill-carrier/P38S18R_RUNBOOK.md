# P38s18r2 GCS-side Round 0 analysis runbook — retired direct v1 flow

> **Do not execute the direct-classifier procedure below again.** Commit
> `a514c3bf` completed it and returned rc 1 at
> `duplicate seam token-prefix record`. The source inventory is closed, but
> overlapping observer records require deterministic alias auditing before
> classification. The current operator card is
> `P38S18R2_ALIAS_REDUCTION_RUNBOOK.md`; the governing phase is
> `phases/p38-2s-round0-alias-aware-seam-tail-reduction.md`. The remainder of
> this file is retained only to reproduce the failed v1 receipt.

This was the v1 operator card for P38s18r2. It remains historical provenance,
not an executable next step.

The local evaluator intentionally has no access to the producer's GCS
credentials. The raw seam/tail NPZ corpus is also too large to use as the
normal handoff object. Keep those raw bytes in GCS, run the registered
classifier on a GCS-authorized machine, and return a small SHA-sealed derived
bundle.

Do not use `run_reduce_p38s18l_on_gcp.sh` for this run. That wrapper is tied to
the older P38s18l `attempt-0/live/<snapshot>` layout and requires at least two
round capsules. P38s18r2 has one immutable round directory. Direct
whole-directory classification was attempted and failed on overlapping
token-prefix records; it must now be handled by the registered seam-plus-tail
reducer.

## 1. Current verdict and why the run stopped

P38s18r2/source `10fe951f0186256aa106627c4323de1f5aa168be` ran on 64 TPU
(`DP16xTP4`, concurrency 256). Round 0 completed the numerical precheck:

- `S_prefill_vs_T_old`: exactly 0 differing bytes;
- `S_decode_vs_S_prefill`: 45 differing bytes / 32 differing elements among
  45,559 actions, with `max_abs=0.10101699829101562` in the committed
  pre-alignment record;
- backward: 0; optimizer commits: 0.

It is not a completed three-round target run. The learner waited 900 seconds
for `round-000000.ack`, while the worker uploaded and then downloaded/verified
3,776 small files serially. The worker took about 57 minutes and wrote the ACK
only after the learner had raised `AlignmentGateError`; rounds 1 and 2 never
started. Root-level `COLLECTED.json`, `COMPLETE.json`, controlled exit 42, and
the three-round classifier contract are absent.

Overall run verdict:

```text
INCONCLUSIVE_DURABILITY_SEAL_TIMEOUT
```

The Round 0 numerical record is analysis-grade. The committed input directly
contradicts the old hand-written “all mismatches are on 256-token boundaries”
claim: only 1 of 32 mismatch elements has
`logical_kv_prefix_length % 256 == 0`. Do not cite a Pallas-boundary root cause
unless the official classifier independently measures it.

## 2. No-TPU remote analysis task

Run this only on a machine that can list and read the registered bucket. Use a
clean checkout containing the exact version of
`scripts/classify_p38_seam.py` being executed. Do not alter source objects and
do not overwrite an existing derived prefix.

```bash
set -euo pipefail

: "${REPO:?set REPO to a clean canon-zero-tim checkout}"
cd "$REPO"
test -z "$(git status --porcelain --untracked-files=no)"

ROUND_URI="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18r2-10fe951f/attempt-0/rounds/000000"
DERIVED_URI="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18r2-10fe951f/attempt-0/derived/p38s18r2-round0-seam-tail-v1"
CLASSIFIER="canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_seam.py"
SCRATCH="$(mktemp -d /tmp/p38s18r2-round0.XXXXXX)"
SOURCE="$SCRATCH/source"
RETURN="$SCRATCH/return"
mkdir -p "$SOURCE" "$RETURN"

command -v gcloud >/dev/null
test -s "$CLASSIFIER"
if gcloud storage ls "$DERIVED_URI/files/SHA256SUMS" >/dev/null 2>&1; then
  echo "REFUSING: immutable derived prefix already exists" >&2
  exit 3
fi

git rev-parse HEAD > "$RETURN/analysis_source_commit.txt"
sha256sum "$CLASSIFIER" > "$RETURN/classifier_source.sha256"
gcloud storage ls --recursive "$ROUND_URI/**" \
  > "$RETURN/OBJECT_LISTING.txt"
gcloud storage rsync --recursive "$ROUND_URI" "$SOURCE"
```

Before classification, require and verify the producer's round contract:

```bash
set -euo pipefail
for name in ROUND_COMPLETE.json ROUND_INVENTORY.json SHA256SUMS \
  mismatch-capsule.npz pre-alignment.jsonl; do
  test -s "$SOURCE/$name"
done
(cd "$SOURCE" && sha256sum -c SHA256SUMS --quiet)
cp "$SOURCE/ROUND_COMPLETE.json" "$RETURN/SOURCE_ROUND_COMPLETE.json"
cp "$SOURCE/ROUND_INVENTORY.json" "$RETURN/SOURCE_ROUND_INVENTORY.json"
cp "$SOURCE/SHA256SUMS" "$RETURN/SOURCE_SHA256SUMS"

python3 - \
  "$RETURN/SOURCE_ROUND_COMPLETE.json" \
  "$RETURN/SOURCE_ROUND_INVENTORY.json" \
  "$RETURN/SOURCE_SHA256SUMS" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

complete = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
inventory = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
manifest_sha = hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest()
assert complete["schema"] == "canon-p38-round-completion-v1"
assert complete["diagnostic_round"] == 0
assert complete["manifest_sha256"] == manifest_sha
assert inventory["schema"] == "canon-p38-round-stage-v1"
assert inventory["diagnostic_round"] == 0
assert inventory["seam_records"] == 915
assert inventory["tail_records"] == 971
print(json.dumps({
    "diagnostic_round": 0,
    "manifest_sha256": manifest_sha,
    "seam_records": inventory["seam_records"],
    "tail_records": inventory["tail_records"],
}, sort_keys=True))
PY
```

Run the official classifier exactly once over the complete remote Round 0
directory. `--require-tail` is mandatory; a layer-only answer is not the
registered s18r2 question.

```bash
set +e
python3 "$CLASSIFIER" \
  --directory "$SOURCE" \
  --capsule "$SOURCE/mismatch-capsule.npz" \
  --mode layer \
  --require-tail \
  --output "$RETURN/classification.json" \
  > "$RETURN/classifier.stdout" \
  2> "$RETURN/classifier.stderr"
CLASSIFIER_RC=$?
set -e
printf '%s\n' "$CLASSIFIER_RC" > "$RETURN/classifier.rc"
```

Do not convert a nonzero classifier return into a hand-written PASS. Preserve
the stderr and return the incomplete-input result as `INCONCLUSIVE`. If the
classifier succeeds, inspect the machine output rather than manually joining
records:

```bash
python3 - "$RETURN/classification.json" <<'PY'
import json
from pathlib import Path
import sys

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert record["status"] == "PASS"
assert record["red_points"] == 32
assert record["joined_red_points"] == 32
assert record["tail_observer_required_and_joined"] is True
print(json.dumps({
    "classification": record["classification"],
    "red_points": record["red_points"],
    "joined_red_points": record["joined_red_points"],
    "mixed_first_difference_signatures": record[
        "mixed_first_difference_signatures"],
    "first_difference_signatures": record[
        "first_difference_signatures"],
}, sort_keys=True))
PY
```

The `32` assertion is intentionally source-specific: it prevents a partial or
wrong capsule from silently producing a plausible classification.

Create a machine-readable wrapper verdict and provenance manifest. These do
not replace the official `classification.json`; they record whether it ran and
which immutable source/classifier bytes were used.

```bash
python3 - "$RETURN" "$ROUND_URI" "$CLASSIFIER_RC" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
round_uri = sys.argv[2]
classifier_rc = int(sys.argv[3])

def sha(name):
  return hashlib.sha256((root / name).read_bytes()).hexdigest()

classification_path = root / "classification.json"
classification = None
if classification_path.is_file():
  classification = json.loads(classification_path.read_text(encoding="utf-8"))

admitted = (
    classifier_rc == 0
    and classification is not None
    and classification.get("status") == "PASS"
    and classification.get("red_points") == 32
    and classification.get("joined_red_points") == 32
    and classification.get("tail_observer_required_and_joined") is True
)
manifest = {
    "schema": "p38s18r2-remote-analysis-manifest-v1",
    "source_round_gcs_uri": round_uri,
    "source_round_complete_sha256": sha("SOURCE_ROUND_COMPLETE.json"),
    "source_round_inventory_sha256": sha("SOURCE_ROUND_INVENTORY.json"),
    "source_round_manifest_sha256": sha("SOURCE_SHA256SUMS"),
    "object_listing_sha256": sha("OBJECT_LISTING.txt"),
    "analysis_source_commit": (root / "analysis_source_commit.txt").read_text(
        encoding="utf-8").strip(),
    "classifier_source_sha256": (root / "classifier_source.sha256").read_text(
        encoding="utf-8").split()[0],
    "classifier_rc": classifier_rc,
    "classification_sha256": sha("classification.json")
        if classification is not None else None,
    "raw_observer_npz_transport": "remote-only",
    "claim_grade": "analysis-grade",
}
(root / "REMOTE_ANALYSIS_MANIFEST.json").write_text(
    json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
verdict = {
    "schema": "p38s18r2-remote-analysis-verdict-v1",
    "verdict": (
        "ANALYSIS_GRADE_CLASSIFIED"
        if admitted else "INCONCLUSIVE_REMOTE_CLASSIFICATION"),
    "classifier_rc": classifier_rc,
    "red_points": classification.get("red_points")
        if classification is not None else None,
    "joined_red_points": classification.get("joined_red_points")
        if classification is not None else None,
    "classification": classification.get("classification")
        if classification is not None else None,
    "run_contract_complete": False,
}
(root / "verdict.json").write_text(
    json.dumps(verdict, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps(verdict, sort_keys=True))
PY
```

## 3. Small returned bundle

Raw `p38_seam_*.npz` and `p38_tail_*.npz` remain in GCS. Return only these
files:

```text
OBJECT_LISTING.txt
SOURCE_ROUND_COMPLETE.json
SOURCE_ROUND_INVENTORY.json
SOURCE_SHA256SUMS
analysis_source_commit.txt
classifier_source.sha256
classifier.stdout
classifier.stderr
classifier.rc
classification.json                  # only when the official classifier wrote it
REMOTE_ANALYSIS_MANIFEST.json
verdict.json
PACKAGING.txt
SHA256SUMS
```

Create and verify the small bundle manifest, then upload it to the immutable
derived prefix:

```bash
printf '%s\n' \
  'P38s18r2 Round 0 remote-classifier receipt.' \
  'Raw observer NPZ files remain under the source Round 0 GCS prefix.' \
  'This receipt is analysis-grade because the full three-round run did not complete.' \
  > "$RETURN/PACKAGING.txt"
(
  cd "$RETURN"
  find . -type f ! -name SHA256SUMS -printf '%P\0' \
    | sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS --quiet
)
gcloud storage rsync --recursive "$RETURN" "$DERIVED_URI/files"
echo "P38S18R2_REMOTE_ANALYSIS_READY path=$RETURN destination=$DERIVED_URI classifier_rc=$CLASSIFIER_RC"
```

The external agent must return the terminal marker above, the local `$RETURN`
path, the derived GCS URI, and the compact classification summary. It must not
commit or push until the user explicitly approves that separate action.

Because the local evaluator cannot access this GCS bucket, `OBJECT_LISTING.txt`
and the source round manifest are mandatory. A SHA manifest proves only the
objects it lists; it does not prove that the expected set was complete.

## 4. Decision table

| Remote result | Interpretation | Next action |
|---|---|---|
| Classifier PASS; all 32 red points join A/B seam and tail | Round 0 localizes the first measured divergent signature | Review `first_difference_signatures`; build the smallest operator-specific repair or focused observer. Do not rerun merely to collect the same data. |
| Classifier reports mixed signatures | More than one first-divergence site exists | Preserve every signature; do not collapse them to one root cause. |
| Missing record, ambiguous join, bad SHA, capsule mismatch, or nonzero classifier rc | `INCONCLUSIVE_REMOTE_CLASSIFICATION` | Return the complete small failure receipt. Decide whether a transport repair and fresh run are needed only after reviewing the missing-key inventory. |
| Round 0 classifier is sufficient but full three-round claim is desired | Scientific direction is available; target contract remains incomplete | Fix round transport, then run a new three-round ID. Never promote s18r2 itself to signed. |

## 5. Transport repair before any fresh three-round run

Do not make a longer timeout the primary fix. At the observed rate, three
rounds would spend hours serially uploading and downloading tiny objects.

Preferred repair: stage each round exactly as today, keep its inner
`SHA256SUMS`, package that immutable directory into one compressed archive,
upload the archive plus its digest, download the archive once, verify the outer
digest and every inner digest, then upload `ROUND_COMPLETE.json` last and write
the ACK. This preserves byte-level evidence while removing thousands of GCS
RPCs. A configurable timeout may be added as a safety margin only after the
bounded archive path is measured.

The repair is not admitted until fake-GCS tests cover: three distinct rounds,
archive corruption, missing inner files, wrong-round content, abrupt learner
exit, ACK-after-verification ordering, and an offline official-classifier
replay from an unpacked archive. Only then render a fresh run ID; never reuse
`p38s18r2` or its GCS prefix.

## 6. Claim ceiling

P38s18r2 proves one analysis-grade Round 0 alignment record and, if the remote
classifier passes, one analysis-grade first-divergence classification. It does
not prove a three-round stochastic signature, controlled exit, `COLLECTED`,
`COMPLETE`, a production repair, or zero-TIM closure.
