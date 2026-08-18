# P38s21 terminal-discriminator operator runbook

This is the current P38 operator card. P38s20 is retired as
`INCONCLUSIVE_DURABILITY_SEAL_TIMEOUT`; do not relaunch it and do not use the
historical P38s18r2 runbook.

Read, in order:

1. `phases/p38-2u-terminal-discriminator.md` — scientific discriminator;
2. `phases/p38-2v-bounded-object-durability.md` — transport repair and gates;
3. this file — exact launch and return contract.

## 1. Publication prerequisite

Do not launch from a dirty checkout. The archive-transport implementation must
first pass every local command in P38.2v, then be committed and pushed only
after explicit user approval. Substitute the resulting full SHA below.

```bash
set -euo pipefail
git fetch origin yuxzhang/canon-zero-tim
SOURCE_COMMIT="<USER_APPROVED_FULL_SHA>"
test "$(git rev-parse FETCH_HEAD)" = "$SOURCE_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
```

Required source properties:

```bash
grep -Fq 'single-deterministic-tar-v1' \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/persist_p38_gcs.sh
grep -Fq 'ROUND_ARCHIVE.tar' \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/persist_p38_gcs.sh
grep -Fq '_TERMINAL_MAX_BYTES = 4 * 1024 * 1024 * 1024' \
  canon-zero-tim/cluster/render_p38_serving_jobsets.py
```

## 2. Render and launch exactly once

```bash
set -euo pipefail
RUN_ID=p38s21
OUT="/tmp/p38-serving-$RUN_ID"
test ! -e "$OUT"

python3 canon-zero-tim/cluster/render_p38_serving_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT" \
  --stock-only \
  --max-concurrency 256 \
  --seam-mode layer \
  --terminal-tail \
  --terminal-discriminator

YAML="$OUT/jobset-p38-serving-stock.yaml"
grep -Fq 'name: CANON_P38_TERMINAL_DISCRIMINATOR' "$YAML"
grep -Fq 'name: CANON_P38_TERMINAL_MAX_BYTES' "$YAML"
grep -Fq 'value: "4294967296"' "$YAML"
grep -Fq 'name: CANON_P38_DIAGNOSTIC_ROUNDS' "$YAML"
kubectl apply --dry-run=server -f "$YAML"
kubectl apply -f "$YAML"
```

Do not edit the YAML, enable prefix cache, enable backward, enable evaluation,
change concurrency, or reuse an old GCS prefix.

## 3. Live monitoring

The run is healthy only if these events occur in order for rounds 0, 1, 2:

```text
[CANON_P38] PRECHECK_ROUND_COMPLETE
[CANON_P38] ROUND_SEAL_REQUESTED
[P38.GCS] ROUND_COMPLETE ... remote_objects=3
[P38.GCS] LIVE_ROUND_PASS
```

After round 2 require:

```text
p38_terminal.classification.json written
controlled exit code 42
[P38.GCS] LIVE_COLLECT_PASS
[P38.GCS] LIVE_COMPLETE_PASS
[P38.GCS] LIVE_WORKER_COMPLETE
```

One missing event means `INCONCLUSIVE`; preserve attempt-0 logs and do not
blindly relaunch.

## 4. Verify and materialize a sealed round beside GCS

Raw NPZ archives stay in GCS. On the GCS-authorized machine, verify a round as
follows; repeat for `000000`, `000001`, and `000002`:

```bash
set -euo pipefail
ROUND=000000
PREFIX="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s21-${SOURCE_COMMIT:0:8}/attempt-0/rounds/$ROUND"
SCRATCH="$(mktemp -d /tmp/p38s21-$ROUND.XXXXXX)"

gcloud storage cp "$PREFIX/ROUND_ARCHIVE.tar" "$SCRATCH/ROUND_ARCHIVE.tar"
gcloud storage cp "$PREFIX/ROUND_COMPLETE.json" "$SCRATCH/ROUND_COMPLETE.json"
ARCHIVE_SHA="$(python3 - "$SCRATCH/ROUND_COMPLETE.json" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["archive_sha256"])
PY
)"
python3 \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_evidence_archive.py \
  verify --archive "$SCRATCH/ROUND_ARCHIVE.tar" \
  --expected-sha256 "$ARCHIVE_SHA"
python3 \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_evidence_archive.py \
  extract --archive "$SCRATCH/ROUND_ARCHIVE.tar" \
  --output "$SCRATCH/files"
(cd "$SCRATCH/files" && sha256sum -c SHA256SUMS --quiet)
```

## 5. Return contract for the analysis agent

Do not commit multi-gigabyte archives. Leave them at their immutable GCS
prefix. Commit or hand back the following small evidence only:

- full source SHA and exact rendered YAML SHA;
- attempt-0 head log from byte zero through controlled exit;
- all three pre-alignment JSON records;
- `p38_terminal.classification.json` and classifier stdout/stderr;
- three `ROUND_COMPLETE.json` files;
- root `COLLECTED.json`, `COMPLETE.json`, and `SHA256SUMS`;
- one text receipt per round containing GCS URI, archive SHA, manifest SHA,
  logical-file count, and successful `p38_evidence_archive.py verify` output.

The receiving agent must be able to distinguish these classifier outcomes:
`pre_lm_head_hidden`, `lm_head_logits`, `vocab_block_reduction`,
`logits_processing`, `processed_vocab_block_reduction`,
`production_tail_only`, mixed, or `INCONCLUSIVE`.

## 6. Stop conditions

- Any restart attempt other than Attempt 0: stop and archive as infrastructure
  evidence.
- Any archive/manifest mismatch: stop; never regenerate a manifest by hand.
- Any 900-second ACK timeout: return worker log plus the relevant archive size
  and transfer timestamps; do not merely increase the timeout.
- Any missing/ambiguous capsule red join: classifier result is
  `INCONCLUSIVE`, even if other rows look decisive.
