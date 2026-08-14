# P38 FrozenLake: P38s13a HOLD pending P38.2l instrument freeze

This runbook is diagnostic-only. It never launches FrozenLake full training,
evaluation, backward, optimizer commit, prefix cache, or unified KV. P48 is a
separate workstream and waits for its DP16 resources.

## Current fact

P38s12f completed the concurrency discriminator: concurrency 32 remained red
at 11 / 46,390 elements (`max_abs=0.16271209716796875`) with exact B-C and
sufficient depth. Do not run concurrency 32 again. Its replay artifacts were
not committed, so the current action is one stock concurrency-256 P38s13a with
the same numerical path and new GCS durability only.

P38.2k final-artifact GCS transport is locally green, but it uploads after the
workload returns. Do not execute the operator sequence below until
`phases/p38-2l-terminal-incident-capture.md` passes its mid-run snapshot and
one-host dress-rehearsal gates.

The frozen source must render this exact attempt-0 prefix:

```text
gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/<jobset>/attempt-0
```

Require `PREFLIGHT.json` before rollout. After the pod terminates, recover
`COLLECTED.json`, `SHA256SUMS`, `mismatch-capsule.npz`, and
`serving-capture.tar` directly from GCS. `COMPLETE.json` is written last and is
the only GCS completion marker. This supersedes the P38s12f launch instructions
below, which remain as historical provenance.

## Current P38s13a operator sequence

Run this only after P38.2l is locally complete, committed, and pushed. Do not
edit the rendered YAML.

```bash
set -euo pipefail
git fetch origin yuxzhang/canon-zero-tim
SOURCE_COMMIT="$(git rev-parse FETCH_HEAD)"
RUN_ID=p38s13a
WORKTREE="/tmp/canon-zero-tim-$RUN_ID"
OUT="/tmp/p38-serving-$RUN_ID"
EVIDENCE="/tmp/p38-return-$RUN_ID"
test ! -e "$WORKTREE"
test ! -e "$OUT"
test ! -e "$EVIDENCE"
git worktree add --detach "$WORKTREE" "$SOURCE_COMMIT"
cd "$WORKTREE"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git status --porcelain)"
mkdir -p "$EVIDENCE"
printf '%s\n' "$SOURCE_COMMIT" > "$EVIDENCE/source_commit.txt"

python3 canon-zero-tim/cluster/render_p38_serving_jobsets.py \
  --source-commit "$SOURCE_COMMIT" --run-id "$RUN_ID" \
  --output-dir "$OUT" --stock-only --max-concurrency 256 | \
  tee "$EVIDENCE/render.txt"
STOCK="$OUT/jobset-p38-serving-stock.yaml"
JOBSET="canon-p38-fl-stock-${RUN_ID}-${SOURCE_COMMIT:0:8}"
PREFIX="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/$JOBSET/attempt-0"
grep -Fq "value: $PREFIX" "$STOCK"
kubectl apply --dry-run=server -f "$STOCK" | \
  tee "$EVIDENCE/dry-run-stock.txt"
kubectl apply -f "$STOCK" | tee "$EVIDENCE/apply.txt"
```

Prove GCS access before waiting for the expensive rollout:

```bash
for unused in $(seq 1 180); do
  if gcloud storage ls "$PREFIX/PREFLIGHT.json" >/dev/null 2>&1; then
    echo "P38_GCS_PREFLIGHT_PASS $PREFIX"
    break
  fi
  sleep 10
done
gcloud storage ls "$PREFIX/PREFLIGHT.json"
```

After the JobSet becomes terminal, recover GCS independently of pod logs:

```bash
mkdir -p "$EVIDENCE/gcs"
gcloud storage rsync -r "$PREFIX" "$EVIDENCE/gcs"
test -s "$EVIDENCE/gcs/COLLECTED.json"
(cd "$EVIDENCE/gcs" && sha256sum -c SHA256SUMS --quiet)
if test -s "$EVIDENCE/gcs/COMPLETE.json"; then
  echo P38_GCS_COMPLETE
else
  echo P38_GCS_COLLECTED_ONLY
fi
```

Always collect the exact JobSet/pod YAML and events as a separate Kubernetes
bundle. `COLLECTED_ONLY` is valuable failure evidence but is not target
admission. A complete bundle must contain exact B-C, a successful journal
join, sufficient depth, full trajectory coverage, and the controlled-exit
contract before strict E0 begins.

## Historical P38s12f instructions

The bundle published under `p38s12b` used `--max_concurrency=256`, so account
it as P38s12a analysis-level evidence. It reproduced A-B red and exact B-C,
but outer postflight saw `rc=137` and the infrastructure archive was
incomplete. Do not call it the concurrency-32 arm and do not rerun it.

P38s12d then rendered concurrency 32 correctly but used source `bdc96818`,
whose FrozenLake recipe still hard-required 256. It failed before rollout with
`P32 FrozenLake geometry mismatch: {'max_concurrency': 32}` and has no
numerical verdict. The source selected below must contain
`validate_frozenlake_max_concurrency`; do not reuse P38s12d's YAML or SHA.

The directory published as P38s12e is not a new run. Its `head.full.log`
contains only P38s12d/source-`bdc96818` output: one 199-line pod log repeated
five times and one 113-line pod log repeated 360 times. Its pre-alignment file
is empty and its classification file concatenates five JSON objects. Checksums
verify transport of those wrong files, not experiment validity. Do not reuse
P38s12e artifacts, names, commands, pods, or conclusions.

## A. Completed one-host row-231 E0-lite

Do not rerun this arm. It completed with
`E0_LITE_ENVELOPE_NOT_REPRODUCED`: REF reproduced all 566 production-B action
values, while R0/R1 differed from production A at 470 values. The exact
numbers and hashes are in
`tasks/p38-pathways-decode-prefill-carrier/artifacts/p38_2j_row231_e0lite_0813.md`.

The command below is retained only for provenance:

On the authorized v5p host, from a clean source containing P38.2j:

```bash
set -euo pipefail
CAPSULE=/path/to/p38s12a-mismatch-capsule.npz
bash \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_frozenlake_replay.sh \
  "$CAPSULE" p38s12a_row231_e0lite 231
```

Require equal actor/engine weights, exact repeated R0/R1/REF measurements, a
working one-bit negative control, no backward, and zero optimizer commits.

Interpret exactly one result:

- `E0_LITE_REPRODUCED`: REF equals production B and R0 equals production A;
  proceed to strict E0 construction/first-divergence instrumentation.
- `E0_LITE_ENVELOPE_NOT_REPRODUCED`: REF equals B but R0 does not equal A;
  do not interpret R0/R1 operator counterfactuals. Capture missing live state.
- `E0_LITE_PREREQUISITE_FAILED`: source row/B-C/REF identity failed; repair the
  input or weight/reference contract before continuing.

E0-lite is mask-derived and never proves a production cause by itself. This
result blocks R0/R1 operator interpretation and the first-divergence walk.

## B. Pin one immutable source for P38s12f

```bash
set -euo pipefail
git fetch origin yuxzhang/canon-zero-tim
SOURCE_COMMIT="$(git rev-parse FETCH_HEAD)"
RUN_ID=p38s12f
WORKTREE="/tmp/canon-zero-tim-$RUN_ID"
BASE_OUT="/tmp/p38-serving-${RUN_ID}-intent256"
OUT="/tmp/p38-serving-$RUN_ID"
EVIDENCE="/tmp/p38-return-$RUN_ID"
test ! -e "$WORKTREE"
test ! -e "$BASE_OUT"
test ! -e "$OUT"
test ! -e "$EVIDENCE"
git worktree add --detach "$WORKTREE" "$SOURCE_COMMIT"
cd "$WORKTREE"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git status --porcelain)"
rg -q '^def validate_frozenlake_max_concurrency' tunix/rl/dp_workloads.py
git merge-base --is-ancestor 6c3938a6f2fe "$SOURCE_COMMIT"
mkdir -p "$EVIDENCE"
printf '%s\n' "$SOURCE_COMMIT" > "$EVIDENCE/source_commit.txt"
```

## C. Preflight intent-diff before apply

Render a same-source, same-run-id concurrency-256 intent baseline and the
concurrency-32 candidate. Never apply the baseline.

```bash
python3 canon-zero-tim/cluster/render_p38_serving_jobsets.py \
  --source-commit "$SOURCE_COMMIT" --run-id "$RUN_ID" \
  --output-dir "$BASE_OUT" --stock-only --max-concurrency 256 \
  > "$EVIDENCE/render-intent256.txt"
python3 canon-zero-tim/cluster/render_p38_serving_jobsets.py \
  --source-commit "$SOURCE_COMMIT" --run-id "$RUN_ID" \
  --output-dir "$OUT" --stock-only --max-concurrency 32 | \
  tee "$EVIDENCE/render.txt"

BASELINE="$BASE_OUT/jobset-p38-serving-stock.yaml"
STOCK="$OUT/jobset-p38-serving-stock.yaml"
cp "$BASELINE" "$EVIDENCE/rendered-intent256.yaml"
python3 \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/check_p38_intent_diff.py \
  --baseline "$BASELINE" --candidate "$STOCK" \
  --output "$EVIDENCE/intent-diff.json"
grep -q '"verdict": "PASS"' "$EVIDENCE/intent-diff.json"
cp "$STOCK" "$EVIDENCE/rendered-stock.yaml"
kubectl apply --dry-run=server -f "$STOCK" | \
  tee "$EVIDENCE/dry-run-stock.txt"
```

The intent gate permits only `--max_concurrency=256 -> 32` and its matching
attestation label. The candidate must also pin:

- 32 prompts, mini-batch four, eight generations, DP16xTP4;
- prefix cache disabled and stock `CANON_KV_UNIFIED=0`;
- capsule row cap 16;
- controlled diagnostic exit 42;
- minimum action logical-KV 1686;
- four bands `1536,1664,1792,1920,2048`;
- Attempt 0 / `maxRestarts: 0`.

Stop instead of editing rendered YAML.

## D. Apply candidate only and preserve every artifact

```bash
set -euo pipefail
JOBSET="canon-p38-fl-stock-${RUN_ID}-${SOURCE_COMMIT:0:8}"
HEAD_JOB="${JOBSET}-pathways-head-0"
if kubectl get jobset -n default "$JOBSET" >/dev/null 2>&1; then
  echo "refusing to reuse existing JobSet: $JOBSET" >&2
  exit 1
fi
kubectl apply -f "$STOCK" | tee "$EVIDENCE/apply.txt"
POD=""
for unused in $(seq 1 180); do
  POD="$(kubectl get pods -n default -l "job-name=$HEAD_JOB" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [ -n "$POD" ] && break
  sleep 10
done
test -n "$POD"
printf '%s\n' "$JOBSET" > "$EVIDENCE/jobset-name.txt"
printf '%s\n' "$POD" > "$EVIDENCE/head-pod-name.txt"
set +e
kubectl logs -n default -f "$POD" -c jax-tpu | \
  tee "$EVIDENCE/head.follow.log"
printf '%s\n' "${PIPESTATUS[0]}" > "$EVIDENCE/log-follow-rc.txt"
set -e
```

Wait for the exact JobSet/pod to become terminal; do not delete it. Then:

```bash
kubectl get jobset -n default "$JOBSET" -o yaml > "$EVIDENCE/jobset.final.yaml"
kubectl get pod -n default "$POD" -o yaml > "$EVIDENCE/head-pod.final.yaml"
kubectl describe pod -n default "$POD" > "$EVIDENCE/head-pod.describe.txt"
kubectl logs -n default "$POD" -c jax-tpu > "$EVIDENCE/head.full.log"
kubectl logs -n default "$POD" -c pathways-proxy > "$EVIDENCE/pathways-proxy.log" 2>&1 || true
kubectl logs -n default "$POD" -c pathways-rm > "$EVIDENCE/pathways-rm.log" 2>&1 || true
kubectl logs -n default "$POD" -c jax-tpu --previous > "$EVIDENCE/head.previous.log" 2>&1 || true
kubectl get events -n default \
  --field-selector "involvedObject.name=$POD" --sort-by=.lastTimestamp > \
  "$EVIDENCE/head-pod.events.txt"
```

Never substitute a tail/UI excerpt for `head.full.log`. Never write
`head.full.log` inside a polling loop and never append to it with `>>`.
`head.follow.log` is the streaming view; `head.full.log` is fetched exactly
once with `>` after the pod is terminal.

## E. Extract, verify, and seal the whole bundle

```bash
python3 \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/extract_p38_capsule.py \
  --log "$EVIDENCE/head.full.log" \
  --output "$EVIDENCE/${RUN_ID}-mismatch-capsule.npz"
python3 \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/extract_p38_serving_archive.py \
  --log "$EVIDENCE/head.full.log" \
  --output "$EVIDENCE/${RUN_ID}-serving-capture.tar"
sed -n 's/^\[CANON_PRE_ALIGN_ARTIFACT_JSON\] //p' \
  "$EVIDENCE/head.full.log" > "$EVIDENCE/pre-alignment.jsonl"
sed -n 's/^\[CANON_P38_SERVING_CLASSIFICATION_JSON\] //p' \
  "$EVIDENCE/head.full.log" > "$EVIDENCE/serving-classification.json"

# Semantic provenance gate. Byte hashes alone cannot detect a wrong run.
test "$(rg -c '^\[entrypoint\] JOBSET_ATTEMPT ' \
  "$EVIDENCE/head.full.log")" -eq 1
test "$(rg -c '^\[sync\] HEAD=' "$EVIDENCE/head.full.log")" -eq 1
grep -Fxq "[sync] HEAD=$SOURCE_COMMIT" "$EVIDENCE/head.full.log"
test "$(rg -c '^\[run\] cmd:' "$EVIDENCE/head.full.log")" -eq 1
rg -q '^\[run\] cmd: .*--max_concurrency=32( |$)' \
  "$EVIDENCE/head.full.log"
! rg -q 'p38s12[de]|bdc96818|P32 FrozenLake geometry mismatch|admitted P33 evidence path already exists' \
  "$EVIDENCE/head.full.log"
test -s "$EVIDENCE/pre-alignment.jsonl"
test -s "$EVIDENCE/serving-classification.json"
python3 - "$EVIDENCE/pre-alignment.jsonl" \
  "$EVIDENCE/serving-classification.json" "$SOURCE_COMMIT" <<'PY'
import json
import pathlib
import sys

pre_path = pathlib.Path(sys.argv[1])
classification_path = pathlib.Path(sys.argv[2])
source = sys.argv[3]
pre_records = [
    json.loads(line)
    for line in pre_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(pre_records) != 1:
  raise SystemExit(f"expected one pre-alignment record, got {len(pre_records)}")
classification = json.loads(classification_path.read_text(encoding="utf-8"))
if classification.get("verdict") != "PASS":
  raise SystemExit(f"serving classification not PASS: {classification}")
if classification.get("source_commit") != source:
  raise SystemExit(
      "classification source mismatch: "
      f"{classification.get('source_commit')} != {source}"
  )
PY
bash \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/seal_p38_evidence.sh \
  "$EVIDENCE" "$RUN_ID"
```

The sealer refuses missing Kubernetes/Pathways files, excludes `SHA256SUMS`
from its own manifest, and immediately validates every digest.

## F. Verdict

Require exact B-C, no capture errors, full 256-trajectory coverage, journal
joins for every selected row, controlled exit 42 accepted, and
`DEPTH_SUFFICIENCY min=1686 ... verdict=PASS`.

- Red A-B: concurrency 32 is insufficient to remove the carrier.
- Exact A-B: repeat one depth-sufficient concurrency-32 arm before claiming
  concurrency/churn is necessary.
- Depth below 1686, missing bundle items, `rc=137`, or failed intent-diff:
  `INCONCLUSIVE`; do not relaunch automatically.

Neither red nor repeated exact identifies RoPE, page lifecycle, cache content,
or another operator. Leave all P38 variables unset to roll back the diagnostic.
