# P38 FrozenLake debug runbook

## Current operator card: P38s23 / P38.2x fixed lm-head

The current target is one 64-chip stock diagnostic, not FrozenLake full
training. P38s21 localized the first measured red interval to `lm_head_logits`;
P38s22 rejected the generic BF16/FP32 algorithm preset. P38s23 tests the first
constructive repair: both local M16 and M256 lm-head calls use one fixed
M256/K4096/N38144 Pallas tile geometry. It keeps prefix cache off, freezes
weights, performs zero backward and zero optimizer commits, and uses
concurrency 256.

The authoritative design and claim ceiling are in
`tasks/p38-pathways-decode-prefill-carrier/phases/p38-2x-fixed-tile-pallas-lm-head.md`.
The only copy/paste render/apply/return contract is
`tasks/p38-pathways-decode-prefill-carrier/P38S23_RUNBOOK.md`. Do not reuse an
old P38 YAML or add env values by hand.

Mandatory render flags:

```text
--stock-only --max-concurrency 256 --fixed-lm-head
```

The 2026-08-18 real-weight one-host receipt is construction-only: 4/4 fixed-M
comparisons exact and negative=1. Before applying, the clean source must have
the user-approved full SHA. On target, interpret only three independently
sealed endpoint rounds; missing round/root evidence is inconclusive.

Rollback is omission of `--fixed-lm-head`; the numerical flag then remains
unset. No production/full-training default changes.

## History: P38.2o decode seam localization

This runbook is diagnostic-only. It never launches FrozenLake full training,
evaluation, backward, optimizer commit, prefix cache, or unified KV. P48 is a
separate workstream and waits for its DP16 resources.

## Current fact and run admission

P38s17 has already run. Do not execute the historical P38s17 operator sequence
below. Reclassification from the six observer records and three immutable
round capsules returns `live_kv_fingerprint_equal_on_red_row` over every valid
extent. The committed directory lacks `COLLECTED.json` and `COMPLETE.json`, so
it is analysis-level evidence, not a terminal bundle.

The P38.2o local gates are complete:

1. immutable-round-only capsule selection and input-SHA provenance;
2. invalid-tail and valid-bit negative controls;
3. corrected P38s17 evidence and a verifying manifest; and
4. one-host endpoint-neutrality for the ordered decode seam observer.

The real Qwen3-8B DP1xTP4 layer rehearsal is endpoint-neutral and produced 130
bounded seam records. After review and source publication, render O2a with the
command in the CURRENT handoff: one stock DP16xTP4 `--seam-mode layer`
diagnostic, new run id `p38s18-layer`, concurrency 256, prefix cache off, three
frozen rounds, backward zero, and optimizer commits zero. No U, concurrency,
batch-size, E0-lite, full-training, or repair arm is allowed.

O2a finds the first divergent layer. Only then does O2b use
`--seam-mode full --seam-layer <N>` to find the internal checkpoint. The first
measured checkpoint chooses the repair. See
`phases/p38-2o-evidence-reconciliation-and-seam-walk.md`.

## Historical P38s17 operator sequence — do not rerun

P38s16 is complete and must not be rerun. The exact host audit joins all 60
mismatch elements and identifies one natural single-active red call (4223) at
the unchanged production fixed-M geometry. The archive has no live KV bytes,
so it cannot decide stale content versus a decode-program seam.

P38.2n N3 is locally complete. Patch 16 captures bounded live A and exact
same-prefix clean B KV fingerprints with one shared callable. The real
Qwen3-8B DP1xTP4 r6 rehearsal produced exactly three A/B pairs, exact
token/extent/provenance joins, and
`observer_pairs_valid_red_join_pending`, with no backward or optimizer commit.
Both pinned model overlays verify all 30 manifest entries and pass 29 runner
tests. Details are in:

```text
canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/
  phases/p38-2n-live-kv-content-discriminator.md
  artifacts/p38_2n_kv_observer_onehost_0815.md
```

The worker-owned terminal protocol is part of that admission: the worker must
ACK `collect`, all postflight checks must pass, then it must ACK `complete`.
`COLLECTED` without `COMPLETE` is useful crash/failure evidence, not admission.

Exactly one new production-shape stock run is admitted after the reviewed
worktree is committed and pushed. Do not run from an uncommitted tree. Do not
render unified KV, concurrency 32, E0-lite, backward, full training, or any
repair arm.

## P38s17 operator sequence — historical stock live-KV discriminator

Run from a clean clone after publication:

```bash
set -euo pipefail
git fetch origin yuxzhang/canon-zero-tim
SOURCE_COMMIT="$(git rev-parse FETCH_HEAD)"
RUN_ID=p38s17
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
rg -q '16-tpu-runner-p38-kv-observer.patch' canon-zero-tim/install.sh
rg -q 'observer_pairs_valid_red_join_pending' \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_kv_observer.py
mkdir -p "$EVIDENCE"
printf '%s\n' "$SOURCE_COMMIT" > "$EVIDENCE/source_commit.txt"

python3 canon-zero-tim/cluster/render_p38_serving_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT" \
  --stock-only \
  --max-concurrency 256 | tee "$EVIDENCE/render.txt"

STOCK="$OUT/jobset-p38-serving-stock.yaml"
test -s "$STOCK"
test ! -e "$OUT/jobset-p38-serving-unified.yaml"
cp "$STOCK" "$EVIDENCE/rendered-stock.yaml"
grep -Fq 'name: CANON_P38_KV_OBSERVER_DIR' "$STOCK"
grep -Fq 'name: CANON_P38_KV_OBSERVER_MAX_CANDIDATES' "$STOCK"
grep -Fq 'value: "3"' "$STOCK"
grep -Fq -- '--max_concurrency=256' "$STOCK"
grep -Fq 'name: CANON_KV_UNIFIED' "$STOCK"
kubectl apply --dry-run=server -f "$STOCK" | \
  tee "$EVIDENCE/dry-run-stock.txt"
kubectl apply -f "$STOCK" | tee "$EVIDENCE/apply.txt"
```

The terminal bundle is valid only if all of these hold:

```text
[CANON_P38_KV_OBSERVER_INIT] exactly 1
[CANON_P38_KV_OBSERVER_CANDIDATE] exactly 3
[CANON_P38_KV_OBSERVER_RECORD] arm=A exactly 3
[CANON_P38_KV_OBSERVER_RECORD] arm=B exactly 3
[CANON_P38] PRECHECK_ROUND_COMPLETE exactly 3
B-C exact in every round
backward=0 and optimizer_commits=0
p38_kv_observer.classification.json status=PASS
red_joins >= 1
COLLECTED.json and COMPLETE.json both present
SHA256SUMS verifies
```

Interpret only the classifier:

- `live_kv_fingerprint_differs_on_red_row`: localize the first differing
  layer/logical page/prefix extent and repair that cache writer/lifecycle path.
- `live_kv_fingerprint_equal_on_red_row`: reject KV content for that observed
  incident and start the ordered in-situ decode seam walk.
- `observer_pairs_valid_red_join_pending`, missing pairs, missing terminal
  markers, or no red join: INCONCLUSIVE; do not claim a cause or launch a
  repair.

## Historical fact

P38s15/source `58a0ed84` successfully executed all three Frozen-Weight
diagnostic rounds (768 trajectories total, 51,330 action tokens) on 64 TPU
(`DP16xTP4`, concurrency 256) with zero backward, zero optimizer commits, and
controlled exit 42. It measured exact B-C (0 mismatches, bitwise identical)
and measured A-B red at 20 / 51,330 elements (`33` differing bytes,
`max_abs=0.203777`). Mismatch rows `rows=[215, 223, 231, 254, 255]` were
captured along with 1,915 incident ledger records (53.3 MB). Evidence is
archived under `evidence/p38s15/`. The next step is single-host strict E0
replay on rows 215 / 223.

P38.2l is locally green. It adds immutable live GCS snapshots, three
frozen-weight diagnostic rounds, all-red-row capsules, and a round-scoped
exact-call incident ledger. A real Qwen3-8B DP1xTP4 capture-on/off rehearsal
proved observer neutrality and no backward/optimizer commit. Execute the
operator sequence below only from a clean immutable source containing P38.2l.

The frozen source must render this exact attempt-0 prefix:

```text
gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/<jobset>/attempt-0
```

Require `PREFLIGHT.json` before rollout. While it runs, immutable
`live/NNNNNN/LIVE.json` snapshots preserve changed host evidence every 30
seconds. After termination, recover `COLLECTED.json`, `SHA256SUMS`, the stable
and per-round mismatch capsules, classification, and serving archive directly
from GCS. `COMPLETE.json` is written last and is the only completion marker.
This supersedes the P38s12f launch instructions below, which remain as
historical provenance.

## Current P38s15 operator sequence

Run this only from a clean source containing P38.2l. It performs three
rollout/alignment rounds with frozen weights: 768 trajectories total,
zero backward, zero optimizer commits, then controlled exit 42. Do not edit the
rendered YAML.

```bash
set -euo pipefail
SOURCE_COMMIT=dc529871d7654ad1ec2cdefe1e4d50e07824393c
P38_2L_COMMIT=bd3090154ee894354e5c09e88b3a76825488aa3d
git fetch origin yuxzhang/canon-zero-tim
git cat-file -e "$SOURCE_COMMIT^{commit}"
git merge-base --is-ancestor "$P38_2L_COMMIT" "$SOURCE_COMMIT"
test -z "$(git diff --name-only "$P38_2L_COMMIT..$SOURCE_COMMIT" -- \
  ':(exclude)canon-zero-tim/tasks/**/evidence/**')"
RUN_ID=p38s15
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
git merge-base --is-ancestor "$P38_2L_COMMIT" HEAD
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

Before treating the run as started, the head log must report the exact pinned
source and the P38.2l live worker:

```text
[sync] HEAD=dc529871d7654ad1ec2cdefe1e4d50e07824393c
[P38.GCS] LIVE_WORKER_LAUNCHED
```

After requests enter the registered prefix interval, require at least one
`[CANON_P38_INCIDENT_LEDGER]` marker. If any of these markers is absent, stop
and classify the launch as wrong-source/instrumentation-inconclusive.

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
and round-scoped incident-ledger join for every selected red row, sufficient
depth, three pre-alignment rows, three round markers, full trajectory coverage,
and the controlled-exit contract before strict E0 begins. Return all files;
do not send only a UI excerpt or the final JSON line.

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
