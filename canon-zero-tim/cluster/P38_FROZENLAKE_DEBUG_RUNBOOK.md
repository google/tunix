# P38 FrozenLake P38s12a request-journal runbook

This runbook launches one strict 64-chip stock diagnostic. It does not launch
full training, evaluation, backward, an optimizer commit, prefix cache, or
unified KV. The complete return/admission contract is
`../tasks/p38-pathways-decode-prefill-carrier/HANDOFF.md`; stop if the two
documents disagree.

## Why this run exists

P38s11 already proved that the full 32-prompt / 256-trajectory stock workload
is red while B-C remains exact. Its four global snapshots could be joined to
rows 199 and 206 offline, but they did not observe the rows at the mismatch
times. P38s12a keeps the known-red concurrency-256 workload and adds a bounded
host-only per-request journal. It also expands the capsule to eight rows and
uses four reachable bands: `1536,1664,1792,1920,2048`.

The journal records host scheduler state only. Its page generations are
observation generations, not allocator generations. This run cannot by itself
prove stale KV, page reuse, RoPE, residual/cast, or another numerical cause.

Do not rerun U/KV-unified; it was already materially red. Do not change
concurrency to 32 in this run. P38s12b is a later, separate single-variable
arm after this evidence is admitted.

## 1. Pin a clean published source

Run from an existing clean `google/tunix` clone after the user has reviewed,
committed, and published P38.2i:

```bash
set -euo pipefail
git fetch origin yuxzhang/canon-zero-tim
SOURCE_COMMIT="$(git rev-parse FETCH_HEAD)"
RUN_ID="p38s12a"
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
test -f \
  canon-zero-tim/patches/tpu_inference/13-tpu-runner-p38-request-journal.patch
test -f \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/phases/p38-2i-request-journal-concurrency-discriminator.md
mkdir -p "$EVIDENCE"
printf '%s\n' "$SOURCE_COMMIT" > "$EVIDENCE/source_commit.txt"
```

## 2. Render and dry-run stock only

```bash
python3 canon-zero-tim/cluster/render_p38_serving_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT" \
  --stock-only | tee "$EVIDENCE/render.txt"

STOCK="$OUT/jobset-p38-serving-stock.yaml"
test -f "$STOCK"
test ! -e "$OUT/jobset-p38-serving-unified.yaml"
cp "$STOCK" "$EVIDENCE/rendered-stock.yaml"
kubectl apply --dry-run=server -f "$STOCK" | \
  tee "$EVIDENCE/dry-run-stock.txt"
```

The YAML must have `CANON_KV_UNIFIED=0`, precheck-only, capsule rows 8,
request journal inside the capture directory, the four bands above,
`batch_size=32`, `mini_batch_size=4`, `num_generations=8`, `mesh_dp=16`, and
`maxRestarts=0`. Stop rather than editing a rendered value.

## 3. Apply and collect the full log

```bash
set -euo pipefail
kubectl apply -f "$STOCK" | tee "$EVIDENCE/apply.txt"
JOBSET="canon-p38-fl-stock-${RUN_ID}-${SOURCE_COMMIT:0:8}"
HEAD_JOB="${JOBSET}-pathways-head-0"
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

Wait until that exact JobSet/pod is terminal. Do not delete it and do not use
`--tail` or timestamps. Fetch again from byte zero:

```bash
kubectl get jobset -n default "$JOBSET" -o yaml > \
  "$EVIDENCE/jobset.final.yaml"
kubectl get pod -n default "$POD" -o yaml > \
  "$EVIDENCE/head-pod.final.yaml"
kubectl describe pod -n default "$POD" > \
  "$EVIDENCE/head-pod.describe.txt"
kubectl logs -n default "$POD" -c jax-tpu > \
  "$EVIDENCE/head.full.log"
kubectl logs -n default "$POD" -c pathways-proxy > \
  "$EVIDENCE/pathways-proxy.log" 2>&1 || true
kubectl logs -n default "$POD" -c pathways-rm > \
  "$EVIDENCE/pathways-rm.log" 2>&1 || true
kubectl logs -n default "$POD" -c jax-tpu --previous > \
  "$EVIDENCE/head.previous.log" 2>&1 || true
kubectl get events -n default \
  --field-selector "involvedObject.name=$POD" \
  --sort-by=.lastTimestamp > "$EVIDENCE/head-pod.events.txt"
```

## 4. Admit or reject the evidence

Require all of these in `head.full.log`:

- Attempt 0 and the exact source SHA;
- one standard INIT, positive OBSERVE, zero CAPTURE_ERROR;
- full 32-prompt / 256-trajectory coverage;
- four pre/post pairs for the registered bands;
- finite A-B red and exact B-C;
- positive request-journal markers;
- classifier PASS with every selected row journal-joined;
- mismatch capsule, classifier JSON, and serving archive payloads;
- terminal precheck accepted with backward=0 and optimizer_commits=0; and
- final PATHTRACE with U=0, journal>0, and coverage=1.

Anything missing is `INCONCLUSIVE`; do not relaunch automatically. A different
A-B count is allowed because trajectories are stochastic. B-C red, invalid or
nonfinite data, capture errors, missing coverage, backward, or optimizer
commit is a hard failure.

## 5. Extract and return everything

```bash
python3 \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/extract_p38_capsule.py \
  --log "$EVIDENCE/head.full.log" \
  --output "$EVIDENCE/p38s12a-mismatch-capsule.npz"
python3 \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/extract_p38_serving_archive.py \
  --log "$EVIDENCE/head.full.log" \
  --output "$EVIDENCE/p38s12a-serving-capture.tar"
sed -n 's/^\[CANON_PRE_ALIGN_ARTIFACT_JSON\] //p' \
  "$EVIDENCE/head.full.log" > "$EVIDENCE/pre-alignment.jsonl"
sed -n 's/^\[CANON_P38_SERVING_CLASSIFICATION_JSON\] //p' \
  "$EVIDENCE/head.full.log" > "$EVIDENCE/serving-classification.json"
test -s "$EVIDENCE/pre-alignment.jsonl"
test -s "$EVIDENCE/serving-classification.json"
tar -tf "$EVIDENCE/p38s12a-serving-capture.tar" | \
  grep -q './p38_request_journal.jsonl'
sha256sum "$EVIDENCE"/* | tee "$EVIDENCE/SHA256SUMS"
```

Return the entire `$EVIDENCE` directory. The next action is review, then exact
whole-vector E0 replay. Do not launch concurrency 32, RoPE repair, page poison,
or another U arm until that review selects it.

Rollback: leave all P38 capture variables unset. The diagnostic is
default-off and does not change ordinary training or evaluation.
