# P38 FrozenLake decode-versus-prefill debug runbook

This runbook launches the strict 64-chip P38 serving-envelope diagnostic. It
does **not** launch FrozenLake full training, evaluation, backward, an optimizer
commit, or unified KV. The canonical evidence-return contract is
`../tasks/p38-pathways-decode-prefill-carrier/HANDOFF.md`; if this short
runbook and that handoff disagree, stop and use the handoff.

## Current status

- Stock P38s1 and unified-KV P38u1 both reproduced a sparse A-B red while B-C
  remained exact. Unified KV is therefore not a sufficient repair and must not
  be rerun.
- P38s5 and P38s6 are nonterminal. The files re-added by `42139ffa` are exact
  duplicates of those already audited logs.
- P38s8 is only an interior compile excerpt. It proves module INIT but contains
  no byte-zero preamble, OBSERVE, capture record, alignment result, terminal
  state, classifier, archive, or postflight.
- A zero-OBSERVE result cannot be blamed on `min_prefix=1536` unless it comes
  from a terminal byte-zero log. The observer runs before prefix filtering.
- P38s10 reached a terminal numerical precheck, but it covered only four
  prompts / 32 trajectories (`N_action=2731`, solve ratio 1.0). Its exact A-B
  and B-C values are a subset PASS, not a carrier repair. Three typed-PRNG-key
  capture errors prevented an admitted serving archive.

The next unique run is `p38s11`, after the local P38.2g9 changes are reviewed
and published. It keeps four-prompt producer units so each unit is
DP16-divisible, but the consumer waits for all eight units. Alignment therefore
covers all 32 prompts / 256 trajectories and rejects a partial tail. Do not run
the current unpublished worktree on the cluster.

Two tempting substitutions are not admitted:

- A local v5p RoPE decode-shape/prefill-shape probe is a cheap screen, not a
  production-cause proof. It may prioritize the later seam walk but cannot
  skip the exact E0 replay gate.
- Do not add P38 variables to P45 full training. The current P38 environment
  requires backward-no-commit, precheck-only stop, four complete records, and
  fail-closed classifier/archive postflight. A production “shadow capture”
  needs its own default-off, nonblocking CL and tests before it is safe.

## 1. Fetch and pin the source

Run from an existing `google/tunix` clone. Do not run from a dirty checkout.

```bash
set -euo pipefail
git fetch origin yuxzhang/canon-zero-tim
SOURCE_COMMIT="$(git rev-parse FETCH_HEAD)"
git merge-base --is-ancestor \
  4a2cb8cd2bff2e1e9f5f82a6d2e0575d166759bd "$SOURCE_COMMIT"

RUN_ID="p38s11"
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
rg -q 'program_path="standard"' \
  canon-zero-tim/patches/tpu_inference/10-tpu-runner-p38-standard-capture.patch
rg -q '_p38_capture_leaf' \
  canon-zero-tim/patches/tpu_inference/12-tpu-runner-p38-prng-key-capture.patch
rg -q '_DIAGNOSTIC_UNITS = 8' \
  canon-zero-tim/cluster/render_p38_serving_jobsets.py
rg -q 'DIAGNOSTIC_COVERAGE_CONTRACT' \
  tunix/rl/agentic/agentic_rl_learner.py
mkdir -p "$EVIDENCE"
printf '%s\n' "$SOURCE_COMMIT" > "$EVIDENCE/source_commit.txt"
```

## 2. Render and dry-run both manifests

The renderer produces stock and unified manifests so its paired contract can
be validated. Only stock may be applied.

```bash
python3 canon-zero-tim/cluster/render_p38_serving_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT" | tee "$EVIDENCE/render.txt"

STOCK="$OUT/jobset-p38-serving-stock.yaml"
UNIFIED="$OUT/jobset-p38-serving-unified.yaml"
cp "$STOCK" "$EVIDENCE/rendered-stock.yaml"
kubectl apply --dry-run=server -f "$STOCK" | \
  tee "$EVIDENCE/dry-run-stock.txt"
kubectl apply --dry-run=server -f "$UNIFIED" | \
  tee "$EVIDENCE/dry-run-unified.txt"
```

Before apply, inspect the stock YAML and require:

```text
CANON_KV_UNIFIED=0
CANON_P38_PRECHECK_ONLY=1
CANON_P38_SERVING_CAPTURE_MAX_CALLS=4
CANON_P38_SERVING_CAPTURE_MIN_PREFIX=1536
CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS=1536,1792,2048,2304,2560
CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard
CANON_RUN_CMD: batch_size=32, mini_batch_size=4, num_generations=8, mesh_dp=16
maxRestarts: 0
```

## 3. Apply stock only and collect from process start

```bash
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
follow_rc="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$follow_rc" > "$EVIDENCE/log-follow-rc.txt"
```

Do not use `--tail` or `--timestamps`. A stopped `kubectl logs -f` stream is
not a terminal JobSet verdict. Wait until the exact JobSet/pod is Completed or
Failed, do not delete it, then fetch the canonical log again from byte zero:

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

## 4. Read the result without guessing

Use only `head.full.log` plus final JobSet/pod state:

| Evidence | Meaning |
|---|---|
| INIT=1, OBSERVE=0 | standard hook was not reached; prefix threshold is not the explanation |
| OBSERVE>0, observed maximum <1536 | current diagnostic traffic misses the registered prefix range |
| OBSERVE crosses a registered range, capture=0 | request/packed-row selection or mapping failed |
| any `CAPTURE_ERROR` | capture failed; numerical values do not admit D1 |
| no full 32-prompt/256-trajectory coverage marker | subset result; numerical verdict is workload-inconclusive |
| four pre/post pairs, no classifier/archive | capture worked; postflight/artifact transport failed |
| four pairs + classifier PASS + archive + terminal postflight | serving capture is admitted; proceed to E0 replay |

A numerical A-B red or exact result by itself is not a capture PASS. Even a
full-coverage exact result is one stochastic observation and must be repeated
before a repair claim. Missing evidence is
`INCONCLUSIVE`, never equality and never a root-cause finding. Page ownership,
stale block tables, RoPE, residual/cast seams, and scheduler lifecycle remain
hypotheses until the archive and exact replay localize the first divergence.

## 5. Return the whole directory

Return `$EVIDENCE` without editing its logs. At minimum include:

```text
source_commit.txt
render.txt
rendered-stock.yaml
dry-run-stock.txt
dry-run-unified.txt
apply.txt
jobset-name.txt
head-pod-name.txt
head.follow.log
head.full.log
jobset.final.yaml
head-pod.final.yaml
head-pod.describe.txt
head-pod.events.txt
pathways-proxy.log
pathways-rm.log
head.previous.log
```

If classifier/archive markers exist, also run the two extractors and return the
run-specific mismatch capsule, serving tar, classifier JSON, pre-alignment
JSONL, and `SHA256SUMS` exactly as specified in `HANDOFF.md`.

## After P38s9

- Admitted capture: run exact E0 replay against the same request and full
  action vector; only then localize RoPE/RPA/residual/logits seams or inspect
  page topology.
- Incomplete capture: make exactly one change selected by the table above.
  Do not rerun unified KV, lower the prefix bound speculatively, or start a
  repair arm.

Rollback is documentation-only. Runtime remains unchanged because all P38
capture controls are default-off and this runbook changes no manifest or
kernel.
