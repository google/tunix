# P45 FrozenLake DP8xTP8 resident handoff

The canonical operator entry point is
[`../../cluster/P45_FROZENLAKE_RESIDENT_RUNBOOK.md`](../../cluster/P45_FROZENLAKE_RESIDENT_RUNBOOK.md).
The P42 evaluation runbook is a separate DP16xTP4 carrier and must not be used
for a new P45 DP8xTP8 launch. Optimizer defaults have changed over time, so do
not identify the carrier from placement alone; require the P45 profile,
DP8xTP8, and `model_dir=qwen8b_tp8` together.

## Purpose and claim boundary

P45 is the fast 64-chip FrozenLake production carrier requested for training:

- Qwen3-8B on one 64-chip v5p slice;
- DP8xTP8, not the existing DP16xTP4 debug/offload geometry;
- 32 prompts x 8 generations = 256 global trajectories;
- one global trajectory microbatch contains 8 trajectories, one per DP rank;
- each DP rank owns 32 trajectories and the optimizer transaction consumes 32
  ordered gradient groups;
- canonical logprob M is 256 per rank and `MIN_TOKEN_BUCKET=2048` globally;
- Adam state remains on TPU (`device-resident`), with no optimizer H2D/D2H
  round trip;
- learning rate `1e-6`, prompt/response limits 4096/2048, 450 updates;
- alignment drift is warning-only, but non-finite values, topology, placement,
  transaction, replica-equality, and OOM failures remain blocking.

This run is a convergence/throughput experiment. It is not evidence that the
remaining FrozenLake decode/prefill carrier is bitwise closed. The classifier
must report `convergence-only` when warning-only alignment is enabled.

Attempt `p45r5` from source `42139ffa` on 64 TPU (`DP8xTP8`, resident optimizer)
successfully passed model loading, compilation, rollout, and sustained training for
~60 hours (2.5 days), reaching `train_steps=47` and `[CANON_ALIGN] step=1535 verdict=PASS`.
The job terminated at `Sat, 15 Aug 2026 06:03:58 UTC` due to Linux kernel host RAM
exhaustion (`Exit Code: 137 (OOMKilled)` on container `jax-tpu`), having reached the
Kubernetes pod memory limit of 200GiB from multi-day trajectory/logging/cache accumulation.
TPU HBM remained fully healthy throughout. For subsequent full runs, increase the head pod
Host memory limit to 350GiB+ and incorporate periodic host GC cycles.

## Operator: fetch and verify one immutable source

Run from an existing clone in one Bash shell. Replace `p45r1` if that run ID
already exists; never overwrite a prior render/evidence directory.

```bash
set -euo pipefail
git fetch origin yuxzhang/canon-zero-tim
SOURCE_COMMIT="$(git rev-parse FETCH_HEAD)"
RUN_ID="p45r4"
WORKTREE="/tmp/canon-zero-tim-${RUN_ID}"
OUT="/tmp/p45-render-${RUN_ID}"
EVIDENCE="/tmp/p45-evidence-${RUN_ID}"

test ! -e "$WORKTREE"
test ! -e "$OUT"
test ! -e "$EVIDENCE"
git worktree add --detach "$WORKTREE" "$SOURCE_COMMIT"
cd "$WORKTREE"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git status --porcelain)"
mkdir -p "$EVIDENCE"
printf '%s\n' "$SOURCE_COMMIT" > "$EVIDENCE/source_commit.txt"
test -f canon-zero-tim/src/engine_shims/models/qwen8b_tp8/MANIFEST.sha256
grep -Fxq 'export CANON_MODEL_DIR_NAME=qwen8b_tp8' \
  canon-zero-tim/cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-resident.env
```

## Run the local admission gate, then render

Use the exact pinned image. Do not substitute a newer image and call the result
equivalent.

```bash
bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh \
  tunix_frozenlake_image:vllm-tpu0.25.0 | \
  tee "$EVIDENCE/local-gate.txt"

python3 canon-zero-tim/cluster/render_p45_frozenlake.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT" | tee "$EVIDENCE/render.txt"

FULL="$OUT/jobset-p45-frozenlake-full-dp8-tp8-resident.yaml"
EVAL="$OUT/jobset-p45-frozenlake-full-dp8-tp8-resident-eval.yaml"
test -s "$FULL"
test -s "$EVAL"
cp "$FULL" "$EVIDENCE/"
cp "$EVAL" "$EVIDENCE/"
kubectl apply --server-side --dry-run=server -f "$FULL" | \
  tee "$EVIDENCE/dry-run-full.txt"
kubectl apply --server-side --dry-run=server -f "$EVAL" | \
  tee "$EVIDENCE/dry-run-eval.txt"
```

Both manifests must contain these resolved values before either is applied:

```text
CANON_P32_WORKLOAD=frozenlake-dp8-tp8
CANON_PROFILE_FILE=cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-resident.env
CANON_P33_SHARED_MESH=8,8
CANON_DP_SIZE=8
CANON_TP_SIZE=8
CANON_LOCAL_TRAJECTORIES=32
MIN_TOKEN_BUCKET=2048
CANON_OPT_STATE_RESIDENT=1
CANON_P30_OPT_STATE_OFFLOAD=0
CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY=1
--train_trajectory_micro_batch_size=8
--vllm_max_num_seqs=32
--vllm_max_num_batched_tokens=256
--learning_rate=1e-6
--max_steps=450
```

The local gate must show seven `P45_QWEN8B_TP8_SITE_PASS` lines plus:

```text
P45_QWEN8B_TP8_CONTRACT_SELFTEST_PASS cases=7/7 tp4_negative=1
P45_QWEN8B_TP8_IMPORT_PASS chain=linear_p22xk model=qwen8b_tp8 tp=8 sites=7
P45_QWEN8B_TP8_FORWARD_VJP_PASS ... forward_exact=1 vjp_exact=1
P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8
```

At runtime, before model loading, require:

```text
[env] profile=qwen3-8b-dp8-tp8-frozenlake-resident model_dir=qwen8b_tp8
```

## Choose exactly one launch

- Fastest throughput run: apply `$FULL`; evaluation is disabled.
- Training plus held-out metrics: apply `$EVAL`; it evaluates 100 prompts x 8
  generations every 10 policy steps and is expected to be slower.

Do not apply both variants on the same slice. The user requested evaluation,
so `$EVAL` is the default choice unless current priority is pure throughput.

The first P45 evidence run is deliberately `maxRestarts: 0`. With no training
checkpoint, an automatic restart begins again at step 0 and can hide whether
the first attempt OOMed. After target capacity is proven, restart policy can be
changed in a separate reviewed CL; do not edit the rendered YAML in place.

```bash
TARGET="$EVAL"  # use "$FULL" only for the throughput-only variant
kubectl apply -f "$TARGET" | tee "$EVIDENCE/apply.txt"

JOBSET="$(python3 - "$TARGET" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1]))['metadata']['name'])
PY
)"
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
kubectl logs -n default -f "$POD" -c jax-tpu | \
  tee "$EVIDENCE/head.follow.log"
```

Do not use `--tail` for the canonical evidence log. The end of `logs -f` is not
proof that the JobSet is terminal.

## First-update capacity and placement gate

As soon as the first `[CANON_P33_DP8] update_step_committed` marker appears,
copy the live JSONL before the pod can terminate:

```bash
kubectl exec -n default "$POD" -c jax-tpu -- \
  sh -c 'test -s /tmp/canon-state/updates.jsonl && head -n 1 /tmp/canon-state/updates.jsonl' \
  > "$EVIDENCE/update-step0.json"
kubectl exec -n default "$POD" -c jax-tpu -- \
  sh -c 'test -s /tmp/canon-state/pre_alignment.jsonl && head -n 1 /tmp/canon-state/pre_alignment.jsonl' \
  > "$EVIDENCE/pre-alignment-step0.json"
kubectl exec -n default "$POD" -c jax-tpu -- \
  sh -c 'test -s /tmp/canon-state/alignment.jsonl && tail -n 32 /tmp/canon-state/alignment.jsonl' \
  > "$EVIDENCE/alignment-step0.jsonl"
```

The first update is admitted only if all of the following are present:

1. `[entrypoint] JOBSET_ATTEMPT 0 (first attempt)`;
2. 64 devices and shared mesh `8,8`;
3. local/global M `256/2048`, local trajectories 32, microsteps 32;
4. `[P41.OPTIMIZER] before_reverse placement=device-resident memory_kind=device`;
5. `[P41.OPTIMIZER] after_commit placement=device-resident memory_kind=device`;
6. `[CANON_P33_DP8] update_step_committed train_steps=1`;
7. `optimizer_placement=device-resident` and optimizer memory kinds `["device"]`;
8. `optimizer_timing.optimizer_h2d_seconds == 0.0` and
   `optimizer_d2h_seconds == 0.0`;
9. finite gradient/parameter delta, valid optimizer transaction, exact DP
   replicas, and no accumulator/reference mutation;
10. finite HBM snapshots at before-reverse, after-accumulation, and
    after-commit with no TPU OOM.

Throughput must be measured rather than assumed. P45 removes the optimizer
host round trip, but it also executes 32 ordered local gradient groups per
update instead of the old DP16 path's 16. Return both optimizer timing and the
full first-update wall time so the two effects can be separated.

Alignment differences may appear only as
`PASS_WITH_ALIGNMENT_WARNINGS`. A non-finite value, invalid shape/hash,
placement drift, transaction failure, replica mismatch, or OOM is a hard
failure and must not be downgraded to a warning.

For the evaluation variant, require exactly one
`[CANON_P33_EVAL] ENABLED workload=frozenlake cadence=10 held_out_rows=100 generations=8`
marker plus finite `[CANON_FROZENLAKE_P42_JSON]` summaries at policy steps
0,10,...,440. The plain full variant must instead attest evaluation disabled.

## Terminal evidence and return bundle

After the exact JobSet reaches Completed or Failed, do not delete it before
collecting evidence:

```bash
kubectl get jobset -n default "$JOBSET" -o yaml > "$EVIDENCE/jobset.final.yaml"
kubectl get pod -n default "$POD" -o yaml > "$EVIDENCE/head-pod.final.yaml"
kubectl describe pod -n default "$POD" > "$EVIDENCE/head-pod.describe.txt"
kubectl logs -n default "$POD" -c jax-tpu > "$EVIDENCE/head.full.log"
kubectl logs -n default "$POD" -c pathways-proxy > \
  "$EVIDENCE/pathways-proxy.log" 2>&1 || true
kubectl logs -n default "$POD" -c pathways-rm > \
  "$EVIDENCE/pathways-rm.log" 2>&1 || true
kubectl logs -n default "$POD" -c jax-tpu --previous > \
  "$EVIDENCE/head.previous.log" 2>&1 || true
kubectl get events -n default \
  --field-selector "involvedObject.name=$POD" \
  --sort-by=.lastTimestamp > "$EVIDENCE/head-pod.events.txt"
sha256sum "$EVIDENCE"/* > "$EVIDENCE/SHA256SUMS"
```

Return the complete `$EVIDENCE` directory. At minimum include source SHA, run
ID, rendered full/eval YAML, both dry runs, the applied variant, complete raw
log, step-0 update/alignment snapshots, final JobSet/pod YAML, describe/events,
Pathways proxy/RM logs, previous log if any, and SHA256SUMS. Also report the
final JobSet condition, pod exit code/reason, and whether evaluation was
enabled.

## Rollback

Do not mutate the old P33/P38 manifests. To abandon P45, stop rendering
`render_p45_frozenlake.py` and return to the existing DP16xTP4 profile. To
restore strict alignment for this carrier, set
`CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY=0` in a reviewed renderer change; do not
hand-edit generated YAML.
