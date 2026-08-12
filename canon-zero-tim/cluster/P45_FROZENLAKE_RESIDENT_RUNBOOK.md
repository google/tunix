# P45 FrozenLake DP8xTP8 resident full/eval runbook

This is the recommended 64-chip FrozenLake DP8xTP8 full-training route. It is
separate from the P42/P33 DP16xTP4 carrier and preserves those manifests for
diagnosis and historical comparison. Current branch defaults may place the
optimizer on device in both routes; topology and model overlay, not only
optimizer placement, distinguish P45.

P45 runs Qwen3-8B on one 4x4x4 v5p slice with:

```text
DP8xTP8
32 prompts x 8 generations = 256 trajectories
450 committed updates
CANON_OPT_STATE_RESIDENT=1
CANON_P30_OPT_STATE_OFFLOAD=0
```

The evaluation variant evaluates 100 held-out prompts x 8 generations every
10 policy steps. Finite FrozenLake alignment drift is warning-only; non-finite
values, OOM, placement drift, topology drift, failed optimizer transactions,
replica mismatch, Pathways failure, and W&B failure remain hard errors. This is
a convergence/throughput run, not proof that strict bitwise zero-TIM is closed.

Attempt `p45r3` did not test resident HBM: it selected the TP4-only `qwen8b`
overlay and failed during model import with a TP-size contract mismatch before
rollout or optimizer construction. The isolated `qwen8b_tp8` overlay now has a
pinned-image import/forward/VJP gate, but no 64-chip result has yet proved
resident HBM capacity or multi-update stability. The first committed update is
therefore still the capacity gate.

## 1. Fetch one immutable source

Run in one Bash shell. Use a new lowercase run ID of at most 16 characters.
Never place HF or W&B secret values in commands, manifests, logs, or handoffs.

```bash
set -euo pipefail
git fetch origin yuxzhang/canon-zero-tim
SOURCE_COMMIT="$(git rev-parse FETCH_HEAD)"
RUN_ID="p45r4"
WORKTREE="/tmp/canon-zero-tim-${RUN_ID}"
OUT="/mnt/disks/linchai_data/launch_manifests/${RUN_ID}"
EVIDENCE="/mnt/disks/linchai_data/launch_evidence/${RUN_ID}"

test ! -e "$WORKTREE"
test ! -e "$OUT"
test ! -e "$EVIDENCE"
git worktree add --detach "$WORKTREE" "$SOURCE_COMMIT"
cd "$WORKTREE"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git status --porcelain)"
mkdir -p "$OUT" "$EVIDENCE"
printf '%s\n' "$SOURCE_COMMIT" > "$EVIDENCE/source_commit.txt"
```

The source must include the isolated TP8 overlay and profile binding:

```bash
test -f canon-zero-tim/src/engine_shims/models/qwen8b_tp8/MANIFEST.sha256
grep -Fxq 'export CANON_MODEL_DIR_NAME=qwen8b_tp8' \
  canon-zero-tim/cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-resident.env
```

## 2. Run the fixed-image gate and render both variants

```bash
sudo docker image inspect tunix_frozenlake_image:vllm-tpu0.25.0 >/dev/null
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

The local gate is valid only if it ends with all of these markers:

```text
P45_QWEN8B_TP8_CONTRACT_SELFTEST_PASS cases=7/7 tp4_negative=1
P45_QWEN8B_TP8_IMPORT_PASS chain=linear_p22xk model=qwen8b_tp8 tp=8 sites=7
P45_QWEN8B_TP8_FORWARD_VJP_PASS ... forward_exact=1 vjp_exact=1
P45_QWEN8B_TP8_PROBE_PASS sites=7/7 padding=none tp=8
P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8
```

Do not use `render_p33_jobsets.py` for this launch. That renderer produces the
separate P42/P33 DP16xTP4 carrier.

## 3. Verify the rendered contract

Before applying either YAML, inspect it and require all of these values:

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

If either optimizer value is `0/1` instead of `1/0`, stop: the wrong renderer
or manifest was selected. Do not hand-edit generated YAML.

`CANON_MODEL_DIR_NAME` is resolved when the profile is sourced inside the pod,
not duplicated in the generated YAML. Before launch the source-level grep in
Section 1 must pass; at runtime require this exact line before model loading:

```text
[env] profile=qwen3-8b-dp8-tp8-frozenlake-resident model_dir=qwen8b_tp8
```

## 4. Choose and apply exactly one manifest

- Use `$EVAL` for full training plus held-out evaluation. This is the default.
- Use `$FULL` only when measuring pure training throughput without evaluation.
- Never apply both to the same slice.

```bash
TARGET="$EVAL"
kubectl apply -f "$TARGET" | tee "$EVIDENCE/apply.txt"
```

The first P45 capacity run deliberately keeps `maxRestarts: 0`: without a
checkpoint, a restart begins again at step 0 and can hide the first-attempt OOM.
Do not change this in the rendered YAML.

## 5. Prove device residency on the first update

The complete log must show:

```text
[entrypoint] JOBSET_ATTEMPT 0 (first attempt)
[P41.OPTIMIZER] before_reverse placement=device-resident memory_kind=device
[P41.OPTIMIZER] after_commit placement=device-resident memory_kind=device
[CANON_P33_DP8] update_step_committed train_steps=1
```

The update record must additionally contain:

```text
optimizer_placement=device-resident
optimizer memory kinds=["device"]
optimizer_h2d_seconds=0.0
optimizer_d2h_seconds=0.0
```

Also require finite HBM snapshots before reverse, after accumulation, and after
commit; a finite gradient and parameter delta; a valid optimizer transaction;
exact DP replicas; and no TPU OOM. Seeing `pinned-host-offload` is not an
optimization fallback—it violates the P45 placement contract.

For `$EVAL`, require exactly one evaluation enablement marker and finite
`[CANON_FROZENLAKE_P42_JSON]` summaries at policy steps `0,10,...,440`:

```text
[CANON_P33_EVAL] ENABLED workload=frozenlake cadence=10 held_out_rows=100 generations=8
```

## 6. Return evidence

Before deleting the JobSet or Pods, preserve the complete head log, rendered
YAML, source SHA, JobSet and Pod YAML, Pod describe/events, Pathways proxy/RM
logs, first-update JSON, alignment JSONL, evaluation summaries, and SHA-256
manifest. The exact commands and full return checklist are maintained in
[`../tasks/p45-frozenlake-dp8-tp8-resident/HANDOFF.md`](../tasks/p45-frozenlake-dp8-tp8-resident/HANDOFF.md).

## Rollback

Stop selecting `render_p45_frozenlake.py` for future runs and return to the
separate DP16xTP4 carrier when reproducing that topology. A running P42 JobSet
cannot be converted in place; relaunching P45 is a separate operator decision.
To restore strict alignment, make a reviewed renderer change setting
`CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY=0`; do not edit rendered YAML.
