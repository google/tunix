# P58 Qwen3-4B DeepSWE native-first runbook

P58 retains a paired 128-chip study design. Each arm uses one `4x4x8` v5p
slice, synchronously split into a 64-device rollout role and a 64-device
trainer role. Both roles are DP8 x TP8. The two arms share data, seeds,
sampling, loss, optimizer, deadlines, artifacts, and update horizon.

- `native` preserves the stock serving/trainer numerical programs from the
  pinned DeepSWE quality-fix lineage. Finite A-B mismatch is the measured
  treatment dose. B-C, nonfinite values, invalid shapes, replica/transaction
  failures, and corrupt evidence remain fatal.
- `zero` enables the complete canonical numerical bundle. A, B, and C must be
  exact at every admitted boundary.

P58 does not modify `main`. Rendering and local validation do not authorize a
Kubernetes apply. An operator must separately approve image publication and
each launch.

Current execution decision (2026-08-21): run only the native three-update
canary. The optional one-host phase was explicitly waived, not passed. Zero is
deferred while its optimization work continues and must not be rendered or
applied. A native result cannot be reported as a paired comparison.

Attempt history: native `p58c01` Attempt-0 failed in `00_env.sh` before any
TPU program because the stock reduction admission was checked as canonical and
three FrozenLake-only zeros were unset. The failed root is immutable and has no
resumable trajectory state. The fix implementation commit is
`acd3136267214b367a6755d0ba28d80e883d6753`; use the final operator-branch
readback SHA that contains it and fresh run-id `p58c02`. Never reuse the
p58c01 YAML or root.

## 1. Frozen recipe

| Field | Value |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| Clean data | 1,012 promoted P46 tasks |
| Clean SHA-256 | `ec297c9cbc39cd67db15b0b9db6a229b15671b848df5ec3101de9ef8df7c9973` |
| Prompt batch / generations | B8 x G16 = 128 raw trajectories |
| Prompt / response / turns | 4,096 / 16,384 / 50 |
| Sampling | temperature 1.0, top-p 1.0, top-k 0 |
| Roles | rollout DP8 x TP8 + trainer DP8 x TP8 |
| Objective | RLOO; `sequence-mean-token-scale`; fixed norm 16,384 |
| PPO | epsilon 0.20, epsilon-high 0.28, beta 0 |
| Optimizer | Adam 1e-6, betas 0.9/0.99, weight decay 0.01, grad clip 1.0 |
| Optimizer placement | TPU device-resident; host offload forbidden |
| Update geometry | prompt mini-batch 8; 128 trajectory mini-batch; trajectory micro-batch 16; accumulation depth 8 |
| Optional interventions | sampler-IS off; group clip/filter off; degenerate masking off; flat-group resampling off |
| Prefix cache | off |
| Canary / campaign | exactly 3 commits / exactly 1,000 commits |

Compact filtering is part of the shared recipe. These terminal statuses are
journaled but get an all-zero policy mask:

```text
MAX_STEPS_REACHED
MAX_CONTEXT_LIMIT_REACHED
TIMEOUT
ENV_TIMEOUT
MODEL_TIMEOUT
REWARD_TIMEOUT
```

Partial filtering uses
`sum(mask * token_loss) / (B_eff * 16384)`. If all 128 rows are filtered,
the transaction is discarded without an optimizer commit and the next data
batch is consumed. It is not resampled. `batch_index` still advances while
`optimizer_step` does not; this separation makes the journal resumable and
prevents an all-filtered batch from overwriting the preceding artifact.

Timeout nesting is fixed: turn 300 s, step/reward 600 s, trajectory 3,000 s,
sandbox 3,300 s, cleanup 300 s, and the shared rollout-batch deadline 3,600 s.

## 2. Fetch, pin, and validate the published source

Only launch a clean, freshly read-back publication on
`yuxzhang/canon-zero-tim`. The publication SHA is deliberately resolved at
execution time so this versioned document never contains a stale or
self-referential commit ID.

```bash
git fetch origin yuxzhang/canon-zero-tim
git switch --detach origin/yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git ls-remote origin refs/heads/yuxzhang/canon-zero-tim | awk '{print $1}')"
test "$SOURCE_SHA" = "$REMOTE_SHA"
test "$(printf '%s' "$SOURCE_SHA" | wc -c)" -eq 40
test -z "$(git status --porcelain)"
```

Run the pinned-image gate with the exact launch image. A registry digest is
required by the renderer; a local Docker image ID is only suitable for the
local gate.

```bash
bash canon-zero-tim/tests/p58_deepswe_native_zero/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

Required terminal marker:

```text
P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1
```

That marker proves CPU/image wiring, not TPU execution, HBM, real R2E rollout,
native mismatch dose, or zero exactness.

Before rendering, verify read-only that the mounted PVC contains
`Qwen3-4B-Instruct-2507`, the R2E dependency imports, the clean JSONL has 1,012
lines, and its digest matches the frozen value. Never print secret values.

## 3N. Render and launch the native three-update canary

Use the exact source SHA, image digest, CPU pool, TPU pool, PVC, and a unique
run id. Never hand-edit rendered YAML. This phase permits only `native`.

```bash
CLIENT_IMAGE_DIGEST='registry.example/tunix@sha256:<64-hex-digest>'
CPU_NODEPOOL='deepswe-cpu-pool'
TPU_NODEPOOL='<4x4x8-v5p-nodepool>'
MODEL_PVC='haoyugao-cpu-np-pvc'
RUN_STEM='p58c02'
STAGE='three-update'

ARM='native'
OUTPUT="/tmp/p58-${ARM}-${STAGE}-${RUN_STEM}.yaml"
python3 canon-zero-tim/cluster/render_p58_deepswe_tim.py \
  --base canon-zero-tim/cluster/jobset-64chip.yaml \
  --output "$OUTPUT" \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_STEM" \
  --stage "$STAGE" \
  --arm "$ARM" \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC"
sha256sum "$OUTPUT"
kubectl apply --server-side --dry-run=server -f "$OUTPUT"
```

The renderer must emit
`P58_DEEPSWE_TIM_RENDER_PASS arm=native stage=three-update`.

The explicit launch boundary, only after operator approval, is:

```bash
kubectl apply -f /tmp/p58-native-three-update-${RUN_STEM}.yaml
```

Do not produce or apply a zero YAML in this phase. Preserve the exact native
YAML and digest with the returned run.

## 4. Evidence and canary interpretation

Each run root is:

```text
/mnt/disks/linchai_data/deepswe_zero_tim/<jobset-name>/
```

Important artifacts:

```text
run.log
weight_attestation.jsonl
pre_alignment.jsonl
alignment.jsonl
updates.jsonl
p58_deepswe_<arm>_<stage>.classification.json
debug/run_manifest.json
debug/batch_metrics.jsonl
debug/batch-<batch_index>.trajectories.jsonl.gz
```

The gzip files contain the complete redacted conversation/tool trajectory,
raw final reward, training reward, advantage, status, task identity,
`batch_index`, and `optimizer_step`. Inspect without editing:

W&B receives both counts and ratios for solved trajectories, all-solved,
all-failed, mixed, incomplete, and effective prompt groups, plus compact-filter
counts/ratios. `effective` and `nonzero_advantage` mean usable policy signal:
compact-filtered rows retain their raw advantage in the journal but do not
inflate these metrics. A separate `raw_nonzero_advantage_ratio` is retained for
audit.

```bash
RUN_ROOT='/mnt/disks/linchai_data/deepswe_zero_tim/<jobset-name>'
jq . "$RUN_ROOT/debug/run_manifest.json"
jq -c '{step,optimizer_step,trajectory_solve_ratio,all_solved_prompt_groups,all_failed_prompt_groups,mixed_prompt_groups,incomplete_prompt_groups,effective_prompt_groups,compact_filtered_trajectories,status_histogram}' \
  "$RUN_ROOT/debug/batch_metrics.jsonl"
gzip -cd "$RUN_ROOT/debug/batch-000000.trajectories.jsonl.gz" \
  | head -n 1 | jq .
jq . "$RUN_ROOT/p58_deepswe_<arm>_<stage>.classification.json"
```

Canary PASS requires exactly three committed update records. There may be more
than three trajectory batches if an entire batch was compact-filtered; every
such extra batch must have a zero-commit receipt, unchanged state, and the
same optimizer step as its successor. Any partial journal, missing digest,
duplicate/missing trajectory, wrong task identity, or non-signed filtered
status is fatal.

The native classifier also requires at least one finite, nonzero
`S_decode_vs_S_prefill` mismatch. Exact native A-B is `NO_TREATMENT`, not a
successful comparison. Native `S_prefill_vs_T_old` and
`T_old_vs_T_current` remain exact. The zero classifier requires all boundaries
exact. Both require device-resident optimizer evidence and no blocking reds.

Useful scan:

```bash
grep -aE '\[P58\.|CANON_ALIGN_PRE_JSON|CANON_ALIGN\]|COMPACT_FILTER|update_step_committed|optimizer_transaction|ONLINE_RUN_PASS|Traceback|OOM|RESOURCE_EXHAUSTED|CANCELLED|IFRT' \
  "$RUN_ROOT/run.log"
```

The run may not be promoted merely because Python exits zero. Require the
classification JSON verdict `PASS` and preserve its digest with the rendered
manifest and raw log.

## 5. Follow-up decision after the native canary

Return the packaged native canary and classifier before proposing another
launch. A native PASS does not itself authorize either full native training or
a zero canary. The user may separately choose one of those paths after review.

```bash
STAGE='full'
RUN_STEM='p58f01'
# Use section 3N only after a new explicit full-native approval.
```

Full-native classification requires exactly 1,000 optimizer commits.
Checkpoint cadence, evaluation cadence, artifact capacity, and analysis
sampling must be reviewed before apply. A later zero canary must restore the
paired invariants and pass its own strict classifier before any comparison.
P58 does not claim Qwen3-32B or 256-chip production readiness.

## 6. Stop and escalation rules

Stop rather than retrying the same manifest if any of these occurs:

- source/image/data digest drift;
- native has no observed mismatch dose;
- native B-C or any zero boundary differs;
- NaN/Inf, invalid shape, replica drift, optimizer/weight attestation failure;
- host optimizer offload, prefix cache, sampler-IS, group filtering, or flat
  resampling appears;
- fewer/more than 128 raw trajectory records in any batch;
- journal continuity or digest failure;
- sandbox cleanup failure, OOM, IFRT/CANCELLED, or deadline nesting drift; or
- classifier verdict is not `PASS`.

Archive the exact YAML, its SHA-256, source SHA, image digest, raw log,
artifacts, and classification. A failed prerequisite or interrupted target run
is `INCONCLUSIVE`, never PASS.
