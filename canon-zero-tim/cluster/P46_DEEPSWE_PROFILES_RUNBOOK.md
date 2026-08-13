# P46 DeepSWE evaluation and training profiles

P46 is the operator entrypoint for the current DeepSWE campaign. It maintains
three immutable workload families, and renders each family for either a
64-chip `4x4x4` slice or a 256-chip `4x8x8` slice. The renderer writes the
signed parameters directly into the JobSet; do not add a second shell override
layer.

This package has local CPU and direct one-host development evidence only; it
has no 64-chip or 256-chip target evidence. A rendered YAML is not a target
PASS. Do not apply it until the exact implementation commit has been published
to `origin/yuxzhang/canon-zero-tim`, read back, and separately approved for
launch.

Required implementation ancestry:

```text
e1b4009394c49ea015919bda0cfdb97c12c221b5
```

P46.5 true reward-only evaluation is newer than that published ancestry and is
currently **unpublished**. Do not infer its presence from `e1b40093`. Before
using the reward-only instructions below, require a later explicitly published
operator SHA containing `evaluation_mode=reward_only`, the true no-logprob
request construction, and `probe_reward_only_v5p.py`. Never apply a manifest
rendered from the dirty development worktree.

The remote branch may advance with documentation or returned evidence. Resolve
and record its exact current 40-character HEAD at execution time, and require
that the resolved HEAD contains the implementation commit above.

## Returned-run correction and required fix

The archived Qwen3-32B P34r03 run is a failed attempt, not a rollout PASS. It
constructed the correct DP16 x TP8 role meshes and returned this cardinality
marker:

```text
[P34.LOGPS_BATCH] configured_prompts=8 generations=8 execution_trajectories=64 observed_trajectories=64
```

However, all 64 returned records were clipped as `ENV_TIMEOUT`; all 64 hung
environment steps were killed only after their shared trajectory budget had
already become negative. The run then failed before forward/backward or an
optimizer commit with `KeyError: 'fsdp'`. Its trainer mesh was named `dp,tp`,
while `RLTrainingConfig.data_sharding_axis` still named `fsdp`.

The published implementation repair has two parts:

1. one shared rollout-batch deadline, per-trajectory deadline, bounded model
   request cancellation and bounded R2E cleanup; and
2. trainer input sharding derived from the actual leading trainer-mesh axis.

Every training launch after publication must emit exactly one pre-rollout
marker whose data axis is `dp` and whose mesh contains `dp,tp`:

```text
[DEEPSWE.DATA_SHARDING] PASS axes=('dp',) mesh=('dp', 'tp')
```

Absence of this marker, `axes=('fsdp',)` on a canonical DeepSWE profile, any
negative remaining timeout, or any `KeyError: 'fsdp'` means the wrong source
SHA/profile ran. Archive the manifest and logs and stop; do not retry the same
manifest. A log-prob cardinality marker proves only that the expected number of
objects reached the trainer-side boundary. It does not prove those objects are
complete, valid, solved, or capable of producing a nonzero advantage.

## Workload matrix

| Family | Model and purpose | Signed work | Hard batch boundary |
|---|---|---|---:|
| `q4-debug` | Qwen3-4B-Instruct-2507 end-to-end training debug | 16K response, B4 x G4, exactly 3 optimizer updates | 3600 s |
| `q4-clean-eval` | Qwen3-4B-Instruct-2507 clean-data evaluation | 16K response, logical 32 tasks x N16; physical 4 tasks x N16 | 3600 s per physical shard |
| `q32-train` | Qwen3-32B full training | 16K response, B8 x G8, exactly 1000 optimizer updates | 5400 s |

Both training families keep AdamW state on TPU
(`CANON_OPT_STATE_RESIDENT=1`, `CANON_P30_OPT_STATE_OFFLOAD=0`, and
`--no-optimizer-offload`). There is no automatic host-offload fallback.

Training uses separated rollout and trainer roles:

| Allocation | Per-role mesh | Per-role devices | Local trajectories | Global M |
|---|---|---:|---:|---:|
| 64 chips | DP4 x TP8 | 32 | Q4: 4; Q32: 16 | 1024 |
| 256 chips | DP16 x TP8 | 128 | Q4: 1; Q32: 4 | 4096 |

Evaluation has no trainer role and uses every visible device: DP8 x TP8 on 64
chips or DP32 x TP8 on 256 chips. Its semantic batch is still exactly four
tasks x 16 samples = 64 trajectories. Prefer 64 chips for evaluation when both
allocations are available; the 256-chip form exists so the same workload can
run when only that slice is schedulable.

## True reward-only evaluation candidate

`q4-clean-eval` has one configuration source:

```text
CANON_P46_EVALUATION_MODE=reward_only
```

That mode derives no sampled or prompt logprob request, no host logprob
extraction, no rescore, no trainer, no alignment and no optimizer. A caller
that supplies reward-only while enabling any trainer/alignment/logprob or
optimizer switch is rejected before the profile can overwrite the
contradiction. In vLLM, integer zero still requests a logprob; the only off
request is:

```text
SamplingParams.logprobs=None
SamplingParams.prompt_logprobs=None
```

Trajectory logprob fields are absent or `null`, never numeric `0.0`. Every
configuration, trajectory, task report and summary carries both:

```text
trajectory_mode=reward_only_no_logprobs
sampled_by=stock@<exact-source-sha>
```

It also carries `sampling_rng_mode=engine_global_sequential`. The TPU/JAX vLLM
backend rejects per-request seeds. Engine seed 42 drives an ordered RNG split
stream; `sample_nonce` is only a stable task/sample artifact identity. Do not
claim that each pair is independently replayable. The L2 one-host diagnostic
can restore the exact idle engine RNG snapshot before its two arms, but target
evaluation does not rewind the engine between requests.

Promotion is layered:

1. **L1 mechanical, mandatory:** exact mode/config, `None/None`, extraction
   bypass, legal schema, complete non-logprob fields, provenance, durable
   artifact and cleanup evidence.
2. **L2 token stream, diagnostic:** exact identity is preferred. A clean suffix
   divergence from one sample boundary is recorded as
   `LAW1_SUFFIX_DIVERGENCE` and does not block reward-only evaluation.
3. **L3 target statistics, mandatory for default promotion:** the same N16
   task/sample identities in observer and reward-only arms, exact paired
   McNemar/binomial PASS, plus valid trajectories/hour and Kubernetes cleanup
   comparison on 64 chips.

The direct-attached development gate passed Qwen3-4B DP1 x TP4 with one pinned
clean R2E Docker task:

```text
P46_REWARD_ONLY_ONEHOST_PASS l1=PASS l2=IDENTICAL_OBSERVER
report=/mnt/disks/tunix-data/deepswe-reward-only-evidence/reward-only-onehost-20260813T061510Z-696010/report.json
report_sha256=db3305413817ffe5c4d0085098475a12753cea6b698e15e4263b0c7d0835ba7c
```

It proved a real rollout, parser and `search` tool action, final reward 0, a
valid trajectory, null logprobs and zero residual Docker containers. The
median two-token diagnostic calls were 0.0330 s with logprobs versus 0.0310 s
reward-only; sampler payloads were 117 versus 70 bytes. This proves the flag
really changes request/extraction and payload. It does not prove a useful
cluster throughput gain, Kubernetes cleanup, TP8, DP8/DP32 or L3.

After publication, reproduce the local gate only on a direct-attached four-chip
v5p host with the pinned local model, dataset and R2E checkout:

```bash
bash canon-zero-tim/tests/p46_deepswe_profiles/run_onehost_reward_only_v5p.sh
```

The isolated smoke is one turn, 256 generated tokens per call and 512 total so
model, parser, tool, final reward and cleanup fit cheaply. Those limits do not
change the production evaluator, which remains 16K and 50 turns. Prefix cache
also remains off in the signed profile; do not combine optimizations until
their own gate is reviewed.

## Immutable data and sampling contract

All three families start from the reviewed clean whitelist:

```text
clean_data/final_filter_result/task_report_good_qwen3_128_retry_20260713_090141.jsonl
sha256=2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7
rows=1851
```

It joins exactly against `R2E-Gym/R2E-Gym-Subset`, split `train`, revision
`2e8108ff942f24fcb5686badfaf7f9a8808566d5`, with 4578 source rows. The
evaluator uses Qwen3-4B-Instruct-2507, temperature 1.0, top-p 1.0, top-k 0,
prefix cache off, engine seed 42 and a stable identity nonce for every
`(docker_image, sample_index)`. The nonce is not a per-request sampling seed.

The evaluation report classifies a task only after exactly 16 valid samples:

- `partial`: Q4 solved between 1 and 15 samples; written to `q4_learnable`;
- `all_fail`: Q4 solved 0 of 16; kept in the separate Q32 hard tier;
- `all_pass`: Q4 solved 16 of 16; reported, not used as a Q4 learning tier;
- `broken` or `incomplete`: never promoted as curriculum data.

`q32_candidates` is the union of `partial` and `all_fail`. These reports are
advisory artifacts. Neither report silently replaces the 1851-row training
whitelist. Using one in a later training run requires a new reviewed manifest
with its exact SHA-256.

## Evaluation lifecycle and artifacts

One logical report covers 32 tasks x 16 samples. It is executed as eight
physical JobSets, each four tasks x 16 samples with concurrency 64. The final
logical shard contains 27 tasks and therefore has only physical indices 0-6.
The full 1851-task campaign is 58 logical shards and 463 physical JobSets.

Every completed trajectory is appended and fsynced before another result is
accepted. Records contain the complete redacted trajectory, exact task/sample
identity, seed, status, reward, elapsed time, source SHA, data revision,
whitelist digest, client-image digest, topology, model and sampling
fingerprint. Resume accepts only an identical fingerprint and rejects duplicate
sample identities. A timed-out shard returns nonzero after preserving completed
records; relaunch the same logical and physical indices to resume.

Reports are immutable and digest-bearing. Concurrent final shard writers are
accepted only when their bytes are identical; any content drift is a hard
failure. Durable output is under the mounted disk at:

```text
/mnt/disks/linchai_data/deepswe_eval/<run-id>/trajectories/
/mnt/disks/linchai_data/deepswe_eval/<run-id>/reports/
/mnt/disks/linchai_data/deepswe_eval/<run-id>/logs/
```

The useful terminal markers are:

```text
P46_EVAL_SUBSHARD_PASS ...
P46_EVAL_LOGICAL_REPORT_PASS ... summary_sha256=<64 hex>
```

The first marker proves only one physical shard was safely persisted. The
second proves the complete exact-N logical report. Neither marker proves
training, Qwen3-32B quality, or production readiness.

## Fetch and verify before rendering

From the remote execution checkout:

```bash
git fetch origin yuxzhang/canon-zero-tim
git switch --detach origin/yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git ls-remote origin refs/heads/yuxzhang/canon-zero-tim | awk '{print $1}')"
test "$SOURCE_SHA" = "$REMOTE_SHA"
test -z "$(git status --porcelain)"
bash canon-zero-tim/tests/p46_deepswe_profiles/run_cpu.sh
```

Require `P46_DEEPSWE_PROFILES_CPU_PASS`. Also run the pinned-image gates named
in the P34/P44 runbooks with the actual registry-digest client image. Do not
print or modify `HF_TOKEN`, `WANDB_API_KEY`, or `.env`.

Before rendering, confirm the published checkout actually contains the
returned-run repair:

```bash
rg -n 'training_data_sharding_axis|DEEPSWE.DATA_SHARDING' \
  examples/deepswe/train_deepswe_nb.py
python3 -m unittest \
  canon-zero-tim/tests/p34_deepswe/test_script_contract.py \
  canon-zero-tim/tests/p44_deepswe_qwen4b_parity/test_integration_contract.py
```

The source must derive `training_data_sharding_axis` from
`train_axis_names[0]`; a hard-coded production `("fsdp",)` is stale and must
not launch.

## Render one JobSet

Set operator-specific inputs without embedding credential values:

```bash
TOPOLOGY=64
BASE=canon-zero-tim/cluster/jobset-64chip.yaml
RUN_ID=p46q4d01
CPU_NODEPOOL=deepswe-cpu-pool
TPU_NODEPOOL=mlperf-v5p-64-np-0
MODEL_PVC=haoyugao-cpu-np-pvc
```

For 256 chips, set `TOPOLOGY=256`, use
`canon-zero-tim/cluster/jobset-256cluster-64chip.yaml`, and select the `4x8x8`
worker node pool.

Render Qwen3-4B three-update debug:

```bash
python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
  --base "$BASE" \
  --output "/tmp/p46-q4-debug-${TOPOLOGY}.yaml" \
  --workload q4-debug \
  --topology "$TOPOLOGY" \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_ID" \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC"
```

Render one Qwen3-4B evaluation physical shard:

```bash
python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
  --base "$BASE" \
  --output "/tmp/p46-eval-${TOPOLOGY}-l0-p0.yaml" \
  --workload q4-clean-eval \
  --topology "$TOPOLOGY" \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_ID" \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC" \
  --logical-shard-index 0 \
  --physical-shard-index 0
```

Render the two validation-only 64-chip parity arms. They are one clean task x
N16 each and write into separate directories under the same run id:

```bash
test "$TOPOLOGY" = 64
for EVAL_MODE in logprob_observer reward_only; do
  python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
    --base "$BASE" \
    --output "/tmp/p46-parity-${EVAL_MODE}-64.yaml" \
    --workload q4-clean-eval \
    --topology 64 \
    --source-commit "$SOURCE_SHA" \
    --source-branch yuxzhang/canon-zero-tim \
    --client-image "$CLIENT_IMAGE_DIGEST" \
    --run-id "$RUN_ID" \
    --cpu-nodepool "$CPU_NODEPOOL" \
    --worker-nodepool "$TPU_NODEPOOL" \
    --model-pvc "$MODEL_PVC" \
    --logical-shard-index 0 \
    --physical-shard-index 0 \
    --evaluation-mode "$EVAL_MODE" \
    --parity-canary
done
```

The renderer rejects `logprob_observer` without `--parity-canary`, rejects the
canary on 256 chips, and rejects evaluation-only controls on either training
family. Apply each arm only with explicit launch approval. After both arms
return, obtain their JobSet wall times from Kubernetes rather than summing
per-trajectory latency, and build the promotion report:

```bash
PARITY_ROOT="/mnt/disks/linchai_data/deepswe_eval/$RUN_ID/parity"
python3 examples/deepswe/deepswe_reward_only_parity.py \
  --observer-jsonl \
    "$PARITY_ROOT"/logprob_observer/outputs/trajectories/*.jsonl \
  --reward-only-jsonl \
    "$PARITY_ROOT"/reward_only/outputs/trajectories/*.jsonl \
  --observer-wall-secs "$OBSERVER_WALL_SECS" \
  --reward-only-wall-secs "$REWARD_ONLY_WALL_SECS" \
  --output "$PARITY_ROOT/l3-report.json"
```

The classifier requires exactly the same 16 valid identities, one task, one
`sampled_by=stock@<SHA>`, sampled-token logprobs in every observer trajectory,
no numeric logprob in reward-only artifacts, and paired statistical PASS. It
also reports valid trajectories/hour for both arms.

Render Qwen3-32B full training:

```bash
python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
  --base "$BASE" \
  --output "/tmp/p46-q32-train-${TOPOLOGY}.yaml" \
  --workload q32-train \
  --topology "$TOPOLOGY" \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_ID" \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC"
```

The renderer refuses to overwrite an existing output, rejects floating images,
bad source SHAs, topology drift, unreviewed whitelist input, empty evaluation
shards, and any signed command-field change. A remote agent may render and
server-side dry-run these files after pulling the publication; applying them
still requires the operator's explicit launch approval.

## Promotion order and claim ceiling

The remote agent must advance one gate at a time. Both 64 and 256 chips are
first-class signed variants; use whichever allocation is available. Prefer 64
only when both are simultaneously available because it is cheaper, not because
it is a prerequisite. Keep one topology for a given resumable evaluation
run-id because topology is part of its fingerprint.

1. Before any full evaluation shard, require a clean published P46.5 SHA and
   complete the 64-chip paired N16 L3 canary. Compare identical task/sample
   identities with `classify_l3_paired_solve_rate`, require its exact paired
   verdict `PASS`, compare valid trajectories/hour, and prove all R2E pods were
   deleted in both arms. The current direct one-host PASS supplies L1/L2 only.
   Use only the validation-only manifests rendered above from the same SHA; do
   not use a historical solve rate as the control.
2. Run one `q4-clean-eval` physical shard at logical index 0 and physical
   index 0 on the available topology. The 64-chip form is DP8 x TP8; the
   256-chip form is DP32 x TP8. Both still evaluate exactly four tasks x N16
   with concurrency 64 and a one-hour boundary. Require
   `P46_EVAL_SUBSHARD_PASS` and
   `[P46.EVAL.POSTFLIGHT] PASS`, 64 unique `(task, sample_index)` records, full
   redacted conversations, finite rewards, no duplicate identity, and proof
   that every R2E pod was deleted. A timeout preserves resumable records but is
   not a PASS.
3. Manually inspect at least one successful and one failed trajectory from the
   persistent JSONL. Confirm that assistant actions alternate with real R2E
   observations, statuses agree with terminal events, and reward 1.0 is used
   only for a valid solved trajectory. Summary-only JSONL is insufficient.
4. Run `q4-debug` on the available topology for exactly three updates. The
   64-chip form splits into DP4 x TP8 rollout/trainer roles; the 256-chip form
   splits into DP16 x TP8 roles. Both retain B4 x G4, 16 trajectories and the
   one-hour shared batch boundary. Require the `dp` data-axis marker once,
   three `P44.LOGPS_BATCH` markers, three trajectory files and digests, three
   batch-metrics rows, finite nonzero gradient activity, train steps
   `0->1->2->3`, exactly three commits, device-resident optimizer state, at
   least 8 GiB classifier-observed HBM margin, and a P44 classifier JSON whose
   `verdict` is `PASS`.
5. Complete all 58 logical N16 evaluation reports through 463 resumable
   physical JobSets only if the curriculum report is wanted. Never classify a
   task from a partial N16 sample set.
6. Launch `q32-train` against the original clean 1851-row whitelist. Require
   the same `dp` data-axis marker before rollout, 16K response, B8 x G8, at most
   64 concurrent sandboxes, a 5400-second shared rollout-batch deadline, and
   TPU-resident optimizer state. Start at update 0 and let the signed profile
   run to 1000 updates; do not edit the rendered YAML to create an ad hoc
   one-update or context variant.

If Q4 evaluation exposes a cleanup leak or malformed trajectory, stop before
training. If Q4 three-update times out, OOMs, has a nonfinite value, lacks
gradient activity, fails optimizer placement/transaction checks, or fails its
classifier, stop before Q32. Do not switch to host optimizer, increase a
deadline, resample a zero-signal batch, inject reward, or loosen a gate inside
the same attempt.

For every attempted gate, return the exact source SHA, image digest, rendered
YAML plus SHA-256, run directory, full head/worker/R2E logs, JobSet events,
trajectory/report SHA-256 values, classifier JSON, optimizer placement, HBM
evidence and the first fatal traceback. Applying any JobSet still requires the
operator's explicit launch approval.

A 64-chip PASS proves the DP4 training carrier or DP8 evaluation carrier that
actually ran. A 256-chip PASS proves its DP16 training or DP32 evaluation
carrier. Results are functionally comparable because model, data, sampler,
loss, optimizer, batch and deadline semantics are identical, but they are not
bitwise or performance-equivalent across DP sizes. Local CPU gates and rendered
YAML prove no TPU, Pathways, R2E, HBM, convergence, or zero-TIM claim.
