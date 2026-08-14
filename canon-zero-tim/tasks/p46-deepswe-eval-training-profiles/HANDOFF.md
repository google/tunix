# P46 full-washing execution handoff

## Current status

The next Q4 target run is the complete data-washing campaign, not another
standalone `l0/p0` smoke. The implementation commit is
`a989af34054434e6567f88e99b45ed67faf15a44` on baseline
`c33ba5f50d606210ca9f2c94fca003b63ea6e326`. A remote executor must fetch
`yuxzhang/canon-zero-tim`, record its exact current HEAD, and require that
`a989af34` is in its ancestry. Never modify or push `main`.

No cluster launch is authorized by this handoff alone. Render/dry-run/apply only
within the operator's explicit instruction.

## What the returned run proved

Evidence `p46e12804` ran source
`2c160bf931d4d94756f5200472de8070615c0e9f` on 128 TPU chips, Qwen3-4B-
Instruct-2507, DP16 x TP8. It passed the exact 1851-row clean-data join and
captured four tasks x N16 = 64 unique reward-only trajectories:

- 54 `SUCCEEDED`, nine `MAX_CONTEXT_LIMIT_REACHED`, one `MODEL_TIMEOUT`;
- seven reward-one and 57 reward-zero outcomes;
- 59 records accepted and five rejected by the old adapter policy; and
- about 21 minutes end to end, roughly ten minutes of which was repeated model
  initialization/JIT.

The trajectory itself showed two separate classes of behavior:

1. Q4 generated nonstandard tool syntax. That is model behavior and must count
   as an unsolved N16 outcome rather than trigger resampling.
2. Our greedy compatibility regex converted
   `<parameter=cmd=ls</parameter>` into a malformed double closing tag. That is
   a harness bug and is fixed in P46.6.

Absolute archived evidence:

```text
canon-zero-tim/tasks/p46-deepswe-eval-training-profiles/evidence/p46e12804/head.full.log
canon-zero-tim/tasks/p46-deepswe-eval-training-profiles/evidence/p46e12804/q4i16k-n16-128-eaa3d1e7f2987b72.p0.20260814T000028Z.jsonl
canon-zero-tim/tasks/p46-deepswe-eval-training-profiles/evidence/p46e12804/SHA256SUMS
```

This is returned debug evidence, not a washed list.

## P46.6 semantics

Q4 evaluation explicitly sets:

```text
action_compat_mode=q4_r2egym_xml_v2
evaluation_mode=reward_only
trajectory_mode=reward_only_no_logprobs
```

Compatibility v2 performs only deterministic repairs seen in returned data:

- inline values with or without a real tail closing tag;
- nested `parameter=path` key spelling; and
- top-level `view/create/str_replace/insert/undo_edit` mapped to
  `file_editor` plus the same command.

Contradictory commands are not guessed. Every trajectory stores raw
`model_response`, canonical executed `action`, repair count, action mode, and
model-action-error count. A malformed/ambiguous model tool call is a completed
model outcome, usually reward zero; it does not get a fresh sample. Genuine
`ENV_TIMEOUT`, `REWARD_TIMEOUT`, `FAILED`, or malformed trajectory structure
may retry only within the current wave's shared one-hour wall-clock budget.
Compatibility-layer-created corruption is a hard job failure.

Qwen3-32B is deliberately different: ordinary `SWEAgent()` and
`train_deepswe_nb.py` keep `action_compat_mode=strict_xml`. P46.6 does not
change Q32 parsing, sampler semantics, loss, reward, precision, optimizer, or
training geometry.

## Full campaign contract

| Field | Exact value |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| Admitted topology | 64-chip `4x4x4` or 128-chip `4x4x8` |
| Preferred next allocation | available admitted topology; current handoff example is 128 |
| Evaluation mesh on 128 | DP16 x TP8, all devices |
| Source tasks | reviewed 1851-row clean whitelist |
| Sampling | N16, temperature 1.0, top-p 1.0, top-k 0 |
| Context | max model length 20,480; response budget 16,384 |
| Agent budget | at most 50 model/environment steps |
| Physical wave | four tasks x N16 = 64 concurrent trajectories |
| Wave deadline | 3600 seconds, including cleanup margin contract |
| Runtime | one resident Q4/vLLM runtime across all waves |
| Campaign size | 58 logical shards, 463 waves, 29,616 identities |
| Last wave | three tasks x N16 = 48 identities |
| Logprobs/trainer/optimizer | absent; this is stock reward-only evaluation |
| Prefix cache | disabled |

The full campaign mode does not enlarge concurrency or weaken the one-hour
physical boundary. It only avoids initializing/compiling Q4 separately for
each group of 64. Every trajectory is appended, flushed, and fsynced before
the next result is accepted. A pod/job interruption is resumed by relaunching
the same published source SHA, topology, and run id.

## Publication/read-back gate

From a clean checkout after publication:

```bash
git fetch origin yuxzhang/canon-zero-tim
git switch --detach origin/yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse HEAD)"
test "$(git status --porcelain)" = ""

rg -n 'q4_r2egym_xml_v2|strict_xml|CANON_P46_FULL_CAMPAIGN' \
  examples/deepswe canon-zero-tim/cluster
rg -n 'trajectory.v5|config.v3|model_action_errors' \
  examples/deepswe/deepswe_eval_artifacts.py
bash canon-zero-tim/tests/p46_deepswe_profiles/run_cpu.sh
```

Required CPU marker:

```text
P46_DEEPSWE_PROFILES_CPU_PASS cases=40
```

The suite now contains 49 unittest cases even though the retained stable
release marker remains `cases=40`. It includes a complete-scale orchestration
test proving one runtime, 463 waves, 29,616 identities, and the final 48-row
wave. CPU PASS is not TPU/Kubernetes or campaign-completion evidence.

## Render the next full 128-chip washing JobSet

Use a new run id because config-v3/trajectory-v5 and the source fingerprint
must not resume v4 evidence:

```bash
TOPOLOGY=128
BASE=canon-zero-tim/cluster/jobset-256cluster-64chip.yaml
RUN_ID="p46q4wash-$(date -u +%Y%m%dT%H%M%SZ)"
CPU_NODEPOOL=deepswe-cpu-pool
TPU_NODEPOOL=<actual-4x4x8-nodepool>
MODEL_PVC=haoyugao-cpu-np-pvc

python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
  --base "$BASE" \
  --output "/tmp/p46-q4-wash-128-${RUN_ID}.yaml" \
  --workload q4-clean-eval \
  --topology "$TOPOLOGY" \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_ID" \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC" \
  --full-campaign
```

Before apply, verify the rendered manifest contains:

```text
canon.zero-tim/full-campaign: "1"
CANON_P46_FULL_CAMPAIGN=1
CANON_P46_EVALUATION_MODE=reward_only
CANON_P46_TOPOLOGY=128
CANON_P46_LOGICAL_SHARD_INDEX=0
CANON_P46_PHYSICAL_SHARD_INDEX=0
state-campaign
logs/campaign.log
4x4x8
32 Pathways workers
```

The renderer rejects parity plus full campaign, nonzero externally supplied
shard indices, Q4 topology 256, Q32 topology 128, floating images, or an
unreviewed whitelist digest. Do not hot-patch the YAML.

## Runtime interpretation

Healthy progress markers are:

```text
P46_EVAL_CAMPAIGN_WAVE_START ... pending=64 runtime_reused=1
P46_EVAL_TRAJECTORY ... completed=<n>/<wave-size>
P46_EVAL_CAMPAIGN_LOGICAL_PASS ... runtime_reused=1
P46_EVAL_CAMPAIGN_PASS tasks=1851 n_sample=16 valid_trajectories=29616 logical_shards=58 ...
[P46.EVAL.POSTFLIGHT] PASS
```

The following stop nonzero and preserve already fsynced artifacts:

```text
P46_EVAL_CAMPAIGN_WAVE_TIMEOUT ... resume_same_run_id=1
P46_EVAL_CAMPAIGN_LOGICAL_INCOMPLETE ...
```

On a wave timeout, first archive the full log and Kubernetes events and verify
sandbox cleanup. Relaunch the same manifest/run id only after the cause is
understood; exact completed identities will be skipped. Never change topology,
source SHA, model, sampling, or run id while calling it a resume.

`MAX_CONTEXT_LIMIT_REACHED`, `MAX_STEPS_REACHED`, `MODEL_TIMEOUT`, and signed
trajectory `TIMEOUT` are valid unsolved fixed-budget outcomes. They do not
block washing and must not be resampled. A raw bad Q4 action also does not
block the campaign merely because the tool rejected it; that is part of the
model's solve-rate measurement.

## Outputs and completion

Persistent root:

```text
/mnt/disks/linchai_data/deepswe_eval/<run-id>/outputs/trajectories/
/mnt/disks/linchai_data/deepswe_eval/<run-id>/outputs/reports/
/mnt/disks/linchai_data/deepswe_eval/<run-id>/outputs/campaign/
/mnt/disks/linchai_data/deepswe_eval/<run-id>/logs/campaign.log
```

The final washed lists are:

```text
outputs/campaign/p46-campaign.q4_learnable.jsonl
  exactly tasks with solved count 1/16 through 15/16

outputs/campaign/p46-campaign.q32_candidates.jsonl
  Q4 partial plus Q4 all-fail tasks; advisory input for later Q32 review

outputs/campaign/p46-campaign.all_pass.jsonl
outputs/campaign/p46-campaign.all_fail.jsonl
outputs/campaign/p46-campaign.summary.json
```

Do not declare washing complete from a 64-row trajectory file, a logical
summary, or `SUCCEEDED` counts. Completion requires the exact global PASS
marker, 29,616 valid identities, 1851 unique tasks, 58 immutable logical
summaries, verified referenced SHA-256 digests, and cleanup PASS.

Return the absolute trajectory/report/campaign paths, `wc -l`, SHA-256 for
every final manifest and summary, the complete campaign log, JobSet events,
and cleanup evidence. Preserve real trajectories so model/tool behavior can be
audited later.

## Claim ceiling

A successful Q4 campaign proves only Qwen3-4B clean-data evaluation under this
fixed budget and produces an advisory curriculum list. It does not prove
Qwen3-32B training, training/rollout alignment, optimizer correctness, or
production zero-TIM. Q32 remains a separate explicitly approved launch after
the washed list and its digests are reviewed.
