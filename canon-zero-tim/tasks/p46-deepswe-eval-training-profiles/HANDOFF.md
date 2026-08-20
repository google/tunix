# P46 full-washing execution handoff

## Current status

The next Q4 target run is the complete data-washing campaign, not another
standalone `l0/p0` smoke. Resume-tag hardening and frozen legacy-v5 adoption
are published as implementation commit
`c3a960acdc94173440144559bb95f1de36d31537`. Publication checkpoint
`dc6b5b32a90ad0e12b1b9ae50ef7cc060b450abf` was read back from
`origin/yuxzhang/canon-zero-tim` with that commit in its ancestry. A remote
executor must resolve the exact current branch HEAD and repeat the ancestry
gate before rendering. Never modify or push `main`.

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

The later 128-chip `p46e12805` campaign uses source
`18d5d2ac1603a26a221af9d5fc430b084ec002df`, config-v3/trajectory-v5 and the
fixed `q4_r2egym_xml_v2` adapter. The repository archive contains its first
64 valid identities: ten reward one and 54 reward zero. Its head log reached
`logical_shard=0 physical_shard=1`, so it had started the next wave. The
operator reports the remote job is still running. **Do not stop it and do not
snapshot its live directory.** Repository evidence does not establish its
current live row count.

Archived transition evidence:

```text
canon-zero-tim/tasks/p46-deepswe-eval-training-profiles/evidence/p46e12805/head.full.log
canon-zero-tim/tasks/p46-deepswe-eval-training-profiles/evidence/p46e12805/q4i16k-n16-128-0d06152434768e31.p0.20260814T020445Z.jsonl
canon-zero-tim/tasks/p46-deepswe-eval-training-profiles/evidence/p46e12805/SHA256SUMS
```

## P46.6 semantics

Q4 evaluation explicitly sets:

```text
action_compat_mode=q4_r2egym_xml_v2
evaluation_mode=reward_only
trajectory_mode=reward_only_no_logprobs
resume_tag=<stable campaign identity>
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
the next result is accepted. `resume_tag` identifies the durable campaign;
`run_id` identifies one Kubernetes launch. A pod/job interruption is resumed
with the same resume tag, harness SHA, sampling-source SHA, topology,
model/data/sampling contract, and either the original manifest or a new launch
run id. A complete fsynced trajectory is skipped. A trajectory interrupted
before its complete JSONL row
is durable restarts from the beginning; token-by-token continuation is not
claimed.

The evaluator writes an immutable `resume_contract.json` before TPU model
initialization and holds an exclusive `flock` lease for the campaign. A second
writer with the same tag fails closed. Each launch gets isolated setup state
and an immutable attempt log. Before a resumed full campaign creates new
sandboxes, it deletes and confirms only R2E pods carrying the same resume-tag
label. The original harness SHA is checked out even if the operator branch has
advanced, after proving it remains in that branch's ancestry.

For a reviewed legacy adoption, `source_commit` remains the old sampling
lineage and `sampled_by=stock@<old-sha>`; `harness_commit` is the exact new
resume-capable checkout. Both are fingerprinted. This is the only admitted
cross-schema path and does not authorize arbitrary cross-SHA mixing.

## Publication/read-back gate

From a clean checkout after publication:

```bash
git fetch origin yuxzhang/canon-zero-tim
git switch --detach origin/yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse HEAD)"
test "$(git status --porcelain)" = ""
git merge-base --is-ancestor \
  c3a960acdc94173440144559bb95f1de36d31537 "$SOURCE_SHA"

rg -n 'q4_r2egym_xml_v2|strict_xml|CANON_P46_FULL_CAMPAIGN' \
  examples/deepswe canon-zero-tim/cluster
rg -n 'trajectory.v6|config.v4|resume_tag|model_action_errors' \
  examples/deepswe/deepswe_eval_artifacts.py
bash canon-zero-tim/tests/p46_deepswe_profiles/run_cpu.sh
```

Required CPU marker:

```text
P46_DEEPSWE_PROFILES_CPU_PASS cases=65
```

The suite includes complete-scale orchestration, torn-tail recovery,
17-of-64 interruption followed by a 47-identity resume, immutable contract,
single-writer lease, attempt-log, full-campaign postflight, and digest/fingerprint
checked v5-to-v6 adoption across logical shards. CPU PASS is not
TPU/Kubernetes or campaign-completion evidence.

## Transition `p46e12805` without stopping it

The old job remains authoritative while it is running. Wait for natural
termination, archive JobSet status/events, and prove no producer/sandbox pod
remains. Then copy, never move, its complete trajectory tree into the exact
staging path below under a **fresh** resume tag:

```text
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/imports/p46e12805/trajectories/
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/imports/p46e12805/SHA256SUMS
```

The copy/seal commands and terminal-state checklist are in
`canon-zero-tim/cluster/P46_DEEPSWE_PROFILES_RUNBOOK.md` under “Preserve and
adopt the running p46e12805 campaign”. Do not place v5 JSONL directly in the
new `outputs/trajectories/` directory. Do not use a resume tag that already has
v6 trajectory evidence.

The first adoption manifest must use exactly:

```text
--source-commit c3a960acdc94173440144559bb95f1de36d31537
--sampling-source-commit 18d5d2ac1603a26a221af9d5fc430b084ec002df
--legacy-import-id p46e12805
--resume-tag <fresh stable tag>
--full-campaign
```

Before TPU initialization it must emit:

```text
[P46.RESUME] LEGACY_IMPORT_PASS import_id=p46e12805 records=<n> valid_records=<n> manifest_sha256=<sha256> receipt=<absolute-path>
```

Any manifest, fingerprint, task-order, identity, attempt, logprob or provenance
mismatch is a hard stop. The importer does not reclassify or rewrite the old
files; it writes immutable v6 copies with per-row `imported_from` provenance
and a receipt. Later relaunches omit `--legacy-import-id` but keep both exact
source SHAs, the resume tag, topology, image, model, data and sampling fields.

## Render the next full 128-chip washing JobSet

For a from-scratch config-v4/trajectory-v6 campaign, use a new resume tag and
keep it stable for the whole washing campaign; a launch run id may change
after interruption. To preserve `p46e12805`, use the transition procedure
above instead of this from-scratch command:

```bash
TOPOLOGY=128
BASE=canon-zero-tim/cluster/jobset-256cluster-64chip.yaml
RESUME_TAG="p46q4wash01"
RUN_ID="p46r01a0"
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
  --resume-tag "$RESUME_TAG" \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC" \
  --full-campaign
```

Before apply, verify the rendered manifest contains:

```text
canon.zero-tim/full-campaign: "1"
CANON_P46_FULL_CAMPAIGN=1
CANON_P46_RESUME_TAG=p46q4wash01
CANON_P46_EVALUATION_MODE=reward_only
CANON_P46_TOPOLOGY=128
CANON_P46_LOGICAL_SHARD_INDEX=0
CANON_P46_PHYSICAL_SHARD_INDEX=0
state-launches/p46r01a0
logs/campaign.log
4x4x8
32 Pathways workers
```

The renderer rejects parity plus full campaign, nonzero externally supplied
shard indices, Q4 topology 256, Q32 topology 128, floating images, or an
unreviewed whitelist digest. Do not hot-patch the YAML.

## Resume after interruption

The simplest recovery is to delete only the failed JobSet object and reapply
the original rendered YAML. If a new Kubernetes name is required, render from
the original published source checkout with a new short launch id and the same
resume tag:

```bash
ORIGINAL_SOURCE_SHA=<sha recorded by the first manifest>
RESUME_TAG=p46q4wash01
RUN_ID=p46r01a1

python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
  --base "$BASE" \
  --output "/tmp/p46-q4-wash-resume-${RUN_ID}.yaml" \
  --workload q4-clean-eval \
  --topology "$TOPOLOGY" \
  --source-commit "$ORIGINAL_SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_ID" \
  --resume-tag "$RESUME_TAG" \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC" \
  --full-campaign
```

Do not replace `ORIGINAL_SOURCE_SHA` with the branch's newer HEAD. The launch
will fetch the branch, prove the original SHA is still in its ancestry, and
check out the original SHA. Contract drift fails before model initialization.

## Runtime interpretation

Healthy progress markers are:

```text
P46_EVAL_CAMPAIGN_WAVE_START ... pending=64 runtime_reused=1
[P46.RESUME] LEASE_ACQUIRED resume_tag=... launch_id=... contract_sha256=...
[P46.RESUME] LEGACY_IMPORT_PASS ...  # first adopted launch only
[P46.RESUME] ORPHAN_SANDBOX_CLEANUP_PASS ... remaining=0
P46_EVAL_TRAJECTORY ... completed=<n>/<wave-size>
P46_EVAL_CAMPAIGN_LOGICAL_PASS ... runtime_reused=1
P46_EVAL_CAMPAIGN_PASS tasks=1851 n_sample=16 valid_trajectories=29616 logical_shards=58 ...
[P46.EVAL.POSTFLIGHT] PASS
```

The following stop nonzero and preserve already fsynced artifacts:

```text
P46_EVAL_CAMPAIGN_WAVE_TIMEOUT ... resume_tag=... resume_same_tag=1
P46_EVAL_CAMPAIGN_LOGICAL_INCOMPLETE ...
```

On a wave timeout, first archive Kubernetes events and identify the immutable
attempt log printed by `[P46.RESUME] ATTEMPT_LOG`. After the cause is
understood, either reapply the original manifest or render a new launch run id
with the **same** resume tag, harness SHA and sampling-source SHA. Exact
completed identities will be skipped. Never change either SHA, resume tag,
topology, model, data, or sampling while calling it a resume. An adopted
campaign keeps the sampling-source commit
`18d5d2ac1603a26a221af9d5fc430b084ec002df` on every later render, but omits
`--legacy-import-id` after the receipt exists.

`MAX_CONTEXT_LIMIT_REACHED`, `MAX_STEPS_REACHED`, `MODEL_TIMEOUT`, and signed
trajectory `TIMEOUT` are valid unsolved fixed-budget outcomes. They do not
block washing and must not be resampled. A raw bad Q4 action also does not
block the campaign merely because the tool rejected it; that is part of the
model's solve-rate measurement.

## Outputs and completion

Persistent root:

```text
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/outputs/resume_contract.json
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/outputs/resume_lease.json
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/imports/<legacy-run-id>/
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/outputs/imports/<legacy-run-id>.receipt.json
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/outputs/trajectories/
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/outputs/reports/
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/outputs/campaign/
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/logs/campaign.attempt-*.log
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
# Optional P38.2y2 fixed output head (not yet TPU-certified)

Qwen3-4B debug and Qwen3-32B training renders accept the explicit
`--fixed-lm-head` option. It is default-off, forbidden for `q4-clean-eval`,
and must not be hand-added to YAML. The pinned-image construction gates pass,
but no 4B/32B TPU target has run. Launch and return requirements are in
`../p38-pathways-decode-prefill-carrier/P38_TP8_FIXED_LM_HEAD_RUNBOOK.md`.
