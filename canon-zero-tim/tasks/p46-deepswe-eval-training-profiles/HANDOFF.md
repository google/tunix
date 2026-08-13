# P46 remote-agent execution handoff

Publication status: **P46.1-P46.5 AND INVALID-ATTEMPT/CAMPAIGN-FINALIZER FIX
PUBLISHED; P46E25609 ACTION-ADAPTER/STATUS FIX AND Q4 64/128 TOPOLOGY
MIGRATION LOCAL AND UNPUBLISHED; TARGET CAMPAIGN INCOMPLETE**.

The bounded lifecycle, evaluator, dual-topology profiles and trainer data-axis
repair are anchored by implementation commit
`e1b4009394c49ea015919bda0cfdb97c12c221b5`; true reward-only evaluation is
anchored by `a4d165e854cc4c2320d8120e89aed185eaf61465`. The execution SHA
must still be read back from the current `origin/yuxzhang/canon-zero-tim`
because later
documentation/evidence commits may advance the branch. Require that the exact
read-back SHA contains both `e1b40093` and `a4d165e8`; never substitute the
older reconciled base
`99c3f7af761c859caa6c81ab509446cc3cc47dc0`. Never modify or push `main`.

Historical operator HEAD
`63b092b001864e4e9a4822b4354a665bb00b1c6b` contains the returned
`p46e25608` evidence and the old false-positive physical-shard completion
behavior; do not launch from it. The invalid-attempt retry and campaign
finalizer are published by
`a642ab267425a5b08b0cebb6e12c607f50f71831`. Resolve and record the exact
current 40-character operator HEAD, require `a642ab26` in its ancestry, and
require `attempt_index`, `P46_EVAL_PHYSICAL_INCOMPLETE`, the 40-case P46 CPU
gate, `r2egym_action_compat.py`, trajectory schema v4, Q4 topology 128 and
`finalize_deepswe_eval.py` before rendering. Until the local correction is
published and read back, stop before rendering; do not recreate it as a YAML
or shell hot patch.

The archived P34r03 Qwen3-32B run generated 64/64 rollout records, but every
record ended as `ENV_TIMEOUT`. It then failed before forward/backward with
`KeyError: 'fsdp'`: the trainer mesh was `dp,tp` while the launcher passed a
stale `fsdp` data-sharding axis. The published implementation derives the data
axis from the trainer mesh and prints `[DEEPSWE.DATA_SHARDING] PASS` before rollout. Do
not rerun from the reconciled base alone; use the exact read-back operator SHA
containing the implementation commit above.

`observed_trajectories=64` is only a cardinality statement. It was not evidence
of 64 valid trajectories in P34r03. The published implementation also contains
the bounded request/trajectory/batch/cleanup lifecycle needed to prevent a sandbox
step from running for hours after its deadline.

P46.5 fixes a separate evaluation-only problem: vLLM integer zero still asks
for logprobs. The published path uses
`evaluation_mode=reward_only` as its single source, sends
`logprobs=None,prompt_logprobs=None`, skips host extraction, forbids numeric
fake logprobs, and records
`trajectory_mode=reward_only_no_logprobs` plus
`sampled_by=stock@<SHA>`. Contradictory trainer/alignment/logprob/optimizer
caller inputs fail closed. TPU/JAX rejects per-request seeds, so artifacts
truthfully record `sampling_rng_mode=engine_global_sequential`; `sample_nonce`
is identity metadata, not a replayable request seed.

A dirty-worktree development run on one direct-attached v5p-8 host passed L1
and L2 with Qwen3-4B DP1 x TP4, one pinned clean R2E Docker task, one real
`search` action, final reward 0, a valid trajectory and no residual container:

```text
P46_REWARD_ONLY_ONEHOST_PASS l1=PASS l2=IDENTICAL_OBSERVER
/mnt/disks/tunix-data/deepswe-reward-only-evidence/reward-only-onehost-20260813T061510Z-696010/report.json
sha256=db3305413817ffe5c4d0085098475a12753cea6b698e15e4263b0c7d0835ba7c
```

This is not clean publication evidence and does not prove L3, Kubernetes or
target throughput. The observer/reward-only diagnostic call medians were
0.0330/0.0310 seconds and sampler payloads 117/70 bytes; do not advertise a
cluster speedup from that micro-measurement.

The first returned 256-chip reward-only physical shard is also not a PASS.
Run `p46e25608`, source
`bdc9681824743911d0691659604dec090dd42bc4`, initialized Qwen3-4B at DP32 x
TP8 and attempted all 64 identities, but finished with 62 `SUCCEEDED` and two
`MODEL_TIMEOUT` records:

```text
namanjain12/aiohttp_final:006fbe03fede4eaa1eeba7b8393cbf4d63cb44b6 sample=6
namanjain12/aiohttp_final:04deab71cc804311016159548e5dcdfb9c2698d3 sample=5
```

The old evaluator counted records that were invalid under its own policy as
completed resume identities and printed
`P46_EVAL_SUBSHARD_PASS ... pending_logical_tasks=30`; revoke that historical
claim. The current policy deliberately counts `MODEL_TIMEOUT` as a valid
unsolved result under the fixed call budget and records
`validity_reason=completed_model_timeout`. This policy change does not
reclassify the old run in place. The fixed evaluator durably records every
attempt, allows only a valid record to complete an identity, retries actual
environment/reward/harness failures with consecutive `attempt_index` values,
and rejects attempts after a valid result. Because the fixed source SHA changes
the fingerprint, start with a new run id and rerun all 64 l0/p0 identities; do
not transplant the old records.

Returned run `p46e25609` is a second failed 256-chip attempt, not a 64-chip
run and not clean evaluation evidence. The exact artifact source provenance is
`stock@8c0e90f3b995f457c1dbb2199639f7f47962ed2b`. It has four tasks x N16,
64 unique identities, 1,102 nonempty action/observation steps, null logprobs,
and terminal status 59 `SUCCEEDED`, four `MAX_CONTEXT_LIMIT_REACHED`, one
`MODEL_TIMEOUT`; all rewards are zero. It stopped after the first wave because
the evaluator counted the four signed context-budget outcomes as invalid,
reported five pending identities, and correctly exited nonzero under that old
classification.

Full action/observation inspection changes the conclusion: Q4 repeatedly
emitted inline-valued tags such as `<parameter=command=view>`. R2E parsed
`command=view` as a key and passed `--command=view` to a positional CLI. The
shard contains 347 `unrecognized arguments` observations, 363 editor usage
errors, 172 `/parameter` shell errors, and 40 missing-argument errors; every
trajectory contains a recognizable adapter leak. This proves streaming and
schema capture, but **zero trajectories are eligible for curriculum
classification**.

The unpublished fix canonicalizes the observed dialect before R2E, preserves
raw `model_response`, records the canonical executed action, and invalidates
any surviving adapter signature as
`validity_reason=r2egym_action_parameter_adapter`. It treats max-step,
max-context, model-timeout and whole-trajectory budget terminals as completed
unsolved model outcomes; model timeout is labeled
`validity_reason=completed_model_timeout`. Environment/reward failures remain
retryable, and recognized adapter/parser corruption overrides any accepted
status. The first published fixed run must use a new run id and rerun all 64
identities; never resume or reclassify `p46e25609` in place.

The full secondary evaluation/data-washing campaign is not complete. It still
requires 1851 x N16 = 29,616 valid trajectories, 58 logical reports and 463
physical JobSets. Neither returned l0/p0 attempt is promotable. No candidate
washed whitelist has been produced or approved.

Before execution, read these files completely:

1. `cluster/P46_DEEPSWE_PROFILES_RUNBOOK.md`
2. `tasks/p46-deepswe-eval-training-profiles/state.md`
3. `tasks/p46-deepswe-eval-training-profiles/plan.md`
4. `tasks/p46-deepswe-eval-training-profiles/phases/p46-4-remote-execution.md`
5. `tasks/p46-deepswe-eval-training-profiles/phases/p46-5-reward-only-evaluation.md`
6. `cluster/P34_DEEPSWE_RUNBOOK.md`
7. `cluster/P44_DEEPSWE_QWEN4B_PARITY_RUNBOOK.md`

Then fetch the operator branch, detach at its exact remote SHA, require a clean
checkout, and run:

```bash
git fetch origin yuxzhang/canon-zero-tim
git switch --detach origin/yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git ls-remote origin refs/heads/yuxzhang/canon-zero-tim | awk '{print $1}')"
test "$SOURCE_SHA" = "$REMOTE_SHA"
test -z "$(git status --porcelain)"
bash canon-zero-tim/tests/p46_deepswe_profiles/run_cpu.sh
rg -n 'training_data_sharding_axis|DEEPSWE.DATA_SHARDING' \
  examples/deepswe/train_deepswe_nb.py
```

The grep must show that `training_data_sharding_axis` comes from
`train_axis_names[0]` and is passed into `RLTrainingConfig`. Stop if production
still hard-codes `fsdp`.

For P46.5, the exact detached publication must also contain all of:

```bash
rg -n 'evaluation_mode=reward_only|prompt_logprobs = None|host_extraction' \
  canon-zero-tim examples/deepswe tunix/generate/vllm_sampler.py
rg -n 'canonicalize_r2egym_action|DEEPSWE.R2E_ACTION_COMPAT' \
  examples/deepswe/r2egym_action_compat.py examples/deepswe/swe_agent.py
rg -n 'trajectory.v4|validity_reason|MAX_CONTEXT_LIMIT_REACHED' \
  examples/deepswe/deepswe_eval_artifacts.py
test -f examples/deepswe/probe_reward_only_v5p.py
```

If those checks fail, reward-only is not published. Stop; do not reconstruct
it with YAML or shell hot patches.

Also require the read-back renderer to admit Q4 exactly on 64/128, emit
`4x4x8` with 32 workers for topology 128, and reject Q4-256/Q32-128. Until the
local topology migration is published, do not render a replacement manifest
by hand.

## Gate 0 — reward-only publication and layered parity

Require `a4d165e8`, the invalid-attempt repair, and the later action-adapter
repair in the exact read-back operator ancestry. Run the 40-case P46 CPU gate and the two
targeted `VllmSamplerConfigTest` cases. If a direct four-chip v5p host is
available, rerun the one-host command from a clean published checkout:

```bash
bash canon-zero-tim/tests/p46_deepswe_profiles/run_onehost_reward_only_v5p.sh
```

Before any target manifest, require this audit too:

```bash
rg -n 'attempt_index|P46_EVAL_PHYSICAL_INCOMPLETE|physical_pending|validity_reason' \
  examples/deepswe/deepswe_eval_artifacts.py \
  examples/deepswe/eval_deepswe.py
rg -n 'canonicalize_r2egym_action|DEEPSWE.R2E_ACTION_COMPAT' \
  examples/deepswe/r2egym_action_compat.py examples/deepswe/swe_agent.py
test -f examples/deepswe/finalize_deepswe_eval.py
```

Stop unless the CPU marker is exactly `cases=40` or if any repair/finalizer
marker is absent.

L1 is a hard gate. L2 token identity is diagnostic; a clean
`LAW1_SUFFIX_DIVERGENCE` is recorded but does not block. Before reward-only
becomes the Q4 clean-evaluation default, run the same N16 task/sample identities
through the validation-only `logprob_observer` and `reward_only` canary arms on
a 64-chip small shard. Render both manifests exactly as documented in
`P46_DEEPSWE_PROFILES_RUNBOOK.md`; each arm is one clean task x N16 and neither
is a production workload default. Require exact paired McNemar/binomial L3
PASS, compare JobSet-level valid trajectories/hour, and prove Kubernetes
cleanup in both arms. Never compare against a historical run or another source
SHA.

## Gate 1 — one Q4 clean-evaluation physical shard

Use whichever admitted Q4 topology is available; 64 chips are not a prerequisite. Render
logical index 0, physical index 0 exactly as documented in
`P46_DEEPSWE_PROFILES_RUNBOOK.md`. On 64 chips evaluation is DP8 x TP8; on 128
chips it is DP16 x TP8. Both are Qwen3-4B-Instruct-2507, a 16,384-token total
response budget per trajectory, at most 50 environment/model steps, four tasks
x 16 samples, concurrency 64, prefix cache off, complete trajectory streaming,
and a one-hour physical-shard deadline. Keep one topology for a resumable
run-id because it is part of the evaluation fingerprint. Server-side dry-run
the rendered YAML first. Apply it only after the operator explicitly approves
the launch.

Require both markers:

```text
P46_EVAL_SUBSHARD_PASS ...
[P46.EVAL.POSTFLIGHT] PASS
```

`P46_EVAL_PHYSICAL_INCOMPLETE` or any nonzero evaluator exit is a failed,
resumable attempt. Relaunch the same fixed source SHA, run id, topology and
l/p indices so only invalid identities receive their next consecutive
`attempt_index`. Do not advance to the next physical shard until this one has
its exact valid identity count.

Return the rendered YAML and digest, full logs, JobSet events, cleanup evidence
and the persistent files below:

```text
/mnt/disks/linchai_data/deepswe_eval/<run-id>/outputs/trajectories/
/mnt/disks/linchai_data/deepswe_eval/<run-id>/outputs/reports/
/mnt/disks/linchai_data/deepswe_eval/<run-id>/logs/
```

Inspect complete trajectory JSON, not only a sample/task summary. Verify the
exact valid identity count, prompt plus alternating assistant/environment
messages, terminal status, finite reward, elapsed time, source/data/model
fingerprint and no credential material. Invalid retries remain visible as
separate durable attempts; they do not count toward N16. A shard timeout is
resumable evidence and a failed gate, even if some records were safely written.
Every accepted record must have either
`validity_reason=completed_under_signed_budget` or
`validity_reason=completed_model_timeout`; any
`r2egym_action_parameter_adapter`, stored inline-valued parameter tag,
`cannot open /parameter`, or tool `unrecognized arguments` adapter signature
blocks the shard. `[DEEPSWE.R2E_ACTION_COMPAT]` warnings are allowed because
the raw response is preserved, but the stored executed action must be canonical
and the resulting tool observation must be real output rather than CLI usage.

Before handing evidence back, follow the return-package commands in the P46
runbook. Return every trajectory JSONL absolute path, `wc -l`, per-file
SHA-256, the archive path/digest and full logs. The archived head log alone is
insufficient to inspect action/observation/tool-call content.

## Gate 2 — Q4 three-update training

Only after Gate 1 passes, render a new `q4-debug` JobSet on whichever topology
is available and obtain separate launch approval. The 64-chip form uses DP4 x
TP8 per rollout/trainer role; the 128-chip form uses DP8 x TP8 per role. Both
keep Qwen3-4B-Instruct-2507, B4 x G4, 16 trajectories, 16K response, three
updates and the one-hour shared rollout-batch boundary. Require:

- one `[DEEPSWE.DATA_SHARDING] PASS` with `axes=('dp',)` and a `dp,tp` mesh;
- three `P44.LOGPS_BATCH` markers, three durable trajectory batches and three
  matching batch-metrics rows;
- finite nonzero gradient activity and train steps `0->1->2->3`;
- exactly three optimizer commits with `optimizer_placement=device-resident`;
- no host optimizer round trip and at least 8 GiB classified HBM margin; and
- a generated P44 classification JSON with `"verdict": "PASS"`.

Any timeout, cleanup leak, malformed trajectory, OOM, IFRT, nonfinite value,
zero gradient activity, optimizer transaction/placement failure or classifier
failure stops promotion to Q32.

## Gate 3 — complete the full Q4 evaluation/data-washing campaign

The first 64-trajectory shard is a smoke and resume unit, not campaign
completion. Continue the same fixed source SHA, topology and run id until all
1851 clean tasks have exactly N16 valid trajectories: 29,616 valid identities,
58 logical reports and 463 physical JobSets. Logical indices 0-56 each have
physical indices 0-7; logical index 57 has physical indices 0-6. Every normal
physical shard is four tasks x N16 = 64 valid identities; the final l57/p6
shard is three tasks x N16 = 48.

Render, server-side dry-run, launch and wait one physical index at a time under
the operator's campaign approval. On `P46_EVAL_PHYSICAL_INCOMPLETE`, retry the
same index until its exact valid count is complete; never fan out all 463
JobSets at once, because sandbox/CPU-node pressure and cleanup are part of the
gate. Advance only after postflight cleanup passes. At the last physical index
of each logical shard, require `P46_EVAL_LOGICAL_REPORT_PASS`; after l57/p6,
require all 58 immutable logical reports and verify their digests.

Finalize the campaign only after all 58 summaries exist:

```bash
RUN_ROOT="/mnt/disks/linchai_data/deepswe_eval/$RUN_ID"
python3 examples/deepswe/finalize_deepswe_eval.py \
  --summary-json "$RUN_ROOT"/outputs/reports/*.summary.json \
  --output-dir "$RUN_ROOT/outputs/campaign"
```

Require exactly:

```text
P46_EVAL_CAMPAIGN_PASS tasks=1851 n_sample=16 valid_trajectories=29616 logical_shards=58 ...
```

The finalizer rejects missing/duplicate task identities, missing shards,
digest drift, cross-shard contract changes, broken/incomplete reports or any
task without exact valid N16. Return and archive `outputs/campaign` together
with all trajectories, logical reports and logs.

The production evaluator retains `max_response_length=16384`, `max_steps=50`,
N16, temperature 1.0, top-p 1.0, top-k 0 and a 3600-second physical deadline.
A task may be categorized only after exact valid N16. `partial`, `all_fail`
and `all_pass` reports are the completed secondary-evaluation output; broken
or incomplete tasks are never promoted. Candidate whitelists remain advisory
and do not replace the original clean whitelist without a separate reviewed
manifest, digest and operator decision.

## Gate 4 — Qwen3-32B training

Only after Gate 2 and the required Gate 3 campaign pass, render `q32-train` for
the available 64- or 256-chip topology and obtain explicit launch approval.
Keep the signed profile unchanged:
Qwen3-32B, original 1851-row clean whitelist, 16K response, B8 x G8, 64
trajectories, maximum concurrency 64, 5400-second shared batch boundary, 1000
updates and TPU-resident optimizer state. Require the `dp` data-axis marker
before the first rollout. Do not reuse the P34r03 manifest.

At each gate, report the exact publication SHA and image digest, manifest
SHA-256, persistent run directory, trajectory/report digests, head/worker/R2E
logs, cleanup state, HBM, optimizer placement, classifier JSON and first fatal
traceback. A missing prerequisite is `INCONCLUSIVE`; a violated signed contract
is `FAIL`.

Do not add host optimizer fallback, relax the one-hour/ninety-minute deadlines,
change the clean whitelist, classify partial N16 tasks, hide trajectories,
emulate a missing topology, or edit a rendered JobSet. Missing prerequisites or
target failure are INCONCLUSIVE/FAIL as appropriate, never PASS.
