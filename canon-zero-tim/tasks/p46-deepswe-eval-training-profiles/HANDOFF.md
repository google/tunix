# P46 remote-agent execution handoff

Publication status: **P46.1-P46.4 PUBLISHED; P46.5 REWARD-ONLY UNPUBLISHED;
TARGET CAMPAIGN NOT RUN**.

The bounded lifecycle, evaluator, dual-topology profiles and trainer data-axis
repair are anchored by implementation commit
`e1b4009394c49ea015919bda0cfdb97c12c221b5`. The execution SHA must still be
read back from the current `origin/yuxzhang/canon-zero-tim` because later
documentation/evidence commits may advance the branch. Require that the exact
read-back SHA contains `e1b40093`; never substitute the older reconciled base
`99c3f7af761c859caa6c81ab509446cc3cc47dc0`. Never modify or push `main`.

The archived P34r03 Qwen3-32B run generated 64/64 rollout records, but every
record ended as `ENV_TIMEOUT`. It then failed before forward/backward with
`KeyError: 'fsdp'`: the trainer mesh was `dp,tp` while the launcher passed a
stale `fsdp` data-sharding axis. The current worktree derives the data axis from
the trainer mesh and prints `[DEEPSWE.DATA_SHARDING] PASS` before rollout. Do
not rerun from the reconciled base alone; use the exact read-back operator SHA
containing the implementation commit above.

`observed_trajectories=64` is only a cardinality statement. It was not evidence
of 64 valid trajectories in P34r03. The eventual publication also contains the
bounded request/trajectory/batch/cleanup lifecycle needed to prevent a sandbox
step from running for hours after its deadline.

P46.5 fixes a separate evaluation-only problem: vLLM integer zero still asks
for logprobs. The unpublished path uses
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
test -f examples/deepswe/probe_reward_only_v5p.py
```

If those checks fail, reward-only is not published. Stop; do not reconstruct
it with YAML or shell hot patches.

## Gate 0 — reward-only publication and layered parity

First reconcile P46.5 onto the exact current operator branch, but commit/push
only with explicit operator approval. Run the 31-case P46 CPU gate and the two
targeted `VllmSamplerConfigTest` cases. If a direct four-chip v5p host is
available, rerun the one-host command from a clean published checkout:

```bash
bash canon-zero-tim/tests/p46_deepswe_profiles/run_onehost_reward_only_v5p.sh
```

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

Use whichever topology is available; 64 chips are not a prerequisite. Render
logical index 0, physical index 0 exactly as documented in
`P46_DEEPSWE_PROFILES_RUNBOOK.md`. On 64 chips evaluation is DP8 x TP8; on 256
chips it is DP32 x TP8. Both are Qwen3-4B-Instruct-2507, 16K, four tasks x 16
samples, concurrency 64, prefix cache off, complete trajectory streaming, and
a one-hour shard deadline. Keep one topology for a resumable run-id because it
is part of the evaluation fingerprint. Server-side dry-run the rendered YAML
first. Apply it only after the operator explicitly approves the launch.

Require both markers:

```text
P46_EVAL_SUBSHARD_PASS ...
[P46.EVAL.POSTFLIGHT] PASS
```

Return the rendered YAML and digest, full logs, JobSet events, cleanup evidence
and the persistent files below:

```text
/mnt/disks/linchai_data/deepswe_eval/<run-id>/trajectories/
/mnt/disks/linchai_data/deepswe_eval/<run-id>/reports/
/mnt/disks/linchai_data/deepswe_eval/<run-id>/logs/
```

Inspect complete trajectory JSON, not only a sample/task summary. Verify 64
unique identities, prompt plus alternating assistant/environment messages,
terminal status, finite reward, elapsed time, source/data/model fingerprint and
no credential material. A shard timeout is resumable evidence and a failed
gate, even if some records were safely written.

## Gate 2 — Q4 three-update training

Only after Gate 1 passes, render a new `q4-debug` JobSet on whichever topology
is available and obtain separate launch approval. The 64-chip form uses DP4 x
TP8 per rollout/trainer role; the 256-chip form uses DP16 x TP8 per role. Both
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

## Gate 3 — optional full Q4 evaluation campaign

Only if a new curriculum report is wanted, complete all 58 logical reports via
463 resumable physical JobSets. A task may be categorized only after exact
N16. `partial` and `all_fail` reports remain advisory and do not replace the
original clean whitelist without a separate reviewed manifest and digest.

## Gate 4 — Qwen3-32B training

Only after Gate 2 passes, render `q32-train` for the available 64- or 256-chip
topology and obtain explicit launch approval. Keep the signed profile unchanged:
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
