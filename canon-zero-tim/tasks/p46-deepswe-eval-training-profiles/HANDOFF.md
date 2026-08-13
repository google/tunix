# P46 remote-agent execution handoff

Publication status: **IMPLEMENTATION PUBLISHED; TARGET CAMPAIGN NOT RUN**.

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

Before execution, read these files completely:

1. `cluster/P46_DEEPSWE_PROFILES_RUNBOOK.md`
2. `tasks/p46-deepswe-eval-training-profiles/state.md`
3. `tasks/p46-deepswe-eval-training-profiles/plan.md`
4. `tasks/p46-deepswe-eval-training-profiles/phases/p46-4-remote-execution.md`
5. `cluster/P34_DEEPSWE_RUNBOOK.md`
6. `cluster/P44_DEEPSWE_QWEN4B_PARITY_RUNBOOK.md`

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
