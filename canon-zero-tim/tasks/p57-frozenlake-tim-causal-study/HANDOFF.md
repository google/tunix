# P57 300-update execution handoff

## START HERE — Zero reference must use the registered optimized path

This section supersedes the older statement that the Zero-TIM pair is merely
deferred on the baseline P57 profile.

Wave 04 P45 ran source `f7adb4e6fb4b86698c0386079b3a17da031a4578`
as `canon-p57-fl-zero-f45w04-f7adb4e6`, but its resolved log selected
`qwen3-8b-dp8-tp8-frozenlake-tim`, not the registered
`qwen3-8b-dp8-tp8-frozenlake-v1-hp` profile. Step-0 strict pre-alignment was
real and green over 47,169 action tokens with A-B/B-C `0/0`, but the baseline
profile left P59 rank-parallel backward off. The serial report adjoint returned
an engine six-axis `data/.../model` gradient and then passed it to a trainer
reducer expecting axis `dp`; construction stopped before DP reduction, AdamW,
weight sync, evaluation, or checkpoint.

The incident report's proposed `_p59_replicated_data_mesh` six-axis whitelist
is not the fix for the active Zero reference: the failing source line is the
serial reducer branch and that helper is not on the traceback. Preserve the
failed run and report unchanged as historical evidence.

The active decision is now fail-closed: every primary P45 or M15/main
300-update `zero` train must use the V1 high-performance profile and its P59
rank-parallel backward, checked-VMA repair, P67 serving scope, first-update
gate, strict alignment, APC-off contract, final-only checkpoint, and existing
evaluation schedule. `render_three_arm_wave.sh zero ...` supplies the HP mode;
the renderer and base P57 profile independently reject a baseline zero train,
and the manifest verifier checks the resolved P59/P66/P67 receipts.

W&B comparison identity remains unchanged. Native/no-IS, native/IS, and the
optimized Zero reference all write to project
`zero-tim-p57-frozenlake-tim`; groups and run names continue to distinguish
the arm and workload. Do not move Zero to a V1-specific project.

No Wave 05 manifest or TPU run exists yet. After publication approval, render
fresh P45 and M15 Zero manifests from the exact pushed SHA, return both YAML
hashes plus resolved-env receipts, and obtain separate launch approval. The
first target acceptance remains one complete 32-group P59 backward, finite
nonzero first-update receipt, valid AdamW `0 -> 1`, weight sync, and policy
step 1 with no strict alignment failure.

## Assignment

You are the execution agent. Run reviewed scripts; do not edit code, profiles,
rendered YAML, flags, or scientific parameters. Do not commit, push, apply a
JobSet, or delete an artifact without explicit user approval.

Read in order: `state.md` → `plan.md` →
`phases/p57-1b-three-arm-baselines.md` → `RUNBOOK.md`.

The immediate queue is four **fresh** jobs: P45 and M15 under native/no-IS and
native/token-IS. None may resume an earlier attempt. The Zero-TIM pair is
deferred.

## Current P45 native incident

Do not resume or reuse `canon-p57-fl-mism-n45j-2a89eef3`. It committed update
1, then failed before weight sync because the evaluation-cycle receipt treated
the pre-sync `rl_cluster.global_steps=0` as the completed timing row. At this
exact boundary the authoritative committed counter is
`actor_trainer.train_steps=1`; the cluster counter deliberately remains at the
evaluated policy step until `sync_weights()` finishes.

The repair keeps the public receipt unchanged:

~~~text
[P57.EVAL.CYCLE] policy_step=0 enclosing_global_step=1
~~~

It now fail-closes on both lifecycle facts: committed actor step must be
`policy_step+1`, and deferred cluster step must still be `policy_step`. Before
rendering replacements, require the P57 CPU suite to report 136/136 and the V1
suite to report 12/12. Use fresh run ids, output roots, campaign roots, and
checkpoint tags for all four jobs. A replacement is accepted only after it
crosses update 1, emits the receipt above, completes weight sync, and begins policy step 1; host
tests alone do not certify the target fix.

## Contract you must not reinterpret

- Qwen3-8B, DP8xTP8, resident optimizer, 300 updates.
- P45: original generator, five turns, 4,096/2,048.
- M15: materialized `m15/main`, 15 turns, 4,096/8,192.
- 32 prompts x eight generations, temperature 0.7, identical optimizer and
  GSPO/RLOO recipe within each workload.
- Exactly one `--seed=42`; runtime vLLM global seed `0`. The backend reports
  per-request seed unsupported, so require equal seed/data contracts but do
  not promise bitwise-identical stochastic trajectories across launches.
- Held-out rollout-only evaluation inside the training JobSet at exactly
  `0,50,100,150,200,250,300`.
- Each evaluation point is 100 prompts x eight generations = 800 rewards.
- Evaluation does not run trainer eval/backward and does not write a checkpoint.
- One scheduled checkpoint at final update 300, `LatestN(1)`, no retained
  50-step milestones. A partial primary run is not resumable by design.
- Native finite A-B is warning-only treatment; B-C and all structural safety
  gates remain fatal.

Do not run the separate evaluation schedule for the primary curve. The old
450-update / 20-evaluator procedure is superseded.

The fixed dataset hashes and generator namespaces are authoritative in the
`Reproducibility and data identity` section of `RUNBOOK.md`. Every arm within a
workload must report the same train/eval hashes. Step 0 means before update 1;
step 50 means rollout still holds the weights after 50 updates and update-51
weights have not been synced; the same policy-version rule holds through 250.
Step 300 runs only after update 300 and rollout weight sync.

## Before rendering

Use an approved, pushed, full 40-character source SHA. In that exact checkout:

~~~bash
bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh
bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh
git diff --check
~~~

Require the full marker list in `RUNBOOK.md`, including
`P57_INPROCESS_EVAL_CLASSIFIER_PASS steps=7` and the pinned-image runtime matrix.
Stop if any marker is absent.

Confirm every earlier P57 JobSet is terminal and packaged. Choose fresh,
previously unused run IDs, output roots, campaign roots, and checkpoint tags.
All four cells are rerendered together after the shared receipt fix so source,
checkpoint cadence, and launch generation remain matched; do not selectively
resume a cell that happened to run farther.

## Render

The following IDs are examples and may be used only if unused:

~~~bash
cd /home/yuxuan/code_rl_repro/worktrees/p57_frozenlake_tim_0820
SOURCE=<approved-pushed-full-40-character-sha>
OUT_NATIVE=/tmp/p57-primary-native-300-c
OUT_IS=/tmp/p57-primary-is-300-c

bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  native "$SOURCE" "$OUT_NATIVE" n45e n15e p57-native-300-c
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  is "$SOURCE" "$OUT_IS" i45e i15e p57-is-300-c
~~~

Return the four YAML paths, their SHA-256 values, and all render/preflight PASS
lines. Stop for user review and launch approval.

## Apply only after approval

~~~bash
kubectl apply -f "$OUT_NATIVE/p45/jobset-p57-frozenlake-mismatch-300.yaml"
kubectl apply -f "$OUT_NATIVE/m15/jobset-p57-frozenlake-mismatch-m15-main-300.yaml"
kubectl apply -f "$OUT_IS/p45/jobset-p57-frozenlake-is-300.yaml"
kubectl apply -f "$OUT_IS/m15/jobset-p57-frozenlake-is-m15-main-300.yaml"
~~~

Monitor all four independently. Preserve the first attempt from byte zero.
Never hide a restart or relaunch under the same scientific identity.

## Terminal acceptance

Each successful run must have:

- update completion at 300 and exactly the registered final checkpoint at 300;
- one eval-enabled receipt with cadence 50;
- P42 JSON policy steps exactly `0,50,100,150,200,250,300`;
- `n=800` and finite metrics at every point;
- one `[P57.EVAL] FINAL policy_step=300 ...` receipt;
- `P57_INPROCESS_EVAL_PASS ... rewards_per_step=800`;
- one `[P57.SEED] CONTRACT_PASS data_shuffle_seed=42 vllm_global_seed=0 ...`;
- one `[P57.DATASET] MATERIALIZED_PASS ...` with the registered train/eval
  hashes for that workload;
- training and evaluation classifier JSON/SHA receipts;
- the arm-specific purity receipt documented in `RUNBOOK.md`;
- native stock-fast/canonical-absence receipts, finite A-B, and exact B-C.

If any item is missing, do not call the run complete.

## What to return

Return one record per JobSet with:

~~~text
workload/arm:
source SHA:
image:
jobset/run/attempt:
YAML path + SHA256:
raw log path + SHA256:
resolved env path + SHA256:
train classifier path + SHA256 + verdict:
eval classifier path + SHA256 + verdict:
checkpoint tag/latest durable step/path:
purity receipt:
stock/zero runtime receipt:
evaluation steps/n/solve/reward:
experiment/vLLM seed receipt:
train/eval dataset SHA-256 receipt:
A-B dose and B-C verdict:
updates/checkpoints/restarts:
W&B run URL/name:
timing/tokens per second/grad and update norms:
IS mean/max/clip fraction (IS only):
infra events:
final classification:
~~~

Package failures with the same completeness as successes. If artifacts remain
in GCS, execute the repository classifier there and return its full JSON plus
an input inventory and SHA ledger.

## Current code status

The 300-update contract, rollout-only evaluation path, final step-300 eval,
seven-point classifier, renderer/profile/env gates, and documentation are
implemented in the working tree. At handoff-writing time they are uncommitted
and unpushed; the user must explicitly approve commit/push before any render
uses them as a source SHA.
