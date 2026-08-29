# P57 300-update execution handoff

## START HERE — optimized Zero references use the P74-enabled two-full wrapper

For the next strict Zero P45 and M15/main full references, do not use the
baseline P57 profile, `render_three_arm_wave.sh`, the legacy P45 resident
renderer, or any previously rendered YAML. The registered route is now the
Phase4 two-full wrapper:

```bash
bash canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/prepare_p67_frozenlake_two_full_wave.sh \
  <approved-40-character-sha> \
  <fresh-output-dir> \
  <fresh-campaign-root> \
  <fresh-p45-run-id> \
  <fresh-m15-run-id>
```

This command is render-only and requires a clean checkout of the exact
published source. The current rollout implementation is local and uncommitted;
commit/push and either target launch each need separate approval.

Both manifests must resolve checked-VMA/P67/first-update protection plus
`CANON_DP_COMPARE_MODE=fingerprint-hybrid`,
`CANON_DP_DISTINCT_SCHEDULE=first-group-warmup`,
`CANON_DP_FINITE_FETCH=batched-commit`, and `CANON_P71_SCAN=fwd`.
`CANON_DP_COLLECTIVE_REDUCE` remains absent. P74 itself has no flag: it is the
checked-VMA source path selected by `CANON_P59_CHECKED_VMA=1`.

This optimized Zero route is no-eval, checkpoint-disabled, strict, APC-off,
and fixed at 300 updates. It does not change the Native/no-IS or Native/IS
comparison arms described below. Target performance and full-horizon results
remain `TARGET NOT RUN` for this rollout.

## START HERE — P57.1c Perf v2 step-boundary repair passed one-host G4

This section supersedes the Wave 15 incident queue below.

Wave 15's Step-1 crash is localized to the learner publishing the next prompt
batch before committing the completed step's Perf v2 timeline.  The producer
opened a new `rollout` span, then the old `Timeline.commit_step()` purged it;
the context later exited against an empty stack.  This was an observer
lifecycle failure after one valid strict update, not a numerical or Pathways
failure.

P57.1c source CL `ec9884e9` now:

1. loads the next batch but exports/commits Perf v2 before publishing it to the
   asynchronous producer;
2. publishes the next batch before P45 host GC, preserving the existing GC /
   rollout overlap;
3. serializes host span entry/exit with step commit; and
4. rejects an active-span commit before mutating the timeline instead of
   purging the span and creating a delayed underflow.

Local gates are green: the old learner-order contract reproduced RED, the
repaired contract passes 2/2, P57 CPU passes 172/172, the pinned image passes
the new `P57_PERF_V2_STEP_BOUNDARY_PASS` and the complete
`P45_EXACT_IMAGE_CPU_PASS`, full pinned-image timeline/tracer suites pass
17/17 and 34/34, V1 Phase4 passes 90/90, flag audit passes 395/395, and diff
hygiene passes.

Approved one-host v5p G4 is also green.  Fresh `r7` completed three optimizer
commits, 12/12 strict alignment rows with zero differing bytes, finite nonzero
updates, and Step-1 rollout without purge/underflow.  Its target-step Perfetto
is readable and contains the five semantic operations executed by this beta-0
workload.  `reference_inference` is correctly absent because beta and forced KL
are both disabled; the classifier now requires that absence instead of
inventing an operation.  The add-only PASS is
`/mnt/disks/tunix-data/logp_probe_1host/p57_perf_v2_p57c_g4_cb38cf67_r7/classification.beta0.json`.
The original over-constrained RED classifier is preserved.

Claim ceiling: `IMPLEMENTED / HOST + PINNED-IMAGE + ONE-HOST G4 PASS / FULL
TARGET NOT RUN / SOURCE CL ec9884e9`.  Commit/push approval applies
only to this source-and-ledger stack.  Fresh P45/M15 full render and launch
remain separate later approvals.  Do not render a production manifest or
launch a full target from this handoff alone.

## START HERE — f45w15 Step-1 Timeline Tracer Underflow Incident Sealed

This section supersedes every later `START HERE` block for the next P45 action.

Wave 15 P45 (`canon-p57-fl-zero-f45w15-799a0bd1`, 64 TPU v5p) completed Step 0
with bitwise exact pre-alignment (0 differing bytes over 46,596 elements,
`verdict: PASS`) and optimizer commit 1 (`stable_norm=0.5510`). During Step 1
Rollout, parallel trajectory workers threw `ValueError: host-139531592390336: no
more spans to end.` from `tunix/perf/experimental/tracer.py:346` / `timeline.py:236`
due to an asynchronous span stack underflow under multi-threaded rollout.

All 19 component logs (3 head logs + 16 worker logs), `RAW_ERROR.log`, and
`INCIDENT_REPORT.md` are sealed under
`evidence/f45w15_timeline_tracer_incident/` with verified `SHA256SUMS`.
GCS mirror: `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p57/f45w15-799a0bd1/`.

## Historical — f45w10 is an unresolved source-worker loss, not a training-math red


This section supersedes every later `START HERE` block for the next P45 action.

Published source `96544812026677c7aeb5bb08b24d1ec1d554d3bd` ran optimized
Zero P45 as `canon-p57-fl-zero-f45w10-96544812`. The committed incident
summary reports Step 63/300, 44.5% solve, and about 2.9 minutes per step before
failure. All 14 retained non-source Pathways worker logs independently report
that worker 2 stopped sending at about 03:32:41; the 10-second pipe deadline
then caused distributed fail-closed teardown. The concurrent M15 run was
reported unaffected and continuing.

The evidence is incomplete at the decisive boundary: worker-2 stdout/stderr,
its Pod termination reason and exit code, Pod/JobSet events, node conditions,
the head log, terminal package, and SHA ledger are absent. Therefore the
correct classification is analysis-grade `INCONCLUSIVE_INFRA_SOURCE_MISSING`,
not a proven network/hardware fault and not a Zero-TIM/backward/loss/optimizer
failure. Do not increase the Pathways deadline or alter training mathematics
from this evidence.

Before another P45 launch, the user must choose one contract:

- maximum throughput: keep eval and checkpoint disabled, use a fresh run ID,
  and accept that any infrastructure loss restarts from step 0;
- resilient full train: implement a separately registered latest-1 rolling
  checkpoint/resume mode, measure its I/O cost, and rerun host/image admission
  before target use.

For either contract, arrange failure-time capture of the first disappearing
worker's current/previous logs, Pod termination JSON, events, node conditions,
and the head log before cleanup. No relaunch is authorized by this handoff.

The operator-side collector implementing that requirement is
`scripts/collect_jobset_logs_to_gcs.py`. Before applying any fresh P45 or M15
JobSet, start one collector per JobSet in its own persistent terminal with a
never-used local output directory and run-specific GCS prefix. It may start
before the JobSet exists and will wait without mutating the cluster. It follows
the head, both Pathways head sidecars, and worker indices `0..15`; snapshots
JobSet/Pod/event/node state; mirrors open files under `live/`; and seals a
checksummed terminal package under `sealed/`. Exact commands and acceptance
criteria are in `RUNBOOK.md` under `External worker-log collection`.

Collector `PASS` means only that the evidence package is complete. A failed
training JobSet can and should still have collector `PASS`. A missing worker
log, missing head log, nonterminal interruption, or upload error is collector
`INCONCLUSIVE`; it must never be rewritten as a training or numerical verdict.
The collector is host-tested only and remains `TARGET COLLECTOR NOT RUN` until
a real JobSet exercises it. This handoff does not authorize a launch, commit,
push, Pod restart, or cleanup.

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
implemented.  The P57.1c source-and-ledger stack has explicit publication
approval.  Any production render must still use the verified published source
SHA, and a full-target launch requires a separate explicit approval.
