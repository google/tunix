# P57 two-workload, three-arm runbook

This runbook renders and verifies the six concept-study cells. It never launches
a JobSet automatically. Every `kubectl apply`, commit, and push requires separate
user approval. Never hand-edit a rendered YAML.

## Experimental matrix

| Wave | Renderer arm | Numerical program | `sampler_is` | Runtime receipt |
|---|---|---|---|---|
| Queue now: `native` | `mismatch` | complete zero-TIM bundle off | `none` | old=A, TIS absent |
| Queue now: `is` | `is` | identical native program | `token` | old=C, TIS present |
| Deferred: `zero` | `zero` | complete zero-TIM bundle on | `none` | old=A, TIS absent |

Each wave has two 64-chip DP8xTP8 JobSets:

- P45 original: generated train/eval parameters use seeds 42/123, grid side
  2–9 and p 0.60–0.85, five turns, prompt/response 4,096/2,048. The historical
  recipe and every paired P57 treatment run 450 updates.
- M15: materialized `m15/main`, grid side 5–12 and p 0.82, 15 turns,
  prompt/response 4,096/8,192, 450 updates.

Both use 32 prompts x eight generations, temperature 0.7, GSPO-token/RLOO,
AdamW 1e-6 and resident optimizer state. Training evaluation is disabled.
Checkpoints are written every 10 updates. The recovery policy still keeps only
the latest ordinary checkpoint; P57 additionally preserves every 50-step
milestone through update 450 so isolated evaluation can restore an exact older
step after training finishes. These are temporary experiment evidence, not
extra recovery generations.

Storage warning: this workload trains the full Qwen3-8B actor in FP32 and its
checkpoint also contains optimizer state. A retained milestone is therefore a
full distributed checkpoint; TP/DP sharding reduces each worker's transfer but
does not reduce aggregate GCS bytes. Nine retained positive milestones can be
on the order of one terabyte per arm (several terabytes for the immediate four
arms). Before launch, the owner must explicitly accept the bucket quota/cost.
Do not replace this with segmented stop/eval/resume or in-process evaluation:
either would change the registered execution contract. Delete milestones only
after every classifier artifact is durable and only with separate destructive
approval.

## Local gates

From the exact source worktree:

~~~bash
bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh
bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh
git diff --check
~~~

Require terminal `P57_FROZENLAKE_TIM_CPU_PASS`,
`P57_STOCK_RUNTIME_MATRIX_PASS variants=5 stages=train,eval`,
`P57_STOCK_POST_BACKWARD_MODULE_C_PASS arms=mismatch,is`,
`P57_STOCK_POST_BACKWARD_MODULE_C_NEGATIVE_PASS arm=unknown`,
`P57_STOCK_OBSERVER_EXACT_IMAGE_PASS targets=absolute values=processed`, and
`P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. Local gates are construction
evidence, not target evidence.

The runtime matrix marker is mandatory. It proves the pinned production image
accepted exactly the five registered stock tuples: the historical M15
selection discovery cell plus P45/M15-main under `mismatch` and `is`. Arbitrary
arm/workload/split combinations remain fail-closed.

## Queue now — fresh four-job 450-update campaign

The earlier 200-update identities are immutable historical evidence, not valid
members of the new 450-update comparison. Package their current terminal state,
ensure none remains live, and do not resume their checkpoints into this
campaign. Render both native-program waves from the approved pushed horizon
change using fresh run IDs, output roots, campaign roots, and checkpoint
namespaces. Never hand-edit a rendered manifest.

~~~bash
cd /home/yuxuan/code_rl_repro/worktrees/p57_frozenlake_tim_0820
SOURCE=<approved-pushed-450-horizon-40-character-sha>
OUT_NATIVE=/tmp/p57-primary-native-450-a
OUT_IS=/tmp/p57-primary-is-450-a
OUT_EVAL_NATIVE=/tmp/p57-eval-native-450-a
OUT_EVAL_IS=/tmp/p57-eval-is-450-a
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  native "$SOURCE" "$OUT_NATIVE" n45c n15c p57-native-450-a
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  is "$SOURCE" "$OUT_IS" i45c i15c p57-is-450-a
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_eval_schedule.sh \
  native "$SOURCE" "$OUT_EVAL_NATIVE" en p57-native-450-a
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_eval_schedule.sh \
  is "$SOURCE" "$OUT_EVAL_IS" ei p57-is-450-a
~~~

The suggested `*c` IDs are valid only if cluster inspection confirms they were
never used; otherwise choose new four-character IDs. The first attempts used
`p45n/m15n/p45i/m15i` and `n45a/n15a/i45a/i15a`; never reuse them. Four-character
IDs avoid the Kubernetes 63-character Pod-name limit. All four jobs use
`checkpoint-mode=new`.

The two evaluation renderers must each report
`P57_EVAL_SCHEDULE_PASS ... manifests=20
steps=0,50,100,150,200,250,300,350,400,450` and
`P57_EVAL_RENDER_PASS ... manifests=20`. They reuse the exact training campaign
tags by construction. Do not change a campaign root between its training and
evaluation renders.

Before applying a training YAML, inspect the target GCS quota and record the
storage decision in the campaign evidence. `max_to_keep=1` limits ordinary
recovery points; the nine `EveryNSteps(50)` evidence points are additional
retained full checkpoints. Insufficient storage blocks launch—it is not
permission to silently drop intermediate evaluations.

For each command require two `P57_THREE_ARM_MANIFEST_PASS` lines and its
terminal markers:

~~~text
P57_THREE_ARM_WAVE_PASS wave=native manifests=2
P57_THREE_ARM_RENDER_PASS wave=native ...
P57_THREE_ARM_WAVE_PASS wave=is manifests=2
P57_THREE_ARM_RENDER_PASS wave=is ...
~~~

The manifests are:

~~~text
$OUT_NATIVE/p45/jobset-p57-frozenlake-mismatch-450.yaml
$OUT_NATIVE/m15/jobset-p57-frozenlake-mismatch-m15-main-450.yaml
$OUT_IS/p45/jobset-p57-frozenlake-is-450.yaml
$OUT_IS/m15/jobset-p57-frozenlake-is-m15-main-450.yaml
~~~

### Launch order

Step-0 evaluation must run before training because its `new`-mode provenance
gate requires the campaign checkpoint namespace to be empty. After separate
launch approval, launch and fully package these four step-0 evaluators first:

~~~bash
kubectl apply -f "$OUT_EVAL_NATIVE/p45/step-0/jobset-p57-frozenlake-mismatch-eval-0.yaml"
kubectl apply -f "$OUT_EVAL_NATIVE/m15/step-0/jobset-p57-frozenlake-mismatch-m15-main-eval-0.yaml"
kubectl apply -f "$OUT_EVAL_IS/p45/step-0/jobset-p57-frozenlake-is-eval-0.yaml"
kubectl apply -f "$OUT_EVAL_IS/m15/step-0/jobset-p57-frozenlake-is-m15-main-eval-0.yaml"
~~~

Only after all four step-0 evaluations classify PASS, obtain launch approval
for the uninterrupted training jobs:

~~~bash
kubectl apply -f "$OUT_NATIVE/p45/jobset-p57-frozenlake-mismatch-450.yaml"
kubectl apply -f "$OUT_NATIVE/m15/jobset-p57-frozenlake-mismatch-m15-main-450.yaml"
kubectl apply -f "$OUT_IS/p45/jobset-p57-frozenlake-is-450.yaml"
kubectl apply -f "$OUT_IS/m15/jobset-p57-frozenlake-is-m15-main-450.yaml"
~~~

Only apply after confirming none of the earlier P57 JobSets remains live and
after separate user launch approval. The two waves share source, image, data,
horizon, optimizer, and objective. Their registered treatment difference is
only token importance sampling and the corresponding old-logprob identity.

Once all four trains close durably at 450, obtain separate evaluation launch
approval. Apply the step 50 through 450 manifests from each schedule. They may
run in any resource-aware order because they are read-only and restore their
explicit `CANON_P57_EVAL_CHECKPOINT_STEP`; they must not silently restore the
latest checkpoint. Every positive milestone requires a restore receipt for the
requested step and the no-update completion marker. Do not delete milestones
until every corresponding classifier artifact is packaged; cleanup is a
separate destructive action requiring explicit approval.

## Deferred Zero-TIM wave

Do not render or launch this wave as part of the current four-job queue. After
the native/no-IS versus native/token-IS evidence is packaged, the user may
separately promote the two Zero-TIM cells. Keep the same approved source SHA
unless a later reviewed repair requires a new immutable source for all affected
comparisons.

~~~bash
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  zero "$SOURCE" /tmp/p57-deferred-zero-450-a z45a z15a p57-zero-450-a
~~~

Rendering success does not authorize this deferred wave. Obtain a separate
launch decision after classifying and packaging the first four jobs.

## Runtime receipts

Every full train must produce exactly one arm receipt:

~~~text
# native/no-IS and zero/no-IS
[P57.TIM_PURITY] PASS sampler_is=none old_logps=rollout tis_weights=absent trainer_rescore=observer-only

# native/token-IS
[P57.TIM_PURITY] PASS sampler_is=token old_logps=trainer tis_weights=present trainer_rescore=training-input
~~~

Native arms must additionally emit zero-TIM-off and stock-route receipts,
including `canonical_markers=0`; zero arms must execute the canonical markers
and remain strict A=B=C. Native A-B is warning-only. B-C, nonfinite,
structural, transaction, optimizer and checkpoint failures are fatal in every
arm. Missing or duplicate purity receipts invalidate the run.

Every train must finish update 450. Healthy trains are
not intentionally paused. A restart or node loss is `INCONCLUSIVE` until the
user chooses whether to resume from the retained checkpoint.

## Evidence return

For each JobSet, return and package:

~~~text
workload/arm/source_sha/image_id:
jobset/run_id/attempt/exit:
yaml_path/sha256:
complete_log_path/sha256:
resolved_env_path/sha256:
arm_purity_marker:
stock_off_or_canonical_receipts:
segment_preflight/segment_complete:
final_checkpoint_step/uri:
alignment: A-B dose, B-C verdict, nonfinite/structural verdict
training: solve curve, sampled tokens/s, seconds/update, grad/update norms
sampler_is: weight mean/max/clip fraction or verified absent
infra_events:
classification/verdict artifact/sha256:
wandb_run:
~~~

Use `canon-zero-tim/scripts/package_run.sh` for every returned run directory.
Incomplete evidence is preserved and classified `INCONCLUSIVE`, never silently
rerun or overwritten.

## Milestone evaluation and analysis

For each valid cell, classify all ten isolated checkpoints at
`0,50,100,150,200,250,300,350,400,450`. Evaluation is deterministic
temperature-0, uses the same immutable held-out maps within a workload, and
performs no backward, optimizer commit, or checkpoint write. Positive steps
restore exactly the named checkpoint even when a later checkpoint exists.
Compute only within-workload contrasts and compare equal steps:

- IS effect: `native-is - native-no-IS`;
- zero-TIM effect: `zero-no-IS - native-no-IS`;
- exactness versus mitigation: `zero-no-IS - native-is`.

One curve per cell is a concept study. Multi-seed replication is a later gate,
not something the first six curves can claim.

## Rollback

The renderer/profile is P57-only and default-empty. Leaving P57 fields unset or
reverting the isolated P57 concern restores the historical P45 paths. Do not
edit or replace the existing P45 production renderer/profile.
