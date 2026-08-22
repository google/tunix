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
  recipe ran 450 updates; this study freezes it to the common 200-update horizon.
- M15: materialized `m15/main`, grid side 5–12 and p 0.82, 15 turns,
  prompt/response 4,096/8,192, 200 updates.

Both use 32 prompts x eight generations, temperature 0.7, GSPO-token/RLOO,
AdamW 1e-6 and resident optimizer state. Training evaluation is disabled.
Checkpoints are written every 10 updates to GCS with LatestN(1).

## Local gates

From the exact source worktree:

~~~bash
bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh
bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh
git diff --check
~~~

Require terminal `P57_FROZENLAKE_TIM_CPU_PASS`,
`P57_STOCK_OBSERVER_EXACT_IMAGE_PASS targets=absolute values=processed`, and
`P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. Local gates are construction
evidence, not target evidence.

## Queue now — render native/no-IS and native/token-IS

Use the approved, pushed 40-character source SHA. The output path must not
already exist. Campaign root plus `-p45/-m15` and the rendered arm becomes the
checkpoint namespace, so a fresh `new` rerun must use a new campaign root.

~~~bash
cd /home/yuxuan/code_rl_repro/worktrees/p57_frozenlake_tim_0820
SOURCE=<approved-pushed-40-character-sha>
OUT_NATIVE=/tmp/p57-primary-native
OUT_IS=/tmp/p57-primary-is
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  native "$SOURCE" "$OUT_NATIVE" p57p45n1 p57m15n1 p57-native-is-a
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  is "$SOURCE" "$OUT_IS" p57p45is1 p57m15is1 p57-native-is-a
~~~

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
$OUT_NATIVE/p45/jobset-p57-frozenlake-mismatch-200.yaml
$OUT_NATIVE/m15/jobset-p57-frozenlake-mismatch-m15-main-200.yaml
$OUT_IS/p45/jobset-p57-frozenlake-is-200.yaml
$OUT_IS/m15/jobset-p57-frozenlake-is-m15-main-200.yaml
~~~

After separate launch approval only:

~~~bash
kubectl apply -f "$OUT_NATIVE/p45/jobset-p57-frozenlake-mismatch-200.yaml"
kubectl apply -f "$OUT_NATIVE/m15/jobset-p57-frozenlake-mismatch-m15-main-200.yaml"
kubectl apply -f "$OUT_IS/p45/jobset-p57-frozenlake-is-200.yaml"
kubectl apply -f "$OUT_IS/m15/jobset-p57-frozenlake-is-m15-main-200.yaml"
~~~

All four may run concurrently when four independent 64-chip slices are
available because their checkpoint tags and JobSet identities are disjoint.
Do not launch two jobs with the same arm/workload campaign tag.

## Deferred Zero-TIM wave

Do not render or launch this wave as part of the current four-job queue. After
the native/no-IS versus native/token-IS evidence is packaged, the user may
separately promote the two Zero-TIM cells. Keep the same approved source SHA
unless a later reviewed repair requires a new immutable source for all affected
comparisons.

~~~bash
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  zero "$SOURCE" /tmp/p57-deferred-zero p57p45z1 p57m15z1 p57-zero-a
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

Every run must finish update 200. Healthy runs are
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

## Final evaluation and analysis

After all six full horizons are valid, render isolated final-checkpoint evals
from the same source/campaign/arm, using `checkpoint-mode=resume` and checkpoint
step 200 for both workloads. All arms use the same held-out set within their
workload. Compute only within-workload contrasts:

- IS effect: `native-is - native-no-IS`;
- zero-TIM effect: `zero-no-IS - native-no-IS`;
- exactness versus mitigation: `zero-no-IS - native-is`.

One curve per cell is a concept study. Multi-seed replication is a later gate,
not something the first six curves can claim.

## Rollback

The renderer/profile is P57-only and default-empty. Leaving P57 fields unset or
reverting the isolated P57 concern restores the historical P45 paths. Do not
edit or replace the existing P45 production renderer/profile.
