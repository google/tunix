# P57 three-arm FrozenLake execution handoff

## Mission

Run the first four cells of a two-workload x three-treatment concept study. The
immediate assignment is P45 and M15-main under native/no-IS and the identical
native program with token-IS (four JobSets). The two complete Zero-TIM/no-IS
cells are explicitly deferred. The execution agent runs reviewed scripts and
returns artifacts; it does not edit code, YAML, profiles, or scientific
parameters.

Read in order: `state.md` → `plan.md` →
`phases/p57-1b-three-arm-baselines.md` → `RUNBOOK.md`.

## Current truth

- M15 was selected before any zero learning result: initial solve 24.625%, 56%
  mixed groups, max context 7,403, no cap hit.
- P45 is the original generator-backed workload, not P57 `l0`.
- Runtime arm names are `mismatch` = native/no-IS, `is` = native/token-TIS,
  and `zero` = complete zero-TIM/no-IS.
- All four immediate jobs use Qwen3-8B DP8xTP8, resident optimizer, in-process
  evaluation off, and 450 updates. Checkpoints save every 10; recovery keeps
  the latest ordinary point and the P57 evidence policy retains every 50-step
  milestone for isolated evaluation.
- These are full FP32-actor plus optimizer checkpoints, not small LoRA deltas.
  Nine retained milestones can approach one terabyte per arm. Record explicit
  bucket quota/cost acceptance before launch; storage insufficiency is a hard
  blocker, not permission to alter the schedule.
- The first four-job attempt (`p45n/m15n/p45i/m15i`) is `INCONCLUSIVE`: all
  256 chips provisioned, but a stale discovery-only Python validator rejected
  every job before step 0. This is not training or TPU numerical evidence.
- The closed five-tuple registry repair is target-proven: `n45a` committed two
  steps and `n15a` committed one step before continuing rollout. Those
  200-update identities are now superseded for the 450-update comparison; keep
  and package them, but do not resume their checkpoints into the new campaign.
- `i45a` is `INCONCLUSIVE`. It passed rollout, pre-backward warning handling,
  arm purity, and backward, then failed in post-backward because the shared
  alignment gate recognized only stock `mismatch`, not stock `is`, when
  `CANON_ENGINE_MODULE_C=0`. The repair is exact-image certified; its repaired
  target path remains `TARGET NOT RUN` until a replacement IS job commits.
- The user superseded the common 200-update horizon with 450 before inspecting
  final arm outcomes. All four native-program cells must restart from the same
  initial checkpoint under fresh identities; the deferred zero cells must also
  use 450 when separately authorized.
- The first 450-update identities (`n45c/n15c/i45c/i15c`) are also immutable
  `INCONCLUSIVE` attempts. All four completed a real Step-0 rollout, then
  trajectory packaging lost the FrozenLake-rendered `prompts` when the newer
  policy-seeded environment record was selected. No trainer alignment,
  backward, optimizer update, or checkpoint occurred. The local compatibility
  repair preserves the DeepSWE environment task when it already has a prompt
  and merges only when FrozenLake carries the prompt in the trajectory task.
  It has passed both P45/P57 and P58 pinned-image gates; the repaired 64-chip
  path remains `TARGET NOT RUN` until a fresh attempt commits an update.

## Operator procedure for the fresh four-job campaign

1. Confirm the user supplied an approved, pushed, full 40-character SHA.
2. In that exact checkout, run:

~~~bash
bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh
bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh
git diff --check
~~~

Require the exact-image marker
`P57_STOCK_RUNTIME_MATRIX_PASS variants=5 stages=train,eval` in addition to
`P57_TRAJECTORY_PROMPT_PROVENANCE_PASS frozenlake=merge deepswe=environment reset_timeout=preserved missing_prompt=fail_closed`,
`P57_STOCK_POST_BACKWARD_MODULE_C_PASS arms=mismatch,is`, its unknown-arm
negative, and the terminal P57/P45 PASS markers. Missing a marker forbids
launch.

3. Confirm and package every earlier P57 JobSet and verify none remains live.
   Then render both two-job waves with no hand edits:

~~~bash
SOURCE=<approved-pushed-prompt-provenance-repair-40-character-sha>
OUT_NATIVE=/tmp/p57-primary-native-450-b
OUT_IS=/tmp/p57-primary-is-450-b
OUT_EVAL_NATIVE=/tmp/p57-eval-native-450-b
OUT_EVAL_IS=/tmp/p57-eval-is-450-b
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  native "$SOURCE" "$OUT_NATIVE" n45d n15d p57-native-450-b
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  is "$SOURCE" "$OUT_IS" i45d i15d p57-is-450-b
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_eval_schedule.sh \
  native "$SOURCE" "$OUT_EVAL_NATIVE" fn p57-native-450-b
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_eval_schedule.sh \
  is "$SOURCE" "$OUT_EVAL_IS" fi p57-is-450-b
~~~

The `*c` IDs and their `*-450-a` output/campaign roots were consumed by the
failed attempts and must never be reused. The commands above preregister fresh
`*d` train IDs, `fn/fi` evaluation roots, and `*-450-b` namespaces. Do not reuse
any earlier run ID, output directory, campaign root, or checkpoint namespace.
All four jobs remain
`checkpoint-mode=new`; no 200-update checkpoint is admitted as their initial
state.

4. Stop unless each train renderer reports two manifest passes plus its
   terminal wave/render markers, and each eval renderer reports exactly 20
   manifests plus the schedule `0,50,...,450` PASS marker. Record every YAML
   SHA-256 value. Also record explicit acceptance of the GCS budget for nine
   full positive-milestone checkpoints per arm.
5. Ask for explicit launch approval. First launch only the four step-0
   evaluators listed in `RUNBOOK.md`. They must complete and classify PASS
   before any training checkpoint exists.
6. Ask separately for training launch approval. Only after approval:

~~~bash
kubectl apply -f "$OUT_NATIVE/p45/jobset-p57-frozenlake-mismatch-450.yaml"
kubectl apply -f "$OUT_NATIVE/m15/jobset-p57-frozenlake-mismatch-m15-main-450.yaml"
kubectl apply -f "$OUT_IS/p45/jobset-p57-frozenlake-is-450.yaml"
kubectl apply -f "$OUT_IS/m15/jobset-p57-frozenlake-is-m15-main-450.yaml"
~~~

7. Monitor all four jobs independently. Do not cancel one arm because another
   fails and do not modify a campaign tag and relaunch automatically.
8. Package every success or failure with `scripts/package_run.sh`; preserve the
   raw log from byte zero and the resolved environment.
9. After all four trains close durably at 450, ask separately before launching
   the step-50 through step-450 evaluation manifests. Each evaluator must
   restore its named step, emit the no-update COMPLETE marker, and be classified
   with `scripts/classify_checkpoint_eval.py`. Never substitute latest=450 for
   an earlier requested milestone.

## Required four-job treatment proof

The two native/no-IS logs require exactly once:

~~~text
[P57.TIM_PURITY] PASS sampler_is=none old_logps=rollout tis_weights=absent trainer_rescore=observer-only
~~~

The two native/token-IS logs require exactly once:

~~~text
[P57.TIM_PURITY] PASS sampler_is=token old_logps=trainer tis_weights=present trainer_rescore=training-input
~~~

All four training logs require the stock-fast zero-TIM-off receipt,
`canonical_markers=0`, stock observer receipt, warning-only A-B treatment dose,
B-C validity, segment preflight, and terminal completion. Every job completes
at step 450. No Zero-TIM receipt is expected in this queue.

Finite A-B is the treatment, not a failure. B-C mismatch, nonfinite values,
structural/replica/transaction/optimizer/checkpoint failures, missing arm
receipt, wrong sampler mode, canonical marker leakage, restarts, and incomplete
logs are hard stop or `INCONCLUSIVE` conditions.

## What to return

Return one block per workload using the template in `RUNBOOK.md`, including:

- exact source/image/jobset/run/attempt identity;
- YAML, full log, resolved env, classification and checkpoint hashes/paths;
- exact purity, stock-route, segment-preflight and completion marker lines;
- A-B dose and B-C/nonfinite/structural verdict;
- solve curve, step timing, sampled tokens/s, grad/update norms;
- confirmation that sampler-IS weights are absent in `mismatch` and present in
  `is`, including mean/max/clip fraction for the IS jobs;
- every infrastructure event and the W&B run identity.

For every isolated evaluation also return requested/restored checkpoint step,
checkpoint tag, evaluation JSON and classifier JSON with SHA-256, exact
held-out dataset SHA, solve/reward, wall time, and the single
`backward=0 optimizer_commits=0 checkpoint_writes=0` COMPLETE line. A complete
curve has exactly ten steps: `0,50,100,150,200,250,300,350,400,450`.

Do not summarize away a failure. If GCS artifacts are too large, run the
repository classifier next to the bucket and return its complete JSON plus
input inventory/SHA ledger; never claim completeness from SHA validity alone.

## Deferred Zero-TIM wave

Do not launch `zero` in the current assignment. After these four runs are
packaged, the user may separately authorize the same script with `zero` for P45
and M15. The final six-cell analysis still compares arms only within P45 or
only within M15.

## Rollback

No historical P45 file is replaced. The study is isolated behind the P57
profile and explicit arm selector; leaving those fields unset restores existing
behavior. Do not use broad reset/checkout operations to undo a failed run.
