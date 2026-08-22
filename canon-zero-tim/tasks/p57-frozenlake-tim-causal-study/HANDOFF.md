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
- All four immediate jobs use Qwen3-8B DP8xTP8, resident optimizer, evaluation off,
  checkpoint every 10 and LatestN(1). Every arm/workload runs 200 updates.
- The first four-job attempt (`p45n/m15n/p45i/m15i`) is `INCONCLUSIVE`: all
  256 chips provisioned, but a stale discovery-only Python validator rejected
  every job before step 0. This is not training or TPU numerical evidence.
- The repair replaces that single hardcoded tuple with a closed five-tuple
  registry. It is locally exact-image certified but is not target-certified
  until a published immutable repair SHA passes the gates below. Do not render
  from a dirty or unpushed tree.

## Operator procedure for the four native-program jobs

1. Confirm the user supplied an approved, pushed, full 40-character SHA.
2. In that exact checkout, run:

~~~bash
bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh
bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh
git diff --check
~~~

Require the exact-image marker
`P57_STOCK_RUNTIME_MATRIX_PASS variants=5 stages=train,eval` in addition to
the terminal P57/P45 PASS markers. Missing this marker means the repair was not
tested and launch is forbidden.

3. Render all four manifests with no hand edits:

~~~bash
SOURCE=<approved-pushed-repair-40-character-sha>
OUT_NATIVE=/tmp/p57-primary-native-b
OUT_IS=/tmp/p57-primary-is-b
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  native "$SOURCE" "$OUT_NATIVE" n45a n15a p57-native-is-b
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  is "$SOURCE" "$OUT_IS" i45a i15a p57-native-is-b
~~~

Do not reuse the failed attempt's run IDs, output directories, or
`p57-native-is-a` campaign root. Four-character replacement IDs are deliberate:
longer examples have already exceeded generated Pod-name limits. Both waves
must remain `checkpoint-mode=new`; there is no step-0 checkpoint to resume.

4. Stop unless each renderer reports two manifest passes plus its terminal wave
   and render PASS markers. Record all four YAML SHA-256 values.
5. Ask for explicit launch approval. Only after approval:

~~~bash
kubectl apply -f "$OUT_NATIVE/p45/jobset-p57-frozenlake-mismatch-200.yaml"
kubectl apply -f "$OUT_NATIVE/m15/jobset-p57-frozenlake-mismatch-m15-main-200.yaml"
kubectl apply -f "$OUT_IS/p45/jobset-p57-frozenlake-is-200.yaml"
kubectl apply -f "$OUT_IS/m15/jobset-p57-frozenlake-is-m15-main-200.yaml"
~~~

6. Monitor all four independently. Do not cancel healthy jobs because another
   fails. Do not modify the campaign tag and relaunch automatically.
7. Package every success or failure with `scripts/package_run.sh`; preserve the
   raw log from byte zero and the resolved environment.

## Required four-job treatment proof

The two native/no-IS logs require exactly once:

~~~text
[P57.TIM_PURITY] PASS sampler_is=none old_logps=rollout tis_weights=absent trainer_rescore=observer-only
~~~

The two native/token-IS logs require exactly once:

~~~text
[P57.TIM_PURITY] PASS sampler_is=token old_logps=trainer tis_weights=present trainer_rescore=training-input
~~~

All four logs require the stock-fast zero-TIM-off receipt,
`canonical_markers=0`, stock observer receipt, warning-only A-B treatment dose,
B-C validity, segment preflight, and terminal completion. Every job completes
at step 200. No Zero-TIM receipt is expected in this queue.

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
