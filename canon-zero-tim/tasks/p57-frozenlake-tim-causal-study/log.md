# Log

## 2026-08-20 UTC — P57 task bound and causal order preregistered

- Type: decision
- Fact: the existing P45 Qwen3-8B full-training geometry is DP8xTP8, while the fixed-lm-head registry has no K4096/TP8 entry. P45 in-training evaluation is also intentionally disabled after a prefix-cache reset timeout; a separate evaluator is required.
- Action: bound a new phase workflow that first admits the zero/mismatch treatment contract, then selects FrozenLake difficulty using zero-TIM outcomes only, freezes the recipe, admits a measurable mismatch dose, and finally runs a paired multi-seed study.
- Command: `git fetch origin yuxzhang/canon-zero-tim`; `git worktree add -b local/p57-frozenlake-tim-study /home/yuxuan/code_rl_repro/worktrees/p57_frozenlake_tim_0820 5f2d016147a55c032ea7b89b156a583d3b4ca7e8`
- Result: clean named worktree created; P57.0 is the only active phase. No source code changed, no TPU workload launched, and no commit or push performed.
- Files/artifacts: `state.md`; `plan.md`; `phases/p57-0-readiness.md` through `phases/p57-5-analysis.md`
- Rollback: remove the uncommitted task directory and worktree; no production path has changed.
- Next: user reviews the preregistered flow. If approved, implement P57.0 only.

## 2026-08-21 UTC — P57.0 local machinery implemented

- Type: implementation/evidence
- Fact: Qwen3-8B DP8xTP8 needed a distinct K4096/TP8 fixed-head registration and P45's quarantined in-process evaluation could not serve as a scientific endpoint.
- Action: added the N18992→19200 fixed-head geometry and receipts; created paired P57 train/eval renderer and profile; added strict zero vs warning-only A-B treatment admission; added an isolated held-out evaluator with step-0 base-weight and positive-boundary checkpoint provenance; added no-update classifier and checkpoint/renderer/receipt tests; registered P57 flags and wrote `RUNBOOK.md`.
- Correction: the first draft proposed one eval generation and update 25. GRPO rejects groups of one, and update 25 is not a retained 10-step checkpoint. The admitted contract uses two deterministic generations and schedules 0/20/50/... .
- Command: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh`
- Result: `Ran 44 tests ... OK`; `P57_FROZENLAKE_TIM_CPU_PASS`. Profile sourcing passed for zero/mismatch training, step-0 eval, and resumed eval. No TPU launched.
- Files/artifacts: fixed-head registry/TP8 contract; P57 profile/renderer/classifier/tests; learner evaluation-only path; checkpoint provenance validator; `RUNBOOK.md`.
- Rollback: all changes are additive/default-off under the P57 profile except the new dormant registry entry; discard the uncommitted worktree diff to restore the base exactly.
- Next: run final static and exact-image gates. Target hardware admission remains user-gated.

## 2026-08-21 UTC — P57.0 pinned-image admission passed

- Type: validation/evidence
- Fact: the bare host lacks the pinned image's complete Tunix dependency set, so the evaluator lifecycle belongs to the exact-image gate rather than the host-only test runner.
- Action: kept the 44 dependency-light P57/profile/receipt tests in `run_cpu.sh`; added the no-training-update evaluator lifecycle test to the qwen8b_tp8 exact-image suite; ran both gates and checked the 307-name flag inventory.
- Command: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh`; `bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh`.
- Result: host `44/44` PASS. Pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` matched all 34 overlay files; base suite `108/108`, P45 suite `40/40`, PEFT `2/2`, Agentic `3/3`; fixed-head K4096/TP8 forward/VJP and overlay probes PASS; terminal marker `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. Flag appendix is 307/307 unique. No TPU launched.
- Files/artifacts: exact-image runner; P57 evaluator lifecycle test; `state.md`; `RUNBOOK.md`.
- Rollback: discard the uncommitted worktree diff; no commit, push, or external workload exists.
- Next: after explicit approval, render from a committed immutable SHA and run the step-0 isolated evaluation before the bounded 20-update pair.

## 2026-08-21 UTC — workload selection changed to stock-only before zero unblinding

- Type: decision/implementation
- Fact: the scientific target is a workload whose untreated stock run finishes near 60–70% solve. Using zero-TIM to select that workload is slower and can bias the later arm comparison. LatestN(1) also cannot retain an uninterrupted 200-step run's intermediate checkpoints.
- Action: preregistered c1/c2/c3 deterministic materialized workloads with disjoint scout/confirm/main splits; made scout/confirm renderer paths stock-only with fixed head off; set scout endpoints to 0/20 and confirmation endpoints to 0/200; added an endpoint selection classifier with an ideal 60–70% band and a 55–75% review guardrail; added compact log markers, cold-agent handoff, and an exact execution runbook.
- Result: discovery cannot render a zero arm, paired scout/confirm rendering is rejected, dataset/candidate/split are signed checkpoint/evaluation provenance, and raw evaluation logs can feed the aggregate classifier. Final repeated local/image gates remain pending. No TPU launched, commit, or push performed.
- Files/artifacts: `examples/frozenlake/p57_workloads.py`; P57 renderer/profile/evaluator/classifiers/tests; `plan.md`; `HANDOFF.md`; `RUNBOOK.md`; `phases/p57-1-stock-discovery.md`.
- Rollback: discard the uncommitted P57 worktree changes; all new paths are additive/default-off.
- Next: run final gates and diff review, then ask the user for commit approval. The first authorized target action is c1 stock eval-0, not zero-TIM.

## 2026-08-21 UTC — stock-discovery implementation passed final local admission

- Type: validation/evidence
- Fact: stock-only workload discovery must be reproducible without observing a zero-TIM learning outcome, and the later causal study must reject any mismatch-arm run whose discrepancy escapes the A-B treatment boundary.
- Action: materialized every c1/c2/c3 scout/confirm/main dataset pair; ran the dependency-light P57 suite; reran the exact pinned-image contract after adding candidate/split provenance, stock endpoint classification, and the P57 A-B-only postflight classifier.
- Command: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh`; deterministic full-materialization gate for all nine candidate/split pairs; `bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh`; `git diff --check`.
- Result: host `75/75` PASS with `P57_FROZENLAKE_TIM_CPU_PASS`; all nine 10,000-train/100-eval materializations PASS with disjoint seeds and maps; pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` matched all 34 overlay files; base `109/109`, P45 `40/40`, PEFT `2/2`, Agentic `3/3`; fixed-head K4096/TP8 forward/VJP and overlay probes PASS; terminal marker `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. Flag inventory is 309/309 unique. `git diff --check` is clean. No TPU launched, commit, or push performed.
- Files/artifacts: stock P57 renderer/profile; deterministic workload materializer; isolated evaluator and endpoint classifier; P57 A-B-only scientific postflight; `HANDOFF.md`; `RUNBOOK.md`; phase/state/log documents.
- Rollback: discard the uncommitted P57 worktree diff; no external state changed.
- Next: review the final diff, then seek separate approval for an immutable commit. The first hardware action remains c1 stock eval-0; neither zero-TIM nor a paired arm is authorized during discovery.

## 2026-08-21 UTC — train-20 discovery superseded by four-recipe rollout calibration

- Type: decision/implementation/evidence
- Fact: a 20-update screen cannot cheaply establish the intended 200-update convergence target, while immutable base-policy rollouts can measure initial solve, group heterogeneity, usable advantage, and context pressure without bias from zero-TIM. The user replaced Easy-Mix's 9x9 ceiling with 10x10.
- Action: replaced c1/c2/c3 scout/confirm discovery with deterministic L0 (2x2–9x9/5 turns/6144), M10 (5x5–10x10/10/8192), M15 (5x5–12x12/15/12288), and M20 (5x5–15x15/20/16384). Added balanced constructive maps, calibration/selection/main data namespaces, a two-JobSet stock renderer, raw-rollout receipts that skip trainer recomputation, a fail-closed classifier, profile gates, exact runbook, and cold-agent handoff. Calibration loads the model twice total (greedy and stochastic), not once per recipe.
- Command: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh`; `bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh`; `git diff --check`.
- Result: host `68/68` PASS with `P57_FROZENLAKE_TIM_CPU_PASS`, including a physical prompt/response cap-hit negative. The hardest M20 `selection` path materialized and attested 10,000 train + 100 eval rows (`train_sha=ed34e1ca...`, `eval_sha=9175a509...`). Pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` matched 34/34 overlay files; base `109/109`, P45 `40/40`, PEFT `2/2`, Agentic `4/4`; fixed-head and Qwen8B TP8 probes PASS; terminal `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. The raw-rollout unit proved trainer recomputation unreachable and training steps unchanged. No TPU, commit, or push.
- Files/artifacts: `examples/frozenlake/p57_workloads.py`; `cluster/render_p57_calibration.py`; P57 profile and paired renderer; rollout-only learner/trajectory receipts; stock classifier/tests; `RUNBOOK.md`; `HANDOFF.md`; phase/plan/state/flag records.
- Rollback: discard the uncommitted P57 worktree diff. All new admission paths are default-off outside the P57 profile.
- Next: user reviews the diff and separately authorizes an immutable commit. First future hardware action is the two-manifest stock calibration; zero-TIM remains blinded.

## 2026-08-21 UTC — calibration reduced to one stochastic M10/M15/M20 JobSet

- Type: user decision/implementation/evidence
- Fact: temperature-0.7 eight-generation rollouts directly measure the distribution used by training, including mixed groups and usable advantage. The greedy mode required a second model load, and L0 could not be selected; neither was necessary for workload admission.
- Action: reduced the calibration inventory to M10/M15/M20 and the execution contract to one stochastic DP8xTP8 JobSet (100 maps x 8 generations per recipe). Reconciled renderer, profile, training entrypoint, classifier, tests, flag registry, plan, phase, runbook, and handoff. Selection now chooses the eligible stochastic solve rate closest to 20%, with exact ties M15 then M10 then M20. Added hard negatives for greedy/L0 intent.
- Command: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh`; `bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh`; `git diff --check`.
- Result: host `70/70` PASS with `P57_FROZENLAKE_TIM_CPU_PASS`. Pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` matched 34/34 overlay files; base `109/109`, P45 `40/40`, PEFT `2/2`, Agentic `4/4`; fixed-head and Qwen8B TP8 probes PASS; terminal `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. No TPU, commit, or push.
- Files/artifacts: `cluster/render_p57_calibration.py`; P57 profile/training entrypoint/classifier/tests; `plan.md`; `phases/p57-1-stock-discovery.md`; `RUNBOOK.md`; `HANDOFF.md`; `state.md`.
- Rollback: discard the uncommitted P57 concern. All admission behavior remains default-off outside the P57 profile.
- Next: review and separately approve an immutable commit. The only future calibration launch is `jobset-p57-frozenlake-calibration-stochastic.yaml`.

## 2026-08-21 UTC — stock calibration corrected from fixed-head-off to full zero-TIM-off

- Type: user correction/implementation/evidence
- Fact: `CANON_P38_FIXED_LM_HEAD=0` did not produce an untreated inference stack because `_canonical_engine.env` and the P45 parent still enabled fixed AR, pinned RPA, processed logprobs, canonical Pallas trunk/VJP, engine-module C, and segmented alignment/training machinery. Calling that configuration stock-fast would have confounded workload selection.
- Action: registered `CANON_P57_INFERENCE_REGIME=stock-fast`; made calibration unset 12 presence-sensitive numerical switches and set 25 boolean/admission gates to zero; removed the canonical excess-precision XLA pin; added render, resolved-profile, training-entrypoint, receipt-v2, manifest-preflight, and classifier enforcement; allowed the unadmitted DP8xTP8 mesh only for this exact rollout-only regime; rewrote the phase plan, runbook, and cold-agent handoff. The later paired study is now explicitly a complete numerical zero-TIM bundle comparison and remains unadmitted until P57.2.
- Command: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh`; `bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh`; `git diff --check`.
- Result: host `73/73` PASS with `P57_FROZENLAKE_TIM_CPU_PASS`. The resolved-profile test executed `00_env.sh` and observed `[P57.STOCK_FAST] ZERO_TIM_OFF_PASS absent=12 zero=25`. Pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` matched 34/34 overlay files and passed base `109/109`, P45 `40/40`, PEFT `2/2`, Agentic `4/4`, stock-fast contract `3/3`, fixed-head and Qwen8B TP8 probes; terminal `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. `git diff --check` is clean. No TPU, commit, or push.
- Files/artifacts: P57 calibration renderer/profile; `cluster/steps/00_env.sh`; `examples/frozenlake/train_frozenlake_qwen3.py`; `tunix/rl/dp_workloads.py`; manifest verifier; receipt classifier/tests; `FLAGS.md`; `plan.md`; phases 0–3; `RUNBOOK.md`; `HANDOFF.md`; `state.md`.
- Rollback: discard the uncommitted P57 concern. Ordinary P45 and all non-P57 paths retain their existing contracts.
- Next: user reviews the diff and separately approves an immutable commit. Then render one stock-fast calibration manifest, pass its mechanical verifier, and separately approve the 64-chip launch. Stop after offline classification.
