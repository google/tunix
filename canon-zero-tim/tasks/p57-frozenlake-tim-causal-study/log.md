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

## 2026-08-21 UTC — p57cal2 exposed import-time canonical overlay dependency

- Type: target failure/correction
- Fact: the 64-chip attempt connected all devices, materialized all three 100-row datasets, and reached vLLM model loading. It then failed importing the overlaid `linear_p22xi.py`: `RuntimeError: P22.XI: CANON_PALLAS_MPAD=1 required`. The committed run log has 393 lines, SHA-256 `a14d460b1e0954e5ec39a7e126611bf9f0bdca453d9cb28f0925739a00dbc2ef`.
- Correction: the prior claim that an installed canonical overlay would fall through to vendor/native code when its flags were absent was false. Several shims validate dependencies at import time.
- Action: route the exact P57 stock-fast calibration around canonical install/overlay/verify while retaining the pinned-image probe, RoPE compatibility decision, and R2E gym install. Add the opposite postflight contract: stock-fast requires all counted canonical runtime markers to be zero; canonical runs still require fixed-order markers. Add host routing/negative controls and a pinned-image import of all six untouched stock engine modules.
- Result: local gates pass. P57 host contracts passed `79/79`; the production stock preflight verified six stock SHA-256 entries and imported all six untouched modules from the pinned image; a deliberate `linear.py` drift negative was rejected; the exact-image suite retained base `109/109`, P45 `40/40`, PEFT `2/2`, Agentic `4/4`, fixed-head/TP8 forward+VJP probes, terminal `P45_EXACT_IMAGE_CPU_PASS`, and exit 0. `p57cal2` remains `INCONCLUSIVE` and contains no completed rollout recipe. No commit or push was performed.
- Files/artifacts: `evidence/p57cal2/run.log`; `cluster/entrypoint.sh`; `cluster/steps/p57_runtime_contract.sh`; `cluster/steps/90_run.sh`; P57 host/exact-image tests; `RUNBOOK.md`; `HANDOFF.md`.
- Rollback: revert the local stock-route concern; published history remains unchanged.
- Next: review the validated diff, then seek separate commit and push approval before any target rerun.

## 2026-08-21 UTC — p57cal3 proved stock routing and exposed a package-entrypoint dependency

- Type: target failure/correction
- Fact: `p57cal3` used immutable source `e4179511e6594d476460d355bd62086a6408ce54`. It passed `ZERO_TIM_OFF_PASS`, all six stock file hashes and imports, the stock route marker, and the zero-canonical-marker postflight. The workload command then failed before the first `RECIPE_START` with `ModuleNotFoundError: No module named 'examples'`. The complete 146-line log is SHA-256 `b6068e56bbf7452b9fce7b5a6630bdc4d94d5ba634252144610a8d30f2ca20da`.
- Cause: invoking `examples/frozenlake/train_frozenlake_qwen3.py` by file path made the script directory, rather than the repository root, the package import root. Canonical runs accidentally hid that defect because their overlay `PYTHONPATH` ended in an empty path component. The stock route correctly removed that incidental behavior. The same route also skipped three nonnumerical packages that canonical Step 30 had historically installed as a side effect.
- Action: render every P57 calibration/train/eval command as `python3 -u -m examples.frozenlake.train_frozenlake_qwen3`; reject file-path commands in the profile and manifest verifier; add a stock-only runtime step for pinned `gymnasium`, `sentencepiece`, and `tiktoken`; and extend the stock preflight to import the complete workload before model load.
- Result: dependency-light P57 tests passed `80/80`. The pinned image installed/imported all six required runtime packages, verified the six untouched engine modules, imported the full module entrypoint, rejected a deliberately modified stock engine, rejected the historical file-path entrypoint, accepted the module entrypoint, and completed with `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8` and exit 0. No TPU was launched and no commit or push was performed.
- Files/artifacts: `evidence/p57cal3/run.log`; P57 calibration/paired renderers; P57 profile and manifest verifier; `cluster/steps/37_install_stock_runtime.sh`; stock preflight; host/exact-image tests; `RUNBOOK.md`; `HANDOFF.md`; `state.md`.
- Rollback: discard the local entrypoint/runtime concern; published history remains unchanged.
- Next: review the validated diff, then seek separate commit and push approval. Only after publication may a separately approved `p57cal4` target run be rendered; it must reach `RECIPE_START` before startup readiness is considered closed.

## 2026-08-21 UTC — p57cal4 proved target capacity and exposed a canonical-only attestation call

- Type: target failure/correction
- Fact: `p57cal4` used source `762152dc3395f59ec4eace10f927f2e27f7fc90d`, connected all 64 TPU devices, materialized M10/M15/M20, loaded the model, initialized the KV cache, and reported approximately 34.3/95 GiB HBM per device. It failed before the first `RECIPE_START` at `attest_actor_anchor_matches_engine()` with `RuntimeError: canonical weight attestation requires the registered engine adapter`. The committed 759-line workload-only log has SHA-256 `073288beee03d579533bb147dbc7af2c80e6d986eb8c74a767781263d2a04bfa`; it lacks wrapper preflight/postflight and exit markers and is therefore analysis-grade, not a complete run bundle.
- Cause: stock-fast correctly skipped the canonical adapter, while the shared P45 resume/P57 no-update block incorrectly treated exact adapter-backed attestation as universally available after `sync_weights_for_resume()`. The transport sync itself completed; only the later canonical-only proof crashed.
- Action: centralized the boundary in `sync_rollout_for_no_update()`. Both regimes execute the real `update_params` synchronization. Stock returns a registered transport-only receipt and emits one exact marker with `exact_weight_attestation=unavailable-by-design`; canonical resume/evaluation still performs exact live-leaf attestation and fails closed. The runtime postflight and offline classifier require the stock marker/receipt and reject fabricated exact equality.
- Command: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh`; `bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh`; `git diff --check`.
- Result: P57 host suite `81/81` PASS with `P57_FROZENLAKE_TIM_CPU_PASS`. The pinned-image suite matched 34/34 overlay files, passed base `110/110`, P45 `40/40`, PEFT `2/2`, Agentic `4/4`, stock import/drift/entrypoint gates, fixed-head and TP8 forward/VJP probes, and terminated `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8` with exit 0. No new TPU launch was performed; publishing this repair does not authorize a target rerun.
- Files/artifacts: `evidence/p57cal4/run.log`; `tunix/rl/frozenlake_checkpoint.py`; FrozenLake entrypoint; stock runtime postflight/classifier/tests; `RUNBOOK.md`; `HANDOFF.md`; `state.md`; readiness phase.
- Rollback: revert this change set; the prior published behavior is the p57cal4 failure at `225bd6ad`.
- Next: render `p57cal5` only from the published immutable repair SHA, then obtain separate launch approval; acceptance requires the complete wrapper log and real rollout progress.

## 2026-08-21 UTC — p57cal5 exposed a learner/cluster API ownership error

- Type: target failure/correction
- Fact: `p57cal5` used source `7a77b32f2cd2dc08078e175fa0c407ca1cf33539`, again connected 64 TPU devices, materialized all three recipes, initialized model and KV cache, and held approximately 34.3/95 GiB HBM per device. Before `RECIPE_START`, the new helper raised `AttributeError: 'RLCluster' object has no attribute 'should_sync_weights'`. The committed 697-line workload-only log has SHA-256 `41c43de99606effe05029fff5572c82413e236c5a0d8d59cab040e6b78d61067`; wrapper start/postflight/exit markers are absent, so the artifact is analysis-grade only.
- Cause: the local behavioral fake incorrectly put `should_sync_weights` on `FakeCluster`, masking that production owns the field on `GRPOLearner`; only synchronization and attestation methods live on `RLCluster`.
- Action: changed `sync_rollout_for_no_update()` to accept the learner, read `learner.should_sync_weights`, and invoke transport/proof through `learner.rl_cluster`. Rebuilt the behavioral test with separate `FakeLearner` and `FakeCluster` roles and added a static wiring assertion that the workload passes `grpo_trainer`.
- Command: focused 24-test contract suite; `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh`; `bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh`; syntax and `git diff --check`.
- Result: focused `24/24` PASS; P57 host suite `81/81` PASS with `P57_FROZENLAKE_TIM_CPU_PASS`; pinned-image base `110/110`, P45 `40/40`, PEFT `2/2`, Agentic `4/4`, stock route/import negatives, fixed-head and TP8 forward/VJP probes all pass; terminal `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8` with exit 0. No new TPU launch was performed.
- Files/artifacts: `evidence/p57cal5/run.log`; `tunix/rl/frozenlake_checkpoint.py`; FrozenLake entrypoint; P45 checkpoint contract test; P57 state/handoff/runbook/lessons.
- Rollback: revert this ownership-repair change set; the prior published behavior is the p57cal5 failure at `39e77bdd`.
- Next: after publication, render `p57cal6` from the immutable repair SHA and obtain separate launch approval. Accept only real rollout progress plus the complete wrapper evidence contract.

## 2026-08-21 UTC — p57cal6 selected M15 after auditable provenance repair

- Type: target evidence/correction/decision
- Fact: p57cal6 completed all M10/M15/M20 stock-fast rollouts, but the recorder wrote sentinel values for `p57_index`, grid side, shortest path, and map SHA because it read the post-construction trajectory task instead of the original dataset row. Every record retained exact `group_id` and pair index; no measured outcome was missing.
- Action: preserved the original receipt byte-for-byte, rematerialized the signed deterministic calibration rows, joined `group_id` to the source row, required complete 100x8 pair coverage per recipe, and wrote a new derived receipt plus separate SHA proof. Repaired future recording at the learner boundary by joining the orchestrator's registered group-id ordering back to its prompt inventory.
- Result: source SHA `b34084dc...` unchanged; derived receipt SHA `ec03fe33...`; proof SHA `4328123d...`; classifier SHA `b6b6e04e...`. The unchanged classifier returned `PASS / FREEZE_M15`. M10/M15/M20 solve rates are 32.125/24.625/25.625%; M15 has 56% mixed groups, max context 7,403, max completion 6,223, and no cap hit.
- Files/artifacts: `evidence/p57cal6/p57_calibration.json`; `p57_calibration.derived.json`; `provenance_derivation.json`; `classification.derived.json`; derivation script and regression test.
- Next: freeze M15 and implement only its stock `selection` curve. No zero-arm outcome is admitted.

## 2026-08-21 UTC — full-bundle stock M15 segmented curve prepared

- Type: implementation/plan advance
- Fact: merely zeroing profile flags was insufficient: the runtime router still selected the canonical overlay outside calibration, and the old evaluator would therefore assess stock checkpoints with a different numerical program. LatestN(1) also requires evaluation before advancing to the next retained boundary.
- Action: extended pristine stock runtime routing to exact mismatch train/eval; added independent stock train/eval environment validators; pinned M15 selection, horizon 200, prompt/response 4096/8192, and stop boundaries 50/100/150/200; added durable segment preflight/completion postflight; kept finite A-B warning-only while structural/nonfinite/non-treatment failures remain fatal; and rewrote runbook/handoff for eval-0 → train-50 → eval-50 → resume.
- Command: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh`; `bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh`; shell/Python syntax checks; `git diff --check`.
- Result: the completed-tree P57 host suite passed `86/86` with terminal `P57_FROZENLAKE_TIM_CPU_PASS`. The pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` matched all 34 Qwen3-8B TP8 overlay files, passed base `110/110`, P45 `40/40`, PEFT `2/2`, Agentic `4/4`, all stock import/drift/entrypoint gates, the seven TP8 projection sites, fixed lm-head, and forward/VJP probes; terminal `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`, exit 0. Shell/Python syntax and `git diff --check` are clean. No TPU, commit, or push occurred.
- Rollback: discard this uncommitted concern; all behavior is P57-env-gated.
- Next: review this uncommitted concern, then request separate commit and push approvals. Only the resulting immutable 40-character SHA may be used for separately approved stock eval-0 and train 0→50 launches.

## 2026-08-21 UTC — eval-0 attempt 1 exposed stale leaf-step admission

- Type: target failure/correction/evidence
- Fact: `p57_eval0_att1` used source `200b244cc400c3bca6281cf2d7b4ed074a2ed734`. It passed resolved-environment validation, source provenance, the six-file stock image probe, RoPE compatibility, and cache setup. It then stopped before model load at Step 37 with `[P57.STOCK_FAST] FATAL: stock runtime install used outside calibration`. The 48-line head-container log is SHA-256 `6e994961d517440b2568c767ce686dc9570ac1264d4ca4175109dc1ae50030d1`; it has the startup prefix through the fatal marker but no terminal wrapper bundle, so the run is `INCONCLUSIVE` and contains no evaluation result.
- Cause: the top-level entrypoint had correctly changed from `p57_is_stock_fast_calibration` to `p57_is_stock_fast_runtime`, but both leaf steps 37 and 38 retained their calibration-only guards. The previous exact-image test exercised only calibration, so it certified the wrong coverage surface.
- Action: changed both leaf guards to the exact aggregate stock-runtime predicate. Extended host coverage to assert both scripts use that predicate. Extended the pinned-image gate to execute Step 37 and Step 38 under calibration, mismatch training, and mismatch evaluation, plus a zero-arm rejection and the existing stock-byte drift rejection.
- Command: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh`; `bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh`; `git diff --check`.
- Result: host suite `87/87` PASS with terminal `P57_FROZENLAKE_TIM_CPU_PASS`. The pinned-image gate emitted `P57_STOCK_RUNTIME_MODE_PASS` for calibration, train, and eval; rejected the zero arm; retained the six-file drift negative; and ended `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`, exit 0. No TPU rerun, commit, or push was performed.
- Files/artifacts: `evidence/p57_eval0_att1/head_container.log`; Steps 37/38; runtime-contract and exact-image tests; `HANDOFF.md`; `state.md`; `lessons.md`.
- Rollback: discard this uncommitted guard-and-test concern; published source remains at the attempt-1 failure behavior.
- Next: review and separately approve commit/push. Relaunch eval-0 in `new` mode from the new immutable SHA; do not resume attempt 1 and do not launch training before eval-0 completes.

## 2026-08-21 UTC — eval-0 attempt 2 exposed a DP8 evaluation-row contract error

- Type: target failure/correction/evidence
- Fact: `p57_eval0_att2` used source `861128387b638f9b05a4811d89923a9109db7d91`. It passed the repaired stock runtime, loaded Qwen3-8B, initialized the KV cache, synchronized rollout weights, and completed real held-out rollouts. Before writing an evaluation receipt, ordinary EVAL conversion entered trainer-side rescore with `q_block=[2,32,12288,128]`, `k/v=[2,8,12288,128]`, and segment rows `[2,12288]`; Splash Attention maps axis 0 over DP8 and rejected 2 as non-divisible. The committed `run.log` has 985 lines and SHA-256 `bd490e0cb91b7cd1502cb6f275b183797617bb708ae9363bf03c58cad49de0fb`; `env.sh` SHA-256 is `3f12838480242bceab5f4babc85fdddd5f7be61b71e415ae1002915fda428b5b`. No COMPLETE/terminal package exists, so the run is analysis-grade `INCONCLUSIVE` and contains no solve-rate result.
- Shape ledger: caller-global M was 2 trajectories per map; shard-local M was undefined because DP8 could not divide it; the trainer sequence width was 12,288; semantic valid rows were 2; scheduler capacity remained 32 sequences per DP rank. The repaired contract uses caller-global M=8, shard-local M=1, semantic rows=8, the same 12,288 width, and unchanged scheduler capacity.
- Action: retained the authoritative EVAL rescore path and changed isolated evaluations from two to eight deterministic generations. Added a renderer divisibility assertion and negative, changed the profile contract, updated classifier coverage from 200 to 800 rewards, required the eight greedy replicas within each map to agree before computing 100-map capability, updated the evaluator lifecycle test, and reconciled the P57 runbook/phase/handoff.
- Command: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh`; `bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh`; `python3 -m py_compile canon-zero-tim/cluster/render_p57_frozenlake_tim.py canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/classify_checkpoint_eval.py canon-zero-tim/tests/p57_frozenlake_tim/test_eval_classifier.py canon-zero-tim/tests/p57_frozenlake_tim/test_renderer.py tests/rl/agentic/agentic_rl_learner_test.py`; `git diff --check`.
- Result: host suite `89/89` PASS with `P57_FROZENLAKE_TIM_CPU_PASS`. The pinned image matched the 34-file Qwen3-8B TP8 overlay, passed base `110/110`, P45 `40/40`, PEFT `2/2`, Agentic `4/4`, executed the P57 evaluator lifecycle with `num_generations=8`, passed all three stock runtime modes and registered negatives, and ended `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`, exit 0. No TPU target rerun, commit, or push was performed.
- Files/artifacts: `evidence/p57_eval0_att2/run.log`; `evidence/p57_eval0_att2/env.sh`; P57 renderer/profile/classifier/tests; evaluator lifecycle test; `RUNBOOK.md`; `HANDOFF.md`; phase/state/lessons.
- Downside: each isolated evaluation now runs 800 deterministic trajectories instead of 200. Capability remains map-level; the eight replicas are not independent samples.
- Rollback: revert the uncommitted DP8 evaluation-row concern; published source remains at the attempt-2 failure behavior.
- Next: review and separately approve commit/push. Relaunch a fresh eval-0 in `new` mode from the new immutable SHA; do not resume attempts 1 or 2 and do not launch training before the 100-map/800-reward classifier passes.

## 2026-08-21 UTC — eval-0 attempt 3 exposed stale entrypoint geometry admission

- Type: target failure/correction/evidence
- Fact: `p57_eval0_att3` used source `8acfe784b6fa8eacb8eb4e41406dd6681173f9c7`. Its resolved environment and command both carried `num_generations=8`, but the process stopped before model load with `P32 FrozenLake geometry mismatch: {'num_generations': 8}`. The committed `run.log` has 76 lines and SHA-256 `b4c5bd426b1b23224e1becfd43dfe18dc5f239d2f59916e06dde1927117d9e6e`; `env.sh` SHA-256 is `c3efb8229c5b1dd90a935b4c22ad0ffab6c443a77f7503a4fc663e36b1429c77`. There is no complete terminal package or evaluation receipt, so the run is analysis-grade `INCONCLUSIVE`.
- Cause: the DP8 repair updated the renderer, profile, classifier, and evaluator lifecycle, but missed the real workload entrypoint's older `expected_generations = 2 if CANON_P57_EVALUATION else 8` assertion. Existing tests proved the outer contract without exercising that inner admission line.
- Action: added `p57_workloads.GENERATIONS_PER_PROMPT=8`, made both the renderer and real entrypoint consume it, and added a regression that rejects the obsolete conditional in the actual entrypoint source.
- Command: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh`; `bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh`; Python syntax checks; `git diff --check`.
- Result: host suite `90/90` PASS with `P57_FROZENLAKE_TIM_CPU_PASS`. The pinned-image gate matched all 34 Qwen3-8B TP8 overlay files, passed base `110/110`, P45 `40/40`, PEFT `2/2`, Agentic `4/4`, all stock runtime modes and negatives, fixed-head/TP8 probes, and ended `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. No 64-chip target rerun was performed.
- Downside: none beyond the already approved eight-generation evaluation cost; training, calibration, and rendered evaluation counts remain eight.
- Next: review and separately approve commit/push. Then launch a new eval-0 attempt in `new` mode from the new immutable SHA; attempts 1–3 are not resumable.

## 2026-08-21 UTC — P57.1 changed to one direct stock 0→200 run

- Type: user decision / plan correction
- Fact: three eval-0 startup attempts consumed launch time without producing a scientific receipt. Source `7b55f6f2...` publishes the repaired eight-generation contract, but the user chose not to spend another launch on baseline or intermediate held-out evaluations during stock workload discovery.
- Decision: launch one uninterrupted mismatch/stock-fast M15 `selection` training JobSet from update 0 through 200. Do not run eval-0 and do not intentionally pause at 50/100/150. Ten-step LatestN(1) checkpoints remain enabled only for infrastructure recovery. An isolated eval-200 is optional after the training curve completes.
- Claim change: P57.1 freezes or rejects the recipe using the preregistered trailing on-policy training solve statistic and treatment/trajectory-health receipts. It no longer claims held-out improvement from a same-split eval-0 or held-out AUC for the discovery curve. The later paired causal campaign must separately preregister an equal evaluation schedule for both arms.
- Action: updated the direct-run handoff/runbook/phase/state; added a renderer regression proving the default stock train resolves to stop 200 without `--evaluation_only`; changed the terminal marker to `next_action=complete` at the signed horizon while retaining `isolated-eval` for a nonterminal recovery segment.
- Downside: less held-out information during workload discovery and no same-split initial baseline. The benefit is one launch instead of eval/train segmentation.
- Next: pass local gates, then seek separate commit/push and TPU launch approval. The intended manifest is `checkpoint_mode=new`, `run_kind=train`, horizon/stop 200.

## 2026-08-21 UTC — stock full attempt 1 exposed a processed-B observer contract gap

- Type: target failure / correction
- Fact: `p57_stock_full_att1` used source `7e608682ea21c501b8ed737b58ffe5591125d6eb`. It passed checkpoint-new admission, materialized the signed M15 selection data, loaded the stock engine, completed a real 256-trajectory rollout, and reported solve ratio 0.289. Before backward or optimizer update 0, the alignment sidecar requested processed `S_prefill` and `VllmRollout` rejected `CANON_PROMPT_PROCESSED_LOGPROBS=0`. The committed evidence contains `run.log` and `env.sh` only, so it is analysis-grade `INCONCLUSIVE`; no checkpoint was written and the run is not resumable.
- Cause: the stock contract classified processed prompt logprobs as part of the training treatment and forced them off, while warning-only A-B observation requires B to apply the same temperature/top-k/top-p semantics as processed decode A. Raw B cannot be relabeled as processed. Source inspection confirms the rescore result is attached only to the alignment sidecar; rollout `S_decode` remains `old_per_token_logps` and processed B is not consumed by loss, backward, or optimizer code.
- Action: enable `CANON_PROMPT_PROCESSED_LOGPROBS=1` only for `train:mismatch`; keep calibration/eval at zero; move the switch from the stock-train zero set to the observer/admission one set; add preflight and runtime markers plus train/eval negative controls; update the P57 plan, phase, runbook, handoff, and state with the observer-only claim boundary.
- Shape ledger: rollout caller-global M=256, shard-local M=32, scheduler capacity=32 per DP rank, and stock serving shapes are unchanged. The extra B call retains the existing stock prompt-logprob implementation because `CANON_LOGPROB_M` remains absent; only its sampling transform is applied before the observer gathers selected-token logprobs.
- Downside: each training update still pays for processed prefill rescore, and the stock arm is no longer accurately described as every numerical flag being zero. The trained-policy program remains untreated; processed-B is shared measurement instrumentation that must also be enabled in the later zero arm.
- Validation: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh` passed `91/91` with `P57_FROZENLAKE_TIM_CPU_PASS`. `bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh` matched all 34 overlay files; passed base `110/110`, P45 `40/40`, PEFT `2/2`, Agentic `4/4`, all three stock runtime modes and registered negatives, fixed-head and seven Qwen3-8B TP8 sites plus forward/VJP; and ended `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`, exit 0.
- Next: review and separately approve commit/push. Then, with separate launch approval, render a fresh `new` stock full run from the immutable repair SHA; do not resume attempt 1.
