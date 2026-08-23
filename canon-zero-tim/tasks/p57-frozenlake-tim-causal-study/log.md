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

## 2026-08-21 UTC — stock full attempt 2 exposed raw/rolled stock prompt observer

- Type: target failure / correction / contract repair
- Fact: `p57_stock_full_att2` used source `c5cc71b57619a866e61e5e75288cc699eca3e1e8`. It passed stock routing, signed M15 materialization, checkpoint-new admission, model/engine startup, one 256-trajectory rollout, and the repaired processed-B host admission. Solve ratio was 0.332 and a real B call ran for 27.114 seconds. Before backward or update 0, extraction failed because selected token 304 was absent from `{5795: Logprob(...)}`. The only committed artifact is a 2,715-line `run.log`, SHA-256 `f49f35f4243cbe98af6b12f9632b88224c530f415be1386c6f360604da0cb749`; there is no complete terminal package or checkpoint, so the attempt is analysis-grade `INCONCLUSIVE` and not resumable.
- Correction to the previous entry: setting `CANON_PROMPT_PROCESSED_LOGPROBS=1` was necessary for Tunix admission but insufficient on the intentionally unoverlaid stock engine. Pinned-image source inspection proved stock `compute_prompt_logprobs` always scores raw `full_logits` and obtains targets with `jnp.roll(input_ids, -1)` over the whole DP-packed buffer. Thus the old statement that the stock B call retained its implementation while only applying a transform was false: no prompt transform was installed, and target identity could cross request/padding boundaries.
- Action: retain two training treatments. Stock A, model/trainer C, loss, backward, reducer, and optimizer remain untouched. For mismatch training only, verify all six stock engine files first, then install a signed two-file observer delta: a minimal runner branch plus helper. Only explicit prompt-logprob B calls apply decode-equivalent temperature/top-k/top-p transforms and gather absolute target IDs from immutable request history. Calibration/evaluation install nothing. The observer delta has no `CANON_LOGPROB_M` reference and never supplies `old_per_token_logps` or a gradient input.
- Gates: host P57 suite `91/91` PASS. Disposable pinned-image patch application with `--fuzz=0`, Python compile, and two-file SHA manifest PASS. Boundary negative proves packed roll differs from absolute target IDs; processed-value test matches the direct temperature definition; end-to-end CPU helper returns target IDs `[20,30,40,0]`, exact processed selected-token logprobs, and the correct three-logit request snapshot. Full pinned-image suite emits `P57_STOCK_OBSERVER_EXACT_IMAGE_PASS targets=absolute values=processed` and ends `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`.
- Claim boundary: the mismatch training treatment is still stock; the runner file is no longer globally describable as byte-identical during training because B instrumentation is installed. Any comparison must say “stock A/C treatment plus observer-only processed B,” not “all six engine files untouched.”
- Next: review and separately approve commit/push. A separately approved target attempt must use a new run id and checkpoint mode `new`; accept startup only with the two observer markers, then require progress past B into backward/update 0 before treating the launch repair as proven.

## 2026-08-21 UTC — observer repair published; runbook attempt id corrected

- Type: publication reconciliation / local preflight failure / documentation repair
- Fact: observer-only processed B was published as `de0f350fc60f436fd9540b41b6f130ddca7a87f4`. A fetch of all remote refs found no P57 target result after `p57_stock_full_att2`; commits after `de0f350f` on the delivery branch concern P58 only. The prior state entry incorrectly continued to call the repair local and publication pending.
- Local failure: executing the direct-train RUNBOOK command on current source `4c3accc3b597af3a54594db84ae1ae39c03768a7` failed before YAML creation because run id `p57-m15-stock-full200` exceeds the renderer's 1–16 character lowercase DNS limit.
- Action: reconcile `state.md` to published/TARGET NOT RUN, replace the overlength example with registered next attempt id `p57m15att3`, and mirror that id and negative warning in `HANDOFF.md`. No numerical, training, renderer, profile, or cluster runtime code changed.
- Validation: P57 host tests passed `91/91` with `P57_FROZENLAKE_TIM_CPU_PASS`. Rendering `p57m15att3` from current source produced exactly one mismatch M15 selection JobSet and `P57.JOBSET VERDICT PASS count=1 updates=200 checkpoint_mode=new run_kind=train checkpoint_step=none`; the command contains `--max_steps=200`, prompt/response `4096/8192`, and no `--evaluation_only`. `git diff --check` is clean.
- Next: review this documentation-only correction and decide whether to commit/push it. A target launch still requires separate user approval and must use checkpoint mode `new`; do not resume attempts 1 or 2.

## 2026-08-22 UTC — P57 causal arms made free of TIM-aware mitigation

- Type: user decision / causal-contract repair / local validation
- Fact: the FrozenLake recipe still hardcoded token sampler importance sampling. In that mode the learner replaced rollout `S_decode` (A) with trainer recompute C as `old_per_token_logps` and multiplied policy loss by detached, mismatch-dependent `min(exp(C-A), 2)` weights. The planned comparison was therefore mismatch-plus-TIS versus exact-zero with inert TIS, not untreated mismatch versus zero-TIM.
- Action: added the P57-only `--sampler_is=none` command contract to both arms and isolated evaluations while leaving every non-P57 FrozenLake run defaulted to token TIS. The learner now admits this only for the exact P57 profile/workload tuple, proves on the first real training batch that rollout A is present and is the old-logprob object, proves TIS weights are absent, emits one `[P57.TIM_PURITY] PASS`, and postflight rejects missing or duplicate receipts. Processed B/trainer C remain observer-only. Standard GSPO ratio clipping at epsilon 0.003/0.005 remains unchanged in both arms.
- Negatives: renderer rejects duplicate or token sampler modes; profile preflight rejects a command changed from `none` to `token`; helper tests independently reject token sampler mode, missing rollout logps, alternate old-logprob identity, disabled rollout-logp use, and present TIS weights; scope negatives reject wrong phase, arm, profile, or workload.
- Validation: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh` passed `102/102` with `P57_FROZENLAKE_TIM_CPU_PASS`. The exact pinned-image gate matched all 34 Qwen3-8B TP8 overlay files, retained base `110/110`, P45 `40/40`, PEFT `2/2`, Agentic `4/4`, stock runtime/observer negatives, fixed-head/TP8 and forward/VJP probes, and ended `P57_STOCK_OBSERVER_EXACT_IMAGE_PASS targets=absolute values=processed` plus `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. Syntax checks and `git diff --check` pass. No TPU, commit, or push occurred.
- Downside: P57 training now intentionally has larger unclipped mismatch ratios than the prior TIS-stabilized recipe may have had; that is the causal treatment being measured. Ordinary GSPO clipping still bounds the shared optimization rule.
- Next: review and separately approve commit/push. A target launch remains separately gated and must use a fresh `new` M15 selection run from the eventual immutable SHA.

## 2026-08-22 UTC — two-workload, three-treatment campaign locally admitted

- Type: user decision / phase replacement / local validation
- Decision: evaluate native/no-IS, native/token-IS, and complete zero-TIM/no-IS independently on both the original P45 workload and frozen M15-main. The immediate primary comparison is the four-job no-IS pair (P45/M15 native versus P45/M15 zero); the two native/token-IS jobs are an add-on after primary evidence is packaged. Contrasts are only within a workload.
- Action: extended the isolated P57 renderer/profile/runtime contract with an explicit `is` arm; added arm-specific learner and postflight receipts; admitted original P45 at 450 updates and M15-main at 200; added a render-only two-workload wrapper and resolved-manifest verifier; rewrote the active phase, plan, runbook, handoff, and state around the six signed cells. No historical P45 launch path was replaced.
- Contract: native/no-IS uses stock-fast A/C, old=A and no TIS; native/token-IS uses the same stock-fast numerical program, old=C and `min(exp(C-A),2)` TIS; zero/no-IS uses the complete canonical bundle, old=A and no TIS. Standard GSPO ratio and epsilon clipping remain shared in every arm.
- Validation: all six rendered cells passed exact command/env and real `00_env.sh` preflight. Focused tests passed 34/34; the full P57 host suite passed 104/104 with `P57_FROZENLAKE_TIM_CPU_PASS`; flag registry audit passed 320/320. The pinned-image gate passed the `arm=is` stock-runtime positive and zero negative, Qwen3-8B TP8 fixed lm-head and seven projection forward/VJP probes, stock observer exact-image, and terminal `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`, exit 0. Syntax and `git diff --check` pass.
- Claim ceiling: one curve per cell is a concept study. Stability/generalization claims require paired multi-seed replication and counterbalanced order.
- Next: review and separately authorize commit/push. Target status is `NOT RUN`; launch of any of the four primary jobs requires a second explicit approval from the immutable published SHA.

## 2026-08-22 UTC — all six concept-study cells frozen to 200 updates

- Type: user decision / horizon-contract change
- Decision: truncate the P45 treatment cells from the historical 450-update recipe to the same 200-update horizon used by M15. This makes training budget, checkpoint count, final evaluation boundary, and curve support identical across all six cells.
- Claim change: P45 remains the original seed-42/123 workload and geometry, but the study no longer claims a complete reproduction of its historical 450-update training recipe. Historical P45 evidence is context only; the causal study is a six-cell 200-update experiment.
- Action: changed the P57 renderer/profile, two-workload wrapper, manifest verifier, tests, active phase, plan, runbook, handoff, and state. Added a negative that rejects an attempted P45 450-update arm.
- Validation: all six 200-step manifests passed exact command/env checks and real `00_env.sh` preflight. The P57 host suite passed 105/105 with `P57_FROZENLAKE_TIM_CPU_PASS`; the new negative rejected P45 at 450; flag audit passed 320/320; syntax and `git diff --check` passed. The pinned-image gate passed the IS stock-runtime positive, zero negative, fixed-lm-head/TP8 forward+VJP probes, stock observer exact-image, and terminal `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`, exit 0. No target, commit, push, or launch occurred.

## 2026-08-22 UTC — first target queue changed to native/no-IS versus native/token-IS

- Type: user decision / execution-order correction
- Decision: queue P45 and M15 under `mismatch` and `is` first, for four independent 200-update DP8xTP8 jobs. Do not include either `zero` cell in this queue; complete Zero-TIM/no-IS remains a deferred second phase requiring a separate launch decision.
- Rationale: the first queue directly measures whether token importance sampling mitigates native trainer-inference mismatch on the easy P45 and harder M15 workloads. It uses the same native/stock-fast numerical program in both arms and changes only the registered sampler correction, old-logprob identity, and TIS weights.
- Action: updated `RUNBOOK.md`, `HANDOFF.md`, `plan.md`, this phase, and `state.md` so the executable commands render `native` and `is`, the four manifest/apply paths name those arms, and the required receipts distinguish no-IS from token-IS. No renderer, profile, runtime, numerical code, or target state changed.
- Validation: ran `render_three_arm_wave.sh` from the exact source for `native` and `is`. Each command emitted two `P57_THREE_ARM_MANIFEST_PASS` lines plus its wave/render PASS markers. The resulting set was exactly `jobset-p57-frozenlake-mismatch-200.yaml`, `jobset-p57-frozenlake-mismatch-m15-main-200.yaml`, `jobset-p57-frozenlake-is-200.yaml`, and `jobset-p57-frozenlake-is-m15-main-200.yaml`; no `zero` manifest was rendered. `git diff --check` passed.
- Claim boundary: the first four curves can estimate `is - mismatch` within each workload. They cannot estimate a Zero-TIM effect until the two deferred `zero` cells run, and one curve per cell remains concept evidence rather than a stability claim.
- Next: validate both two-workload render waves from the exact source, commit/push the documentation-only correction, then separately approve any `kubectl apply`.

## 2026-08-22 UTC — four-wave launch attempted and blocked on static environment validator

- Type: target failure / evidence collection
- Action: rendered all four 200-update manifests with 4-character run-ids (`p45n`, `m15n`, `p45i`, `m15i`) to satisfy the 63-character Pod name limit; applied all four JobSets simultaneously across 256 TPU chips.
- Result: all 4 JobSets were admitted by Kueue and provisioned all 64 Worker Pods to 1/1 Running. However, during initialization, `train_frozenlake_qwen3.py` called `dp_workloads.validate_p57_stock_train_environment`, which failed because its static `expected` dict in `dp_workloads.py:820-840` hardcodes `TIM_ARM: "mismatch"`, `WORKLOAD_CANDIDATE: "m15"`, and `DATA_SPLIT: "selection"`.
- Classification: `INCONCLUSIVE` execution failure before step 0. All 4 JobSets were deleted immediately to release the 256 TPU chips.
- Evidence: `evidence/four_wave_launch_error.log` recorded all four tracebacks.
- Next: peer agent to update `dp_workloads.py` to validate dynamic environment parameters, then relaunch four waves.

## 2026-08-22 UTC — closed runtime-matrix repair locally validated

- Type: target-failure repair / fail-closed contract expansion / one-host pinned-image validation
- Cause: the runtime validator was a stale discovery-only contract, not a dynamic validation of the new causal matrix. It admitted exactly `(mismatch,m15,selection)` and therefore rejected all four correctly rendered P45/M15-main `mismatch`/`is` jobs before step 0.
- Action: replaced the three hardcoded arm/workload/split expectations in both stock train and eval validators with a closed registry of five tuples: the historical discovery tuple and P45/M15-main under `mismatch` and `is`. Returned attestations now carry the resolved tuple and variant name. Added train+eval positives for all five tuples and a negative for unregistered `(is,m15,selection)`. All zero/one/absent numerical switches, topology, optimizer, horizon, and stock-fast requirements are unchanged.
- Validation: the P57 host suite passed 105/105 with `P57_FROZENLAKE_TIM_CPU_PASS`; syntax and `git diff --check` passed. On the local v5p host, the pinned production image resolved immutable image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`, installed all 34 overlay files, emitted `P57_STOCK_RUNTIME_MATRIX_PASS variants=5 stages=train,eval`, kept the zero and stock-drift negatives red, passed the Qwen3-8B TP8/fixed-lm-head forward+VJP probes and stock observer, and ended `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8` with exit 0. The container reported no `/dev/vfio`; no device computation or target JobSet was claimed.
- Classification: repair is locally GREEN but unpublished and target `NOT RUN`. The four prior attempts remain immutable `INCONCLUSIVE` evidence.
- Next: review before any commit/push. If published, rerender from the new SHA with fresh four-character IDs `n45a/n15a/i45a/i15a`, fresh campaign root `p57-native-is-b`, and `checkpoint-mode=new`; then ask separately before launch.

## 2026-08-22 UTC — wave B launched: native mismatch runs committed steps; token-IS blocked on post-backward check

- Type: target execution / evidence collection
- Action: sequentially launched Wave B manifests (`n45a`, `n15a`, `i45a`, `i15a`) on 26f9f4a2.
- Result:
  - `P45 mismatch` (`n45a`): Step 0 and Step 1 committed (`train_steps=2`), rollout throughput ~3.3s/row.
  - `M15 mismatch` (`n15a`): Step 0 committed (`train_steps=1`), Step 1 rollout underway (~6s/row).
  - `P45 is` (`i45a`): Step 0 rollout and pre_backward alignment passed with warnings, but post_backward `alignment.check_batch` failed because `CANON_ENGINE_MODULE_C!=1`. Deleted JobSet and recorded evidence in `evidence/i45a_alignment_error.log`.
- Next: peer agent to update `alignment.py` to allow `CANON_ENGINE_MODULE_C=0` in stock IS mode, then relaunch IS wave.

## 2026-08-22 UTC — token-IS post-backward Module C scope repaired locally

- Type: target-failure repair / post-backward admission / pinned-image validation
- Cause: `alignment.check_batch` recognized P57 stock training only when `TIM_ARM=mismatch`. The registered `is` arm uses the identical stock-fast A/C program and also intentionally sets `CANON_ENGINE_MODULE_C=0`; its only treatment differences are trainer C as old logprob and token-TIS weights. Consequently `i45a` passed rollout, pre-backward, and backward but was rejected by a stale post-backward attestation before a completed step.
- Action: changed the existing P57 stock predicate from equality with `mismatch` to membership in the two registered stock arms `(mismatch,is)`. Added a positive covering both arms and a negative proving an unknown arm still fails on `CANON_ENGINE_MODULE_C!=1`. Loss, sampler, forward, backward, reducer, optimizer, warning policy, and zero-arm strictness are untouched.
- Validation: syntax and `git diff --check` passed. Host bare Python could not import the dependency-complete Tunix stack because `metrax` is absent, so it was not counted. The full pinned image ended `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8` with exit 0. A focused run in the same image passed both tests: registered arms emitted two post-backward PASS records; the unknown arm emitted a RED and was accepted only because the negative expected that exception.
- Classification: local `CPU PASS`; repaired target path `TARGET NOT RUN`. `i45a` remains immutable `INCONCLUSIVE`; healthy native jobs remain untouched.
- Next: review before commit/push. After publication, determine and package old `i15a` state, then render only the IS wave with fresh IDs `i45b/i15b`, fresh campaign `p57-native-is-c`, and `checkpoint-mode=new`; obtain separate launch approval.

## 2026-08-22 UTC — paired-arm horizon restored to 450 updates

- Type: user decision / horizon-contract supersession / local validation
- Decision: the four immediate P45/M15 native-no-IS and native-token-IS jobs must restart from initialization and run 450 updates. The prior 200-update jobs are preserved as immutable partial evidence but are not resumed into the new causal comparison. The deferred zero-TIM cells must also use 450 when separately authorized so horizon remains controlled across all six cells. Historical M15 selection discovery remains a separate 200-update contract.
- Action: separated `_STOCK_DISCOVERY_UPDATES=200` from `_PAIRED_ARM_UPDATES=450`; updated the profile's train/eval tuple gate and stock runtime attestation so only M15 selection admits 200 while P45/M15-main admit 450; changed the two-workload wave renderer and verifier to 450; replaced the old P45-450 rejection with a paired-200 rejection; synchronized the plan, active phase, future campaign/analysis phases, runbook, handoff, state, and thread row.
- Invariants: batch remains 32 prompts x eight generations, DP8xTP8, AdamW 1e-6, GSPO-token/RLOO, temperature 0.7, resident optimizer, checkpoint every 10 with LatestN(1), and in-process evaluation off. No precision, model, loss, sampler, ratio clipping, backward, reducer, or optimizer semantics changed.
- Validation: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh` passed 105/105 with `P57_FROZENLAKE_TIM_CPU_PASS`. The pinned exact-image gate installed and verified all 34 Qwen3-8B TP8 overlay files and exited 0. Fresh local `native` and `is` wave renders each emitted two `P57_THREE_ARM_MANIFEST_PASS` markers plus wave/render PASS; the four outputs were exactly the P45/M15-main mismatch/is `*-450.yaml` manifests. The resolved-env contract rejected the stale 200 paired horizon while retaining the 200-step stock-discovery tests.
- Claim boundary: this proves construction and pinned-image compatibility only. No 450-update target JobSet has run. The local manifests embed the current committed base SHA for gate purposes and must be rerendered from the eventual approved published SHA; their hashes are not launch artifacts.
- Next: review the local diff. Commit and push only after separate user approval. Then render both waves with fresh four-character IDs and checkpoint namespaces, record four manifest hashes, and request separate launch approval.

## 2026-08-22 UTC — 450 horizon retained; isolated evaluation fixed to every 50 updates

- Type: user decision / checkpoint-retention contract / evaluation orchestration
- Decision: keep every paired treatment at 450 updates and evaluate held-out solve at `0,50,100,150,200,250,300,350,400,450`. Training must remain uninterrupted and must not re-enable in-process evaluation.
- Cause addressed: `LatestN(1)` alone deletes old checkpoints, and the trainer historically restores latest when no exact step is supplied. Rendering ten post-hoc evals without fixing both behaviors would either find missing steps or silently restore 450 for an earlier point.
- Action: added P57-only `LatestN(1) OR EveryNSteps(50)` preservation while keeping saves every 10; added an optional exact restore step to the trainer and wired isolated evaluation to it; restricted paired eval rendering to retained 50-step milestones; extended the 450 stop-boundary gate through 250/300/350/400/450; added a 20-manifest two-workload schedule renderer/verifier per arm; hardened profile/resolved-env/checkpoint provenance gates; updated the plan, active/future phases, runbook, handoff, state, and thread ledger.
- Evaluation order: render everything from one immutable SHA; run eval-0 before training while its namespace is empty; run the 450-step train without pauses; after durable close, evaluate retained positive milestones. Milestone deletion is not automatic and requires separate approval after evidence packaging.
- Validation: Python and shell syntax passed; the full P57 host suite passed
  119/119 with `P57_FROZENLAKE_TIM_CPU_PASS`; native and IS train waves each
  passed both manifest preflights; a native eval schedule rendered 20
  manifests and ended `P57_EVAL_SCHEDULE_PASS ...
  steps=0,50,100,150,200,250,300,350,400,450` plus
  `P57_EVAL_RENDER_PASS`. The pinned production-image gate installed all 34
  Qwen3-8B TP8 overlay files, passed the five-tuple stock train/eval matrix,
  fixed-head forward/VJP and observer gates, and ended
  `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8` with exit 0.
- Claim boundary: construction evidence only; no target evaluation or 450-update train has run. No training mathematics or numerical kernel changed.
- Storage review: the actor is full-parameter FP32 and checkpoints include
  optimizer state. `LatestN(1)` still limits ordinary recovery generations,
  but the nine additional 50-step evidence milestones can approach one
  terabyte per arm. The runbook/handoff now require explicit GCS quota/cost
  acceptance before launch and separate approval before post-analysis cleanup.

## 2026-08-23 UTC — four 450-update attempts exposed FrozenLake prompt provenance override

- Type: target failure / shared trajectory-schema compatibility repair / local pinned-image validation
- Fact: `n45c/n15c/i45c/i15c`, all sourced from `5f449cc8`, provisioned their 64 TPU chips and completed real Step-0 rollout generation in roughly 59–62.5 seconds. All four then failed in `TrajectoryCollectEngine._original_input()` before trainer alignment, backward, optimizer update, or checkpoint with `policy-seeded trajectory original_input is missing required key 'prompts'`. The incoming raw log SHA-256 is `d37f26a109620131325d1f0e8343a20d76e2dddb01784bf6066fef92898a4799`; classification SHA-256 is `989ee6dc9f289213306b83004fa17db4a3f26a08d4812cd6d81969e1d525ce6b`. All four attempts are immutable `INCONCLUSIVE` and not resumable.
- Cause: DeepSWE commit `43614af5` correctly made a policy-seeded environment task durable across reset timeouts, but selected that task wholesale whenever `policy_version` existed. DeepSWE stores its dataset `prompts` in the environment task; FrozenLake initializes an empty environment task and constructs its rendered `prompts` only in `agent.trajectory.task`. Policy seeding therefore left the FrozenLake environment record with metadata but no prompt and discarded the valid trajectory prompt.
- Action: preserve a policy-seeded environment task unchanged when it already owns `prompts` (the DeepSWE contract); otherwise merge its durable metadata into a prompt-bearing trajectory task with environment keys authoritative (the FrozenLake contract). Keep environment-only reset-timeout fallback and missing-prompt fail-closed behavior. Add positive/negative unit coverage and make the four provenance cases permanent members of the P45/P57 pinned-image gate. Re-register fresh replacement identities `n45d/n15d/i45d/i15d`, evaluation roots `fn/fi`, and `*-450-b` campaign/output/checkpoint namespaces.
- Validation: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh` passed 119/119. Pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` passed the FrozenLake merge, DeepSWE environment-authority, reset-timeout preservation, and missing-prompt negative controls and ended `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. The complete P58 pinned-image suite independently ended `P58_EXACT_IMAGE_CPU_PASS ... regressions=1`. Python/shell syntax and `git diff --check` pass. No TPU target rerun, commit, push, or launch occurred.
- Claim boundary: local compatibility and exact-image construction evidence only. The bug is isolated from IS/Zero-TIM numerics, model execution, optimizer, and checkpoints because all target failures preceded those stages. The repaired 64-chip training path remains `TARGET NOT RUN` until a fresh `*d` attempt commits an update.
- Next: review before commit/push. If published, render the four step-0 evaluators and replacement trains from the exact repaired SHA; preserve the existing launch-approval sequence and GCS-budget gate.

## 2026-08-23 UTC — paired campaign superseded by 300 updates with in-process evaluation

- Type: user decision / execution-contract supersession / local implementation
- Decision: replace the unlaunched 450-update plus isolated-milestone design with 300 uninterrupted updates and rollout-only held-out evaluations at `0,50,100,150,200,250,300`. The immediate queue remains P45/M15 x native-no-IS/native-token-IS; the Zero-TIM pair remains deferred. Historical M15 selection discovery stays at 200 with evaluation disabled.
- Evaluation semantics: 100 held-out prompts x eight generations at the common temperature-0.7 recipe. Steps 0–250 run at their pre-update policy. After update 300 and rollout weight sync, one final held-out rollout records policy step 300. Evaluation examples are not passed to the trainer, so evaluation cannot execute trainer forward/backward or interact with the alignment sidecar.
- Action: changed the P57 renderer/profile/resolved-env/runtime registry/wave verifier to the 300-step contract; enabled the P33/P31 rollout-only evaluation trio; added the final evaluation before learner close; added a fail-closed classifier requiring exactly seven JSON records and 800 finite rewards each; wired it into postflight; changed checkpoint evidence retention to `0` while preserving save-every-10 plus `LatestN(1)`; reduced the legacy recovery evaluator to step 0/final only; rewrote runbook/handoff/plan/phase/state/flag/thread records.
- Validation: Python and shell syntax pass; `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh` passes 121/121 and emits `P57_INPROCESS_EVAL_CLASSIFIER_PASS steps=7`; fresh native, IS, and zero renders each produce P45+M15 manifests and pass real `00_env.sh` preflight at 300. The classifier missing-step and 799/800 coverage negatives both fail as required. Pinned exact-image and target TPU evaluation remain unrun for this uncommitted tree.
- Claim boundary: local construction evidence only. No target run has demonstrated the final step-300 rollout or W&B curve. Earlier 200/450 attempts and their failures remain immutable historical evidence; none may be resumed into this campaign.
- Next: user reviews the uncommitted diff. Only after explicit commit/push approval: publish, run pinned exact-image, render fresh native and IS waves from the pushed SHA, then request separate launch approval.

## 2026-08-23 UTC — paired seed and dataset identity made fail-closed

- Type: reproducibility hardening / local construction evidence
- Decision: keep the immediate campaign at experiment/data-shuffle seed 42 and pin the P57 vLLM engine global seed to 0. This backend does not support a stable per-request sampling seed, so independent temperature-0.7 launches are controlled by identical signed data and seed configuration but are not claimed to produce byte-identical token trajectories or identical curves. General stability remains deferred to a preregistered multi-seed replication.
- Action: append exactly one `--seed=42` to every paired train/recovery-eval command; register and check exact primary dataset identities before rollout (P45 train/eval `ddc96fd9...`/`b10add7f...`, M15 `main` `ff1e659b...`/`8edb61cb...`); emit one seed and one dataset receipt; make the seven-point postflight classifier require both receipts and full hashes; document update-count timing for policy steps 0/50/.../300.
- Validation: P57 CPU gates pass 126/126, including seed-43 and dataset-row mutation negatives. Flag audit passes 322/322. Fresh local native, IS, and zero waves render all six 300-update manifests and pass resolved-env preflight with exactly one seed argument. The pinned immutable image installs all 34 Qwen3-8B TP8 overlay files, passes the P57 stock runtime train/eval matrix, rollout-only evaluation tests, prompt-provenance controls, fixed-lm-head/projection probes, and ends `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8` with exit 0.
- Claim boundary: host and pinned-image construction only. No target TPU campaign has certified the final step-300 evaluation or shown how much stochastic launch variance remains. The diff is uncommitted and unpushed pending explicit user approval.
- Next: review the complete diff. After separate commit/push approval, rerender the immediate native and IS waves from the published full SHA and request separate approval before applying the four JobSets.

## 2026-08-23T19:20:00Z — P45 native evaluation-cycle counter false-red

- Type: target incident / local control-flow repair.
- Fact: `canon-p57-fl-mism-n45j-2a89eef3` completed Step-0 rollout, backward, and one optimizer commit (`actor_trainer.train_steps=1`), then raised `P57 evaluation cycle mapping drifted: policy_step=0 enclosing_global_step=0` before weight sync. The flat incoming raw/classification artifacts hash to `8cc48710b78a7273eea3ac2a12467f12fc6e6f86d1000fe73bbad6af49475d42` and `9fac0a896829283ebd67d70a86200cbe020b788353be9aa2ae7c3fc11c743813`.
- Cause: the receipt executes before `RLCluster.sync_weights()`, which is the operation that advances `rl_cluster.global_steps`. It incorrectly used that deliberately deferred counter as the already-completed timing row. Both standard `update_actor` and P28/G6 have already advanced `actor_trainer.train_steps` at this boundary.
- Action: derive `enclosing_global_step` from the committed actor counter while also asserting the deferred cluster counter still equals the evaluated policy step. Added positive coverage for both update regimes and negatives for an uncommitted actor step and an early cluster advance.
- Validation: focused helper/callsite tests 4/4; P57 CPU 132/132 with `P57_FROZENLAKE_TIM_CPU_PASS`; V1 12/12 with `V1_HP_THREE_FULL_CPU_PASS`; syntax and diff hygiene pass. No TPU, image, render, launch, commit, or push.
- Classification: `n45j` is `INCONCLUSIVE_POST_COMMIT_CONTROL_FLOW`; its one update is not a completed training campaign and is not resumable evidence.
- Next: after review and separate commit/push approval, render a fresh P45 native/no-IS identity and request separate launch approval. Target success requires receipt `policy_step=0 enclosing_global_step=1`, completed weight sync, and policy step 1 reachability.

## 2026-08-23T19:05:00Z — four fresh restarts and final-only checkpoint contract

- Type: user decision / storage-and-iteration optimization / phase-contract update.
- Decision: restart the immediate P45/M15 x native-no-IS/token-IS matrix as four fresh 300-update jobs from one future published SHA. Do not resume any earlier partial attempt. Keep the deferred Zero-TIM pair on the same checkpoint contract when it is later authorized, so checkpoint I/O is not a treatment variable.
- Checkpoint mechanism: `LatestN(1)` alone only reduces retained generations; it does not reduce writes. The active P57 primary identity now sets `FixedIntervalPolicy(300)`, so the trainer's existing post-commit save call fires only after update 300. The final actor and resident optimizer state are retained; intermediate primary checkpoints do not exist. Renderer/profile/resolved-env/Python parser all require interval `300`, max-to-keep `1`, milestone interval `0`, and full stop `300`. Legacy P45 and historical M15-selection/200 remain interval `10`/latest `1`.
- Evaluation-counter mechanism: the step-0 receipt continues to derive its enclosing timing row from committed `actor_trainer.train_steps` while proving `rl_cluster.global_steps` is still the pre-sync policy step. This is shared by all four native jobs.
- Validation: focused P45 checkpoint contract 15/15; P57 CPU 136/136 with `P57_FROZENLAKE_TIM_CPU_PASS`; V1 12/12 with `V1_HP_THREE_FULL_CPU_PASS`; Python/shell syntax and `git diff --check` pass. The broad bare-host P45 suite has two unrelated dependency import errors (`datasets`, `metrax` absent), so it is not counted as a green suite. No pinned-image, TPU target, production render, launch, commit, or push was performed.
- Claim boundary: host construction evidence only. A target run must prove the repaired step-0 receipt reaches weight sync and policy step 1, then completes update 300 and writes the sole registered checkpoint at step 300.
- Next: review this local diff. Only after explicit commit/push approval, publish it; then render all four fresh manifests from the full pushed SHA, return their hashes/preflight receipts, and request separate launch approval.
