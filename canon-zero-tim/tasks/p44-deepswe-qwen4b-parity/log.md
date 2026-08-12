# Log

## 2026-08-12T00:01:25Z — P44.1: bind Qwen3-4B parity task

- Type: decision
- Fact: Tunix has a Qwen3-4B model config and registry entry, while the canonical engine package currently has model overlays only for Qwen3-1.7B, Qwen3-8B, and Qwen3-32B.
- Action: Bound the resumable task directory and froze the intended common recipe plus the topology-only difference allowlist.
- Command: `rg -n "Qwen3-4B" tunix canon-zero-tim examples`
- Result: Qwen3-4B registry/config support is present; DeepSWE profile, engine shim, renderer, and classifiers are absent.
- Files/artifacts: `state.md`, `plan.md`
- Rollback: Remove only the isolated `tasks/p44-deepswe-qwen4b-parity/` records if the campaign is abandoned before implementation.
- Next: Add failing parity-contract tests, then implement the shared recipe.

## 2026-08-12T00:04:09Z — P44.1: shared recipe contract passes

- Type: code change
- Fact: Both P44 variants carry the same Qwen3-4B model, 4x4 trajectory workload, rollout limits, GRPO algorithm, optimizer settings, and bounded stages; only the registered topology fields differ.
- Action: Added fail-closed 64/256 workload selection, a normalized recipe signature, mutual exclusion with P39/P43, and CPU positive/negative controls.
- Command: `bash canon-zero-tim/tests/p44_deepswe_qwen4b_parity/run_cpu.sh`
- Result: PASS, 5 tests; marker `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS`.
- Files/artifacts: `tunix/rl/deepswe_contract.py`, `canon-zero-tim/tests/p44_deepswe_qwen4b_parity/test_contract.py`
- Rollback: Revert only the P44 workload/constants and the isolated P44 tests; existing workload objects were not changed.
- Next: Implement and test the Qwen3-4B TP8 engine overlay.

## 2026-08-12T00:06:32Z — P44.2: Qwen3-4B exact-image adapter passes

- Type: code change
- Fact: Qwen3-4B TP8 yields local projection widths 512, 128, and 1216; the down projection requires a model-local BK64 canonical-VJP chunk.
- Action: Added the Qwen3-4B profile and model overlay, changed the canonical VJP replica to import the model contract's BK, and pinned the updated manifests.
- Command: `bash canon-zero-tim/tests/p44_deepswe_qwen4b_parity/run_exact_image.sh sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
- Result: PASS; 29/29 overlay files matched, 9 CPU tests passed, model self-test passed 5/5, marker `P44_EXACT_IMAGE_CPU_PASS overlay=qwen4b`.
- Files/artifacts: `canon-zero-tim/src/engine_shims/models/qwen4b/`, `canon-zero-tim/cluster/profiles/qwen3-4b.env`, `canon-zero-tim/src/engine_shims/p22xk_vjp_ops.py`
- Rollback: Remove the isolated qwen4b profile/overlay and restore the prior `p22xk_vjp_ops.py` plus its single global manifest hash.
- Next: Integrate P44 selection, topology placement, rollout-only behavior, and durable trajectory evidence.

## 2026-08-12T00:11:41Z — P44.3: dual-topology training path passes

- Type: code change
- Fact: Both variants now use Qwen3-4B, 16 trajectories, identical bounded stages and durable artifact schemas; DP-local geometry is DP4/local4/M1024 versus DP16/local1/M4096.
- Action: Added the shared topology-selecting profile, active-workload placement, generalized P43/P44 artifact path and rollout-only control, P44 alignment admission, one dual-topology renderer, and environment/recipe drift controls.
- Command: `bash canon-zero-tim/tests/p44_deepswe_qwen4b_parity/run_cpu.sh`
- Result: PASS, 22 tests; all six topology-stage JobSets render and preflight, normalized rendered recipes match, marker `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS`.
- Files/artifacts: `canon-zero-tim/cluster/profiles/qwen3-4b-dp-parity-deepswe-debug.env`, `canon-zero-tim/cluster/render_p44_deepswe_parity.py`, `tunix/rl/deepswe_debug.py`, `tunix/rl/agentic/`
- Rollback: Remove P44 flags/profile/renderer and the P44 branches in the shared artifact/alignment helpers; P43 schema constants and markers remain separately selected.
- Next: Add topology-aware postflight classification and the operator runbook.

## 2026-08-12T00:16:45Z — P44.4: classifier and operator handoff pass

- Type: code change
- Fact: P44 now has one fail-closed classifier with explicit DP4 and DP16 transaction, scheduler, mesh, artifact, solve-label, HBM, and monotonic-update expectations.
- Action: Routed P44 evidence reservation and postflight ahead of P43/P39/P34, added classifier controls for both allocations, and wrote the publication-aware dual-topology runbook and handoff.
- Command: `bash canon-zero-tim/tests/p44_deepswe_qwen4b_parity/run_cpu.sh`
- Result: PASS, 26 tests; both rollout-only and three-update fixtures pass on 64/256, while topology mismatch and nonmonotonic steps fail closed.
- Files/artifacts: `canon-zero-tim/cluster/steps/90_run.sh`, `canon-zero-tim/tests/p44_deepswe_qwen4b_parity/classify_run.py`, `canon-zero-tim/cluster/P44_DEEPSWE_QWEN4B_PARITY_RUNBOOK.md`, `HANDOFF.md`
- Rollback: Remove the P44-first reservation/classifier branches and isolated P44 docs/tests; existing P43/P39/P34 fallback order remains intact.
- Next: Run exact-image and adjacent DeepSWE release regressions, then audit the complete diff.

## 2026-08-12T00:18:57Z — P44.5: local release gates pass

- Type: verification
- Fact: The shared model-pinned BK import preserves Qwen4B BK64, Qwen8B BK256, and Qwen32B BK128 overlay contracts in the same immutable dependency image.
- Action: Ran the complete P44 gate, adjacent DeepSWE suites, syntax/compile/diff checks, and exact-image installs for all affected model overlays.
- Commands: `run_cpu.sh` for P44/P43/P39, P34 `run_static.sh`; `run_exact_image.sh` for P44/P43/P34 with local image ID `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`; `bash -n`; `python3 -m py_compile`; `git diff --check`
- Result: PASS — P44 27/27, P43 21/21, P39 15/15, P34 10 suites; every affected overlay installs 29/29; Qwen4B self-test 5/5; all three exact-image terminal markers present. Both DeepSWE dataset entrypoints omit the datasets library's removed `trust_remote_code` argument; the remaining occurrence is a supported tokenizer option.
- Files/artifacts: complete P44 worktree, `cluster/P44_DEEPSWE_QWEN4B_PARITY_RUNBOOK.md`, `tasks/p44-deepswe-qwen4b-parity/HANDOFF.md`
- Rollback: The P44 lane is default-off and isolated by explicit flags; removing its profile/renderer/tests plus P44 branches restores prior selection, while the model-pinned BK import can be independently reverted with its manifest hash.
- Next: Publish to `yuxzhang/canon-zero-tim`, read back and detach at its exact SHA, then launch the available topology's first rollout-only stage.

## 2026-08-12T00:51:20Z — P44.7: reconcile first 256-device target failure

- Type: correction
- Fact: Remote attempt `p44r02` used source `5a52cc8c4cdaacce9dbe4983ab141d342d0e5588`, passed the Pathways/R2E/model/CLI/dataset gates, and failed before mesh construction at `P34 physical half split crosses host boundaries: processes=[0]`. The splitter could only reach that check after accepting 256 unique device ids and coordinates with extents `(4, 8, 8)`.
- Action: Pulled archived evidence commit `40cd16ad7ec2b229be15c4c7d50c12498ca323b4`, compared the traceback with the role splitter and existing topology helpers, and re-opened local implementation before target promotion.
- Command: `git pull --ff-only origin yuxzhang/canon-zero-tim`
- Result: Fast-forwarded `5a52cc8c` to `40cd16ad`; raw log SHA-256 is `3d7101454fad0361394fecf06adc30d7734945d333a8c086b7a74b8d26dda944`. Existing successful Pathways logs contain the same `PjRt-IFRT device count: total=1` CPU diagnostic followed by 64 virtual TPU devices carrying repr `logical_task`, so the current README explanation of incomplete worker registration is not established.
- Files/artifacts: `debug_logs/p44_p44r02_deepswe_256_parity.raw.log`, `debug_logs/README.md`, `tunix/rl/deepswe_contract.py`, `tunix/utils/topology.py`
- Rollback: This checkpoint changes only task records; code repair remains isolated to the default-off DeepSWE paths and will retain hard topology failures.
- Next: Compare the available DeepSWE reference branches, then add target-derived placement tests before changing the splitter.

## 2026-08-12T00:53:04Z — P44.7: correct the workload reference branch

- Type: correction
- Fact: The user-confirmed workload reference is `yuxzhang/deepswe-quality-fix`, pinned at `023978b976dd6d94e7a42948c3f3a68e34d73744`; `yuxzhang/deepswe-debug-fix` was a mistaken name and does not exist on origin.
- Action: Removed `deepswe-launch-fix` from the phase input and bound comparison to the pinned quality-fix revision plus the two separately reviewed main commits.
- Command: `git fetch origin yuxzhang/deepswe-debug-fix`
- Result: The mistaken ref returned `couldn't find remote ref`; no branch switch, merge, source edit, or main mutation occurred.
- Files/artifacts: `state.md`, `phases/p44-7-r02-repair.md`
- Rollback: Not applicable; this checkpoint corrects reference provenance only.
- Next: Verify and diff the pinned quality-fix revision against the current P44 implementation.

## 2026-08-12T01:01:51Z — P44.7: repair passes exact-image and adjacent gates

- Type: code change and verification
- Fact: Pathways virtual TPU devices expose a degenerate `process_index=0` but a stable repr `logical_task` per four-device host; agentic generation consumes prompt batches while learner logprob calls consume generated trajectory batches.
- Action: Added a standalone fail-closed DeepSWE host-key resolver, exact host-cardinality controls, deterministic device-inventory evidence, single-conversation prompt wrapping, and trajectory-counted logprob microbatching. Compared but did not merge `yuxzhang/deepswe-quality-fix@023978b976dd6d94e7a42948c3f3a68e34d73744`; ported only the independently reviewed narrow semantics.
- Commands: P44/P43/P39/P34 CPU gates; P44/P43/P34 exact-image gates using `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`; targeted `AgenticRLLearner` and `AgenticGrpoLearner` unit tests in that image; `python3 -m py_compile`; `git diff --check`.
- Result: PASS — P44 32/32 plus both affected learner unit tests; P43 21/21; P39 15/15; P34 static/trajectory/update gates; all three exact-image terminal markers. The branch was safely fast-forwarded from `40cd16ad` to then-current operator baseline `6b529510409fd21fe69c7dfac497c2a117e52913`; the intervening P42 evidence-only commit had no overlapping files.
- Failure note: Importing `tunix.utils.topology` from the standalone contract initially pulled the package-level optional `metrax` dependency on the bare CPU gate, so the host-key parser remains self-contained at this boundary. Two preliminary targeted-test invocations selected the wrong absl/unittest module path; the tests passed once launched from `tests/rl/agentic`.
- Files/artifacts: `tunix/rl/deepswe_contract.py`, `tunix/rl/agentic/agentic_rl_learner.py`, `tunix/rl/agentic/agentic_grpo_learner.py`, `examples/deepswe/train_deepswe_nb.py`, P44 classifier/tests/exact-image gate.
- Rollback: Revert the isolated host-key resolver and P44 evidence requirements plus the two narrow agentic batching changes; no production flag default or existing P34/P39/P43 recipe was relaxed.
- Next: Inventory direct-attached v5p/Qwen3-4B/R2E prerequisites before implementing or running a default-off one-host smoke.

## 2026-08-12T01:04:25Z — P44.8: local smoke inventory is environment-blocked

- Type: verification and documentation
- Fact: The current host cannot initialize the TPU backend because `libtpu.so` is absent, exposes no `/dev/vfio` or `/dev/accel*`, has no existing Qwen3-4B checkpoint in the reviewed local roots, cannot import `r2egym`, and has neither `kubectl` nor a readable user kubeconfig.
- Action: Stopped the optional one-host lane as `BLOCKED_REAL_ENVIRONMENT` without downloading weights, substituting a fake R2E environment, or running a forward/backward/update. Corrected the archived r02 explanation, promoted P44.7 to passed, made P44.6 target promotion active, and added exact host-inventory/logprob evidence requirements to the operator runbook and handoff.
- Commands: Host JAX TPU inventory with `JAX_PLATFORMS=tpu`; read-only device-node, package, model-directory, and kubeconfig checks.
- Result: Local one-host E2E NOT RUN. This does not invalidate the CPU/exact-image repair gates and does not block a publication followed by an operator-owned 64/256 rollout-only retry.
- Files/artifacts: `phases/p44-8-onehost-smoke.md`, `phases/p44-6-target-promotion.md`, `cluster/P44_DEEPSWE_QWEN4B_PARITY_RUNBOOK.md`, `tasks/p44-deepswe-qwen4b-parity/HANDOFF.md`, `debug_logs/README.md` Section 45.
- Rollback: Documentation-only phase transition; the P44 lane remains default-off and no external state changed.
- Next: Audit the complete implementation diff, rerun syntax/diff and final P44 release markers, then wait for explicit commit/push authorization.

## 2026-08-12T01:06:10Z — P44.7: replay repair onto latest operator baseline

- Type: integration
- Fact: While the final audit was running, `origin/yuxzhang/canon-zero-tim` advanced to `7ea2176f807e3e13fde17499e15fef2bd497363b` with the independent P42 duplicate-gradient contract correction.
- Action: Fetched and fast-forwarded the development branch, then replayed all uncommitted P44 changes. The only textual conflict was adjacent documentation in `debug_logs/README.md`; resolution preserves both the P44 Section 45 root-cause correction and upstream P42 Section 46 finding. No DeepSWE code file overlapped.
- Command: `git fetch origin yuxzhang/canon-zero-tim`; temporary local stash; `git merge --ff-only origin/yuxzhang/canon-zero-tim`; restore and resolve documentation.
- Result: Worktree now exactly tracks operator baseline `7ea2176f807e3e13fde17499e15fef2bd497363b` plus the uncommitted P44 repair; all files are unstaged and `git diff --check` passes.
- Files/artifacts: complete P44 diff, preserved P42 upstream changes.
- Rollback: The conflict-preservation stash remains available until the final post-replay gates confirm the restored worktree.
- Next: Rerun affected post-replay gates, then remove only the redundant task-owned stash and wait for explicit commit/push authorization.

## 2026-08-12T01:07:00Z — P44.7: latest-baseline release gates pass

- Type: verification
- Fact: Development HEAD and `origin/yuxzhang/canon-zero-tim` both resolve to `7ea2176f807e3e13fde17499e15fef2bd497363b` before publication of the still-uncommitted P44 repair.
- Action: Re-ran the affected fixed-image and adjacent gates after the upstream replay, verified the restored worktree, then dropped only the redundant task-owned conflict-recovery stash. Pre-existing unrelated stashes were preserved.
- Commands: P44 exact-image gate; P43 and P39 CPU gates; P34 static, trajectory, and update CPU gates; `git diff --check`.
- Result: PASS — `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS` (32 cases), two learner unit tests, `P44_EXACT_IMAGE_CPU_PASS overlay=qwen4b`, `P43_DEEPSWE_DEBUG_CPU_PASS`, `P39_DEEPSWE_PILOT_CPU_PASS`, `P34_STATIC_PASS suites=10`, `P34_TRAJECTORY_CPU_PASS tests=5`, and `P34_UPDATE_CPU_PASS tests=5`.
- Files/artifacts: complete unstaged P44 implementation and documentation diff.
- Rollback: No external state changed; no commit or push exists. The operator branch remains untouched at the upstream baseline.
- Next: Wait for explicit commit/push authorization. After publication, fill in the exact repair SHA in `HANDOFF.md` and begin P44.6 with rollout-only on the available 64- or 256-device allocation.

## 2026-08-12T01:11:42Z — P44.7: repair implementation committed for publication

- Type: publication checkpoint
- Fact: The user explicitly authorized commit and push to the operator branch; main remains forbidden.
- Action: Staged only the 20 P44/DeepSWE implementation, test, runbook, handoff, and phase-ledger files and created repair implementation commit `5f0cf7e04b34932d8c9deb2463f3b205e3ad8b51` with subject `deepswe: repair Pathways placement and batch semantics`.
- Command: `git commit -m "deepswe: repair Pathways placement and batch semantics"`.
- Result: Commit created on local development branch from exact operator baseline `7ea2176f807e3e13fde17499e15fef2bd497363b`; this publication-metadata checkpoint will be committed on top and both commits pushed only to `origin/yuxzhang/canon-zero-tim`.
- Files/artifacts: implementation commit `5f0cf7e04b34932d8c9deb2463f3b205e3ad8b51`, updated `HANDOFF.md` publication contract.
- Rollback: Revert the two publication commits on the operator branch; do not rewrite or touch main.
- Next: Push, read back the exact remote head, and give that head to the launch agent as the execution source.
