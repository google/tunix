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
