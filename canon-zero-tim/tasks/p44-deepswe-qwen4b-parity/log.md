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

## 2026-08-12T02:45:02Z — P44.9: r04 SwiGLU feature repair passes latest-baseline gates

- Type: target-failure reconciliation, implementation, and local verification
- Fact: Archived attempt `p44r04` reached the Qwen3-4B MLP and failed at
  `(M, F)=(4096, 1216)` because the TP8-local intermediate width is not
  divisible by the unchanged BF256 SwiGLU kernel. Audit also found that the
  future Qwen3-32B TP8-local width is `3200`, not BF256-aligned; Qwen3-8B width
  `3072` is already aligned.
- Action: Preserved the base kernel, bf16 formula, and canonical custom VJP;
  added exact model-overlay feature-padding mappings `1216->1280` and
  `3200->3328`; left the 8B mapping empty; sliced the output to the semantic
  width; rejected unregistered widths; exposed `F/Fp` in PATHTRACE; and made
  the P44 classifier require the 4B runtime marker.
- Integration: Fetched and safely fast-forwarded the dirty development
  worktree from `a9dc5f296a5cd1225efba7a66a7249113baefe00` to exact operator baseline
  `e4ead609498771987c011a9cbc16fec7e4b17f69` using a named temporary stash.
  The two intervening commits affect only the P38 task ledger. The task-owned
  stash was restored and dropped; two pre-existing user stashes were not
  changed.
- Commands: P44/P43/P39 CPU gates; P34 static, trajectory, and update gates;
  P44/P43/P34 exact-image gates using immutable local image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`;
  `git diff --check`.
- Result: PASS — P44 34 cases, P43 22 cases, P39 15 cases, P34 10 static
  suites plus trajectory/update gates; all overlays install 29/29. Exact
  Pallas-interpret forward/VJP markers report 4B `1216->1280`, unpadded 8B
  `3072->3072`, and 32B `3200->3328`, each with an adjacent-width negative
  control. P34 terminal marker reports 55 unit and two Pallas cases.
- Files/artifacts: model contracts and manifests for qwen4b/qwen8b/qwen32b;
  `p22xj_padded_swiglu.py`; `qwen2_p22xj.py`; P44 classifier/probe/tests;
  P34/P43 exact-image regressions; P44 phase ledger, handoff, runbook, and r04
  archive interpretation.
- Boundary: No TPU target, rollout, backward, optimizer update, cloud action,
  commit, push, main-branch change, precision change, loss change, or optimizer
  policy change occurred for P44.9.
- Rollback: Do not publish or launch P44.9. The repair is additive and
  model-pinned; no broad branch reset or main-branch operation is required.
- Next: Obtain explicit commit/push authorization, publish only to
  `yuxzhang/canon-zero-tim`, record the exact read-back SHA, then launch a fresh
  rollout-only `p44r05` on the available 64- or 256-device allocation.

## 2026-08-12T02:54:29Z — P44.9: implementation committed for publication

- Type: publication checkpoint
- Fact: The user explicitly authorized commit and push to the operator branch;
  main remains forbidden.
- Action: Staged only the 27 P44.9 engine, model-contract, manifest, test,
  runbook, archive-interpretation, and phase-ledger files and created
  implementation commit `1a058b461496e039a3857c094b109b794027783a` with
  subject `deepswe: pad model-pinned SwiGLU features`.
- Result: The implementation commit is based on exact operator revision
  `e4ead609498771987c011a9cbc16fec7e4b17f69`; the remote branch was fetched
  immediately before commit and had not advanced. Local and exact-image gates
  remain the evidence recorded in the preceding checkpoint.
- Files/artifacts: implementation commit `1a058b46`; updated `HANDOFF.md`,
  `state.md`, `plan.md`, and P44.6/P44.9 phase contracts.
- Boundary: No main-branch action, force push, cloud action, or target launch
  is authorized or performed by this checkpoint.
- Rollback: Revert the P44.9 implementation and publication-metadata commits
  on the operator branch; do not rewrite history or touch main.
- Next: Commit this publication metadata, push both commits only to
  `origin/yuxzhang/canon-zero-tim`, and read back the exact remote head.

## 2026-08-12T02:56:29Z — P44.9: correct the implementation anchor expansion

- Type: publication metadata correction
- Fact: Push of implementation `1a058b46` and metadata `01e0f1c4` succeeded,
  and remote read-back returned
  `01e0f1c4f279d90b2805d0fc46716010f69e3bfc`. A subsequent ancestry check
  exposed that the handoff had expanded the valid short implementation SHA to
  an incorrect 40-character value.
- Action: Read the object id directly with `git rev-parse 1a058b46`, corrected
  every full-SHA occurrence to
  `1a058b461496e039a3857c094b109b794027783a`, and marked P44.9 published.
- Result: The implementation object and code are unchanged; this correction
  repairs only the operator-facing provenance metadata.
- Boundary: No force push, history rewrite, main-branch action, cloud action,
  or target launch occurred.
- Next: Commit and push this correction to the same operator branch, read back
  its exact head, and require the launch agent to verify ancestry from the
  corrected implementation anchor.

## 2026-08-12T03:30:50Z — P44.10: r05 Mosaic repair passes real one-host v5p

- Type: target-failure reconciliation, code change, and real-TPU verification
- Fact: Fast-forwarded to exact operator head
  `3ec5fd7c3074844c62d3a9ff2c95179449a66129`, which archives `p44r05` from
  source `115ef8144a873b5f108ec4b52aafc959032c3f43`. The raw log SHA-256 is
  `51b1674c3c3b2d42e6738a0d66dce3a5f222bbd2c52a296ce75379488e181168`.
  r05 proved P44.9 through all 36 layers, then Mosaic rejected Qwen3-4B
  BN64/BK64 matmul block specs. Gate/up semantic N and down semantic K are
  both 1216, so the archived suggestion to change only BK was incomplete.
- Action: Pinned the Qwen3-4B overlay to BN/BK128; added exact model-pinned
  matmul K/N `1216->1280` padding, semantic output slicing, matching
  padded-K canonical VJP, richer K/Kp/N/Np PATHTRACE, fail-closed classifier
  requirements for both directions, interpret and real-TPU probes, and a
  repeatable `run_onehost_v5p.sh` gate.
- Commands: P44 CPU and exact-image gates; P43/P39/P34 DeepSWE CPU gates;
  P43/P34 exact-image gates; syntax/compile/diff checks; and
  `run_onehost_v5p.sh` with immutable local image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
- Result: PASS — P44 36 cases; Qwen4B/Qwen8B/Qwen32B overlays each install
  29/29; P43 22 cases; P39 15 cases; P34 static 10 suites plus
  trajectory/update. The privileged image exposed four TPU v5 devices and
  passed real Pallas forward/custom-VJP exact comparison at M=4096 for all
  five unique local projection shapes (q, k/v, o, gate/up, down), including
  N padding `2560x1216->2560x1280` and K padding
  `1216x2560->1280x2560`; both unknown-width negative controls passed.
  Terminal markers: `MATMUL_DIM_PADDING_PASS mode=tpu cases=5/5 ... devices=4`
  and
  `P44_ONEHOST_V5P_MATMUL_PASS model=qwen4b devices=4`.
- Boundary: Full one-host DeepSWE E2E remains `BLOCKED_REAL_ENVIRONMENT`
  because the image has no R2E-Gym or Kubernetes access and the reviewed HF
  cache is not a complete initial-weight snapshot. No fake trajectory, model
  download, remote launch, commit, push, main action, or optimizer/loss/
  precision policy change occurred. The unrelated P45 bare-host suite was
  inconclusive because optional `datasets` and `metrax` packages are absent.
- Files/artifacts: engine/model manifests, P44 probes/classifier/tests,
  `phases/p44-10-r05-matmul-padding.md`, runbook, handoff, and corrected r05
  archive interpretation.
- Rollback: Revert only the uncommitted P44.10 engine/test/documentation diff;
  do not reset or touch main. The published operator branch remains at
  `3ec5fd7c`.
- Next: Wait for explicit commit/push authorization. After publication only to
  `yuxzhang/canon-zero-tim`, read back the exact remote head and launch a fresh
  rollout-only `p44r06` on the available 64- or 256-device allocation.

## 2026-08-12T04:35:00Z — P44.11: real one-host DeepSWE chain reaches backward

- Type: implementation, iterative target repair, and real-TPU verification
- Fact: A complete Qwen3-4B-Instruct-2507 snapshot, pinned R2E-Gym checkout
  `0d94c4eb9431cd195c55a7ea3abd54006c9a1735`, cached `R2E-Gym-V1`, reviewed
  whitelist, working Docker daemon, and four direct-attached TPU v5 devices
  are available on the host. This supersedes P44.8's earlier prerequisite
  inventory.
- Action: Added a default-off, mutually exclusive DP1 x TP4 colocated profile,
  one-prompt/two-generation local artifact schemas, persistent solve metrics,
  a fail-closed real-gradient/no-optimizer-commit boundary, state
  fingerprints, optimizer-memory placement and HBM reporting, and a repeatable
  runner. Iteration repaired cached dataset selection, vLLM CPU staging,
  local `dp` data-axis selection, rollout-only's unnecessary trainer forward,
  Splash-attention sequence divisibility, and production-prefix-cache static
  contract preservation.
- Commands: `run_onehost_deepswe_v5p.sh rollout-only` and
  `run_onehost_deepswe_v5p.sh backward-no-commit`; P44/P43/P39 CPU gates; P34
  static/trajectory/update/exact-image gates; P44 Qwen3-4B exact-image gate;
  syntax/compile/diff checks.
- Rollout result: PASS marker `DEEPSWE_ONEHOST_ROLLOUT_PASS`. Two real
  trajectories selected the reviewed Orange3 task and executed real Docker
  `search` actions, then both reached `MAX_CONTEXT_LIMIT_REACHED` under the
  signed response-512/turn-2 bound. Trajectory and solve-metric artifacts are
  complete; terminal episode completion and solve quality are not proved.
- Backward result: the model loaded, sampler/trainer logprobs were exactly
  equal, trainer forward and backward executed, and the no-commit report
  proved `commits=0`, train step `0 -> 0`, device-resident optimizer state,
  and no changed model/reference/optimizer/accumulator paths. The finite
  gradient norm was `0.0` because both rewards and advantages were zero, so
  verdict is deliberately `INCONCLUSIVE_NO_SIGNAL` and the runner exits 3.
- Memory: per-device peak HBM was approximately 35.92 GiB against a 95.74 GiB
  limit.
- Evidence: copied without modification from `/tmp` to
  `/mnt/disks/tunix-data/deepswe-onehost-evidence/20260812-p44-local-dev/`.
  Report and trajectory hashes are recorded in
  `phases/p44-11-onehost-deepswe-integration.md`.
- Integration: During final gates the operator branch advanced from tested
  base `3ec5fd7c3074844c62d3a9ff2c95179449a66129` first to `76cef0ec` and then
  to `d8184123448d0add72b72f09d0a6faf5d326c26e`. The latter adds P38
  capture/precheck hardening, including a guarded shared-learner precheck
  change. The development branch was safely fast-forwarded; resolution
  preserved upstream P38 Section 50, the P38 learner logic, and local P44
  Section 49/one-host logic. Post-reconciliation regressions cover the combined
  source; the actual v5p run remains development evidence from the earlier
  recorded base plus diff.
- Result: PASS — P44 40 tests and Qwen3-4B exact image; P43 22 tests; P39 15
  tests; P34 static 10 suites, trajectory, update, and Qwen3-32B exact image.
  The P44 exact-image gate initially caught a stale local test expectation for
  the prefix-cache expression; restoring the original P34 expression plus a
  separate one-host override made both P34 and P44 contracts pass.
- Boundary: This is development evidence against a recorded base plus
  uncommitted diff. It proves one-host integration wiring, real environment
  action, backward execution, no-commit behavior, optimizer placement, and
  HBM only. It proves no nonzero learning signal, update, TP8, Pathways,
  separated roles, DP4/DP16, 64/256 behavior, Qwen3-32B training, zero-TIM, or
  production admission. No commit, push, main-branch action, remote launch,
  precision/loss/reward/optimizer-policy change, or secret access occurred.
- Next: Audit the complete diff and wait for explicit commit/push
  authorization. After publication only to `yuxzhang/canon-zero-tim`, repeat
  rollout-only from a clean checkout before the independent 64/256 target
  ladder.

## 2026-08-12T04:44:00Z — P44.11: latest-head v5p reconciliation repeats the result

- Type: latest-source target verification
- Fact: Operator head advanced to
  `d8184123448d0add72b72f09d0a6faf5d326c26e` with P38-specific capture and
  alignment-precheck hardening, including a guarded change in the shared GRPO
  learner.
- Action: Preserved the upstream P38 code and evidence, reconciled the local
  P44 changes, reran P44/P43/P39/P34 CPU and Qwen4B/Qwen32B exact-image gates,
  then repeated both real one-host stages. The first rollout attempt was
  stopped before TPU initialization because it was accidentally run inside a
  network-restricted sandbox and libtpu could not reach instance metadata;
  the authorized host rerun is the evidence-bearing attempt.
- Result: All regression gates PASS. Latest-source rollout-only PASS; latest
  backward-no-commit again returned exit 3 `INCONCLUSIVE_NO_SIGNAL` with
  exact sampler/trainer logps, finite zero gradient, zero commits, unchanged
  state, device-resident optimizer, and the same HBM profile. Both stages
  executed real Docker tool actions and wrote persistent artifacts.
- Evidence: paths and SHA-256 are recorded in
  `phases/p44-11-onehost-deepswe-integration.md`. Both manifests explicitly
  record `source_commit=d8184123...` and the local development branch; runner
  inventory recorded `tracked_dirty=1`, so this is not clean-publication
  evidence.
- Boundary: No commit, push, main action, remote cluster launch, one-update,
  or claim promotion occurred.
- Next: Wait for explicit commit/push authorization, then require the launch
  agent to repeat rollout-only from a clean detached operator SHA.

## 2026-08-12T05:05:00Z — P44.10/P44.11: implementation committed for publication

- Type: publication checkpoint
- Action: Audited and committed the Qwen3-4B BN/BK128 plus model-pinned
  matmul K/N-padding repair, the default-off one-host DeepSWE integration
  profile, durable trajectory/solve metrics, backward-no-commit boundary,
  tests, runbook, and handoff. Generated `runs/` artifacts were excluded.
- Result: Implementation commit
  `29cea119259f1f7fe583a3e3dd1cb190acc0bf63` created from exact operator
  baseline `d8184123448d0add72b72f09d0a6faf5d326c26e`. The previously recorded
  P44/P43/P39/P34 and real v5p results remain the release evidence; no new
  claim is inferred from committing them.
- Safety: main was not checked out, modified, merged, or targeted. Publication
  remains scoped only to `origin/yuxzhang/canon-zero-tim`.
- Next: Commit this publication metadata, push both commits to the operator
  branch, and read back the exact remote head before handoff.

## 2026-08-12T05:10:00Z — P44.10/P44.11: operator publication read back

- Type: publication read-back
- Result: Fast-forward push to `origin/yuxzhang/canon-zero-tim` succeeded.
  The first exact remote read-back was
  `0b492277167004743c07f2fe77705d27c1f8cb01`, which contains implementation
  commit `29cea119259f1f7fe583a3e3dd1cb190acc0bf63`.
- Safety: the push used explicit ref
  `HEAD:refs/heads/yuxzhang/canon-zero-tim`; main was untouched and no force
  update was used.
- Next: Resolve the latest remote SHA again at launch time, require the
  implementation commit as an ancestor, repeat clean-source one-host
  rollout-only, then enter the independent 64/256 promotion ladder.
