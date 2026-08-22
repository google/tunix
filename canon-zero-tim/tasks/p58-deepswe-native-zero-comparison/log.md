# Log

## 2026-08-21 UTC — P58 task bound and loss ambiguity preregistered

- Type: decision/research
- Fact: the user approved a two-arm Qwen3-4B-Instruct B8 x G16 comparison and asked to verify `sequence-mean-token-scale` before implementation. The current local DeepSWE notebook and contract, the pinned quality-fix notebook, and the official DeepSWE algorithm description all select fixed maximum-context normalization. The public rLLM launcher instead selects `seq-mean-token-sum`, and an open-source issue records the inconsistency without resolution.
- Fact: current Tunix computes `sequence-mean-token-scale` as masked token sum divided by response width, averaged across rows. The operator branch counts empty rows in that average; pinned quality-fix and current `origin/main` exclude them. The trainer scales each micro-batch gradient before an equal-step gradient accumulator average, so B8 x G16 requires an explicit equal-eight-microbatch invariant.
- Action: created an independent P58 workflow rather than rewriting the historical P44 or P46 ledgers. Preregistered the shared recipe, treatment boundary, algorithm-neutral switches, claim ceiling, fixed-16K loss formula, empty-row policy, and pre-launch tests.
- Source: local HEAD `a8716c27d8d6c65bbce827140ab37464424ce20c`; observed operator remote `762152dc3395f59ec4eace10f927f2e27f7fc90d`; pinned workload reference `023978b976dd6d94e7a42948c3f3a68e34d73744`.
- Result: P58.1 is active. No implementation code, existing task document, manifest, TPU resource, commit, push, branch, image, credential, or external state was changed. Existing dirty P46 work remains untouched.
- Files/artifacts: `state.md`; `plan.md`; `phases/p58-1-loss-aggregation-contract.md`.
- Rollback: remove only the untracked `canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/` directory.
- Next: user reviews the recommended loss contract. Implementation begins only after that decision; hardware launch remains a later explicit approval.

## 2026-08-21 UTC — host loss-test attempt was inconclusive

- Type: validation
- Action: attempted the existing loss suite with `python3 -m unittest tests.rl.common_test` after the source audit.
- Result: the suite did not import because the bare host lacks `metrax`; zero tests executed. This is `INCONCLUSIVE`, not PASS, and does not change the code-reading findings. The P58.1 implementation must run its formula and gradient gates in the pinned exact image or another declared environment with the full dependency set.
- External effects: none; no model, TPU, cluster, optimizer, commit, or push was used.
- Next: keep P58.1 active until the future exact-image loss oracle and reduction tests pass.

## 2026-08-21 UTC — fixed-B empty-row policy tightened (superseded)

- Type: correction/decision
- Fact: excluding empty rows would silently change the effective batch and gradient scale, while counting them would silently dilute the update. Neither is desirable in the signed no-filter B8 x G16 comparison.
- Action: require exactly 128 non-empty completion rows before every P58 optimizer commit. Any empty row is logged and rejects the batch without resampling or committing. This makes the current and pinned denominator implementations equal on every admitted batch.
- Result: no common loss implementation was copied from `main`; the P58.1 gate now treats empty rows as an upstream trajectory/admission failure rather than an alternate loss-normalization policy.

## 2026-08-21 UTC — compact-filter policy correction and isolated worktree

- Type: correction/decision
- Correction: the preceding fixed-B policy incorrectly collapsed a legitimate DeepSWE compact-filtered all-zero loss mask with a malformed trajectory. P58 preserves the official and pinned quality-fix compact filter. Exactly 128 raw trajectory records are required, but `B_eff` is the number of rows with nonzero policy masks. Signed filtered rows remain journaled and are excluded from policy loss; structurally invalid rows remain fatal.
- Math: `sequence-mean-token-scale` is frozen as `sum(mask * token_loss) / (B_eff * 16384)`. Eight raw-equal microbatches must be accumulated by effective-row weight, not by an unweighted mean of local means. `B_eff=0` produces no optimizer commit and no resampling.
- Action: fetched the latest operator tip and created named branch/worktree `local/p58-deepswe-native-zero-0821` at `7a77b32f2cd2dc08078e175fa0c407ca1cf33539`. Mechanically migrated only the untracked P58 workflow documents; the dirty P46 review worktree remains unchanged.
- Validation: repository preflight passes for branch, required package paths, credential-free remote, and runtime-config scan. The clean-state check passed before P58 document migration; current dirtiness is the P58 task directory itself.
- External effects: one read-only remote fetch occurred before the worktree was created. No main mutation, merge, commit, push, image, model download, TPU, Kubernetes resource, credential, or other external state was changed.
- Next: implement P58.1 only, stop at its first failed gate, and leave P58.2 pending until the numerical contract passes.

## 2026-08-21 UTC — P58.1/P58.2 implementation and exact-image gates passed

- Type: implementation/validation
- Action: implemented the additive P58 Qwen3-4B B8 x G16 DP8 x TP8 per-role contract, paired renderer/profile, explicit fixed-16K effective-row loss, denominator-weighted stock-trainer accumulation, canonical global-denominator path, full trajectory journal, W&B signal counts/ratios, native/zero alignment policies, transaction receipts, and arm-aware classifier. Compact-filtered rows retain raw advantages for audit but are excluded from the effective/nonzero-policy-signal metrics. Copied the reviewed 1,012-task clean JSONL byte-for-byte into `canon-zero-tim/clean_data/p46_q4_learnable/` and verified its frozen digest.
- Correction found during integration: the inherited P34 `full` rule incorrectly demanded old large-tensor trajectory capture for P58. P58 has a separate full-trajectory journal, so it is now excluded from that P34-only capture condition. Native/zero x canary/full environment resolution passes.
- Correction found during artifact testing: all-filtered batches do not increment optimizer step, so using optimizer step as the trajectory filename would collide on the next batch. P58 now persists monotonically increasing `batch_index` separately from `optimizer_step`, validates continuity and digests on resume, and refuses partial journals.
- Correction found during paired-path review: the stock native trainer lacked a durable update report, while the zero path already had an explicit segmented transaction report. P58 now records the native stock JAX-sharded transaction without claiming fixed-tree DP reduction. Zero retains explicit DP8 reduction evidence. The classifier understands the two truthful receipt types.
- Correction found during no-signal review: the canonical segmented zero arm would commit a zero gradient when all 128 rows were compact-filtered. It now discards the complete streamed accumulator without changing model, optimizer, or train step, matching the stock path and the preregistered no-commit rule.
- Validation: syntax and shell parsing passed; `git diff --check` passed; P58 loss 5/5, renderer 4/4, profile 2/2, alignment policy 2/2, environment 1/1, durable journal 2/2, classifier 2/2, full alignment 40/40, P34 contract 5/5, P34 environment 7/7, P34 renderer 13/13, P44 renderer 6/6, common loss 60/60, selected real trainer tests 3/3, and compact-filter trajectory test 1/1 passed in the pinned image.
- Terminal marker: `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1`.
- Claim: implementation plus CPU/pinned-image validation only. No model/R2E one-host run, Pathways run, 128-chip target, HBM measurement, native mismatch dose, zero exactness, convergence, image publication, commit, push, or launch exists.
- Next: P58.3 is active. Reconcile the unrelated moving operator tip, then request the appropriate separate approvals for publication and either one-host sanity or direct paired canaries.

## 2026-08-21 UTC — legacy full static wrapper device probe was inconclusive

- Type: validation limitation
- Action: an expanded early gate invoked the complete historical P34 static wrapper. Its first nine suites passed; the final device-probe subprocess reached its own 120-second timeout on this non-TPU host.
- Result: `INCONCLUSIVE`, not FAIL and not TPU PASS. The final P58 exact-image gate directly runs the relevant P34 contract, environment, and renderer regression suites and records their passing counts. The absent TPU probe is retained as a blocker for target claims.

## 2026-08-21 UTC — execution order changed to native-first

- Type: user decision/handoff
- Decision: waive the optional P58.3 one-host sanity without claiming PASS, publish the shared implementation, and activate only the 128-chip native three-update canary. The zero arm remains implemented and covered by CPU regression tests but is explicitly deferred because its optimization work is incomplete.
- Scope: the remote executor may render and launch `arm=native, stage=three-update` from the exact post-push readback SHA and a digest-pinned image. It must not render or apply zero under this decision. A native PASS is an integration/training result only; it cannot establish the paired treatment effect or zero-TIM.
- Gate: exactly three native optimizer commits, complete durable trajectories, finite nonzero A-B dose, exact B-C, TPU-resident optimizer, valid cleanup/checkpoint transactions, and native classifier `PASS`.
- Publication: the user explicitly approved commit and push to `yuxzhang/canon-zero-tim`. `main` remains untouched. The final remote SHA must be obtained and reported by readback after push rather than embedded self-referentially in this commit.
- Next: publish after reconciling the unrelated P57 tip and rerunning focused plus pinned-image gates; the remote executor then follows `cluster/P58_DEEPSWE_TIM_RUNBOOK.md` section 3N.

## 2026-08-21 UTC — P58 implementation published

- Type: publication evidence
- Action: committed the complete P58 native-first concern, rebased it without conflict over operator commits `39e77bdd` and `874ef342`, reran the focused gates and pinned exact-image suite, and performed a normal non-force fast-forward push to `yuxzhang/canon-zero-tim`.
- Published implementation commit: `c5bdc9d993dfaf1a6956335609fbf259f9ed95f7`.
- Validation after rebase: renderer 4/4, profile 2/2, environment 1/1, clean diff, and terminal marker `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1`.
- Readback: local HEAD and the remote-tracking operator branch both resolved to `c5bdc9d993dfaf1a6956335609fbf259f9ed95f7`; ahead/behind was `0/0`; the worktree was clean.
- External effects: one implementation commit and one fast-forward operator-branch push. `main` was untouched. No image, model, credential, YAML render, Kubernetes object, TPU job, or run artifact was created.
- Next: this documentation-only publication checkpoint will advance the branch once more. The executor must fetch and use the final post-checkpoint remote SHA, then follow section 3N for native only.

## 2026-08-21 UTC — p58c01 bootstrap failure diagnosed and fixed locally

- Type: target failure/implementation/validation
- Evidence: `evidence/p58c01/run.log`, SHA-256 `f551712696c9c36dbf4f1f2fb713a4c975ff49f2184cf62e887341679341d0bc`. JobSet attempt was explicitly `0`.
- First failing boundary: `00_env.sh`. The native profile intentionally resolved `CANON_P32_DP_REDUCTION_ADMITTED=0` for the stock JAX-sharded trainer, while the inherited P34 admission loop demanded `1`. The same native stock loop required `CANON_FROZENLAKE_L3=0`, `CANON_FROZENLAKE_P27=0`, and `CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY=0`, but the DeepSWE profile left them unset.
- Classification: bootstrap `INCONCLUSIVE`. The coordinator stayed on the CPU preflight and exited before repository sync/install, Pathways device probing, model initialization, rollout, trajectory journaling, forward, backward, optimizer, or checkpoint work. It provides no TPU or training evidence.
- Fix: keep native reduction admission truthfully at `0`; make only the inherited reduction expectation arm-aware; and export the three unrelated FrozenLake zeros in the P58 profile. Do not set native reduction admission to `1`, because that would falsely claim the zero arm's fixed-tree reducer.
- Regression: added a renderer-to-real-`00_env.sh` test that executes the exact native three-update shell path, requires `P34 contract OK: DP8xTP8`, and verifies the resolved reduction/FrozenLake values. Profile 2/2, environment 2/2, shell syntax, and `git diff --check` pass.
- Pinned-image result: `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1`.
- Publication/rollback: fix is uncommitted and unpushed; `main` is untouched. Reverting the four local source/test changes removes the fix, but no rollback was executed.
- Next: request commit/push approval, then use a new immutable native run-id `p58c02`; never reuse p58c01 and never launch zero under the current decision.

## 2026-08-21 UTC — p58c01 bootstrap fix published

- Type: publication evidence
- Action: committed the admission/profile fix, real `00_env.sh` regression, p58c01 classification, and p58c02 handoff as one concern; the operator tip had not moved, so no rebase was required; performed a normal non-force fast-forward push.
- Fix implementation commit: `acd3136267214b367a6755d0ba28d80e883d6753`.
- Gates on the published tree: `git diff --check`, shell syntax, profile 2/2, environment 2/2, and the previously recorded pinned-image terminal marker all pass.
- Readback: local HEAD and `origin/yuxzhang/canon-zero-tim` both resolved to `acd3136267214b367a6755d0ba28d80e883d6753`; ahead/behind was `0/0`; the worktree was clean.
- External effects: one fix commit and one fast-forward operator-branch push. `main` was untouched. No image, model, secret, YAML render, Kubernetes object, TPU program, or p58c02 run was created.
- Next: publish this documentation-only checkpoint, then the remote executor fetches the final readback SHA and renders only fresh native p58c02.

## 2026-08-21 UTC — p58c02 direct-entrypoint failure diagnosed and fixed locally

- Type: target failure/implementation/validation
- Evidence: `evidence/p58c02/run.log`, SHA-256 `8983ab0a61355a32c9992e09f33f3e42d3bf673463cf0ca500e54b749fba56de`.
- First failing boundary: the canonical wrapper initialized Pathways, then `runpy.run_module("examples.deepswe.train_deepswe_nb")` raised `ModuleNotFoundError: No module named 'examples'`. The signed JobSet invokes the wrapper as `/app/examples/deepswe/canonical_entrypoint.py`; file execution places only its containing directory on `sys.path`, not repository root `/app`.
- Classification: bootstrap `INCONCLUSIVE`. No model initialization, rollout, trajectory, forward, backward, optimizer transaction, checkpoint, or 128-chip training evidence exists.
- Fix: derive repository root from `canonical_entrypoint.py`'s own resolved path and prepend it before the package-qualified import. Keep the renderer and every training hyperparameter unchanged. Change the native stock preflight from the easier module launch to the exact direct-file entrypoint so this failure blocks before the expensive run boundary.
- Regression: the entrypoint isolated-subprocess contract passes 9/9; Python/Bash syntax, `git diff --check`, native environment 2/2, P58 renderer 4/4, P34 renderer 13/13, and P58 profile 2/2 pass. From `/tmp` with a cleared external `PYTHONPATH`, the exact direct-file command reaches the trainer on the bare host (then stops only because that host lacks `datasets`) and exits zero with full DeepSWE CLI help in the pinned image. The complete pinned-image terminal marker is `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1`.
- One-host inventory: Qwen3-4B weights are present, but direct TPU initialization fails because `libtpu.so` is absent. A real one-host v5p test was therefore not run and is not claimed.
- External effects: fetched/fast-forwarded the requested operator branch and ran local/container read-only validation. No commit, push, main mutation, image publication, model download, Kubernetes resource, TPU job, or credential change occurred.
- Next: after explicit commit/push approval, publish and read back the fix, then render only native three-update run `p58c03`. P58c01 and p58c02 remain immutable and must not be resumed; zero remains deferred.

## 2026-08-21 UTC — p58c02 direct-entrypoint fix published

- Type: publication evidence
- Action: committed the direct-file import bootstrap, exact native preflight, isolated subprocess regression, pinned-image gate inclusion, p58c02 classification, and p58c03 handoff as one concern; the operator tip had not moved; performed a normal non-force fast-forward push.
- Published fix commit: `82d82f72a7220d945737d95f6266b5b7e2cfe706`.
- Readback: local HEAD and `origin/yuxzhang/canon-zero-tim` both resolved to the published commit with ahead/behind `0/0`; the worktree was clean before this publication-only checkpoint.
- External effects: one fix commit and one fast-forward operator-branch push. `main` was untouched. No image publication, model download, Kubernetes object, TPU job, credential change, or p58c03 run occurred.
- Next: publish this documentation checkpoint, fetch its final readback SHA, and hand only native run-id `p58c03` to the remote executor. Zero remains deferred.

## 2026-08-21 UTC — p58c03 parent-environment leak diagnosed and fixed locally

- Type: target failure/implementation/validation
- Source intake: fast-forwarded the isolated P58 worktree from `ae5e00ad5742b300d2391e004d4b908374fa1135` to operator tip `10ccdb3012e7a6bd3f0c9ae6bdf29d717cf84440`. The new tip added only the immutable p58c03 evidence. `main` was not touched.
- Evidence: `evidence/p58c03/run.log`, SHA-256 `15aa9968200c55a02ef47c72c5e209277397835e1752a4dbd9699fce3b2c42b4`; `evidence/p58c03/head_container.log`, SHA-256 `d5e8b5b1941aa5632fa6267cfdac445727c175bf8d2bbcc79c1ece7cf7aba1e2`. JobSet attempt was explicitly `0`.
- First failing boundary: after environment validation, exact source sync, pinned R2E install/adapter validation, native stock-engine preflight, Pathways initialization, exact direct entrypoint, device discovery, and bounded runtime patching, `deepswe_contract.validate_environment` rejected `{'CANON_LOGPROB_M': '256'}` before model initialization. The later W&B attestation fatal is derivative of that Python exit.
- Root cause: `00_env.sh` is a child process. The native profile correctly unset `CANON_LOGPROB_M` there, but its generated export-only `env.sh` could only overlay the parent entrypoint's raw renderer environment; it could not delete the stale value. The contract was correct and was not loosened.
- Fix: make generated `env.sh` an authoritative snapshot. When sourced, it first clears all non-secret namespaces managed by `00_env.sh`, then exports exactly the resolved set. `HF_TOKEN`, `WANDB_API_KEY`, and injected secret variables are neither serialized nor cleared.
- Regression: extend the renderer-to-real-`00_env.sh` test through the actual parent reload boundary. It seeds raw native `CANON_LOGPROB_M=256`, sources the generated snapshot, requires both `CANON_LOGPROB_M` and `CANON_FIXED_AR` absent, and calls the Python environment contract. Native and zero contract tests pass.
- Validation: Bash syntax and `git diff --check` pass; P58 profile 2/2, renderer 4/4, environment 3/3, P34 environment 7/7, contract 5/5, renderer 13/13, and P57 adjacent 81/81 pass. The complete pinned-image gate exits zero with `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1`.
- Classification: p58c03 is bootstrap `INCONCLUSIVE`. No model initialization, rollout, trajectory, forward, backward, optimizer transaction, checkpoint, or 128-chip training evidence exists; there is no resumable state.
- External effects: one requested fast-forward pull and local/container validation only. No commit, push, main mutation, image publication, model download, Kubernetes object, TPU job, credential change, or p58c04 render/launch occurred.
- Next: obtain explicit commit/push approval, publish and read back the fix, then use only fresh native run-id `p58c04`. Never reuse p58c01/p58c02/p58c03; zero remains deferred.

## 2026-08-21 UTC — p58c03 environment-snapshot fix published

- Type: publication evidence
- Action: committed the authoritative managed-environment snapshot, symmetric native/zero parent-reload regression, p58c03 immutable classification, and p58c04 handoff as one concern; the operator tip had not moved; performed a normal non-force fast-forward push.
- Published implementation commit: `c0ca41805bd65a4fdede4825ed2835cdce6e13ed`.
- Gates on the published tree: `git diff --check`, Bash syntax, P58 environment 3/3, focused P58/P34 regressions, P57 adjacent 81/81, and terminal marker `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1`.
- Readback: local HEAD and `origin/yuxzhang/canon-zero-tim` both resolved to the published implementation commit with ahead/behind `0/0`; the worktree was clean before this publication-only checkpoint.
- External effects: one implementation commit and one fast-forward operator-branch push. `main` was untouched. No image publication, model download, YAML render, Kubernetes object, TPU job, credential change, or p58c04 run occurred.
- Next: publish this documentation-only checkpoint, fetch its final readback SHA, and hand only native run-id `p58c04` to the remote executor. Zero remains deferred.

## 2026-08-21 UTC — p58c04 sandbox-start failure diagnosed and fixed locally

- Type: target failure/implementation/validation
- Source intake: fast-forwarded the isolated P58 worktree from `d2f57e0bf9ec50a4c70c2f4c404db870dbb6ff7a` through the p58c04 evidence checkpoint to final observed operator tip `8acfe784b6fa8eacb8eb4e41406dd6681173f9c7`. The P57 logs/implementation in the intervening commits were explicitly out of scope; no P57 source or documentation was changed by this work.
- Evidence: `evidence/p58c04/run.log`, SHA-256 `f5caf2efb70bfec083a4454e441ce7f4b5b0632abbd206439ba9497bca5a6a40`; `evidence/p58c04/env.sh`, SHA-256 `a311eb64ee30b1fa0a168b68d9f17661756ed9cb3b272dd19d9bdddbc7f34666`. The signed source was `d2f57e0bf9ec50a4c70c2f4c404db870dbb6ff7a`.
- Reached boundary: p58c04 passed environment validation, exact source sync, pinned R2E install/adapter checks, stock-engine preflight, Pathways/128-device discovery, Qwen3-4B/vLLM initialization, W&B initialization, and entered `run_producers_from_stream` with concurrency 128.
- First failure: 128 RepoEnv creations were attempted, with no log evidence of a sandbox reaching Running before the 1,200-second start deadline and at least 121 readable start-timeout records in the interleaved output. The pinned R2E `start_container` caught the start exception, printed it, deleted the pod, and returned. Construction continued with `container=None`; later setup exec targeted a deleted pod and received Kubernetes 404. The Kubernetes client's subsequent `body.decode` on `None` produced the misleading terminal AttributeError. Websocket content parsing was not the root cause and was not relaxed.
- Classification: `INCONCLUSIVE`. No environment reset completed and there is no model-generated trajectory, forward, backward, optimizer transaction, checkpoint, or resumable journal state.
- Runtime fix: the Kubernetes-only wrapper invokes the bounded start directly, confirms deletion on failure, and re-raises the original exception. It refuses any return with `container=None`; Docker continues through the untouched upstream method. The existing collector maps a start `TimeoutError` raised during reset to signed `ENV_TIMEOUT` and always closes the environment. A bounded marker reports only pod name, phase, and scheduler condition/reason/message, never pod spec/environment, so a repeated Pending failure is actionable.
- Load mitigation: P58 sandbox orchestration concurrency is 64, matching the P34/reference recipe. B8 x G16 remains 128 trajectories, now in two waves. Data, seeds, sampling, loss, role meshes, trainer microbatch/accumulation, optimizer placement, and three-update horizon are unchanged.
- Adjacent compatibility: the newly shared stock-contract checks require `CANON_P28_BATCHED_REVERSE=0` and `CANON_BATCHED_EVIDENCE=0`; the P58 native profile now declares those zeros. This is not a P57 change and not an algorithm treatment.
- Regression: host R2E optional contract 4/4 with two explicit dependency skips; P58 renderer 4/4; P58 environment 3/3; Python syntax and `git diff --check` pass. The pinned image additionally runs the exact start-timeout/cleanup and raised-reset-timeout controls, and exits zero with `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1`.
- External effects: one requested fast-forward pull and local/pinned-container validation only. No commit, push, main mutation, image publication, model download, rendered YAML, Kubernetes object, TPU job, or credential change occurred.
- Next: obtain explicit commit/push approval, publish and read back the fix, then render only fresh native run-id `p58c05`. If p58c05 again has zero confirmed Running sandboxes, collect scheduler/node events and treat CPU-pool capacity as the next boundary; do not patch websocket decode. Zero remains deferred.

## 2026-08-21 UTC — bounded timeout telemetry added locally

- Type: implementation/validation
- Action: added low-cardinality timeout provenance to the trajectory record and P58 durable journal, splitting sandbox start, environment reset/step, model generation, final reward, and trajectory-deadline stages. Scheduler metadata is restricted to fixed `unschedulable` and resource categories; full Kubernetes messages remain only in the bounded raw log marker. P58 W&B now receives per-status/count ratios, sandbox-start and environment-timeout ratios, CPU/memory admission counts, and all-timeout batch flags.
- Interpretation: `deepswe/all_sandbox_start_timeout_batch=1` proves effective R2E environment throughput was zero and the model was not the first bottleneck. Only a zero sandbox-start ratio combined with `deepswe/status/model_timeout_ratio>0` implicates model-serving throughput.
- Regression: syntax, host P58 environment 3/3, renderer 4/4, optional R2E contract 4/4 with two dependency skips, timeout artifact controls, and the complete pinned-image suite pass. Terminal marker: `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1`.
- External effects: local source/tests/documentation only. No commit, push, main mutation, image publication, model download, rendered YAML, Kubernetes object, TPU job, or credential change occurred.
- Next: publish only after explicit user approval, then use fresh native run-id `p58c05` and read the sandbox-start metrics before changing any training or serving hyperparameter.

## 2026-08-21 UTC — p58c04 sandbox repair published

- Type: publication evidence
- Action: committed the fail-closed Kubernetes sandbox start path, 64-concurrency P58 orchestration, bounded timeout trajectory/W&B telemetry, adjacent native-profile zeros, exact regression controls, and p58c05 handoff. The first normal push was safely rejected because the operator branch advanced after pre-push fetch. Fetched the new tip, confirmed it added only P57 attempt evidence, rebased without conflict, and reran focused plus complete pinned-image gates before a normal non-force push.
- Published implementation commit: `174fcf3a42af3e9cd465307843a1c19a08098c99`.
- Validation after rebase: renderer 4/4, environment 3/3, optional R2E contract 4/4 with two dependency skips, syntax/diff checks, and terminal marker `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1`.
- Readback: local HEAD and the remote-tracking operator branch both resolved to the published implementation commit with ahead/behind `0/0` before this documentation checkpoint.
- External effects: one P58 implementation commit and one normal fast-forward push to `yuxzhang/canon-zero-tim`. `main` was untouched. No image publication, model download, rendered YAML, Kubernetes object, TPU job, credential change, or p58c05 run occurred.
- Next: publish this documentation-only checkpoint, fetch its final readback SHA, and hand only native run-id `p58c05` to the remote executor. Zero remains deferred.

## 2026-08-21 UTC — p58c05 Kueue admission diagnosed and direct-full phase activated locally

- Type: evidence analysis/implementation/phase transition
- Pulled source: `a6a9ca2a05cd1a0ec02ccc7171841d20033b0240`, which adds immutable `evidence/p58c05_admission/` artifacts.
- First failure: the Workload remained `QuotaReserved=False`; Kueue reported `couldn't assign flavors to pod set pathways-worker: flavor 0xv5p-8 doesn't match node affinity, flavor cpu-user doesn't match node affinity`. The worker requested 128 TPU devices and exact `4x4x8` topology but also had literal node-pool selector `tpu-v5p-slice`.
- Root cause: P58 inherited P34 rendering that treats every worker-nodepool string as a concrete selector. In this launch, `tpu-v5p-slice` was a Kueue-managed sentinel; making it literal contradicted the `0xv5p-8` ResourceFlavor. No JobSet pod or training process started, so this is admission `INCONCLUSIVE`, not a runtime or throughput failure, and p58c05 has no resumable state.
- Fix: for registered sentinels `auto`, `none`, `tpu-v5p-slice`, and `any`, the P58 renderer omits only literal `cloud.google.com/gke-nodepool` and lets Kueue inject concrete pool affinity. It retains the TPU accelerator and exact `4x4x8` topology. Explicit real node-pool names remain exact. Renderer regressions cover both behaviors.
- Phase decision: by user instruction, P58.4N is superseded without PASS. P58.5N is active: fresh native run `p58f01`, `stage=full`, exactly 1,000 commits. Updates 1–3 are mandatory online monitoring milestones and do not stop a healthy job. Zero remains deferred.
- Validation: focused renderer 6/6, environment 3/3, optional R2E 4/4 with two dependency skips, Python/Bash syntax, `git diff --check`, and the complete pinned-image gate pass. Terminal marker: `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1`. A real full-stage CLI render produced 32 four-chip workers (128 TPU), exact `4x4x8`, and no literal Kueue-sentinel nodepool.
- External effects: one requested fast-forward pull and local source/test/documentation edits only. No commit, push, main mutation, image publication, rendered launch YAML, Kubernetes apply, model download, TPU job, or credential change occurred.
- Reconciliation: while validation and publication preparation ran, the operator branch advanced through two non-overlapping P57-only commits. The P58 edits were preserved, the worktree fast-forwarded twice without conflict, and publication validation uses final base `7e608682ea21c501b8ed737b58ffe5591125d6eb`.
- Next: rerun focused checks on the final tip, then await separate commit/push approval. After publication/readback, the remote executor follows the active P58.5N runbook with fresh `p58f01`.

## 2026-08-21 UTC — P58 Kueue admission repair and native full phase published

- Type: publication evidence
- Action: after explicit user approval, committed the Kueue-managed worker-affinity repair, sentinel/explicit-pool regressions, p58c05 evidence interpretation, and P58.5N direct-full runbook/handoff. The branch had advanced through a non-overlapping P57 full-horizon commit; P58 was fast-forwarded and restored without conflict before final validation.
- Published implementation commit: `abbc76008e0a7fcb63562c27d5cf4608fb4f4e90`.
- Final-base validation: P58 focused 13/13 with two dependency skips; current P57 adjacency 17/17; Python/Bash syntax and `git diff --check`; real full-stage CLI rendering with 32 four-chip workers, 128 TPU, exact `4x4x8`, and absent literal Kueue sentinel nodepool; complete pinned-image terminal marker `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1`.
- Readback: local HEAD and `origin/yuxzhang/canon-zero-tim` both resolved to `abbc76008e0a7fcb63562c27d5cf4608fb4f4e90` with ahead/behind `0/0` before this documentation checkpoint.
- External effects: one normal fast-forward push to `yuxzhang/canon-zero-tim`. `main` was untouched. No image publication, model download, Kubernetes apply, TPU job, credential change, or `p58f01` run occurred.
- Next: publish this documentation checkpoint, fetch its final readback SHA, and hand fresh native full run-id `p58f01` to the remote executor. Zero remains deferred.

## 2026-08-21 UTC — p58f01 sandbox LocalQueue and reset-provenance failures repaired locally

- Type: source intake/evidence analysis/implementation/validation.
- Source intake: fast-forwarded the isolated P58 worktree to operator tip `606b37cf4984a22bcb46391c18834a1006bfb98b`. The new P58 artifact is immutable `evidence/p58f01/run.log`, SHA-256 `16c513c773ac2bfb1542178b4e42b03098bb9114564106b03f83c0195a0d542f`, 1,387 lines and 231,681 bytes. The target run used source `6f18d95b22835fc70326d21bb70c1fb41f7b0e12`. `main` was not touched.
- Reached boundary: exact environment/bootstrap/R2E preflight passed; Pathways reported 128 devices across 32 four-device hosts and the exact 64-device rollout plus 64-device trainer split; Qwen3-4B/vLLM, W&B, checkpoint management, and `run_producers_from_stream` concurrency 64 initialized.
- First failure: every sandbox reset timed out. The log contains 128 `ENV_RESET_TIMEOUT` rows and at least 127 bounded Pod markers with `PodScheduled=False`, reason `SchedulingGated`, message `Scheduling is blocked due to non-empty scheduling gates`. Runtime-created standalone R2E Pods did not inherit the parent JobSet's `kueue.x-k8s.io/queue-name`, so this cluster's Kueue integration gated them without a LocalQueue. This is sandbox admission, not model-serving throughput.
- Secondary failure: the 128-row all-timeout batch finished in 2,413.4 seconds, then strict GRPO processing raised `ValueError: policy_version is missing from trajectory task.` Reset had failed before the first model call, which was the old assignment point. The exception occurred before P58 batch persistence, so p58f01 has no resumable journal or checkpoint.
- Repair: derive `R2E_K8S_QUEUE_NAME` from the parent JobSet queue label, reject absent/invalid values, preserve it through the authoritative `00_env.sh` snapshot, and add it unchanged to every R2E Pod. Seed `env.task["policy_version"]` at environment construction before reset while retaining the strict downstream missing check. Classify `SchedulingGated` separately from `Unschedulable` in bounded trajectory, journal, and W&B metrics.
- Regression: renderer requires exact parent/sandbox queue parity and rejects missing/invalid queues; runtime fake proves the label reaches the Pod body and invalid values fail before create; environment regression proves `R2E_*` survives authoritative reload; learner regression proves policy provenance exists before reset; trajectory/artifact controls prove `scheduling_gated` stays bounded and is exported separately. Host renderer 7/7, environment 3/3, optional R2E 4/4 with two explicit dependency skips, P34 contract/environment/renderer 25/25, and P57 adjacency 91/91 pass. Host artifact import is unavailable because this shell lacks `metrax`; the complete pinned-image gate passes it plus the learner/collector controls with terminal marker `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1`. A full p58f02 CLI render has 32 four-chip workers, exact `4x4x8`, no literal worker nodepool, parent/sandbox queue parity at `multislice-queue`, and `max_steps=1000`.
- Classification: p58f01 is `INCONCLUSIVE`, immutable, and must not be resumed or reused. P58.5N remains active. After publication/readback, the next fresh native full run-id is `p58f02`; zero remains deferred.
- External effects: one requested fast-forward pull and local source/test/documentation edits only. No commit, push, main mutation, image publication, rendered launch YAML, Kubernetes apply, TPU job, model download, or credential change occurred.

## 2026-08-21 UTC — p58f01 repair published

- Type: publication evidence.
- Action: after explicit user approval, committed the sandbox LocalQueue inheritance, authoritative `R2E_*` snapshot, reset-time policy provenance, bounded `scheduling_gated` telemetry, exact regressions, and p58f02 runbook/handoff as one concern. The pre-publication fetch proved the operator branch had not advanced; performed a normal non-force fast-forward push.
- Published implementation commit: `c67e9d5bfa3f1b3b592a2440075eb165e073e6ac`.
- Validation on the published tree: `git diff --check`, Python/Bash syntax, P58 renderer 7/7, environment 3/3, optional R2E 4/4 with two dependency skips, P34 focused 25/25, P57 adjacency 91/91, full p58f02 static render, and terminal marker `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1`.
- Readback: local HEAD, `FETCH_HEAD`, and `origin/yuxzhang/canon-zero-tim` all resolved to the implementation commit with ahead/behind `0/0` before this documentation checkpoint.
- External effects: one normal fast-forward push to `yuxzhang/canon-zero-tim`. `main` was untouched. No image publication, model download, rendered launch YAML, Kubernetes apply, TPU job, or credential change occurred.
- Next: publish this documentation-only checkpoint, fetch its final readback SHA, then hand only fresh native full run-id `p58f02` to the remote executor. Zero remains deferred.

## 2026-08-21 UTC — p58f02/p58f03 intake and native weight-gate diagnosis

- Type: source intake/evidence analysis/reconciliation.
- Source intake: fast-forwarded the isolated P58 worktree to operator tip `5dd865294560899b0438228f458a84acbe61cdb4`. P58f02 raw log `evidence/p58f02/run.log` has SHA-256 `99ce3b378254d95860c20b10b5d76695f171aac4b0d15af29f5aba9bc0d0bff6`, 1,324 lines, and 225,993 bytes. P58f03 raw log `evidence/p58f03/run.log` has SHA-256 `fdb958d5e1db8bafa25b6df8c3223a3c6a642d00c6a1915bb34a8e17b5bcf600`, 7,087 lines, and 631,570 bytes.
- P58f02: the sandboxes remained `SchedulingGated` because the cluster's `cpu-user` flavor requires `nodeSelector: cpu-np`, not `deepswe-cpu-pool`. The user's CPU-node change was the correct fix and was published in `7208d7b330759ac7dc31493ece65d32a6c355308`. A previously drafted generic CPU/original-input fallback is not needed; it was removed from the working tree and retained only as recoverable `stash@{0}`.
- P58f03 reached boundary: source `7208d7b330759ac7dc31493ece65d32a6c355308` passed P34 CLI, exact 128-device/32-host Pathways inventory, and the 64-rollout/64-trainer DP8 x TP8 split. The first real rollout batch completed in 616.3 seconds. It durably wrote 128 trajectories: 126 `SUCCEEDED`, two `MAX_CONTEXT_LIMIT_REACHED`, three solved, two mixed/effective groups, and 32 nonzero advantages. Sandbox-start timeouts were zero.
- Trajectory artifact: `/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-native-full-p58f03/debug/batch-000000.trajectories.jsonl.gz`, SHA-256 `26c92d2153865cc14296303fcb97afd98f857744e50574032b6eba8631f23a9e`.
- First failure: after journaling and before trainer forward/backward/update, the shared P34 gate called `attest_canonical_engine_weights`. Native correctly had no registered canonical adapter (`CANON_ENGINE_MODULE_C=0`), so it raised `canonical weight attestation requires the registered engine adapter`; the subsequent `AlignmentGateError: P34 requires exact rollout/trainer weights before A/B/C` was derivative. This is a gate-routing defect, not a rollout, CPU-throughput, model-timeout, or observed weight-mismatch result.
- Classification: p58f02 and p58f03 are immutable `INCONCLUSIVE`. P58f03's trajectory journal is valid diagnostic evidence, but there is no trainer forward, backward, optimizer commit, or checkpoint. It is not resumable training state.

## 2026-08-21 UTC — arm-aware exact live-weight attestation repaired locally

- Type: implementation/validation/documentation.
- Repair: added a shared observer-only `attest_exact_live_engine_weights` implementation using the existing pure trainer-to-engine mapping and bitwise leaf comparison. The generic cluster gate now invokes an arm-aware rollout interface. Zero still delegates to its registered canonical adapter. Only the signed P58 native route may use the observer with no adapter; any unsigned route, wrong workload flags, native adapter leakage, missing/mismatched leaves, or invalid mesh fails closed.
- Provenance: the observer normalizes internal vLLM mesh axes `data/model` to public contract axes `dp/tp` after validating the exact active-workload DP8 x TP8 shape and singleton remainder. It does not register a canonical adapter, replace serving/forward functions, alter token selection, or change trainer/optimizer math.
- Regression: Python compilation passes; four native/zero/negative rollout routing tests pass; two exact observer/mesh tests pass; the full rollout canonical module passes 15/15; and the complete pinned P58 exact-image gate exits zero with `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1`. A separate broad legacy adapter test invocation had three unrelated environment/device setup errors (missing active-workload environment in two cases and only one available device where four were required), so that invocation is not represented as a whole-suite PASS.
- External effects: one requested fast-forward pull, one local stash preserving the superseded fallback, local source/tests/documentation edits, and local/pinned-container tests only. No commit, push, `main` mutation, image publication, rendered launch YAML, Kubernetes apply, TPU job, model download, or credential change occurred.
- Next: after explicit commit/push approval, publish and read back the repair, then use fresh native full run-id `p58f04`. Require `[P34.WEIGHTS] EXACT` before A/B/C and continue the same full 1,000-commit job; do not render zero.

## 2026-08-21 UTC — p58f03 native weight-gate repair published

- Type: publication evidence.
- Action: after explicit user approval, committed the arm-aware exact-live-weight interface, signed native observer route, canonical/negative regressions, exact-image coverage, and p58f04 runbook/handoff. The pre-publication fetch proved the operator branch had not advanced; the push was normal and non-force.
- Published implementation commit: `234eaddb8e3543083927aa10effe101abef18a91`.
- Validation on the published tree: Python compilation and `git diff --check` pass; native/zero/unsigned/leaked-adapter routes pass; the full rollout canonical module passes 15/15; and the complete pinned-image P58 gate exits zero with `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1`.
- Readback: local HEAD, `FETCH_HEAD`, and `origin/yuxzhang/canon-zero-tim` all resolved to the implementation commit with ahead/behind `0/0` before this documentation checkpoint.
- External effects: one implementation commit and one normal fast-forward push to `yuxzhang/canon-zero-tim`. `main` was untouched. No image publication, Kubernetes apply, TPU job, model download, or credential change occurred.
- Next: publish this documentation checkpoint, fetch its final remote SHA, and hand only fresh native full run-id `p58f04` to the executor. Require `[P34.WEIGHTS] EXACT` before A/B/C; zero remains deferred.

## 2026-08-22 UTC — p58f04 processed-S_prefill failure repaired locally with isolated native observer

- Type: source intake/evidence analysis/implementation/validation/documentation.
- Source intake: after a clean P58 preflight, fast-forwarded the isolated worktree from `18c4ac78` to operator tip `609c8e6d6d2cb9e7ebd0ea8fa0d7a4fe0b877f68`. The only incoming file was immutable `evidence/p58f04/run.log`, 32 lines and 4,468 bytes, SHA-256 `a7b0cda5e7d359c7e320b29f8af197db0dd6c46dc34850aa55ffb350fb766fdd`. `main` was untouched.
- Reached boundary: the first rollout batch completed in 557.2 seconds and durably wrote 128 trajectories: 125 `SUCCEEDED`, three `MAX_CONTEXT_LIMIT_REACHED`, six solved, five all-failed groups, one mixed/effective group, two incomplete groups, and 16 nonzero advantages. Sandbox-start timeout count was zero. The journal is `/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-native-full-p58f04/debug/batch-000000.trajectories.jsonl.gz`, SHA-256 `e39caf5df63ba54406a36427a413dea562e5771f4c52b30c840229d3178c1f3b`.
- Previous repair result: exact live-weight attestation passed for 398 leaves, 4,022,468,096 elements, and the 64-device DP8 x TP8 rollout role. P58f04 therefore closes the p58f03 weight-routing defect.
- First failure: before trainer forward/backward/update, RLCluster requested processed `S_prefill`. Native correctly had `CANON_PROMPT_PROCESSED_LOGPROBS=0`; `VllmRollout` rejected labeling the stock raw prompt-logprob helper as processed. The stock helper is not an acceptable fallback because its packed-buffer roll can choose targets across request/padding/DP boundaries. Enabling the canonical processed engine would destroy the native-vs-zero treatment separation.
- Repair: registered `CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER` as experimental/default-off. P58 native alone resolves it to one while keeping canonical prompt processing, engine module C, fixed AR, logprob M, VJP2, Pallas, precision, and segmented-training switches disabled/absent. The installer verifies all stock hashes first, then applies an exact two-file observer overlay. The helper applies decode-equivalent temperature/top-k/top-p transforms and absolute request-history targets only for post-rollout B. P58 zero explicitly resolves the observer to zero and retains the full canonical bundle. Shell, Python, rollout, installer, and postflight contracts reject mixed or unsigned tuples.
- Validation: Bash/Python compilation and `git diff --check`; P58 profile 2/2, stock-observer static 6/6, environment 4/4; P57 adjacency 91/91; P34 static 10 suites; pinned-image patch/install manifest; three observer target/value probes; and the complete pinned P58 image gate pass. Terminal marker: `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 stock_observer=1 regressions=1`. Host direct Tunix imports lack `metrax`; the pinned-image environment ran those tests successfully.
- Classification: p58f04 is immutable `INCONCLUSIVE`. It has a valid diagnostic trajectory journal but no trainer forward, backward, optimizer commit, or checkpoint, so it is not resumable training state.
- External effects: one user-requested fast-forward pull plus local source/test/documentation edits and local/pinned-container tests. No commit, push, `main` mutation, image publication, rendered launch YAML, Kubernetes apply, TPU job, model download, or credential change occurred.
- Next: after explicit commit/push approval, publish and read back the repair, then render only fresh native full run-id `p58f05`. Require stock preflight, exactly one native observer processed-B marker, exact weights, finite forward/backward, and the first optimizer commit before promoting the boundary. Continue to 1,000 commits if healthy. Do not render or launch zero.

## 2026-08-22 UTC — p58f05 full-stage alignment admission repaired locally

- Type: source intake/evidence analysis/implementation/validation.
- Source intake: fast-forwarded the isolated P58 worktree through immutable p58f05 evidence to operator tip `be66906b10da7deba144290644fc4ab543abb464`; the commit after p58f05 is P57-only. `main` was untouched.
- Evidence: `evidence/p58f05/run.log`, SHA-256 `73def19531ca1a9ef083a30d11ceb89696afcbe4125bd128f7ff0e7152ec06a6`. The 486.4-second batch durably wrote 128 trajectories: 126 `SUCCEEDED`, two `MAX_CONTEXT_LIMIT_REACHED`, six solved, two mixed/effective groups, and 32 nonzero advantages. All timeout dimensions were zero. Exact weights passed for 398 leaves/4,022,468,096 elements and the Native stock observer processed all 2,048 prompt rows.
- First failure: after the alignment sidecar attached and before trainer forward/backward/update, `gsm8k_ab_report_policy()` rejected the signed Native `full/1000` tuple. P58 arm semantics were already recognized, but its workload boolean had been included in an alternative branch whose stage set remained `one-update/three-update`. The existing test fixture exercised only `three-update`, so the real full-stage policy was never called.
- Repair: split production P34 full, P39/P43/P44 registered debug updates, and P58 Native training into explicit predicates. P58 warning admission requires `CANON_P58_TIM_ADMITTED=1`, `CANON_P58_TIM_ARM=native`, no competing DeepSWE mode, and exact `three-update/3` or `full/1000`. Zero remains warning-off and strict; Native still warns only for finite decode-vs-prefill A-B. No flag was added, removed, or repurposed, and all zero-TIM Native disables/absences remain unchanged.
- Regression: host-direct policy tests pass 5/5, including full positive, missing admission, wrong horizon, competing workload, and Zero warning negative controls. Renderer-to-profile/environment tests pass 5/5 and now call the policy using a real rendered Native full environment. Python compilation and `git diff --check` pass.
- One-host inventory: the default Python lacks `libtpu.so`; `/mnt/disks/tunix-data/venvs/train` contains JAX 0.9.2/libtpu and local Qwen3-4B-Instruct weights. A stale empty lock created by the first failed probe was removed after confirming no visible owner. The TPU runtime then loaded but could not obtain `CHIPS_PER_HOST_BOUNDS` from instance metadata and timed out after 55 seconds. Direct-attached v5p execution is therefore `BLOCKED_DIRECT_TPU_METADATA`, not PASS; topology was not emulated.
- External effects: one requested fast-forward pull, removal of the single self-created `/tmp/libtpu_lockfile`, local source/tests/documentation edits, and read-only/local validation. No commit, push, image publication, Kubernetes object, remote TPU job, model download, credential change, or `main` mutation occurred.
- Next: finish adjacent and pinned-image gates. If direct four-device TPU inventory becomes available, run the renderer-derived full-stage gate there; otherwise preserve the blocker. After explicit commit/push approval, publish/read back and use fresh native full run-id `p58f06`; p58f05 is immutable and not resumable training state.

## 2026-08-22 UTC — p58f05 repair validation complete

- Type: validation/handoff checkpoint.
- Host validation: P58 alignment policy 5/5, renderer-to-profile/environment policy 5/5, profile 2/2, renderer 7/7, and adjacent P34 warning policy 3/3 pass. P34 static emits `P34_STATIC_PASS suites=10`; current P57 adjacency passes 102/102 and emits `P57_FROZENLAKE_TIM_CPU_PASS`. Python compilation, Bash syntax, registry audit, and `git diff --check` pass.
- Exact-image validation: pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` exits zero after checking the one-host runner's shell contract and emits `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 stock_observer=1 regressions=1`.
- Real one-host attempt: `/mnt/disks/tunix-data/venvs/train` loaded JAX 0.9.2/libtpu on node `aaron-v5p-node6`, but this container has no `/dev/vfio` and reports zero chips. The bounded runner emitted `P58_ONEHOST_ALIGNMENT_BLOCKED reason=device_inventory_timeout timeout_secs=10` and exited 3. This is an environment blocker, not a code PASS or code failure; no topology or TPU result was emulated.
- Claim ceiling: the one-host runner covers only exact four-device inventory, a TPU matmul, and renderer/profile/alignment-policy admission. Even a future PASS would not prove Qwen/R2E rollout, trainer forward/backward, optimizer commit, Pathways, or DP8 x TP8 behavior.
- External effects: local tests and documentation only. No commit, push, image publication, model download, rendered YAML, Kubernetes object, remote TPU job, credential change, or `main` mutation occurred.
- Next: await explicit commit/push approval. After publication and readback, launch only fresh Native run-id `p58f06`; Zero remains strict, separately configured, and deferred.

## 2026-08-22 UTC — p58f05 alignment-admission repair published

- Type: publication evidence.
- Action: after explicit user approval, committed the signed Native `full/1000` admission repair, positive/opposite-arm/neighboring-workload controls, bounded one-host gate, exact-image coverage, flag guidance, and p58f06 handoff. The final pre-push fetch proved the operator branch had not advanced, and the push was normal and non-force.
- Published implementation commit: `5132d7ad0d3bc7c53de09e20bae835dca18a211a`.
- Readback: local HEAD, `FETCH_HEAD`, and `origin/yuxzhang/canon-zero-tim` all resolved to the implementation commit with ahead/behind `0/0` before this publication checkpoint.
- External effects: one normal fast-forward push to `yuxzhang/canon-zero-tim`. `main` was untouched. No image publication, model download, rendered YAML, Kubernetes apply, TPU job, or credential change occurred.
- Next: publish this documentation-only checkpoint, fetch its final readback SHA, then hand only fresh Native full run-id `p58f06` to the executor. Zero remains deferred.

## 2026-08-22 UTC — p58f06 step-0 rollout and stock observer passed, failed on S_prefill_vs_T_old boundary

- Type: target execution / evidence collection
- Evidence: `evidence/p58f06/run.log`. JobSet `canon-p58-ds4b-native-full-p58f06` ran across 128 TPU v5p chips.
- Result: Step 0 Rollout completed all 128 trajectories in 492.7 seconds with 3 solves and 0 timeouts. Exact live-weight attestation passed (`[P34.WEIGHTS] EXACT step=0 leaves=398 elements=4022468096 devices=64 PASS`). Stock prompt observer processed all 2,048 prompt logprob rows (`[P58.STOCK_OBSERVER] PROCESSED_PROMPT_LOGPROBS_PASS rows=2048 populated=2048`).
- Failure: during `alignment.check_pre_backward`, `S_decode_vs_S_prefill` was warned, but `S_prefill_vs_T_old` had floating-point differences between vLLM Rollout TPU and JAX Trainer TPU and was not in `warning_boundaries` for Native mode, triggering `AlignmentGateError: pre-backward alignment gate RED: ['S_prefill_vs_T_old']`.
- Action: JobSet deleted immediately to release 128 TPU chips; evidence published to branch.

## 2026-08-22 UTC — p58f06 finite Native B-C warning scope repaired locally

- Type: source intake/evidence analysis/implementation/validation/documentation.
- Source intake: fast-forwarded the isolated P58 worktree through immutable p58f06 evidence and the later P57 evidence/execution-log checkpoints to operator tip `68fa7d924ef7138e99cc2864ebbcf9edb6e676d9`. Both upstream's target-execution checkpoint and this repair checkpoint are preserved. `main` was untouched.
- Evidence: `evidence/p58f06/run.log`, 7,094 lines and 1,945,573 bytes, SHA-256 `34c6830d5b4179cf8ccdd697a0b03d9764fc75ffefa9313d5a1910914e774fd9`. The 492.7-second rollout durably wrote 128 trajectories: 126 `SUCCEEDED`, two `MAX_CONTEXT_LIMIT_REACHED`, three solved, five all-failed groups, one mixed/effective group, two incomplete groups, and 31 effective nonzero advantages. All timeout dimensions were zero. The trajectory journal is `/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-native-full-p58f06/debug/batch-000000.trajectories.jsonl.gz`, SHA-256 `ddaefb3c0efc8eb7f29724c80b5aa88ab38e8b49e7bd3cf7134c4916afe2e6f3`.
- Reached boundary: the previous full-stage admission repair passed. Exact live weights passed for 398 leaves and 4,022,468,096 elements over the 64-device rollout role; the Native processed-B observer passed all 2,048 prompt rows. Alignment ran over 405,827 action tokens. A-B differed in 279,909 elements and B-C differed in 314,476 elements; both arrays were shape-valid and finite. The run stopped before trainer forward/backward/update because P58's warning tuple contained only `S_decode_vs_S_prefill`, leaving finite `S_prefill_vs_T_old` blocking. Optimizer step remained zero and no checkpoint exists, so p58f06 is immutable `INCONCLUSIVE`, not resumable training state.
- Root cause: the Native treatment preserves both the serving decode/prefill and serving/trainer numerical programs, but the P58-specific warning scope had been narrowed to only the first seam. This was a policy/classifier defect, not malformed action geometry, nonfinite values, weight drift, rollout failure, or a zero-TIM flag leak.
- Repair: signed P58 Native now treats finite A-B and finite B-C as warning-only treatment observations. Trainer `T_old_vs_T_current` repeat and derived ratio `r` remain exact/fail-closed, and nonfinite/shape, weight, replica, transaction, optimizer, and every Zero-arm difference remain hard. The classifier accepts a finite nonzero treatment dose on either Native serving boundary and independently requires exact trainer repeat. No numerical flag was added, removed, enabled, disabled, or repurposed.
- Validation: host-direct profile 2/2, renderer 7/7, alignment policy 8/8, environment 5/5, and classifier 4/4 pass. P34 static passes 10 suites; current P57 adjacency passes 105/105 with `P57_FROZENLAKE_TIM_CPU_PASS`; shared alignment regression passes 40/40. Python compilation, Bash syntax, flag-registry audit, and `git diff --check` pass. The pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` emits `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 stock_observer=1 regressions=1`.
- One-host evidence: the bounded direct runner loaded the training JAX/libtpu environment, but this container exposes no `/dev/vfio` and reports zero chips. It emitted `P58_ONEHOST_ALIGNMENT_BLOCKED reason=device_inventory_timeout timeout_secs=10`; this is an environment blocker, not a TPU PASS or code failure. No topology was emulated.
- External effects before publication: three requested fast-forward pulls, local source/tests/documentation edits, local tests, one pinned-image test, and removal of the empty lock/cache files created by those tests. No image publication, rendered YAML, Kubernetes apply, remote TPU job, model download, credential change, or `main` mutation occurred.
- Next: await explicit commit/push approval. After publication and exact remote readback, use only fresh Native full run-id `p58f07`; require both serving-boundary warnings, finite forward/backward, exact trainer repeat, TPU-resident optimizer, and the first commit, then continue the same job through 1,000 commits if healthy. Zero remains strict and deferred.

## 2026-08-22 UTC — p58f06 finite Native B-C warning-scope repair published

- Type: publication evidence.
- Action: after explicit user approval, committed the signed P58 Native finite A-B/B-C warning scope, strict trainer-repeat/Zero negative controls, classifier treatment-dose correction, runbook/handoff updates, and preserved upstream execution checkpoint. The pre-push fetch proved the operator branch had not advanced; the push was normal and non-force.
- Published implementation commit: `2ac6383780be57033ddb5f34d348b632bf566011`.
- Readback: local HEAD, `FETCH_HEAD`, and `origin/yuxzhang/canon-zero-tim` all resolved to the implementation commit with ahead/behind `0/0` before this publication checkpoint.
- External effects: one normal fast-forward push to `yuxzhang/canon-zero-tim`. `main` was untouched. No image publication, rendered YAML, Kubernetes apply, TPU launch, model download, or credential change occurred.
- Next: publish this documentation-only checkpoint and verify its final remote readback. The executor must fetch that final tip and launch only fresh Native full run-id `p58f07`; Zero remains strict and deferred.

## 2026-08-22 UTC — p58f07 step-0 rollout and pre-backward passed, failed on post-backward T_old_vs_T_current

- Type: target execution / evidence collection
- Evidence: `evidence/p58f07/run.log`. JobSet `canon-p58-ds4b-native-full-p58f07` ran across 128 TPU v5p chips.
- Result:
  - Step 0 Rollout completed all 128 SWE-bench RepoEnv sandboxes (`N_action=436,464` tokens).
  - Pre-backward alignment passed with warnings: `[CANON_ALIGN_PRE] step=0 verdict=PASS_WITH_ALIGNMENT_WARNINGS bounds=[('S_decode_vs_S_prefill', 830053), ('S_prefill_vs_T_old', 1169723)]`. This verified that the `S_prefill_vs_T_old` policy repair in `2ac63837` worked as expected.
  - Step 0 Rescore B completed in 26.9s. Backward gradient accumulation ran across 8 microsteps on 128 TPUs.
  - In post-backward `alignment.check_batch()`, the trainer failed on `AlignmentGateError: alignment gate RED mode=train: ['T_old_vs_T_current', 'r_all_exactly_1']`.
- Action: deleted JobSet to release 128 TPU chips; recorded evidence in `evidence/p58f07/run.log` and pushed to branch.

## 2026-08-22 UTC — p58f07 trainer-observer program geometry repaired locally

- Type: source intake/evidence analysis/implementation/validation/documentation.
- Source intake: after the clean P58 preflight, fast-forwarded the isolated worktree from `883d2ece81fd1477281bfab3768d0ac6114e593f` to operator tip `1462cdccdd6c39d658fdf8df9786ebb1ddb507e1`. The incoming P58 artifact is immutable `evidence/p58f07/run.log`, 24 lines and 1,396 bytes, SHA-256 `147332c0d9ffc6a4e5016963b18f427efeee683adb2a31defcd671941a1c58ef`; the other incoming changes are P57-only adjacency. `main` was untouched.
- Reached boundary: p58f07 completed 128 real SWE RepoEnv trajectories (`N_action=436,464`), passed pre-backward with finite Native A-B/B-C warnings (`830,053` and `1,169,723` differing bytes), completed Rescore B in 26.9 seconds, and entered real value-and-grad/backward. The first post-backward check stopped on `T_old_vs_T_current` and derived `r_all_exactly_1`. No durable optimizer receipt or checkpoint exists, so p58f07 is immutable `INCONCLUSIVE` and not resumable training state.
- Root cause: P58 inherited prompt-counted `compute_logps_micro_batch_size=8`; the Agentic GRPO conversion multiplied it by G16 and computed standalone trainer `T_old` as one 128-trajectory program. The frozen training contract slices the same ordered batch into eight 16-trajectory value-and-grad programs for `T_current`. Batch shape is part of the stock numerical program, so the hard gate compared different programs rather than a same-program repeat. The strict gate was correct to stop but its observer input geometry was wrong.
- Repair: added a P58-only fail-closed geometry resolver. Signed Native and Zero now compute observer `T_old` in exact 16-trajectory chunks and concatenate the ordered outputs before the unchanged sidecar is sliced. B8 x G16, 128 raw rows, rollout logps, loss, compact filtering, eight-step gradient accumulation, TPU-resident optimizer, commit cadence, and every arm-specific numerical flag remain unchanged. `T_old_vs_T_current` and `r` remain exact/hard; no mismatch was waived. Unsigned arms, partial coverage, and non-divisible geometry are rejected. Non-P58 workloads retain their existing prompt-counted scoring geometry.
- Evidence hardening: `[P58.LOGPS_BATCH]` now records `execution_trajectories=16 observed_trajectories=128 geometry=p58-trainer-trajectory-microbatch`. The P58 classifier requires exactly one such marker per durable batch and rejects the former 128-row observer geometry.
- Validation: Python compilation, `git diff --check`, and deterministic flag-registry audit (`320/320`, `FLAG_AUDIT_PASS`) pass. Host environment geometry 9/9, profile 2/2, renderer 7/7, and alignment policy 8/8 pass. P34 static emits `P34_STATIC_PASS suites=10`; current P57 adjacency passes 105/105 and emits `P57_FROZENLAKE_TIM_CPU_PASS`. In pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`, environment 9/9, classifier 5/5, adjacent stock `AgenticGrpoLearnerTest.test_compute_logps_micro_batch_size`, shared alignment 40/40, P34/P44 neighbors, and stock-observer probes pass. The complete gate emits `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 stock_observer=1 regressions=1`.
- External effects: one requested fast-forward pull plus local source/tests/documentation edits and local/pinned-container tests only. No commit, push, image publication, rendered YAML, Kubernetes object, TPU launch, model download, credential change, or `main` mutation occurred.
- Reconciliation: while validation ran, the operator branch advanced by one P57-only alignment-policy commit. The local P58 edits were placed in a recoverable stash, the worktree fast-forwarded without conflict to final base `963cc2764595eae003b88b868f5818cdc5b659a6`, and the P58 edits were restored exactly. On that final base, P57 again passed 105/105 with `P57_FROZENLAKE_TIM_CPU_PASS`, the flag audit passed 320/320, `git diff --check` passed, and the complete pinned-image P58 gate again emitted its terminal PASS marker.
- Next: await separate explicit commit/push approval. After publication and exact remote readback, launch only fresh Native full run-id `p58f08`; require the 16-row geometry marker, exact trainer repeat, finite backward, a valid device-resident optimizer receipt, and the first commit before promoting this boundary. Zero remains deferred.

## 2026-08-22 UTC — superseded geometry repair removed; Native stock-program mismatch made observational

- Type: user correction / implementation / validation / documentation.
- Correction: before any commit or push of the preceding checkpoint, the user clarified that P58 Native is the untreated `yuxzhang/deepswe-quality-fix` training system. Replacing its standalone 128-trajectory trainer observer with eight 16-trajectory calls would change the Native numerical program and undermine the comparison. The unpublished geometry helper, runtime branch, marker gate, and geometry tests were removed. The preceding checkpoint remains in this ledger as superseded reasoning and had no published effect.
- Runtime semantics: with `use_rollout_logps=true` and sampler-IS disabled, the policy loss uses rollout A as `old_per_token_logps`. Standalone `T_old` is observer-only. Signed P58 Native therefore keeps the stock prompt-counted 128-trajectory observer and records every shape-valid finite A/B/T_old/T_current mismatch plus finite `w`, `r`, and `w*r` consequences as warnings. It still requires a nonzero serving-path treatment dose on A-B or B-C. Zero remains exact on every boundary. NaN/Inf, invalid shape, weight/replica/transaction/optimizer faults, OOM, and corrupt evidence remain hard.
- Classifier: removed `native_trainer_repeat_exact` and the P58-only geometry-marker condition. Native now requires `T_old_vs_T_current` to be present, valid, and finite; Zero retains `zero_all_boundaries_exact`. Added positive finite-drift and negative nonfinite/Zero-drift controls.
- Validation: Python compilation and `git diff --check` pass; flag audit is `320/320` with `FLAG_AUDIT_PASS`; host environment 5/5, profile 2/2, renderer 7/7, and alignment policy 9/9 pass; P34 emits `P34_STATIC_PASS suites=10`; P57 passes 105/105 with `P57_FROZENLAKE_TIM_CPU_PASS`. Pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` passes classifier 5/5, shared alignment 42/42, stock-observer and adjacent regressions, and emits `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 stock_observer=1 regressions=1`.
- Target claim: no new TPU run was made. P58f07 remains immutable `INCONCLUSIVE` with no durable optimizer receipt/checkpoint. Direct one-host TPU remains unavailable because this container exposes no `/dev/vfio`; no TPU PASS is claimed.
- Next: after publication and exact remote readback, launch only fresh Native full run-id `p58f08`; require finite Native boundaries/ratios, a nonzero A-B or B-C dose, finite backward, a device-resident optimizer receipt, and the first commit. Zero remains deferred.

## 2026-08-22 UTC — p58f07 Native stock-program warning policy published

- Type: publication evidence.
- Action: after explicit user approval, committed the stock-quality-fix Native policy, finite/nonfinite/Zero classifier controls, flag contract, P58 runbook/handoff/phase records, and p58f07 evidence index. The pre-push fetch proved the operator branch had not advanced; the push was normal and non-force.
- Published implementation commit: `81622977bf15393798c671e578ee059d1268e78b`.
- Readback: local HEAD, `FETCH_HEAD`, and `origin/yuxzhang/canon-zero-tim` all resolved to the implementation commit with ahead/behind `0/0` before this publication checkpoint.
- External effects: one normal fast-forward push to `yuxzhang/canon-zero-tim`. `main` was untouched. No image publication, rendered YAML, Kubernetes apply, TPU launch, model download, credential change, or failed artifact deletion occurred.
- Next: publish this documentation-only checkpoint and verify its final remote readback. The executor must fetch that final tip and launch only fresh Native full run-id `p58f08`; Zero remains strict and deferred.

## 2026-08-22 UTC — p58f08 worker crashed on Pathways ResourceManager CL mismatch

- Type: target execution / infrastructure evidence collection
- Evidence: `evidence/p58f08/run.log`. JobSet `canon-p58-ds4b-native-full-p58f08` ran across 128 TPU v5p chips.
- Result: Head Pod initialized, verified stock engine, applied bounded R2E patch, and loaded dataset. However, `pathways-worker-0` failed during initialization with `ResourceManagerDone: crashing worker due to failed precondition: FAILED_PRECONDITION: Server pipe /leader_resource_manager id=18098245068127715496: pipes with strict compatibility check require the client and the server binaries to be built at the same CL, but got cl/956357083 (client) vs. cl/42 (server)`.
- Cause: HostNetwork port 29001 on CPU node `gke-mlperf-v5p-cpu-np-ebb0f94d-lf6h` collided with an existing running Pathways Resource Manager (`nt-ds-pw-35b-gsm8k-v1`) that runs at CL/42, causing the worker to connect to the foreign RM.
- Action: Deleted failed JobSet to release resources; recorded evidence in `evidence/p58f08/run.log` and pushed to branch.

## 2026-08-22 UTC — p58f08 foreign ResourceManager collision repaired locally

- Type: source intake/evidence analysis/implementation/documentation.
- Source intake: fast-forwarded the clean isolated P58 worktree from `af852d64a8f6507a72b76d8497ccf14d670a97bb` to operator tip `5c5aca27520e828d788442fd95871a1604b8617b`. The incoming P58 artifact is immutable `evidence/p58f08/run.log`, 12 lines and 764 bytes, SHA-256 `87d4386f1818ab40c87817819549df56d6e7de3995e333665b0021ff111a2f0e`. `main` was untouched.
- Reached boundary: the P58 head verified the stock engine, applied the bounded R2E patch, and loaded the dataset. `pathways-worker-0` then failed strict compatibility with `cl/956357083 (client) vs. cl/42 (server)` before any rollout, trajectory journal, trainer program, optimizer receipt, or checkpoint existed. P58f08 is immutable `INCONCLUSIVE` and not resumable.
- Root cause: P58 inherited `hostNetwork:true` for the CPU head even though the proxy, ResourceManager, and JAX client share one Pod and communicate over localhost. Another Pathways job on the same CPU node already exposed CL/42 RM port 29001, so the P58 worker reached that foreign service instead of its own RM. This is a Kubernetes network/port collision, not DeepSWE, B8 x G16, model, loss, Native numerical, or optimizer failure.
- Repair: the P58 renderer alone sets the CPU head to `hostNetwork:false` with `dnsPolicy: ClusterFirst`. TPU workers retain `hostNetwork:true` and `ClusterFirstWithHostNet`. Workers continue to address port 29001 through the generated JobSet Pod DNS. Validation now rejects head host-network regression, missing `enableDNSHostnames`/`publishNotReadyAddresses`, and any worker ResourceManager or `PATHWAYS_HEAD` drift. No Native/Zero numerical flag, topology, model/data, deadline, algorithm, optimizer, or update setting changed.
- Validation status at checkpoint creation: focused renderer tests pass 12/12; Python compilation, a fresh p58f09 render, and `git diff --check` pass. Full host, adjacency, flag-registry, and pinned-image results are recorded by the following validation checkpoint after they complete.
- External effects: one requested fast-forward pull plus local source/tests/documentation edits only. No commit, push, image publication, Kubernetes apply, TPU launch, model download, credential change, or `main` mutation occurred.
- Next: finish the full validation matrix, then await separate commit/push approval. After publication and exact remote readback, render only fresh Native full `p58f09`; verify the isolated head, JobSet DNS publication, host-network workers, exact RM DNS, and matching Pathways CL before waiting for rollout. Zero remains strict and deferred.

## 2026-08-22 UTC — p58f08 network-isolation repair validation complete

- Type: validation/handoff checkpoint.
- Host validation: P58 renderer passes 12/12, profile 2/2, and environment 5/5. A fresh Native/full p58f09 render emits `P58_DEEPSWE_TIM_RENDER_PASS` and contains `head hostNetwork=false`, `head dnsPolicy=ClusterFirst`, unchanged host-network workers, and the exact generated JobSet RM DNS. Python compilation and `git diff --check` pass.
- Adjacency/registry: P34 emits `P34_STATIC_PASS suites=10`; P57 passes 105/105 with `P57_FROZENLAKE_TIM_CPU_PASS`; deterministic flag audit passes 320/320 with `FLAG_AUDIT_PASS` and `changed_names=0`.
- Exact-image validation: pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` exits zero and emits `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 stock_observer=1 regressions=1`.
- Claim ceiling: these tests prove manifest construction and regressions only. A direct-attached one-host run cannot reproduce a Kubernetes host-port collision, and this container still exposes no `/dev/vfio`; no TPU or Pathways runtime PASS is claimed. Fresh p58f09 is required to prove attachment to the intended RM and resume real training progress.
- External effects: local tests and documentation only. No commit, push, image publication, Kubernetes apply, TPU launch, model download, credential change, or `main` mutation occurred.
- Next: await separate explicit commit/push approval. After publication/readback, launch only fresh Native full p58f09 and collect all head-container plus one worker log immediately if strict-CL attachment fails again. Zero remains deferred.

## 2026-08-22 UTC — p58f08 Pod-network proposal superseded by placement evidence

- Type: user correction / source intake / infrastructure reconciliation.
- Source intake: saved the unpublished Pod-network work as a recoverable stash, passed the clean P58 preflight at `5c5aca27520e828d788442fd95871a1604b8617b`, fast-forwarded to operator tip `3edf480072126145acc2df259419e12dd2737c69`, and restored the local work without conflict. The incoming P58 changes are the completed p58f08 diagnosis and immutable `evidence/p58f09/run.log`. `main` was untouched.
- Corrected p58f08 diagnosis: after adding the required Kueue flavor, a head on `deepswe-cpu-pool` started but TPU workers could not maintain the scheduler pipe across node-pool subnets. On `cpu-np`, six concurrent JobSet heads already occupied six CPU nodes; without Pod anti-affinity, Kubernetes packed the seventh host-network head onto an occupied node and fixed port 29001 reached a foreign CL/42 ResourceManager. The user's CPU-node interpretation was correct: preserve the proven Pathways host network and `cpu-np`, and isolate fixed ports through scheduler placement rather than Pod networking.
- Supersession: the preceding local `hostNetwork:false`/`ClusterFirst` proposal was never committed or pushed and is superseded. Historical reasoning remains in this append-only ledger; current state, phase, runbook, handoff, renderer, and tests now require `hostNetwork:true`, `ClusterFirstWithHostNet`, and hostname-level required anti-affinity selecting the automatic JobSet `pathways-head` replicated-job label.

## 2026-08-22 UTC — p58f09 rollout completed; reset-timeout original input repaired locally

- Type: target evidence analysis / implementation / tests / documentation.
- Evidence: `evidence/p58f09/run.log`, 4,553 lines and 455,785 bytes, SHA-256 `8977eefcb2ef34bc17c4dbb6e129b1d02cacba6b63041ab42d43a3aa8b5f4d0b`. The run used source `933d1516da9703f06d072461bde81d6789e7c8ef`, correct 128-device Pathways inventory, rollout DP8 x TP8 plus trainer DP8 x TP8, the exact 1,012-task clean list, and the frozen Native B8 x G16 / 16K / 1,000-update command.
- Reached boundary: Step-0 rollout completed 128 execution and 128 observed trajectories in 1,699.1 seconds, inside the 3,600-second batch deadline. Several environment resets reached the admitted 3,000-second trajectory deadline before first observation, and one later row reached `MAX_CONTEXT_LIMIT_REACHED`. Learner preprocessing then failed at `rl_utils.merge_micro_batches(original_inputs_list)` with `AttributeError: 'NoneType' object has no attribute 'keys'`. No P58 journal, alignment, forward, backward, optimizer receipt, or checkpoint was produced; p58f09 is immutable `INCONCLUSIVE` and not resumable.
- Root cause: Token-mode trajectory output used only `agent.trajectory.task` for `original_input`. That field is assigned after an environment observation, so a reset deadline can leave it `None`; the environment still retains the exact original dictionary in `env.task`. The learner correctly expects every original input to be a mapping, and filtering the row at merge time would silently change the compact-filter recipe.
- Repair: trajectory construction now prefers the observed agent task, falls back to `env.task` only after pre-observation termination, and fails closed with `TypeError` if neither source is a dictionary. The row retains its signed timeout/context status and existing all-zero policy mask; no trajectory is dropped, resampled, relabeled, or allowed to affect reward/loss. Added positive reset-timeout and missing-input negative controls. The P58 renderer now adds exact required hostname anti-affinity for all JobSet `pathways-head` Pods and validates retained head/worker host networking, JobSet DNS, and RM/PATHWAYS_HEAD routing.
- Numerical boundary: B8 x G16, 128 trajectories, clean data, Native stock numerical program, Zero disables/strict gates, rollout logps, compact status list, loss, gradient accumulation, TPU-resident optimizer, deadlines, and 1,000-commit horizon are unchanged. No flag was added, deleted, enabled, or repurposed.
- External effects: one requested fetch/pull plus local code/tests/documentation edits only. No commit, push, image publication, Kubernetes apply, TPU launch, model download, credential change, or `main` mutation occurred.
- Next: complete the host, adjacency, registry, and pinned-image regressions. After separate commit/push approval, publication, and exact remote readback, launch only fresh Native full `p58f10`; require distinct CPU hostnames for active Pathways heads, a durable 128-row Step-0 journal, finite Native boundaries/backward, and the first TPU-resident optimizer commit. Zero remains deferred.

## 2026-08-22 UTC — p58f09 repair validation complete

- Type: validation / handoff checkpoint.
- Host validation: P58 renderer passes 14/14, including rejection of `deepswe-cpu-pool`; profile passes 2/2, environment 5/5, and alignment policy 9/9. Python compilation and `git diff --check` pass. A fresh Native/full p58f10 render emits `P58_DEEPSWE_TIM_RENDER_PASS` and contains `cpu-np`, head/worker `hostNetwork:true`, head `ClusterFirstWithHostNet`, the exact required `pathways-head`/hostname anti-affinity term, JobSet DNS publication, and matching worker RM/PATHWAYS_HEAD DNS.
- Adjacency/registry: P34 emits `P34_STATIC_PASS suites=10`; P57 passes 105/105 with `P57_FROZENLAKE_TIM_CPU_PASS`; deterministic flag audit passes 320/320 with `FLAG_AUDIT_PASS` and `changed_names=0`.
- Exact-image validation: pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` passes the six targeted agentic/trajectory tests, including reset-timeout task fallback and missing-input fail-closed controls, and emits `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 stock_observer=1 regressions=1` with exit zero.
- Validation correction: the first full rerun after making `cpu-np` fail-closed rejected the environment test's stale placeholder `cpu-pool`. The production contract behaved correctly; the fixture was changed to the admitted `cpu-np`, host environment returned 5/5, and the complete pinned-image rerun then emitted the terminal PASS marker above.
- Claim ceiling: these tests prove manifest construction and host-side trajectory semantics in the pinned dependency image. They do not prove Kubernetes anti-affinity placement, Pathways runtime, a durable 128-row target journal, trainer forward/backward, or an optimizer commit. Only fresh p58f10 can cross those boundaries.
- External effects: local rendering/tests/documentation only. No commit, push, image publication, Kubernetes apply, TPU launch, model download, credential change, or `main` mutation occurred.
- Next: await separate explicit commit/push approval. After publication/readback, launch only fresh Native full p58f10. Zero remains deferred.
