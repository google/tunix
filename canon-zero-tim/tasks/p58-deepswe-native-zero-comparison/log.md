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
