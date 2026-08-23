# P58.6-P58.8 release CL plan

Base: `ccbcf572dc903bb1cce12f897cbdb05aec94922a`.

This ledger is a staging specification, not commit or push approval. Every CL
must be reconstructed by hunk from this exact base. The final tree must be
audited against the pinned-image-tested release tree before any publication.

## CL-A — Repair P59 TP4/TP8 nested engine backward

Proposed subject: `Repair P59 TP backward mesh composition`

Problem: the P59 outer rank-parallel map used the trainer mesh vocabulary while
the real fixed-head and fused-linear kernels entered nested engine shard maps.
TP4/TP8 therefore failed before the first optimizer commit.

Scope:

- `tunix/rl/canonical_qwen3_adapter.py`
- `tests/rl/canonical_qwen3_adapter_test.py`
- `canon-zero-tim/src/engine_shims/p38_fixed_lm_head.py`
- `canon-zero-tim/src/engine_shims/linear_p22xf.py`
- the two corresponding hashes in `canon-zero-tim/MANIFEST.sha256`
- `canon-zero-tim/tests/p59_backward/probe_tp4_installed_shim_composition.py`
- `canon-zero-tim/tests/p59_backward/run_tp4_tp8_installed_shim_exact_image.sh`
- only the P59 gate invocations and P59 marker fields in the P58/V1 exact-image
  runners
- only the P59 TP4/TP8 status receipt in `FLAGS.md` and the P58.8/V1 P59
  result-log hunks

Gate: P59 30/30; installed overlays 36/36 for Qwen3-1.7B/TP4 and
Qwen3-8B/TP8; forced CPU DP2xTP4 and DP2xTP8 fixed-head plus installed
projection VJPs; serial/parallel leaf equality; report adjoint; fixed reducer;
ordinary-global placement negative controls; P58 and V1 exact-image terminal
markers include `p59_real_shim=4`.

Downside: TP cotangents are gathered in FP32 and added in fixed rank order,
with `optimization_barrier` on both operands of every addition. This adds
communication and temporary memory and constrains compiler reassociation of
the FP32 chain. This CL proves gradient correctness, not identity with the
historical serial AdamW trajectory, and it does not certify a real TPU target.

Rollback: keep `CANON_P59_RANK_PARALLEL_BACKWARD=0`, or revert CL-A alone.
The original `f7d22555` failure evidence remains immutable inside this base.

## CL-B — Admit the signed P57 Zero/full W&B project

Proposed subject: `Admit the P57 Zero training W&B project`

Problem: the FrozenLake Zero/full target stopped in environment validation
because its signed profile project differed from the generic workload project.

Scope:

- the exact P57 profile/arm/kind override in `tunix/rl/dp_workloads.py`
- its positive and wrong-arm/profile negative controls in
  `canon-zero-tim/tests/p33_workloads/test_dp_workloads.py`
- only the P57 W&B invocation and marker fields in the P58/V1 exact-image
  runners
- only P57 W&B result-log hunks

Gate: P57 136/136 and both complete pinned-image runners include
`p57_wandb=1`.

Downside: this is an intentionally workload-specific telemetry exception. It
must not broaden admission for Native, evaluation, or another FrozenLake
profile.

Rollback: revert CL-B alone. No numerical or optimizer code is in this CL.

## CL-C — Add matched mutation-free one-host XProf carriers

Proposed subject: `Add matched DeepSWE one-host XProf carriers`

Problem: Native and optimized Zero lacked a matched four-chip capture of the
same trainer/backward work with durable XPlane and semantic Perfetto evidence.

Scope:

- the one-host arm admission and no-commit update repeat in
  `examples/deepswe/train_deepswe_nb.py`, `tunix/rl/deepswe_debug.py`,
  `tunix/rl/agentic/agentic_grpo_learner.py`,
  `tunix/rl/agentic/agentic_rl_learner.py`, `tunix/rl/alignment.py`,
  `tunix/rl/rollout/vllm_rollout.py`, and `tunix/sft/peft_trainer.py`
- one-host admission in
  `canon-zero-tim/cluster/steps/p58_install_stock_prompt_observer.sh`
- the vLLM output test double's stock `num_cached_tokens=0` receipt in
  `tests/rl/rollout/vllm_rollout_canonical_test.py`
- `classify_onehost_xprof.py`, `classify_onehost_xprof_pair.py`, and the three
  `run_onehost_deepswe_xprof_*.sh` scripts
- `test_onehost_xprof.py`, `test_onehost_xprof_pair.py`, and only the one-host
  environment tests/runner marker fields
- the seven new P58 one-host provenance flag registrations and the inventory
  count update in `FLAGS.md`
- the matching settable-flag count in
  `canon-zero-tim/tests/p3_prefix_cache/test_contract.py`
- the P58.6 phase and one-host portions of the runbook/result ledger

The shared continue-decode and Perfetto reader is staged here with one-host-only
admission. CL-D separately widens that reader to the signed P58 Zero/full
profile; this prevents CL-C from silently introducing a production recipe.

Gate: one-arm classifier 5/5, pair classifier 2/2, environment contract,
flags 366/366, and P58 exact-image `onehost_xprof=1`.

Downside: each arm runs one compile/warmup backward and one profiled repeat on
the same in-memory batch. It makes zero optimizer commits, injects a registered
diagnostic `[-1, 1]` cotangent, and therefore measures the update/backward
program rather than end-to-end training throughput or a production gradient.

Rollback: revert CL-C alone; all selectors default to empty/off and no
production P58 profile depends on this CL until CL-D is present.

## CL-D — Add the Qwen3-4B optimized Zero-HP full recipe

Proposed subject: `Add the DeepSWE Qwen3-4B Zero-HP recipe`

Problem: the DeepSWE-derived Qwen3-4B comparison had no strict-Zero optimized
full recipe carrying the already-selected serving and trainer performance
bundle.

Scope:

- `canon-zero-tim/cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env`
- high-performance render/admission/postflight changes in
  `render_p58_deepswe_tim.py`, `00_env.sh`, `90_run.sh`,
  `tunix/rl/deepswe_contract.py`, and the production-only widening in
  `examples/deepswe/train_deepswe_nb.py`
- the upstream `24b1bbcf` P58 `maxRestarts=3` and Pathways/GRPC keepalive
  settings, preserved identically for the optimized Zero-HP renderer
- `classify_zero_hp_full.py` and its renderer/profile/environment/classifier
  tests
- P58.7 bundle/status hunks in `FLAGS.md`
- P58.7 phase, combined runbook, HANDOFF, plan, state, log, and this final CL
  audit ledger
- the current deepswe-train `THREADS.md` row, append-only `EVIDENCE.md` rows,
  and the three SHA-sealed latest-base exact-image/runtime-manifest evidence
  directories
- final P58 exact-image aggregation fields not owned by CL-A, CL-B, or CL-C

Gate: renderer 16/16, profile 4/4, full classifier 3/3, environment negative
controls, host suites, flags 366/366, and complete P58 exact-image
`zero_hp_full=1`.

Downside: this is a 128-chip DP8xTP8 1,000-commit target recipe. It combines
multiple previously selected optimizations, so target performance gains are
not attributable to one kernel. P59 changes the admitted gradient summation
program and does not promise the historical serial AdamW trajectory. APC is
explicitly off. No target TPU execution is claimed by this CL.

Rollback: revert CL-D alone to remove the recipe and renderer reachability;
CL-A/B/C remain independently testable and default off.

## CL-E — Restrict changed-flag auditing to executable files

Proposed subject: `Ignore evidence markers in flag diff audits`

Problem: the final pre-push audit scanned newly committed immutable `run.log`
files and Markdown receipts. Stable non-settable marker names such as
`CANON_ALIGN_PRE_JSON` were therefore reported as missing environment flags,
even though all 366 settable names were registered.

Scope:

- the changed-name pathspec in
  `.claude/skills/manage-canon-flags/scripts/audit_flag_registry.py`;
- a focused temporary-Git-repository regression proving executable names are
  still found while Markdown, evidence, and debug-log markers are ignored;
- only the release-gate and handoff receipts needed to disclose this fifth,
  audit-only CL.

Gate: focused audit regression, full `--changed-base` registry audit at
366/366, the existing host adjacency suites, and `git diff --check`.

Downside: a settable flag written only in Markdown or an immutable evidence
artifact is no longer considered executable and will not be discovered by the
changed-name scan. The independent full `FLAGS.md` inventory count remains
mandatory.

Rollback: revert CL-E alone. It changes no model, rollout, trainer, numerical,
renderer, profile, or target-runtime path.

## Excluded concern

The unrelated APC B-arm `num_cached_tokens` availability hardening found in
the older dirty worktree is deliberately excluded from this release series.
P58.7 has APC off, and including that hunk would create a fifth numerical
concern. The release tree retains the `ccbcf572` APC implementation.

## Publication order

Stage and review A, B, C, D, then the audit-only E. After every approved
commit, run that CL's focused host gates. After E, run `git diff --check`, flag audit, both complete
pinned-image gates, and a file-hash audit against the tested release tree.
Fetch again before any separately approved push; if the operator tip moved,
rebase and repeat focused gates. Hardware launch remains a separate approval.
