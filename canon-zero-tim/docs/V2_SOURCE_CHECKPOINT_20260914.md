# V2 source-review checkpoint — 2026-09-14

This is a source checkpoint for manual review/refactoring, **not final training
certification or a new performance result**. Publication does not close the
remaining image, full-gradient, classification or hardware gates below.

## Source identity and scope

- Fetched delivery base: `321d5944ac5c800abdfbfb4b9a54a09c5eb8ecd0`.
- Tested source: `f378c2a000ec779b0cbe057788041e8a6998acc7` (41 local commits,
  zero merges above that base). The commit adding this document changes only
  documentation: this file and one FLAGS description, not executable files.
- Previous local checkpoint: `8fb01b100032eed6bc3c1cac6ea02cd56e1e628c` on
  fixed base `14c8f16e5b6fe773dc75299d8fdfa42f8d451bda`. It is preserved on
  `backup/wrapup-v2-prepublish-20260914-r1` locally.
- The 37 replayed V2 implementation/cleanup commits and four R1 repairs are
  retained. R1 fixes source-relative P33 imports, reviewed render goldens,
  complete-capture comparisons and failing exit statuses for invalid fp64
  evidence. All four repair patches remain byte-identical after rebase.
- Eight newer upstream commits are preserved, including DeepSWE Fleet and
  explicit P58 treatment/full routing. These are upstream runtime changes,
  **not** evidence-neutral metadata. The generic DeepSWE one-host recipe also
  changed from B1xG2 to B2xG2; older captures do not certify that recipe.
- Independently composing upstream changes with the old local tree reproduces
  every non-FLAGS file. FLAGS reconciles the two added Fleet names: 435 unique
  names, partitioned as 59 current-default / 206 optional / 154 diagnostic /
  16 historical. No new flag/default policy is introduced by this checkpoint.
- The adapter, rollout implementation, engine shims, patches, installer and
  MANIFEST are byte-identical to the previous local checkpoint. This is source
  preservation, not proof of complete old-plain to final-plain gradient parity.

The original V/V2/launch/reference trees and failed evidence remain preserved.
The old two-parent V2 merge is not published as part of this linear interval.
No TPU, Docker training, GKE, image publication or cleanup ran for this checkpoint.

## Verified offline scope

Verified by running the following gates on the tested source above, using the
host train venv with `JAX_PLATFORMS=cpu`, explicit source `PYTHONPATH`, and no
containers. Counts overlap and must not be added into a unique-test total.

| Gate | Actual result |
|---|---|
| R1 complete-capture/fp64 tools, including negative controls | 85 passed |
| P33, render goldens, registry tests, R1 tools, XProf hierarchy | 235 passed; 132 subtests; 3 warnings |
| P45 CPU contract/alignment chains | 133 + 51 passed |
| V1 full-recipe CPU chain | 109 passed |
| Flag registry and four-section inventory | 435/435; exact set/partition |
| Stock path-only installation, qwen1p7b_tp2 / qwen8b_tp2 / qwen4b_tp4 | 37/37 files for each model |
| Fleet CPU suite | 28 passed, 1 skipped: optional Agent Sandbox package absent |
| V1 defaults plus length sorting | 35 passed; 191 subtests; 1 skipped in this invocation |
| Length sorting, separate forced-64-device CPU invocation | 9 passed, no skips |
| P58 profile / environment / renderer / classifier | 16 / 25 / 48 / 10 passed |
| P58 replay / Q4-TP4 contract / capacity probe / one-host XProf contract | 2 / 3 / 10 / 5 passed |
| P34 renderer / environment | 13 / 7 passed |
| P39 renderer / environment | 2 / 3 passed |
| P43 renderer / environment | 3 / 3 passed |
| P44 renderer / environment | 10 / 5 passed |
| P46 renderer / environment | 14 / 7 passed |
| Source composition, patch identity, syntax, diff check, bounded added-secret scan | passed; 57 Python files parsed |

The Fleet skip is not package/import attestation. CPU device forcing is not
TPU evidence. Passing the existing P58 tests does not resolve the additional
adversarial receipt cases discovered during review (see below).

Author-host command/evidence location (outside this Git repository):
`/home/yuxuan/code_rl_repro/tasks/wrapup_v2_default/`.
The successful sequence is `scripts/run_checkpoint_offline.sh` with outputs
in `evidence/r2_offline_checkpoint_20260914_r2/`; source, forced-device and
debt-reproduction receipts are in `evidence/r2_checkpoint_20260914_r1/`.
Every gate log records its command, source and exit status. The first sequence
stopped on a task-harness `PYTHONPATH` omission; its logs are retained separately
as `r2_offline_checkpoint_20260914_r1`, not relabeled successful.
These large task records are **not shipped by Git**; request them when auditing
the historical execution. This document and the following executable entry
points are carried by Git.

### Portable offline entry points

From the repository root, with dependencies provisioned in the review venv:

```bash
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD"
python -m pytest -q -p no:cacheprovider \
  canon-zero-tim/tests/p61_backward/test_compare_capture_bits.py \
  canon-zero-tim/tests/p61_backward/test_compare_full_trees.py \
  canon-zero-tim/tests/p61_backward/test_check_fp64_repin.py
python canon-zero-tim/.claude/skills/manage-canon-flags/scripts/audit_flag_registry.py \
  --repo . --changed-base 321d5944ac5c800abdfbfb4b9a54a09c5eb8ecd0
bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_cpu.sh
bash canon-zero-tim/tests/v1_phase4/run_cpu.sh
bash canon-zero-tim/tests/deepswe_sandbox_fleet/run_cpu.sh
XLA_FLAGS=--xla_force_host_platform_device_count=64 python -m pytest -q \
  -p no:cacheprovider tests/rl/test_p32_length_sort.py
```

Full-tree contracts: [capture comparisons](../tests/p61_backward/CAPTURE_BITS.md)
and [fp64 admission](../tests/p61_backward/FP64_REPIN.md).
Run policy remains in [AGENTS](../AGENTS.md) and the canonical in-package skills.
None of the offline commands grants launch authority.

## Still open — do not promote this checkpoint

Not verified because the required final-source evidence has not been produced:

1. Complete pinned-image validation, including the previously failing adapter,
   PEFT and GRPO chain. Host installation and an unchanged historical failure
   set are not a complete image PASS.
2. Complete old-plain to final-plain gradients and their producer/input/weight/
   recipe provenance. Updates, norm anchors and 12 sampled state leaves are
   insufficient. Historical replica-to-plain fp64/G4 evidence cannot transfer
   until that link is proved; no anchor re-pin is authorized here.
3. Standard FrozenLake/GSM8K classification. Preserve old standard FAIL verdicts,
   the withdrawn P59 36-to-1 reuse target, FL chunk-coverage debt and the
   incomplete GSM8K JSON tail. Small exporter fixtures do not prove full XProf
   coverage; real re-export remains capacity/tool limited.
4. Required workload/geometry/length and real-optimizer coverage. No-commit
   captures do not certify full training; DP4, DP2xTP2, TP4, P45/M15, DeepSWE
   and target-scale claims must be evaluated per supported cell. No new
   performance comparison or speedup is claimed.
5. Final independent numerical/product acceptance. The current independent
   review covers source/history composition and bounded offline checks only.

Two upstream issues found during this checkpoint also remain open:

- [P58 classifier](../tests/p58_deepswe_native_zero/classify_run.py#L386)
  collapses duplicate length-sort fields into a dictionary and ignores a bare
  marker lacking the trailing space. The real classifier with real persisted
  fixture artifacts accepts `rows=127 rows=128` and an extra bare
  `[P32.LENGTH_SORT]`, while rejecting a wrong row and a missing receipt.
  This is a reproduced admission-tool defect, **not** a passing negative
  control. Do not rely on this receipt gate for P58 treatment admission until
  its parsing and adversarial tests are repaired in a separate CL.
- [Fleet installer](../cluster/steps/36_install_agent_sandbox.sh#L34)
  checks reused checkout HEAD but not dirty contents before editable install.
  Its exact-source claim requires a clean/content check before Fleet admission.
  This is a static finding, not a completed runtime exploit test.

### Next acceptance sequence

Keep the checkpoint immutable; do manual cleanup on a separate branch. First
close the complete image gate (CPU-only, source read-only, separately approved
container). Then close complete-gradient/provenance and standard classification;
repair the upstream admission-tool defects before relying on those paths.
Finally run the required specifically approved one-host/optimizer cells and
obtain independent final acceptance. Check disk capacity before captures; do
not delete failed runs or borrow another lane's storage without approval.

Tunix integration and general-name file renames remain separate, minimal
refactoring CLs after reviewing the relevant diff and acceptance scope. Never
merge into or push to `main`. A future push requires a new explicit approval.

## Drawbacks / 本方案的缺点

This checkpoint makes an intentionally incomplete acceptance state available
for code review. A reader who ignores the status table could mistake it for a
training release; the unresolved image, numerical and upstream admission debt
must therefore travel with the source. Detailed historical raw logs still live
on the author host, and manual refactoring will require its own affected gates.

## Source navigation

Verified on the tested source; line numbers are navigation aids, not execution
or certification evidence. Historical `p22`/`p38` names are retained for now.

| Mechanism | Main symbol | Source |
|---|---|---|
| Adapter construction | `Qwen3EngineForwardAdapter` | [canonical_qwen3_adapter.py:6579](../../tunix/rl/canonical_qwen3_adapter.py#L6579) |
| Decode A | `get_per_token_logps` | [vllm_rollout.py:287](../../tunix/rl/rollout/vllm_rollout.py#L287) |
| Prefill B | `get_prefill_rescore_logps` | [vllm_rollout.py:302](../../tunix/rl/rollout/vllm_rollout.py#L302) |
| Trainer C | `compute_per_token_logps` | [canonical_qwen3_adapter.py:14900](../../tunix/rl/canonical_qwen3_adapter.py#L14900) |
| Trainer chunking | `_sequence_group` | [canonical_qwen3_adapter.py:14340](../../tunix/rl/canonical_qwen3_adapter.py#L14340) |
| Shape admission | `_canonical_topology_contract` | [canonical_qwen3_adapter.py:2311](../../tunix/rl/canonical_qwen3_adapter.py#L2311) |
| Shared scorer | `_install_shared_logprob_pipeline` | [canonical_qwen3_adapter.py:3076](../../tunix/rl/canonical_qwen3_adapter.py#L3076) |
| Fixed projection | `matmul` | [p22_pallas_matmul.py:174](../src/engine_shims/p22_pallas_matmul.py#L174) |
| Ordered TP sum | `_contract_parallel` | [linear_p22xf.py:356](../src/engine_shims/linear_p22xf.py#L356) |
| Fixed LM head | `fixed_lm_head` | [p38_fixed_lm_head.py:373](../src/engine_shims/p38_fixed_lm_head.py#L373) |
| Operator VJPs | `matmul`, `swiglu`, `rmsnorm` | [p22xk_vjp_ops.py:107](../src/engine_shims/p22xk_vjp_ops.py#L107) (also 131, 193) |
| Cache VJP | `make_diff_rpa_chunked` | [rpa_diff_chunked.py:18](../src/engine_shims/rpa_diff_chunked.py#L18) |
| Checked VMA | `vma_local_fn`, `mark_data_varying` | [canonical_qwen3_adapter.py:5247](../../tunix/rl/canonical_qwen3_adapter.py#L5247) |
| Group reverse | `_p32_reverse_group` | [canonical_qwen3_adapter.py:11019](../../tunix/rl/canonical_qwen3_adapter.py#L11019) |
| Full RL value/gradient | `segmented_dp_grpo_value_and_grad` | [canonical_qwen3_adapter.py:11749](../../tunix/rl/canonical_qwen3_adapter.py#L11749) |
