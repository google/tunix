# P66 evidence log

## 2026-08-25 — CHECKPOINT: task adopted

- Source base: `1e5f7e835f4babe43a50496a5b998ea32cffcf71`.
- Clean preflight: PASS before P66 edits.
- Historical evidence reviewed: P61n2 had 17/17 strict alignment per arm, zero failures, all seven input hashes identical, exact model-before leaves, and complete full-tree captures.
- Historical numerical result: gradient rel-L2 `0.0158173601`, one-minus-cos `0.0001249273`, norm-ratio error `0.0007164464`, sign mismatch `0.0031906065`; gradient passed its frozen envelope. Real AdamW-delta rel-L2 `0.0997646104` and one-minus-cos `0.0049764518` exceeded the original frozen thresholds, producing `NUMERICAL_REJECT`.
- Historical result artifact SHA256: `6562ebdf...` as recorded by the P61 ledger; current P66 has not revalidated the remote run directory.
- Tier-1 baseline located at `/home/yuxuan/code_rl_repro/tasks/p61-backward-numerical-oracle/artifacts/p61c2_20260823t0012z/tier1.r1.json`, SHA256 `05f704baad09a44b36d944e4be14900d10316ffb3ba9cfb6b769a293de8d6d38`.
- Release completeness defect: `tasks/p61-backward-numerical-oracle/scripts/run_onehost_dp4_numerical_ab.sh` invokes `tests/p61_backward/compare_full_trees.py`, which is absent from the published tree. A reviewed copy and tests exist in the prior P59 worktree; P66.1 will restore and test them before TPU use.
- Claim boundary: P66 evaluates both gradient correctness and optimizer-update consequences. It will not claim that gradient-envelope acceptance implies identical AdamW trajectory.
- Next: restore comparator/tests and make reject/inconclusive evidence packaging durable.

## 2026-08-25 — CHECKPOINT: P66.1 host and pinned-image gates

- Restored comparator/test source is byte-identical to the historical P61n2 implementation: comparator SHA256 `88bc3d27973d161612ac044f5c94b0747e21e4d5202ba99422f672016fd00986`; tests SHA256 `dc02c57c73950cfdf199f0c3a2e977cf5f758680169a912a31a37ea2e11423e8`.
- P61 comparator/wrapper tests: 6/6 PASS. The negative control proves `NUMERICAL_REJECT` returns nonzero, is manifested, and never prints GREEN.
- P59 focused host tests: 37/37 PASS.
- Shell syntax and `git diff --check`: PASS.
- Pinned image admission: `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`; overlay manifest 36/36 and container tests 35/35 PASS. The command output is retained by the execution transcript but was not separately written as a durable raw-log artifact, so this is an admission receipt rather than a signed raw log.

## 2026-08-25 — RESULT: P66.2 current-source P59 same-input A/B

- Tested source: `1e5f7e835f4babe43a50496a5b998ea32cffcf71`; dirty diagnostic diff SHA256 recorded by both arms: `d662a264123ece31af106db30b69e9ebf2b5de660ae4ac46d0c2d410d334d64d`.
- Control: `p59_dp4_numerical-control_p66s1_20260825t1813z`; 1/1 optimizer commit, 17/17 strict PASS, 0 FAIL, pullback invocations 4. Classification SHA256 `4962239e2979060f0c9ca218215465cb4f56aedebbecea9c7f36d4affde92277`; arm-manifest SHA256 `3a92c050ff3b4e72a919be8a2da344d42929cd50f5ab269103bae3e987de891c`.
- Candidate: `p59_dp4_numerical-candidate_p66p1_20260825t1813z`; 1/1 optimizer commit, 17/17 strict PASS, 0 FAIL, pullback invocations 1. Classification SHA256 `1a04485a7f09e44509fb9850f52178f3fc68631d490da4667ed482d28a69e40a`; arm-manifest SHA256 `d2cf76a61943a4e2fb80a1ab6f4fa766ca81cf5967fde791135d9569dc470aaf`.
- Both current arms and historical P61n2 have identical N_action `16127` and identical pre-alignment hashes. Cross-arm seven input hashes match; all 310 model-before leaves are exact.
- Gradient, 1,720,574,976 elements: rel-L2 `0.01581736014170759`, one-minus-cos `0.0001249272969429116`, norm-ratio error `0.0007164463558021472`, sign mismatch `0.0031906064720508445`; all finite, no dead leaves. This passes every frozen gradient threshold.
- Real AdamW parameter delta: rel-L2 `0.09976461037537022`, one-minus-cos `0.004976451772042978`, norm-ratio error `0.000007423376985782326`, sign mismatch `0.0037967687086651125`; all finite, no dead leaves. Rel-L2 and direction exceed the frozen update thresholds.
- Mechanism-only timings with cold compilation and full-tree capture: reverse `330.582s -> 227.621s`; global step `777.81s -> 673.19s`. These are not performance-admission numbers.
- Comparator verdict: `NUMERICAL_REJECT`. Result SHA256 `28f576472dd3c625376d9f93dfa1627e87f31e38631ea1418473a57d2c469454`; driver SHA256 `ca894890e9414474211ce04a9366a7eb68dd67df21446836d2401a63a5417f67`; top-level manifest SHA256 `12e5a28c8b4c6d96580e2b08eb68b3f5cffc567a48ec5ffee230653051d9b0ff`; every manifest entry verifies.
- Evidence root: `/mnt/disks/tunix-data/logp_probe_1host/p61_dp4_numerical_ab_p66ab1_20260825t1813z/`.
- Interpretation: P59 remains gradient-correct under the accepted FP64/Tier-1 policy, but is not a trajectory-preserving optimization. First-step AdamW acts close to a sign transform, amplifying small gradient ordering differences into a roughly 10% relative update-vector difference. This does not by itself prove non-convergence.
- Next: keep P59 out of the causal `Z-min` debug arm and compare the ordinary and segmented backward programs on one hash-bound batch.

## 2026-08-25 — CHECKPOINT: P66.3 ordinary/segmented carrier implemented

- Added an exact DP4xTP1, deterministic, backward-no-commit P66 carrier with two closed arms: `ordinary` invokes the stock whole-model `nnx.value_and_grad`; `segmented` invokes the canonical grouped reverse with no gradient sink. Both bypass AdamW and the persistent gradient accumulator.
- Both arms capture the complete model-before and normalized gradient trees, run the same 16 rank-major alignment checks plus the existing pre-alignment check, require P59 off, and prove sampled model/optimizer/accumulator/reference state plus train step are unchanged.
- Added a fail-closed arm classifier and cross-arm comparator. The latter requires 17/17 strict PASS per arm, identical seven hashes for all 16 groups, byte-exact full model-before trees, verified capture manifests, and the unchanged P61 Tier-1 gradient thresholds.
- Added immutable serial wrapper: ordinary completes before segmented; run labels and comparison bundle labels cannot be reused; full-tree D2H makes the run performance-ineligible.
- Host verification: P66/P61/P59/APC/flag-focused suite 46/46 PASS; Python/Bash syntax and `git diff --check` PASS; flag registry 380/380 with `FLAG_AUDIT_PASS`.
- Claim ceiling: implementation and host contracts only. Pinned image and one-host TPU arms have not run; no backward numerical verdict exists yet.

## 2026-08-25 — RESULT: P66.3 ordinary arm is an infeasible carrier

- Pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` passed the P59 overlay gate and the P66/P61 focused tests (9/9). Two earlier container invocations failed only because the worktree `.git` indirection was not mounted and then because Git safe-directory ownership was unset; the final ephemeral container mounted the real parent repository and set its own safe directory. Those are harness failures, not numerical runs.
- One-host ordinary label: `p66o1_20260825t1903z`; comparison label: `p66ab2_20260825t1903z`. The arm passed strict pre-alignment with `N_action=16127` and the same seven hashes as P66.2, then captured all 310 model-before leaves (`6,882,299,904` bytes).
- The cold whole-model `nnx.value_and_grad` compile failed before producing any gradient: `RESOURCE_EXHAUSTED`, HLO temporaries `450.29G` versus available HBM `95.74G`. The idle post-exception container was stopped without deleting its run directory. The segmented mate was never launched.
- Raw log SHA256: `72ab7e39fa5b458e55534355d74b80109b2308091afac083d726865248b00c0d`; pair-driver SHA256: `a8231597bf9b92b8af55a4852bd87b43e2fb6b556c4e5e20c579bdb382a7e252`.
- Verdict: `INCONCLUSIVE_CARRIER`, not a segmented-backward rejection. The failed run is frozen under `/mnt/disks/tunix-data/logp_probe_1host/p59_dp4_p66-ordinary_p66o1_20260825t1903z/`.

## 2026-08-25 — CHECKPOINT: target gradient explosion supersedes norm diagnosis

- Durable P62 target evidence places the first red at group-0 `engine_vjp`: loss cotangent `max_abs=0.0112169283`, `stable_norm=0.689952`; engine VJP `max_abs=5.79227764e21`, `stable_norm=5.38142010e22`. The trainer rank-local receipt preserves the same maximum and identifies embedding plus layer-0 parameters as the largest leaves.
- Attempt-7 GSM8K full logs report update-scale norms around `1e20`-`1e22` on every step. Historical ordinary-scale gradients and the independent P45/M15 non-finite failures make this a backward regression, not a legitimate finite-gradient distribution. P63 overflow-safe clipping is therefore downstream mitigation and cannot certify or repair the backward.
- Official JAX documentation confirms that `check_vma=False` disables out-spec replication checks and efficient reverse-mode VMA tracking, which otherwise avoids defensive `psum`s. That makes the P59 TP>1 outer/nested manual-axis composition a strong causal candidate, but not yet a verdict; `CANON_FIXED_AR_GATHER` remains the competing single-variable TP transpose candidate.
- Pre-registered the structural VMA probe and the one-host S/P/R full-Qwen TP4 discriminator in `phases/p66-3-vma-causal-bisection.md`. No repair, optimizer commit, source commit, or push has occurred.

## 2026-08-25 — CHECKPOINT: P66.3 G0 repair and first G1 harness red

- G0 exposed a real structural defect before any target inference: enabling
  `check_vma=True` rejected the historical fixed-head transpose and then the
  projection transpose because replicated parameter/state arguments were
  being treated as replicated cotangents even though P59 needs one varying
  cotangent per data rank.
- The default-off diagnostic repair marks parameter/state inputs varying over
  the manual data axis and lets VMA own the TP activation transpose; manually
  summing the TP input cotangent in that checked path would duplicate the
  transpose reduction. The final pinned image passes the installed real-shim
  TP4/TP8 suite in both historical VMA-off and repaired VMA-on modes, including
  fixed head, projection, attention, report adjoint, fixed reducer, exact
  serial/parallel numerics, and 2x36/36 manifests.
- Added a competing padding-row hypothesis without accepting it: every G1
  layer/chunk now records residual RMS and post-pullback `dhidden` split by
  real/padding rows. Small padding RMS is causal only if the padding cotangent
  becomes nonzero; all four arms must emit the 28-layer profile.
- Final host gates before G1: P59 37/37, P66 6/6, P61 6/6, APC 31/31,
  workload 63/63, flag registry 381/381, Python/Bash syntax and diff hygiene
  PASS.
- Frozen failed labels: serial `p66s2_20260825t2005z`, campaign
  `p66g1_20260825t2005z`. The wrapper verified four direct TPU devices and
  DP1xTP4, then the GSM8K entrypoint rejected workload
  `gsm8k-p66-dp1-tp4` from its older deterministic-A/B whitelist. This is
  `HARNESS_ADMISSION_FAIL`: no pre-alignment, model load, backward, alignment
  verdict, or optimizer commit occurred. The failed evidence directories are
  retained.
- Repaired only that exact workload admission and made missing-artifact
  classification fail closed instead of raising its own traceback. Focused
  P66 7/7 and GSM8K/workload adjacency 75/75 pass. Fresh labels are required
  for the rerun.
- Frozen second harness labels: serial `p66s3_20260825t2010z`, campaign
  `p66g2_20260825t2010z`. This attempt passed deterministic rollout and strict
  pre-alignment (`1/1`, `N_action=3972`, zero differing bytes), then stopped
  before the first VJP because `_p32_group_spec` retained the older DP8/DP16
  or P59-DP4 proxy guard. The tracebacked container was stopped after its raw
  log became idle; there was no alignment FAIL, gradient, or optimizer commit.
- Added only the exact workload+arm+DP1xTP4 group-spec admission. P66 7/7,
  workload 63/63, compilation, and diff hygiene pass. Fresh labels remain
  mandatory.
- Frozen third campaign labels: serial `p66s4_20260825t2015z`, historical
  P59 `p66u3_20260825t2015z`, campaign `p66g3_20260825t2015z`. Serial is a
  valid result: 17/17 strict PASS, zero FAIL/commit, engine group-0 VJP norm
  `6.0506024`, mapped gradient norm `0.3781629`, and all 56 padding row-layer
  cotangents exactly zero even though the minimum padding residual RMS is
  `0.0346194`. This rejects the claim that small padding RMS alone caused the
  observed explosion on this real replay.
- The historical P59 arm matched pre-alignment and reached the installed
  fixed-head VJP, then the installed attention shim rejected the diagnostic
  unit-data context at layer 27 because its older local-attention admission
  required `data>1`. No numeric/profile receipt or optimizer commit exists;
  this is another proxy-carrier structural red, not the expected numerical
  red and not a P59 verdict.
- Added the unit-data attention exception only for the exact P66 workload,
  physical DP1xTP4, and the three registered parallel diagnostic arms. The
  installed-attention pinned-image probe now executes that exact DP1xTP4 VJP.
  Updated `MANIFEST.sha256`; host P59 37/37 and P66 7/7 pass. Both final
  VMA-off and VMA-on pinned-image gates pass with 2x36/36 manifests and
  `p66_unit_data_attention=1`. A same-runtime-source fresh four-arm campaign
  is still required; the earlier valid serial result is not reused across the
  shim change.
- Frozen fourth campaign labels: serial `p66s5_20260825t2035z`, historical
  P59 `p66u4_20260825t2035z`, campaign `p66g4_20260825t2035z`. Serial exactly
  reproduced the prior valid receipts (`engine_norm=6.0506024`, mapped
  `gradient_norm=0.3781629`, padding cotangent count zero, 17/17 strict PASS).
- Historical P59 now completed all 28 real layer pullbacks. Padding cotangents
  remained exactly zero, while real-row `dhidden` grew from ordinary deep-layer
  scale to `4.2658e19` at layer 0. It then stopped at the rank-staged gradient
  assembly guard because the diagnostic data axis has size one. This is direct
  evidence against the padding mechanism and for a repeated P59 TP transpose
  amplification, but the arm still lacks the registered full-tree numeric
  receipt.
- Added only the exact P66 parallel-arm exception that preserves the leading
  staged rank axis of length one through full-state assembly. P66 7/7, P59
  37/37, compilation, and diff hygiene pass. No production DP>1 path changed;
  final pinned-image and fresh same-tree G1 reruns remain required.

## 2026-08-25 — RESULT: G1 historical P59 is numerically red

- Serial evidence root:
  `/mnt/disks/tunix-data/logp_probe_1host/p66_tp4_tp4-serial_p66s7_20260825t2112z/`.
  It has 17/17 strict PASS, zero optimizer commits, engine-VJP stable norm
  `6.0506024`, mapped gradient norm `0.3781629`, and zero padding-row
  cotangents in all 56 layer/chunk observations.
- Historical-P59 evidence root:
  `/mnt/disks/tunix-data/logp_probe_1host/p66_tp4_tp4-p59-old_p66u6_20260825t2112z/`.
  It has the same frozen hashes and 17/17 strict PASS. Real-row `dhidden`
  grows from `0.0128` at layer 23 through `153` at layer 17 and `2.338e8` at
  layer 10 to `4.2658e19` at layer 0. The 310-leaf engine gradient has max abs
  `1.833e20` and stable norm `1.5402378e21`; every padding-row cotangent is
  still zero. Verdict: `EXPECTED_RED`, no optimizer commit.
- Interpretation: H3 padding/small-RMS is rejected for this payload; old P59
  TP ownership/transpose semantics are causal. This does not yet prove the
  checked-VMA candidate correct.

## 2026-08-25 — CHECKPOINT: P candidate structural red and P64 reconciliation

- Pinned-image checked-VMA composition passed the installed real-shim TP4/TP8
  suite with `manifests=2x36/36` and zero optimizer commits before the latest
  bridge edit.
- P arm `p66p8_20260825t2155z` passed strict pre-alignment (`N_action=3972`,
  zero differing bytes) and reached the final-norm pullback, then failed before
  numerical receipts: cotangent `bfloat16[256,2048]{V:data}` was presented to
  a pullback whose nested engine map had declared `bfloat16[256,2048]` without
  VMA. Evidence is frozen under
  `/mnt/disks/tunix-data/logp_probe_1host/p66_tp4_tp4-p59_p66p8_20260825t2155z/`;
  optimizer commits remain zero.
- Implemented a default-off P66-only candidate that directly invokes the
  already-local TP engine body inside the outer checked map. This avoids a
  redundant nested map that previously consumed the outer partitions and then
  erased both data/model specs to `None`. Host P66 7/7 and P59 37/37 pass; the
  changed bridge is not yet pinned-image or TPU verified.
- Fetched upstream without modifying the dirty worktree. The newest P64
  evidence is commit `1406cc2d`; the current fetched remote tip is now
  `9f91d930`, whose three later commits only add/correct M15 Attempt-5
  evidence and handoff. Direct pull is deferred because both trees modify
  runtime adapter/learner/FLAGS files and merging now would invalidate
  existing source receipts.
- P64 target evidence: strict DP8xTP8 pre-alignment PASS for 46,276 actions;
  finite loss and group-input cotangents; first non-finite at group-0
  `engine_vjp`, leaf 1, rank 3. Exact-zero input ranks emit exact zero and all
  nonzero-input ranks emit NaN. Log/receipt SHA-256 values are registered in
  the active phase file. This supports the engine-VJP focus but does not
  certify the candidate repair.
- Next: rerun the affected pinned-image gate on the newest bridge, then launch
  a fresh-label P arm only. R remains gated on a finite P numerical receipt.

## 2026-08-25 — CHECKPOINT: checked-VMA reaches the fixed TP sum boundary

- P attempts `p66p9_20260825t2210z`, `p66p10_20260825t2220z`, and
  `p66p11_20260825t2232z` each passed strict pre-alignment with zero differing
  bytes and performed zero optimizer commits. They are retained structural
  RED evidence, not numerical verdicts.
- P9 showed that an unconditional invariant-to-varying pcast was being applied
  to values already varying on data; the marker is now idempotent and rejects
  only reduced/unreduced inputs.
- P10 progressed through final norm and q/k/v projections, then found the
  stock ragged-paged-attention Pallas output `ShapeDtypeStruct` discarded the
  active manual-axis type. A narrow default-off overlay now inherits the
  query/cache output VMA. Both qwen1p7b and qwen8b_tp8 installs verify all
  37 files.
- P11 progressed through the patched attention path to the real layer-27 VJP.
  It rejected `dnext_hidden bfloat16[256,2048]{V:data}` because the actual
  layer output was `bfloat16[256,2048]{V:(data,model)}`. The fixed TP
  all-gather/ring reducers create identical completed sums on every model
  rank, but elementwise local addition cannot prove that fact to VMA. Feeding
  the invariant cotangent as four independent varying copies would recreate a
  per-layer TP amplification, so the cotangent is not relabeled.
- The current P66-only candidate applies `lax.pmean` to the already-identical
  completed TP sum. This leaves its BF16/FP32 value exact on admitted power-of-
  two TP4/TP8 while registering an invariant output and the correct transpose.
  Production/default-off behavior is unchanged.
- Verified after this edit: P66 host suite 4/4, P59 host suite 37/37,
  compilation and diff hygiene pass; VMA-on pinned-image installed fixed-head,
  projection, and attention composition passes on DP2xTP4 and DP2xTP8 with
  `manifests=2x37/37` and zero optimizer commits.
- Next: fresh-label P only. R remains blocked until P reaches a finite
  full-Qwen numerical receipt.

## 2026-08-25 — CHECKPOINT: P12 closes all transformer layers

- P12 label `p66p12_20260825t2246z`, evidence root
  `/mnt/disks/tunix-data/logp_probe_1host/p66_tp4_tp4-p59_p66p12_20260825t2246z/`.
  It passed strict pre-alignment (`N_action=3972`, zero differing bytes),
  performed zero optimizer commits, and traversed the real final norm plus all
  28 checked-VMA transformer pullbacks.
- Its first red moved to the final input-embedding VJP: embedding hidden was
  typed `{V:(data,model)}` while the propagated logical cotangent was
  `{V:data}`. The fixed vocab ppermute ring, like the projection reducer,
  creates an identical completed sum on every TP rank without informing VMA.
- Applied the same P66-only invariant boundary to that already-identical
  embedding result. Relabeling the cotangent varying remains forbidden because
  it would duplicate the logical loss output.
- After regenerating the pinned overlay, both qwen1p7b and qwen8b_tp8 install
  manifests are 37/37. P66 4/4, P59 37/37, diff hygiene, and the complete
  VMA-on pinned-image TP4/TP8 gate pass with zero commits.
- Fresh P13 is running. It is the first candidate able to reach the full-tree
  numerical receipt; R remains gated on that result.

## 2026-08-25 — RESULT: P13 checked-VMA full-Qwen gradient is ordinary

- P13 label `p66p13_20260825t2256z`, evidence root
  `/mnt/disks/tunix-data/logp_probe_1host/p66_tp4_tp4-p59_p66p13_20260825t2256z/`.
- Classification PASS: 17/17 strict Zero-TIM, zero FAIL, identical frozen
  input hashes, zero optimizer commits, and no model/optimizer/accumulator/
  reference state change.
- Full engine VJP: all 310 leaves finite and nonzero, max abs `0.58984375`,
  stable norm `6.05732584`. Mapped gradient stable norm `0.37858307`.
  Serial S was `6.0506024` / `0.3781629`; the P-to-S norm differences are
  `0.11112%` and `0.11111%`, respectively. Historical U was
  `1.5402378e21`, so the repeated TP amplification is removed.
- Layer profile remains bounded (`0.00218` to `0.58984` component max abs),
  and all 56 row-layer observations retain zero padding cotangent despite the
  layer-0 padding RMS `0.0346194`.
- The post-training `weakref` cleanup AttributeError is emitted only after
  `TRAINING_DONE`, docker exit 0, signed classification PASS, and all evidence
  writes; it is not a backward or classifier failure.
- Pre-registered R (`CANON_FIXED_AR_GATHER=0`, every other P flag unchanged)
  is active. One-host P is a causal KEEP candidate, not target certification.

## 2026-08-25 — RESULT: R matches P; fixed gather acquitted

- R label `p66r1_20260825t2304z`, evidence root
  `/mnt/disks/tunix-data/logp_probe_1host/p66_tp4_tp4-gather-off_p66r1_20260825t2304z/`.
- Classification PASS: 17/17 strict Zero-TIM, zero FAIL/commit, all 310 leaves
  finite/nonzero, engine norm `6.05732584`, mapped norm `0.37858307`.
- P and R have exact-equal seven-hash sequences, model-before samples, engine
  and mapped gradient summaries, sampled gradient hashes, and (after removing
  the arm label) layerwise and padding profiles. Therefore disabling
  `CANON_FIXED_AR_GATHER` changes no gradient evidence under repaired VMA; H2
  is rejected as the cause of the `1e21` regression.
- P single diagnostic backward was `158.016s`; R was `165.922s`. This single
  compile-bearing observation is directionally consistent with retaining
  gather but is not a steady-state performance verdict.
- Next: fresh current-source S then U to close the four-arm source freeze. P/R
  already share the final runtime tree.

## 2026-08-25 — RESULT: final-source S control is stable

- S label `p66s8_20260825t2313z`, evidence root
  `/mnt/disks/tunix-data/logp_probe_1host/p66_tp4_tp4-serial_p66s8_20260825t2313z/`.
- Classification PASS: 17/17 strict, zero FAIL/commit, engine norm
  `6.050602436`, mapped norm `0.378162891`, and `134.460s` single diagnostic
  backward. Both norms exactly reproduce the earlier S control at recorded
  precision.
- S, P, and R have exact-equal seven input hashes and model-before samples.
  Current-source U is the final expected-red negative control below.

## 2026-08-25 — RESULT: final-source U reproduces the unsafe transpose

- U label `p66u7_20260825t2320z`, evidence root
  `/mnt/disks/tunix-data/logp_probe_1host/p66_tp4_tp4-p59-old_p66u7_20260825t2320z/`.
- Strict pre-alignment passes with `N_action=3972`, zero differing bytes, and
  the same four-arm input hashes. All 56 padding row/layer cotangent records
  remain zero, while real-row `dhidden` reaches `4.2658096e19` at layer 0.
- The 310-leaf engine gradient is finite but unsafe: max abs
  `1.8331452e20`, stable norm `1.5402378e21`, and naive FP32 norm `inf`.
  The pre-registered `grad_norm > 1e6` fatal sentinel stops the arm before an
  optimizer commit. The retained classification is `EXPECTED_RED`, with no
  alignment failure or classifier reason.
- Because U fails closed before the final report, it has no `update.json` or
  model-before sample. The causal contract uses U's matching signed
  pre-alignment plus the same frozen runtime tree; exact model-before/group
  comparisons are asserted only for the three successful S/P/R arms.
- Classification SHA-256:
  `5d50bad5c99e620ea8ae11e5daa8d2c059dcfce7ff546f2f0dfb944960b74d8f`;
  pre-alignment SHA-256:
  `ff52de9a12a71725a373b6950413ddc08e58b0da724edcec3c629bcb3fa9e208`.

## 2026-08-25 — RESULT: four-arm G1 verdict supports H1 VMA

- The pre-registered classifier returns `H1_VMA_SUPPORTED` with no contract
  reasons. It verifies same-input pre-alignment across S/U/P/R, same S/P/R
  group hashes and model-before samples, zero optimizer commits, S/P/R PASS,
  and U `EXPECTED_RED`.
- S engine/mapped norms are `6.050602436` / `0.378162891`; repaired P and
  gather-off R are `6.057325840` / `0.378583074`. P and R captured gradient
  evidence is exact; each mapped-norm ratio to S is `1.0011111163`. Historical
  U is `1.5402378e21` and fails closed.
- Verdict: the old P59 TP composition's erased VMA/replication ownership is
  the supported cause of the repeated transpose amplification. The fixed
  gather is acquitted, and padding/small-RMS is rejected for this replay.
- Generated classifier receipt SHA-256:
  `be5a160396474666fa214658faadd120dd337ccf16a2705132a3bfcec8c67c8a`.
  Its immutable source classifications are S
  `335b4c9ee134459ed4360dba2aed7604cad736fed7de124990602fa68945e90c`,
  U `5d50bad5c99e620ea8ae11e5daa8d2c059dcfce7ff546f2f0dfb944960b74d8f`,
  P `d317c6616eef697675e79f9e0dabcb8d228d569aa906b8d7b291cfa3bf5558d4`,
  and R `839492e8c5ef77217242da73dcf2304a23beb8b60c08974a26f6c5c2bee97904`.
- Claim ceiling: this completes the one-host DP1xTP4 full-Qwen group-0 causal
  gate. It does not certify the signed P64 DP8xTP8 target, a real optimizer
  update, convergence, or production performance. G2 requires separate launch
  approval.

## 2026-08-26 — RESULT: final-source G1.5 same-point oracle passes

- Implemented the default-unreachable `tp4-vma-oracle` arm and isolated its
  bounded numerical comparator in `tunix/rl/p66_vjp_oracle.py`. The checked-VMA
  candidate is always computed first; the ordinary serial pullback sees the
  same state/input/cache/cotangent and its result is never fed into reverse.
  The arm compares head, norm, layers 27/14/0, and embedding parameter plus
  activation/cache cotangents. A normal-value perturbation is a mandatory
  negative control.
- Added fail-closed single-arm and P13-pair classifiers. The pair gate compares
  exact frozen input/model evidence and exact candidate engine/gradient sample,
  layer profile, and row profile; input drift is `INCONCLUSIVE_INPUT_MISMATCH`
  and candidate drift is `FAIL_OBSERVER_RED`.
- Host after final cleanup: P66 16/16, P59 37/37, flag audit, Python/Bash
  syntax, manifest verification, and `git diff --check` PASS. The first pinned
  overlay generation correctly stopped on a malformed attention patch hunk;
  hunk counts were repaired, actual generated SHA values registered, and the
  complete final-source pinned-image gate passed on immutable image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  with `manifests=2x37/37` and P66 unit-data attention coverage for both P and
  oracle arms.
- The first G1.5 run `p66o1_20260825t2357z` passed numerically, but a subsequent
  deletion of duplicate dead adapter code changed runtime source bytes. It is
  preserved and explicitly superseded rather than treated as the final-tree
  result.
- Final-source run label `p66o2_20260826t0010z`; evidence root
  `/mnt/disks/tunix-data/logp_probe_1host/p66_tp4_tp4-vma-oracle_p66o2_20260826t0010z/`.
  Its carrier records adapter SHA
  `d5f013912b236373bc6e0ad4a3f105675d646d69d27f030e8f41a623eb9177af`
  and oracle-module SHA
  `c784eae8fbeb30f5e6a74385b28fcb67d0c66c5e86adee00fa29b2daa8a50003`.
- Classification PASS: 17/17 strict, zero FAIL/commit, engine norm
  `6.05732584`, mapped norm `0.37858307`, all 310 leaves finite/nonzero, and
  all 56 padding-row cotangent observations zero. Six endpoint rel-L2 values
  are head `5.7114e-7`, norm `0`, layer27 `9.4928e-4`, layer14 `3.3325e-3`,
  layer0 `5.2568e-3`, embed `0`; all are below the frozen `4e-2` cap and all
  other frozen metrics pass. Norm/embed are array-exact.
- P13 observer-neutrality PASS: zero input reasons and zero observer reasons.
  `SHA256SUMS` verifies from the repository worktree. Final SHA values:
  raw `9acac40782e13ac7502df50fa2a962e8f99370b364bfab769e8e7bd2142623a3`,
  classification `cfc78137211609997860cb6ad251be1464b10ffde28e736ace720e14ebc6b8b5`,
  update `31a485e3d019a42701488cb81ba000cd41f27aee618dbe0d155cd978b820ef96`,
  neutrality `d50d093754d07e9bc9bbe2d6d8429664808484e45586c0b0953d8530ba2be366`.
- Verdict: G1.5 COMPLETE and source-freeze review admitted. G2 signed P64
  DP8xTP8 replay remains NOT RUN because no target launch was authorized. No
  optimizer update, convergence run, commit, push, or production promotion was
  performed.
