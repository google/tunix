# P58.8 — P59 TP4/TP8 nested-mesh admission repair

Status: active. The real installed fixed-head/projection P59 composition passes
in the pinned image for forced DP2 x TP4 and DP2 x TP8 on the fresh four-CL
tree rebuilt on `ccbcf572`, including operand barriers. Commit approval and
hardware target certification remain open.

## Goal

Repair the two first-red bootstrap failures fetched from the V1 Phase4 full
campaign without weakening P59 gradient correctness, strict Zero-TIM, or the
P57 signed workload identity:

1. GSM8K DP16 x TP4 entered P59 head pullback, then failed because an outer
   trainer `dp/tp` AbstractMesh with DP manual and TP automatic could not nest
   the fixed-head engine's concrete six-axis `data/.../model/...` shard_map.
2. FrozenLake DP8 x TP8 stopped before model/trainer execution because the
   generic workload W&B project differed from the exact signed P57 Zero/full
   profile project.

The immutable incoming evidence commits are `f7d22555e28270fef8128c287948a5b83ca2cc7d`
and `2a89eef35199429dde5c8d4330dc87ebb4b902bb`. Neither failed run reached an
optimizer commit. The source branch was fetched for analysis but not merged
into this dirty implementation worktree because both commits contain evidence
only.

## Design

- Preserve the certified TP1 P59 path.
- For TP>1, construct a two-axis `data/model` view over the exact live engine
  device order, make both real axes manual in the outer P59 map, and localize
  compatible nested engine maps onto that current AbstractMesh.
- Consume already-applied data/model partition specs exactly once while
  retaining the named `model` axis for fixed-order TP collectives. Bind only
  proven-unit auxiliary engine axes. Reject unknown axes, topology/device-order
  changes, non-unit auxiliary axes, or any unexpected axis transition.
- Relabel outputs back to the trainer `dp/tp` mesh only after the compiled
  boundary, with shape, topology, and device-order checks.
- Admit TP-local fixed-head and fused-linear output boundaries only while P59
  is explicitly enabled and TP is non-unit; ordinary global boundaries remain
  unchanged.
- Override the generic FrozenLake W&B project only for the exact signed P57
  profile, Zero arm, training kind, and DP8 x TP8 workload. Wrong arm/profile
  combinations still fail closed.

This changes the carrier topology, not the numerical acceptance policy. P59
still claims ordinary-JAX FP64 gradient correctness and does not claim the
same AdamW trajectory as historical serial backward.

## Gates

1. Forced 16-device CPU tests exercise DP2 x TP4 and DP2 x TP8 through both
   legacy and modern nested shard-map APIs, TP-column placement, and a
   fixed-order `model` all-gather/sum.
2. Existing P59 DP2 x TP2 numerical, TP1 nested-map, unknown-axis negative,
   and DP4 trainer-mesh tests remain green.
3. P57 exact-profile W&B positive and mismatched-arm negative controls pass.
4. Full P59/P57/V1 host suites, manifest installation, flag audit, syntax, and
   diff hygiene pass.
5. Both P58 and V1 exact-image gates must execute the installed fixed-head and
   projection shims under P59, not only a synthetic nested map. Positive and
   negative controls must prove local/global output shapes and unchanged
   device-index maps.
6. A bounded no-commit gate must execute the modified fixed-head and installed
   projection VJPs through rank-parallel assembly, report adjoint, and the fixed
   reducer; compare serial/parallel leaves plus an FP64 oracle and prove ordinary
   global output placement is unchanged. Production fixed-head registration
   starts at TP4, so this gate is forced CPU DP2 x TP4/TP8. The available
   four-chip host cannot form DP2 x TP4; no artificial TP2 geometry is admitted
   and no one-host TPU composition PASS is claimed.
7. Reconstruct four independent CLs on exact remote tip `ccbcf572`:
   P59 mesh/shims/tests; P57 W&B; P58.6; P58.7.
8. Real DP16 x TP4 and DP8 x TP8 remain target gates. Any real alignment FAIL
   kills the candidate; a new mesh/fixed-head/bootstrap red stops at that
   boundary and is not reinterpreted as PASS.

## Result log

- Implemented by `tunix/rl/canonical_qwen3_adapter.py:323`,
  `canon-zero-tim/src/engine_shims/p38_fixed_lm_head.py:410`, and
  `canon-zero-tim/src/engine_shims/linear_p22xf.py:269`; the two shim hashes
  are updated in `canon-zero-tim/MANIFEST.sha256:4` and `:19`.
- W&B workload admission is repaired in `tunix/rl/dp_workloads.py:892` with its
  positive/negative regression in
  `canon-zero-tim/tests/p33_workloads/test_dp_workloads.py:465`.
- Construction verified by the focused TP4/TP8 topology test, existing P59 numerical and
  negative controls, P59 30/30, P57 128/128, V1 12/12, manifest install
  36/36, flag audit 366/366, and syntax/diff hygiene.
- Verified in pinned image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  by `P58_EXACT_IMAGE_CPU_PASS ... p59_tp4_tp8=2 p57_wandb=1 regressions=1`
  and
  `V1_HP_EXACT_IMAGE_PASS dp16_gathered=1 dp2tp2_parallel=2 p59_tp4_tp8=2 p57_wandb=1 perfetto_window=1 manifests=3`.
- A later bare-host direct import of the two focused tests was
  **INCONCLUSIVE**, not FAIL, because that shell lacks `metrax` and `datasets`;
  the official host P59/P57/V1 runners still pass 30/30, 128/128, and 12/12,
  and the dependency-complete pinned-image executions above are the focused
  topology/W&B verdicts.
- Correction after review: the earlier exact-image marker exercised an
  artificial nested shard-map and did not invoke the two modified
  installed-shim branches. It proved the mesh-carrier principle only; the
  replacement gate below supersedes that construction-only receipt.
- Verified after correction in pinned image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  by `run_tp4_tp8_installed_shim_exact_image.sh`: fixed-head P59-local VJP,
  report adjoint, fixed reducer, installed `linear_p22xf` local split/VJP, and
  ordinary-global negative placement all pass for forced DP2 x TP4 and DP2 x
  TP8, with zero optimizer commits and two 36/36 overlay manifests.
- The first TP8 installed-projection attempt was a useful red: staged weight
  gradients were exact, but BF16 accumulation of eight hidden-cotangent
  partials differed in 32/64 elements and was `0.5` max-abs from FP64. The
  serial result was exact for that probe. `linear_p22xf.py` and
  `p38_fixed_lm_head.py` now gather FP32 TP partials, add in ascending rank
  order with `optimization_barrier` on both operands, and cast once. Both TP4
  and TP8 match the serial probe exactly after this barrier hardening in both
  complete pinned-image gates.
- Full regression verified by P59 30/30, P57 128/128, V1 12/12, flags 366/366,
  manifest 36/36, `P58_EXACT_IMAGE_CPU_PASS ... p59_real_shim=4 ...`, and
  `V1_HP_EXACT_IMAGE_PASS ... p59_real_shim=4 ...`. This is installed-shim
  pinned-image admission, not DP16 x TP4 or DP8 x TP8 TPU target certification.
- The final fresh release worktree is based exactly on `24b1bbcf`; concern and hunk
  ownership, disadvantages, gates, and independent rollback are frozen in
  `RELEASE_CL_PLAN.md`. An unrelated APC B-arm receipt hardening hunk was
  excluded rather than silently creating a fifth numerical concern.
- The final fetch advanced by three commits. Two only add immutable P45/P58
  failure evidence; `24b1bbcf` changes this same P58 renderer to
  `maxRestarts=3` plus Pathways/GRPC keepalives. Those setup changes are
  integrated into both ordinary and Zero-HP P58 rendering rather than being
  overwritten by the release reconstruction.
- Verified after that integration by renderer 16/16, profile 4/4, full
  classifier 3/3, flag audit 366/366, diff hygiene, and the complete pinned
  P58 terminal marker with `onehost_xprof=1 zero_hp_full=1 apc=1
  p59_real_shim=4 p57_wandb=1`. A checksum dry-run against the previously
  tested release tree reports zero differences outside the upstream renderer,
  newly fetched evidence, and task documentation; the renderer's diff against
  `24b1bbcf` contains only the Zero-HP additive selector.
- Excluding that APC hunk first exposed a construction-only test-double red:
  the P58 vLLM mock omitted stock `RequestOutput.num_cached_tokens` while the
  baseline production reader requires it. The test double now carries
  `num_cached_tokens=0`; no production APC decision was reintroduced.
- Verified on the final release tree by APC 31/31, flag audit 366/366,
  `P58_EXACT_IMAGE_CPU_PASS ... onehost_xprof=1 zero_hp_full=1 apc=1
  p59_real_shim=4 p57_wandb=1 ...`, and `V1_HP_EXACT_IMAGE_PASS ...
  p59_real_shim=4 p57_wandb=1 ...` in the same pinned image. The failed first
  P58 rerun is preserved as construction evidence; it did not execute a real
  alignment gate or optimizer commit.
- Latest-tip correction: fetched operator tip
  `ccbcf572dc903bb1cce12f897cbdb05aec94922a` adds the P57 evaluation-cycle
  counter repair, final-only primary checkpoints, and lazy NumPy host-render
  import. The release was rebuilt by migrating only the prior dirty hunks and
  new files, so those upstream fixes remain intact. The P59 FP32 TP sum now
  mirrors the registered fixed reducer's operand barriers; the P57 signed W&B
  test also rejects the correct arm under a wrong profile. Host P59 30/30,
  current P57 136/136, V1 12/12, APC 31/31, flags 366/366, syntax and diff
  hygiene pass. Both complete pinned-image gates pass with the post-barrier
  TP4/TP8 real-shim markers; real TPU targets remain unverified.
- Still unverified because unavailable/unauthorized: approved commits, a real >=8-chip P59 TP
  composition, any optimizer commit with the repaired path, strict target
  alignment, target XProf/performance, and the four committed release CLs.

## Rollback

Leave `CANON_P59_RANK_PARALLEL_BACKWARD` off to make the new TP carrier
unreachable. Revert the P59 adapter/shim/test/manifest concern and the P57 W&B
admission concern independently. Preserve both incoming failed-run artifacts.
