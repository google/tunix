# P44.10 — r05 Mosaic matmul geometry repair

- Status: local and real one-host v5p gates passed; unpublished; remote target pending

## Evidence and diagnosis

- Pulled baseline: `3ec5fd7c3074844c62d3a9ff2c95179449a66129`.
- Archived source run: `p44r05`, target source
  `115ef8144a873b5f108ec4b52aafc959032c3f43`.
- Raw log: `debug_logs/p44_p44r05_deepswe_256_parity.raw.log`, SHA-256
  `51b1674c3c3b2d42e6738a0d66dce3a5f222bbd2c52a296ce75379488e181168`.
- The run emitted `F=1216 Fp=1280 feature_padded=1` through all 36 layers,
  proving P44.9, then Mosaic rejected `canon_matmul_bm128_bn64_bk64` because
  the trailing TPU block dimension was 64 rather than 128-aligned.
- Qwen3-4B gate/up projections have semantic local `N=1216`; the down
  projection has semantic local `K=1216`. Therefore `BK=128` alone does not
  repair every projection: the wrapper must cover both contracted K and
  output N.

## Repair

1. Pin the Qwen3-4B overlay to `BN=128`, `BK=128`.
2. Admit only model-pinned `MATMUL_K_PADDING={1216:1280}` and
   `MATMUL_N_PADDING={1216:1280}`; reject unknown non-aligned widths.
3. Zero-pad x/y contracted K together, pad weight output N, call the unchanged
   Pallas matmul, and slice only output N back to the semantic shape.
4. Reproduce the padded-K BK128 accumulation order in the canonical VJP.
5. Expose K/Kp/N/Np in PATHTRACE and require both MLP directions in the P44
   fail-closed classifier.

## Gates

- P44 CPU: PASS, 36 cases, marker
  `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS`.
- P44 exact-image: PASS, overlay 29/29, two learner tests, exact interpret
  forward/VJP for both padding directions, marker
  `P44_EXACT_IMAGE_CPU_PASS overlay=qwen4b`.
- Adjacent DeepSWE: P43 22 cases, P39 15 cases, P34 static 10 suites plus
  trajectory/update gates: PASS.
- Adjacent overlays: Qwen3-8B and Qwen3-32B exact-image gates: PASS, 29/29.
- Real one-host v5p command:

  ```bash
  bash canon-zero-tim/tests/p44_deepswe_qwen4b_parity/run_onehost_v5p.sh \
    sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
  ```

  The privileged pinned-image container exposed exactly four TPU v5 devices.
  At the r05 semantic M=4096, real Pallas forward and promoted custom VJP were
  exactly equal to the canonical implementation for all five unique local
  projection shapes: q `2560x512`, k/v `2560x128`, o `512x2560`, gate/up
  `2560x1216` padded to `N=1280`, and down `1216x2560` padded to `K=1280`.
  Both unregistered-width negative controls passed. Markers:
  `MATMUL_DIM_PADDING_PASS mode=tpu cases=5/5 ... devices=4` and
  `P44_ONEHOST_V5P_MATMUL_PASS model=qwen4b devices=4`.

## Boundaries

- The one-host result proves only direct-attached Qwen3-4B matmul lowering and
  VJP construction at target-shaped M. It does not prove model initialization,
  a real R2E trajectory, backward across the model, TP8, Pathways, role
  separation, DP4/DP16, 64/256-chip behavior, optimizer placement, or a
  completed P44 stage.
- The fixed image has no importable `r2egym`, no `kubectl`, and no kubeconfig.
  Its local Qwen3-4B Hugging Face cache contains tokenizer artifacts rather
  than a complete initial-weight snapshot. Full one-host DeepSWE E2E remains
  `BLOCKED_REAL_ENVIRONMENT`; no fake environment was substituted.
- The P45 CPU suite was also attempted and was inconclusive on the bare host
  because optional `datasets` and `metrax` packages are absent. P45 is not a
  DeepSWE release gate and no P45 source was changed.
- No commit, push, remote launch, precision/loss/optimizer change, or main
  branch action occurred in this phase.

The R2E/model prerequisite statement describes this phase's pinned-image
inventory only. P44.11 later located complete prerequisites in separate host
paths and executed the real one-host rollout and backward-no-commit chain; it
does not change the narrower P44.10 kernel-gate claim.

## Next

After explicit commit/push authorization, publish only to the operator branch,
read back its exact head, and run a new rollout-only attempt named `p44r06` or
later. Require both matmul-padding PATHTRACE directions before classification.
