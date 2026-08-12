# P38.2g7 local gate — 2026-08-12 UTC

## Scope

This is local construction evidence for the P38-only four-prompt diagnostic
consumer batch. It is not a target Pathways result and makes no numerical
carrier, backward, optimizer, or training claim.

## Pinned environment

- Image reference: `tunix_frozenlake_image:vllm-tpu0.25.0`
- Image ID:
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`

## Results

- P38 renderer unit gate: 6/6 PASS.
- P38 outer postflight shell gate: PASS.
- Qwen3-1.7B exact-image overlay: 20/20 PASS; manifest 29/29 exact.
- Qwen3-8B exact-image overlay: 20/20 PASS; manifest 29/29 exact.
- Adjacent P45 exact-image CPU gate: 83 workload tests and 31 alignment tests
  PASS; Qwen3-8B TP8 seven-site contract, import, and canonical forward/VJP
  probes PASS.
- Python compilation, shell syntax, and `git diff --check`: PASS.

## Target requirement

P38s8 must still prove the runtime marker
`[CANON_P38] DIAGNOSTIC_BATCH_CONTRACT ... trajectories=32 dp=16 verdict=PASS`
and return the complete stock-only serving capture through outer postflight.
