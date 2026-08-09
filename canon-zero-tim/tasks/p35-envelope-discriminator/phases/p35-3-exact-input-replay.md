# P35.3 exact-input replay

Status: locally complete; 64-chip target not run

## Question

Which sub-boundary inside the r28 B-serving versus C-adapter envelope first changes bits:
weight memory placement, physical metadata/cache construction, or the adapter's outer program?

## Arms

- B is the r28-style grouped native serving value.
- R0 directly replays captured B input tensors with live serving leaves.
- R1 repeats R0 with bitwise-equal trainer-mapped leaves.
- R2 keeps the mapped leaves and direct entry but uses adapter-generated metadata and a fresh
  adapter cache.
- R3 replays the unchanged production adapter envelope on the complete original batch.
- C is the original production adapter value from that batch.

## Hard controls

- B/R0 and R3/C must be bitwise exact or the measurement is inconclusive.
- B/C must reproduce the r28 red boundary.
- R0, R1 and R2 repeat measurements must be bitwise exact.
- Every attestation and the injected-drift negative control must pass.
- Missing reports, stale paths, unexpected exit codes or missing markers fail closed.
- The run stops before backward and never commits an optimizer update.

## Local verification

```bash
sudo docker run --rm -v "$PWD:/workspace:ro" -w /workspace \
  -e PYTHONDONTWRITEBYTECODE=1 -e JAX_PLATFORMS=cpu \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  bash canon-zero-tim/tests/p33_workloads/run_cpu.sh
bash canon-zero-tim/tests/p33_workloads/run_exact_image.sh \
  tunix_frozenlake_image:vllm-tpu0.25.0
git diff --check
```

The operator procedure and r29 evidence-return commands are frozen in
`canon-zero-tim/cluster/P35_ENVELOPE_HANDOFF.md`.

## Rollback

Leave `CANON_P35_EXACT_REPLAY` unset. All new calls remain unreachable in ordinary serving and
training. Preserve r28 and every later red or inconclusive artifact.
