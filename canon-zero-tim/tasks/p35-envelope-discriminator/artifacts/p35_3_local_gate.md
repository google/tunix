# P35.3 local gate

Date: 2026-08-09 UTC

## Scope

Default-off exact-input replay producer, fail-closed classifier, bounded 64-chip renderer wiring,
artifact SHA reporting and operator handoff. No cloud resource, TPU target run, backward,
optimizer update, cloud-resource lifecycle change, commit or push is included.

## Results

- Focused pinned-image P35 replay tests: PASS.
- Complete adapter/envelope suite: 40 PASS, 5 skipped.
- Complete P33/P35 CPU gate: PASS.
- Qwen3-1.7B and Qwen3-8B exact-image overlay/install gates: PASS.
- Real one-host v5p TP4 smoke: 4 devices `[0,1,2,3]`; replay plus signed-zero/one-bit exact
  equality controls: 2 PASS in 35.90s.
- Python AST, shell syntax and `git diff --check`: PASS.

One-host artifact:
`canon-zero-tim/debug_logs/p35_3_onehost_tp4_fixed.log`

SHA-256:
`56f110efcebc5d1c934335eacef643a904a7501a7cf67fe0d25c6420343ad9f2`

One-host command, executed from the disposable source copy after placing the reviewed files at
their repository-relative paths:

```bash
sudo docker run --rm --privileged --net=host \
  --name p35_3_onehost_smoke_fix \
  -v /mnt/disks/tunix-data/p35_3_smoke_337ce07c_20260809:/workspace:ro \
  -w /workspace \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e XLA_FLAGS=--xla_allow_excess_precision=false \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  python3 -m pytest -p no:cacheprovider -q \
  tests/rl/canonical_qwen3_adapter_test.py::CanonicalQwen3AdapterTest::test_p35_exact_replay_uses_captured_tensors_and_repeats_exactly \
  tests/rl/canonical_qwen3_adapter_test.py::CanonicalQwen3AdapterTest::test_bitwise_array_equality_detects_signed_zero_and_one_bit
```

The first code-review pass caught an insertion error that had moved the existing
`_bitwise_arrays_equal` return below the new summary helper. The final recorded TPU smoke and full
adapter suite ran only after restoring that return. No published or target code contains the
intermediate error.

These are local implementation gates only. P35.3 target numerical status remains NOT RUN until a
source-pinned 64-chip Attempt 0 returns the required JSON and classification.

## Rollback

Leave `CANON_P35_ENVELOPE` and `CANON_P35_EXACT_REPLAY` unset. The diagnostic code is then
unreachable; serving, training, precision, loss, sampling, gradients and optimizer behavior are
unchanged.
