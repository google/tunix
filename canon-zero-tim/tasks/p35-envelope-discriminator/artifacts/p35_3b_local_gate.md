# P35.3b local bounded-replay gate

Date: 2026-08-10 UTC

## Scope

Default-off P35 diagnostic execution only. The repair preserves the existing sampling and
canonical logprob program boundaries, writes preliminary A/B/C evidence before optional replay,
and serializes every captured replay record. It does not change production precision, sampling,
loss, backward, optimizer, checkpoint or cloud-resource state.

## Results

- Focused CPU replay and preliminary-path tests: 2 PASS.
- Complete P33/P35 CPU gate: PASS, including a negative replay-failure control that preserved and
  hashed the preliminary report, and a complete five-artifact success control.
- Qwen3-1.7B and Qwen3-8B exact-image overlay gates: 10/10 each; both 29-file manifests matched.
- One-host v5p TP4 smoke: four devices, four captured replay arms over two records, 8/8 record
  begin/complete pairs, replay repeat controls exact, signed-zero/one-bit controls effective,
  2 PASS in 34.72s. The first record intentionally has no action predictor.
- `git diff --check` and executable English-text scan: PASS.

The target-only fused-tail candidate was not admitted. Its CPU bitwise gate changed 178/256
target logprobs by about one ULP, so the implementation returned to the original numerical path.

## Reproduce

Complete CPU contracts:

```bash
sudo docker run --rm -v "$PWD:/workspace:ro" -w /workspace \
  -e PYTHONDONTWRITEBYTECODE=1 -e JAX_PLATFORMS=cpu \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  bash canon-zero-tim/tests/p33_workloads/run_cpu.sh
```

Exact-image overlays:

```bash
bash canon-zero-tim/tests/p33_workloads/run_exact_image.sh \
  tunix_frozenlake_image:vllm-tpu0.25.0
```

One-host TP4 smoke:

```bash
sudo docker run --rm --privileged --net=host \
  --name p35_3b_onehost_evidence_r3 \
  -v "$PWD:/workspace:ro" -w /workspace \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e XLA_FLAGS=--xla_allow_excess_precision=false \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  python3 -m pytest -p no:cacheprovider -q -s \
  tests/rl/canonical_qwen3_adapter_test.py::CanonicalQwen3AdapterTest::test_p35_exact_replay_uses_captured_tensors_and_repeats_exactly \
  tests/rl/canonical_qwen3_adapter_test.py::CanonicalQwen3AdapterTest::test_bitwise_array_equality_detects_signed_zero_and_one_bit
```

Raw one-host artifact:
`canon-zero-tim/debug_logs/p35_3b_onehost_tp4_r3.log`

SHA-256:
`2d2aca9c4c25bffd58e48a66ebe4177eeaba9068c8c86d9f983798b3121638b8`

Relevant source hashes for that smoke:

- `tunix/rl/canonical_qwen3_adapter.py`:
  `c7ea104e8263f8768a5f7275df2919213296f5baafe4a5c57e885d9ba47c8738`
- `tests/rl/canonical_qwen3_adapter_test.py`:
  `38675e4c54dff18dbd2b2ce97e2e72d648bd821b35e42e1f5e9bbe17f26fa179`

## Claim boundary

This proves local code mechanics and bitwise neutrality on the direct-attached host. It does not
prove that the 64-chip Pathways IFRT interruption is fixed, and it does not classify the target
adapter-envelope carrier. That requires a source-pinned r30 Attempt 0.

## Rollback

Leave `CANON_P35_ENVELOPE` and `CANON_P35_EXACT_REPLAY` unset. The diagnostic methods and report
paths are then unreachable.
