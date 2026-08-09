# P35.1 local gate artifact

Date: 2026-08-09 UTC

Source commit before local changes:
`ad309a810e35121d7d25db67c32c2712d9f8e086`

Exact-image command:

```bash
sudo docker run --rm \
  -v "$PWD:/workspace:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=cpu \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  bash -lc "python3 -m unittest discover -s tests/rl -p alignment_test.py && \
    python3 -m unittest discover -s canon-zero-tim/tests/p35_envelope -p 'test_*.py'"
```

Observed result:

- `tests/rl/alignment_test.py`: 13 tests, PASS.
- `tests/rl/rollout/vllm_rollout_canonical_test.py`: 6 tests, PASS.
- `canon-zero-tim/tests/p35_envelope`: 5 tests, PASS.
- One-ULP and signed-zero controls produced nonzero bitwise differences.
- A drift outside the action mask changed full hashes while the action-only boundary remained
  exact.
- Missing-arm, red-attestation, ineffective-negative-control and hash/count inconsistency inputs
  were classified `INCONCLUSIVE`.
- `git diff --check`: PASS.
- Python compilation: PASS.
- Executable English-only scan: PASS.

No TPU, cloud resource, W&B, Hugging Face token, commit or push operation occurred.
