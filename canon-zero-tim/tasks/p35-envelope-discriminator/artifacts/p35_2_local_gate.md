# P35.2 local producer gate artifact

Date: 2026-08-09 UTC

Source commit before local changes:
`c660134bababc9123e6820c1f241246cfbf602a7`

Response-contract repair reviewed against:
`b8d3ad8dc84022e88f4b22a919ba60d46fea64c9`

Pinned-image CPU gate:

```bash
sudo docker run --rm \
  -v "$PWD:/workspace:ro" -w /workspace \
  -e PYTHONDONTWRITEBYTECODE=1 -e JAX_PLATFORMS=cpu \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  bash canon-zero-tim/tests/p33_workloads/run_cpu.sh
```

Observed result:

- P33 workload suites: 59 tests, PASS.
- Alignment: 13 tests, PASS.
- Native rollout contracts: 9 tests, PASS.
- P35 producer helpers: 8 tests, PASS, including a bad active page-id rejection.
- P35 classifier and renderer: 11 tests, PASS.
- Canonical adapter focused tests: 2 tests, PASS.
- Explicit DP-axis test: 1 test, PASS.
- P35 preflight and renderer bridge accepted response 256 and rejected response 64 and 65.
- P35 postflight accepted only expected diagnostic exit 1 and rejected three malformed controls.
- Final line: `[P33.WORKLOAD] CPU_GATE PASS workloads=2 p35_postflight=1`.

Pinned overlay command:

```bash
bash canon-zero-tim/tests/p33_workloads/run_exact_image.sh \
  tunix_frozenlake_image:vllm-tpu0.25.0
```

Observed result:

- qwen1p7b: 29 manifest entries match; 10/10 prompt/decode chunk tests PASS.
- qwen8b: 29 manifest entries match; 10/10 prompt/decode chunk tests PASS.
- Final line: `P33_EXACT_IMAGE_PASS decode_chunk_cases=5 prompt_chunk_cases=5 overlays=2`.

Additional exact-weight control:

```bash
sudo docker run --rm \
  -v "$PWD:/workspace:ro" -w /workspace \
  -e PYTHONDONTWRITEBYTECODE=1 -e JAX_PLATFORMS=cpu \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  python3 -m unittest discover -s tests/rl \
    -p canonical_qwen3_adapter_test.py -k bitwise
```

Result: 1/1 PASS, including signed-zero and one-bit negative behavior.

P35 attempt r21 reached rollout but failed in native reference Splash before the producer. It
emitted no P35 report or classification, so the target carrier classification remains NOT RUN.
No W&B/Hugging Face credential change occurred in the response-contract repair.

Rollback: leave `CANON_P35_ENVELOPE` unset and do not render/apply the P35 JobSet.
