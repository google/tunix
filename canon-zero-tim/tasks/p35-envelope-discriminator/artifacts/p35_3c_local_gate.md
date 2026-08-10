# P35.3c local gate

Date: 2026-08-10 UTC

## Scope

Default-off first-record stage localization for the r30 Pathways IFRT disconnect. The probe keeps
the existing model, logits, sampling and canonical target-logprob callables, inserts six explicit
readiness boundaries, appends fsynced JSONL evidence and stops after the first captured/live
record with `NO_NUMERICAL_VERDICT`.

This local gate contains no 64-chip run, Qwen target replay, backward, optimizer update, W&B
mutation, commit or push.

## Results

- Focused stage-classifier and renderer tests: 14 PASS.
- Focused pinned-image adapter stage probe: 1 PASS.
- Complete P33/P35 CPU contract: PASS; 59 P33, 13 alignment, 10 rollout, 14 envelope and 30 P35
  tests passed, followed by the tied-embedding, DP, negative-control and preflight gates.
- Qwen3-1.7B and Qwen3-8B exact-image overlay/install gates: both 29/29 manifest entries and
  10/10 prompt/decode chunk cases PASS.
- Four-logical-device CPU production-shape mechanics: 1 PASS.
- Real four-device one-host v5p TP4 production-shape mechanics: 1 PASS in 31.69 seconds. It
  materialized a synthetic local logits array `(256, 151936)`, 155,582,464 logical bytes, and
  completed the six ordered stages.
- Python AST, shell syntax, executable-language scan and `git diff --check`: PASS.

The real-device test is deliberately scoped: it validates stage instrumentation and the real
local M/vocabulary array shape on TP4. It does not run Qwen, Pathways or the failing 64-chip replay,
so it is not evidence that the target disconnect is fixed.

## Reproducible commands

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu python3 -m unittest \
  canon-zero-tim/tests/p35_envelope/test_classify_stage_probe.py \
  canon-zero-tim/tests/p35_envelope/test_render_p35_jobset.py

sudo docker run --rm -v "$PWD:/workspace:ro" -w /workspace \
  -e PYTHONDONTWRITEBYTECODE=1 -e JAX_PLATFORMS=cpu \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  python3 -m pytest -p no:cacheprovider -q \
  tests/rl/canonical_qwen3_adapter_test.py::CanonicalQwen3AdapterTest::test_p35_exact_replay_uses_captured_tensors_and_repeats_exactly

sudo docker run --rm -v "$PWD:/workspace:ro" -w /workspace \
  -e PYTHONDONTWRITEBYTECODE=1 -e JAX_PLATFORMS=cpu \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  bash canon-zero-tim/tests/p33_workloads/run_cpu.sh

bash canon-zero-tim/tests/p33_workloads/run_exact_image.sh \
  tunix_frozenlake_image:vllm-tpu0.25.0

sudo docker run --rm --privileged --net=host \
  --name p35_3c_full_vocab_tp4 \
  -v "$PWD:/workspace:ro" -w /workspace \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e XLA_FLAGS=--xla_allow_excess_precision=false \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  python3 -m pytest -p no:cacheprovider -q \
  tests/rl/canonical_qwen3_adapter_test.py::CanonicalQwen3AdapterTest::test_p35_stage_probe_full_vocab_array_shape
```

## Reviewed file hashes

```text
7b098106d52bf22751f13b8cb5cdfc6ea5fdc65964ba3d4522deca6fd471f393  tunix/rl/canonical_qwen3_adapter.py
4a47a676d987b48531209d3c291a9db188752e8d310db65cbcf632101cd2bc9f  tests/rl/canonical_qwen3_adapter_test.py
02d10e76bbb50351f09a60c8773e7d26a46c50f36aebc119a57eace7adcee491  canon-zero-tim/cluster/render_p35_jobset.py
5d8a1e142364e2be6f4e90e1294e45bd6c605ff33c1702a9031ecdbddd8c2c3a  canon-zero-tim/cluster/steps/90_run.sh
07c312779d82c1a48cf666b55f21167e97c37b5978441216012945a12b42a425  canon-zero-tim/tests/p35_envelope/classify_stage_probe.py
```

## Target status

P35.3c target status is NOT RUN. After explicit commit/push and 64-chip approvals, r31 must use
the source-pinned `--stage-probe` renderer path and archive coordinator, proxy, resource-manager,
worker and Kubernetes event evidence. A complete r31 identifies infrastructure progress only and
still has `numerical_verdict=false`.

## Rollback

Leave `CANON_P35_REPLAY_STAGE_PROBE` unset or set it to `0`. The barriers, JSONL writes and
diagnostic stop are unreachable. Production serving, training, precision, loss, sampling,
gradients and optimizer behavior remain unchanged.
