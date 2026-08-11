# P38.2g local admission gate

Date: 2026-08-11 UTC

## Result

The capsule loader, mask-derived R0/R1 scheduler, bounded fixed-chunk reference,
fixed-M engine-record lowering, live-adapter entry point, one-host runner,
measurement classifier, and negative controls are locally admitted. No real
P38.2f target-capsule replay was run because no verified target capsule exists
on this host or in the repository. Real Qwen3-8B synthetic deep and shallow
TPU controls subsequently ran; see `p38_2g_onehost_synthetic_0811.md`. They
admit the measurement path but do not isolate the production carrier.

## Commands and observations

Focused pure-CPU contracts:

```bash
python3 -m unittest discover -s tests/rl -p p38_frozenlake_replay_test.py
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/test_classify_p38_frozenlake_replay.py
```

Result: 11/11 schedule and capsule tests passed; 5/5 classifier controls
passed. The tests include embedded-array hash corruption, invalid masks,
fixed-M metadata, exact action-predictor coverage, repeat determinism, and an
ineffective-negative-control or weight-mismatch rejection.

Focused exact-image adapter integration:

```bash
sudo docker run --rm \
  -v "$PWD:$PWD:ro" -w "$PWD" \
  -e PYTHONPATH="$PWD" -e JAX_PLATFORMS=cpu \
  -e XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  python3 -m unittest discover -s tests/rl \
    -p canonical_qwen3_adapter_test.py -k p38_frozenlake
```

Result: 1/1 passed. R0, R1, and the fixed-chunk reference each repeated
bitwise exactly; the one-bit control was detected. The fake model classified
`LOCAL_CARRIER_NOT_REPRODUCED`, which is a valid measurement result and not a
repair claim.

The first exact-image attempt failed during report assembly because the
reference arm returns logprobs separately from its diagnostics dictionary.
The implementation now normalizes that return structure; the unchanged
rerun passed.

Complete adjacent CPU gate:

```bash
sudo docker run --rm -v "$PWD:$PWD:ro" -w "$PWD" \
  -e PYTHONPATH="$PWD" \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  bash -lc 'bash canon-zero-tim/tests/p33_workloads/run_cpu.sh >/dev/null'
```

Result: exit 0. The existing 67-workload and 26-alignment suites, adjacent
P35/P38 gates, both new focused suites, and the adapter integration test all
passed.

Overlay exact-image gate:

```text
P38_OVERLAY_EXACT_IMAGE_PASS models=2 cases=10
```

Qwen3-1.7B and Qwen3-8B each passed 10/10 decode/prompt chunk overlay cases.

## Claim ceiling

The capsule contains tokens and masks, not the original serving scheduler's
per-call metadata. R0 is therefore labeled `mask-derived-v1`; it is a causal
counterfactual, not an exact scheduler replay. A real result must first show
that local R0 reproduces a red R0-versus-reference boundary. R2/R3 are not
implemented or admitted until that prerequisite and the negative control pass
on a verified target capsule.

## Reproduction after capsule capture

```bash
canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_frozenlake_replay.sh \
  /absolute/path/to/recovered-p38-capsule.npz <unique-label>
```

The runner requires the authorized DP1xTP4 v5p host, Qwen3-8B, global/local
M=256, prefix cache disabled, independent fresh runtime KV cache per arm,
policy version zero, and device-side bitwise equality between the actor anchor
and live engine leaves. It exits before the agentic learner and backward, with
zero optimizer commits.

## Rollback

Leave `CANON_P38_FROZENLAKE_REPLAY` unset. Remove the isolated replay files or
revert their bounded change after preserving evidence. No production default,
precision, loss, prefix-cache policy, or attention kernel changed.
