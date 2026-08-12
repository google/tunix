# P38.2g6: standard-runner serving capture

- Status: local implementation and admission gates complete; target P38s7 is
  not run.

## Evidence correction

P38s6 initialized the patched runner but emitted zero
`CANON_P38_SERVING_CAPTURE_OBSERVE` and zero capture records while completing
real rollout traffic. The prior explanation that every active prefix stayed
below 1536 is withdrawn: `_p38_serving_begin` emits its first observation
before applying the prefix-stratum filter.

The exact pinned source gives the causal explanation. FrozenLake leaves
`rollout_vllm_additional_config` unset, `RolloutConfig` defaults it to `None`,
and `vllm_sampler` turns that into an empty additional configuration. The TPU
runner therefore has `enable_continue_decode=False` and executes the standard
`_execute_model`/`sample_tokens` path. P38.2g5 installed its only begin/finish
hook inside `_execute_continue_decode`; module initialization was reachable,
but the production hot path was not.

P38s6 is `INCONCLUSIVE_WRONG_PATH_NONTERMINAL`. It also ended after the final
adapter-forward norm without alignment, child exit, classifier, serving
archive, or outer postflight. A `CANON_PALLAS_CANONICAL_VJP` trace in that
forward is not evidence that backward executed.

## Deliverable

Capture the actual FrozenLake standard/mixed serving path without changing
which executable generates `S_decode`:

1. call the bounded capture after standard `_prepare_inputs`;
2. select request-level one-token decode requests even when the scheduled
   batch also contains prefill rows;
3. map each selected request to its packed token row, stable attention row,
   query range, physical block table, token history, DP rank, and scheduler
   slot;
4. carry the capture sequence through `ExecuteModelState` and complete the
   post record after the unchanged standard `sample_tokens` call;
5. attest `program_path=standard` in init, observation, pre, post, classifier,
   rendered environment, and outer postflight; and
6. reject async scheduling for this bounded diagnostic rather than silently
   producing a partial record.

Do not set `enable_continue_decode=True`. That changes the program under test
and cannot explain the existing production A/B carrier.

## Local exit gate

- exact-image Qwen3-1.7B and Qwen3-8B overlays match every manifest entry and
  pass a fake standard/mixed scheduler reachability test with
  `enable_continue_decode=False`;
- a negative control proves the continue-decode route cannot masquerade as a
  standard capture;
- the mixed-row mapping test proves a decode token after a packed prefill chunk
  uses its token offset rather than its request slot;
- the classifier rejects the wrong program path and accepts complete standard
  records;
- renderer, environment, postflight, Python syntax, shell syntax, and the full
  frozen-image CPU gate pass; and
- the handoff names P38s7 as the only next target attempt.

Completed evidence on the pinned image:

- Qwen3-1.7B and Qwen3-8B exact-image overlays: 20/20 each, with all 29
  manifest entries matching;
- serving classifier: 26/26;
- renderer: 5/5;
- shell postflight: PASS; and
- full pinned-image CPU gate: PASS, including 81 workload tests, 31 alignment
  tests, standard-path preflight, and all adjacent negative controls.

Installed runner SHA-256:
`a7bdc527182ad115385e60005cff8c4e135efd2714eb97a2e929dc3dbc45e890`.

These are construction/admission results only. No real Pathways serving
record, block table, page ownership, numerical repair, backward pass, or
optimizer behavior is claimed.

## Target exit gate

One stock-only Attempt-0 P38s7 run must preserve a complete non-timestamped log
through outer postflight and return four standard-path pre/post records, the
run-specific mismatch capsule, classifier PASS, serving archive, and at least
one exact request/token-history join. Backward and optimizer commits remain
zero. Unified KV is not rerun.

If `INIT` exists but a standard-path `OBSERVE` does not, the run is void. If
observations exist but no stratum is captured, return the complete observation
range and stop; do not auto-adjust bounds. Do not repurpose or delete the slice
when the rollout engine reaches zero running requests: adapter precheck and
outer postflight must also terminate.

## Rollback

Leave `CANON_P38_SERVING_CAPTURE_DIR` and
`CANON_P38_SERVING_CAPTURE_EXPECTED_PATH` unset. The added path is diagnostic
and default-off; it does not alter sampling, attention, loss, precision,
backward, optimizer, or normal training behavior.
