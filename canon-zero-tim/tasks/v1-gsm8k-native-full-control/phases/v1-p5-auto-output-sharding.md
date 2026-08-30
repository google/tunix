# P5 — GSM8K Native Auto/Manual output-sharding legality

Status: `IMPLEMENTED / HOST PASS / PINNED-IMAGE PASS / TARGET NOT RERUN`

## Incident ladder

Attempt 04 at source `0d224e4a0e8c278f1bf9f699af235fdea83ef327`
crossed the prior Splash boundary, then failed at the attention output
projection because an Explicit mesh required the doubly-sharded contraction's
output placement to be named. Source `6c701164` added the existing activation
spec to the four projection sites.

Attempt 05 at source `29c923dc042654a59968f9b062a72c3d30646230`
kept the same DP16xTP4 shape but selected Auto axis types for untreated Native.
Rollout completed at 5,668.9 tokens/s. The first trainer embedder gather then
failed because `.get(out_sharding=...)` received `P('data', None, 'model')`
whose named axes were Auto. JAX accepts named output shardings there only for
Explicit axes.

Immutable raw-error SHAs are
`bb34aa0ab2f7c617d7ba52111b214a5378c9b6b0376a026571ad8e0d55f79b22`
for Attempt 04 and
`2a4a2cfca101a19f179924b4e6ed1756440afe74678d11fa2ab467db06008353`
for Attempt 05.

## Repair

`_activation_out_sharding(spec)` now maps mesh axis names to axis types. It
returns a `NamedSharding` only when every axis actually named by `spec` is
Explicit. Auto or Manual named axes return `None`, so the original compiler
inference path runs. A missing axis name still raises instead of silently
degrading. CPU/no-mesh behavior is unchanged.

This keeps the Attempt-04 Explicit gather/projection repair while making the
Attempt-05 Auto Native path legal. It changes no model math, parameter or
activation partition specs, mesh shape, loss, precision, gradient, optimizer,
profile, flag, or Native/Zero treatment ownership.

## Gates and harness correction

- Host Native renderer: 12 tests, one expected pinned-only skip.
- Flag audit: 409/409, `changed_names=0`.
- Forced eight-device Qwen suite: Explicit projection/gather positives,
  Auto/Manual omission controls, mixed-axis referenced-name control, Splash
  negative/positive, and neighboring data-parallel tests.
- First exact-image invocation exposed a test harness defect: the file's
  `absltest.main()` ran before the Attempt-04 class definition, so 11/13 tests
  executed. Moving the entrypoint to EOF is part of this phase's gate repair.
- Authoritative rerun executes 13/13 and ends:

```text
V1_GSM8K_NATIVE_FULL_EXACT_IMAGE_PASS native_contract=12 qwen_sharding=13 auto_out_sharding=2 zero_neighbor=1
```
- The complete gate was rerun after fast-forwarding to operator parent
  `98d102eb27fe05fcee327688d0aa6d236b32be4a` and emitted the same terminal.
  This remains transcript-only construction evidence, not a target run.

## Claim ceiling

This closes the Attempt-05 JAX API exception in source and pinned image. It
does not prove a real Auto-axis DP16xTP4 trainer forward, backward, optimizer
commit, performance, or convergence. A separately approved fresh Attempt-0
must use a clean published SHA and reach at least one optimizer commit.
