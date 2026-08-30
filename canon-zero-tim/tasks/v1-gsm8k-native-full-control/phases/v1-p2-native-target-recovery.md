# P2 — GSM8K Native target admission recovery

## Goal

Advance the stock DP16xTP4 Native carrier past its first real learner forward
without changing model mathematics or importing any Zero-TIM selector.

## Failure ladder

| Attempt | First failing boundary | Disposition |
|---|---|---|
| 01 | embedder gather output layout ambiguous on Explicit DPxTP mesh | output sharding named; preserved as immutable evidence |
| 02 | activation `with_sharding_constraint` names Explicit axes | Explicit meshes use `reshard`; Auto and CPU legacy path retained |
| 03 | replicated Splash kernel leaf conflicts with model-sharded `shard_map.in_specs` | local repair maps real kernel leaves to their existing manual specs |
| 04 | output projection contracts a model-sharded head axis without naming its Explicit-axis output | projection sites name their existing activation specs |
| 05 | Auto-axis Native embedder passes an illegal named `out_sharding` to `.get` | superseded by P5 axis-type-aware output-sharding guard |

## Repair contract

- Treat `manual_sharding_spec` as the target layout for the corresponding
  Splash kernel pytree leaves.
- Apply `jax.sharding.reshard` only when the physical mesh contains Explicit
  axes.
- Preserve the exact kernel object on Auto meshes.
- Prove the old admission error fires before the repair and the repaired
  values are byte-identical.
- Do not alter attention, loss, precision, gradient, optimizer, profile,
  renderer, YAML, Native/Zero isolation, or training hyperparameters.

## Gate ladder

1. Python/Bash syntax and `git diff --check`.
2. Forced eight-device CPU with a real Splash kernel pytree:
   replicated-input negative, repaired `shard_map` positive, exact values,
   normalized leaf placements, Auto-mesh object identity.
3. Pinned production image: Native contract, all Qwen sharding tests, and one
   adjacent Zero renderer.
4. Separately approved fresh DP16xTP4 Native Attempt-0: cross the prior Splash
   boundary and at least one real optimizer commit.

## Result log

The Attempt-03 repair is target-proven for its Splash boundary because
Attempts 04 and 05 crossed it. Broader training remains incomplete. The
current Attempt-05 repair and authoritative exact-image result live in
`v1-p5-auto-output-sharding.md`.
