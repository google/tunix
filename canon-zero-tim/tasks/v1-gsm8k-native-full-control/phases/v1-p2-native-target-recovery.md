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

`IMPLEMENTED / PINNED-IMAGE PASS / TARGET NOT RUN` on repair base
`2af1197f4d0bb604d7c423f703251fc5187b4594`. The final pinned-image receipt is
`V1_GSM8K_NATIVE_FULL_EXACT_IMAGE_PASS native_contract=10 qwen_sharding=9
zero_neighbor=1`. This phase file ships with the repair CL. No post-fix TPU
run, optimizer commit, image publication, or Kubernetes mutation was
performed during validation.
