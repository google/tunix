# Does `force_on_policy_ratio` change `is_oob`? No.

Addendum to [`README.md`](README.md). On branch
`jfacevedo/trellis-force-on-policy` so the main report stays reviewable as-is.

**Question.** Every run in the report used the launcher default
`USE_ROLLOUT_LOGPS=true`, i.e. **without** `force_on_policy_ratio`. The MLPerf
reference recipe sets `force_on_policy_ratio: true`
(`grpo_qwen35_397b_swe_openhands_async.yaml:75`). Does that gap invalidate the
`is_oob` results?

**Answer: no.** The gate never reads that flag. Confirmed from code and then
measured.

---

## From the code

`algo_core.py:1114-1121` — the quantity behind `seq_geomean`, `is_oob_ratio`
and the whole TIS gate:

```python
rollout_logps = getattr(train_example, "rollout_per_token_logps", None)
log_is_raw = None
if rollout_logps is not None:
    log_is_raw = jax.lax.stop_gradient(per_token_logps) - jnp.astype(
        rollout_logps, jnp.float32
    )
```

`rollout_per_token_logps` is carried unconditionally. `use_rollout_logps` (this
stack's `force_on_policy_ratio`) only decides whether those log-probs also
become `old_per_token_logps`, which is the PPO ratio's denominator. NVIDIA state
the same separation in the reference config itself:

> `force_on_policy_ratio` only fixes the PPO ratio at one; it does not select or
> replace this actor-vs-generation importance-sampling filter.

## Measured

`fop1` is byte-identical to `hp0` in configuration except the flag — verified in
the launch commands (`--no-use_rollout_logps` vs `--use_rollout_logps`).

**Step 0's four loss calls — before any optimizer update, so identical weights
and identical rollouts:**

| metric | hp0 vs fop1 |
|---|---|
| `tis/is_oob_ratio` | **IDENTICAL** |
| `sampler_is/seq_geomean_mean` | **IDENTICAL** |
| `sampler_is/token_logdiff_absmean` | **IDENTICAL** |
| `sample_mask/kept_frac` | **IDENTICAL** |

**The gradient path, where it should and does differ:**

| | hp0 (`force_on_policy` off) | fop1 (on) |
|---|---|---|
| loss | N/A, 0.0000, −0.0568, 0.0406, 0.0000, 0.0000 | N/A, 0.0000, −0.0562, **−0.0078**, 0.0000, 0.0000 |
| `grad_norm` | N/A, 0, 0.5863, 1.154, 0, 0 | N/A, 0, **0.7109**, **0.8626**, 0, 0 |
| `reward_mean` | 0.5000, 0.4688, 0.2188, 0.5000, 0.5000, 0.3438 | *identical* |

**Across all 24 calls**, `is_oob_ratio` agrees on 11/24 and the means are
0.8125 vs 0.8229. That ~1% difference is **second-order**, not the flag acting
on the gate: once step 0's optimizer update lands, the two runs hold different
weights, so every later forward differs. The clean comparison is step 0, where
everything the gate touches is bit-identical.

## What this means

- **The `is_oob` conclusions in the report stand.** Running with
  `force_on_policy_ratio` would not have changed them.
- **It still matters for recipe fidelity.** The reference sets it; we should
  match for the submission. It changes the PPO ratio (pinned to 1, so clipping
  goes inert) and therefore the gradient — visible above as different
  `grad_norm` and loss from step 2.
- Reward is unchanged across all six steps here, which is expected at
  `learning_rate=2e-07` over six steps: the policy barely moves, so the same
  completions come back.

Evidence: [`logs/fop1/`](logs/fop1/) against [`logs/hp0/`](logs/hp0/).

Reproduce:

```bash
USE_ROLLOUT_LOGPS=false RUN_ID=fop1 MAX_STEPS=6 TRAIN_MICRO_BATCH_SIZE=4 \
  bash docs/bringup/2026-09-15-dense-grpo/run_bringup.sh
```
