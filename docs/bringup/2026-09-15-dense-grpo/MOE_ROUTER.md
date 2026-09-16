# MoE: does bf16 flip the router? Plus a blocker for the recommended config.

Addendum to [`README.md`](README.md), on branch `jfacevedo/trellis-followups`.

**Why this matters.** In a dense model, batch-shape-dependent bf16 rounding
perturbs log-probs smoothly. MoE adds a failure mode dense structurally cannot
show: if rounding shifts router logits across a top-k boundary, that token is
routed through **different experts** — not a slightly wrong number, a different
computation.

**Setup.** The ~1M-param MaxText Qwen3.5 MoE fixture from
`tests/rl/router_replay_maxtext_test.py`, scaled up to 8 experts / top-2 /
4 layers / 256-token sequences / 8 sequences, scored through the real
`compute_per_token_logps` at forward batch size 1 vs 4. One chip, matched
`--xla_tpu_scoped_vmem_limit_kib=65536` across every config.

---

## 1. BLOCKER: `precision=HIGH` does not work on the MoE path [M]

The README recommends bf16 weights + fp32 activations + `precision=HIGH`.
**That configuration cannot run on MoE.** Megablox GMM is a Pallas kernel, and
Mosaic accepts only two precisions — `jax/_src/pallas/mosaic/lowering.py:2873-2879`:

```python
    precision_attr = None  # That's the default in Mosaic.
  elif precision == lax.Precision.HIGHEST:
    precision_attr = ir.Attribute.parse("#tpu.contract_precision<fp32>")
  else:
    raise NotImplementedError(f"Unsupported dot precision: {precision}")
```

Observed directly: `NotImplementedError: Unsupported dot precision: HIGH`.

**Consequence.** For MoE the options are DEFAULT or HIGHEST. HIGHEST measured
~25–27% more expensive than HIGH on the dense benchmark, so the MoE version of
this fix is meaningfully dearer than the dense one. Anyone planning to carry the
recommendation to the 35B needs to know this before costing it.

## 2. No evidence of router flips at bf16 [M]

| config | mean \|Δ\| | p99.9 | max | max/mean | frac > 0.05 |
|---|---|---|---|---|---|
| bf16, DEFAULT | 0.00837 | 0.03594 | 0.04002 | 4.8× | **0.000000** |
| fp32, HIGHEST | **0.00026** | 0.00547 | 0.02880 | 109× | 0.000000 |

A flip sends a token through different experts, which should produce O(0.1–1)
jumps. Across 1536 scored tokens there are **zero** above 0.05, and the bf16
distribution is smooth and unimodal (max only 4.8× the mean). On this model, at
this scale, bf16 batch-shape rounding does **not** appear to flip the router.

fp32 + HIGHEST cuts the smooth component **32×** (0.00837 → 0.00026), consistent
with the dense result.

One caveat on the fp32 row: max/mean of 109× is a heavy tail, but p99.9 =
0.00547 against max = 0.02880 means it is **a single outlier token out of 1536**.
That could be one flip, or one token with an unusual hidden state. One event is
not a measurement, and I am not claiming it either way.

## 3. GAP: router observation is not implemented [M]

Flips cannot be counted directly, because the adapter never reports the routing
it used. `TunixMaxTextAdapter.__call__` is typed `-> Tuple[Array, None]` and
ends:

```python
    return logits, None
```

It accepts `forced_routed_experts` (replay **in**) but returns nothing about the
routing it chose (capture **out**). So `compute_per_token_logps(...,
return_routed_experts=True)` can never yield anything on this path — the
`extra` it looks for is always `None`. §2 is therefore an *indirect* test via
the divergence distribution, not a flip count.

**To measure router agreement properly**, the adapter needs to return the
top-k indices alongside the logits. That is the single change that would turn
this from an inference into a measurement.

---

## Caveats — these bound the conclusion tightly

- **Randomly initialised, ~1M params.** No trained checkpoint. A trained router
  has learned, likely more separated, logit gaps; flip probability could go
  either way and this says nothing about the real 35B.
- **One device, no expert parallelism.** With `num_ep=1` there is no
  capacity-based token dropping and no `ragged_all_to_all`. The specific concern
  that capacity-based dispatch makes expert assignment a function of *batch
  composition* is **not tested here** — that mechanism may not even be active at
  this scale, and it is the one most likely not to dilute with sequence length.
- **Random token ids**, not model completions.
- **Sensitivity floor.** 1536 tokens × 4 layers ≈ 6144 routing decisions, so a
  flip rate below roughly 0.02% would be invisible.

## What I would do next

1. Make the adapter return routed experts (small change, turns §2 into a real
   measurement).
2. Rerun with expert parallelism and a capacity factor, which is where the
   batch-composition mechanism would actually appear.
3. Only then consider a real checkpoint.

Reproduce:

```bash
PYTHONPATH=<repo> python probe_moe_router.py --dtype bfloat16
PYTHONPATH=<repo> python probe_moe_router.py --dtype float32 --precision highest
# precision=high will raise NotImplementedError -- that is finding 1
```
