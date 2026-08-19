# P38.2h fixed-lm-head one-host VJP receipt — 2026-08-19

Scope: construction evidence on one direct-attached four-device v5p using the
real Qwen3-8B BF16 lm-head weight `[4096, 151936]`. These runs used the branch
HEAD plus the documented uncommitted P38.2h diff; they are development
evidence, not a clean published-source or 64-TPU result.

## Initial automatic-transpose result

The original M4096 forward was `lax.map` over 16 shared-weight M256 calls. Its
automatic transpose produced exact `dHidden` and repeat-deterministic output,
but its shared `dWeight` did not match 16 completed M256 pullbacks accumulated
in explicit ascending chunk order:

- verdict: `FIXED_LM_HEAD_CHUNK_VJP_NOT_INVARIANT`;
- `dHidden`: 0 differing elements;
- `dWeight`: 11,950 differing elements, `max_abs=2.0`;
- repeat `dHidden/dWeight`: exact;
- normal-value negative: 1 differing element.

SHA-256:

```text
80adc77c139413427dc4603ef981ef839fd104ff0758ce66bd19c85f9879f0e4  p38_fixed_lm_head_vjp_p38_2h_vjp2_20260819.result.json
a69d140ac492d570e4828c9dd67093c5ebc698505c2df0aa19e37434bac9ef2f  p38_fixed_lm_head_vjp_p38_2h_vjp2_20260819.raw.log
```

## Fixed-order outer VJP result

The repair leaves forward unchanged and gives only the M4096 outer
composition a custom VJP. Each M256 pullback completes independently; one
loop-carried `lax.scan` adds the 16 shared-weight cotangents in ascending chunk
order. The real-v5p rerun passed:

- verdict: `FIXED_LM_HEAD_ONEHOST_VJP_PASS`;
- `dHidden`: 0 differing elements, `max_abs=0.0`;
- `dWeight`: 0 differing elements, `max_abs=0.0`;
- repeat `dHidden/dWeight`: exact;
- all gradients finite;
- nonzero `dHidden` elements: 16,777,216;
- nonzero `dWeight` elements: 16,381;
- normal-value negative: 1 differing element.

The scalar candidate loss and the diagnostic sum of 16 scalar chunk losses
differ by about `3.8e-6`; that scalar comparison is intentionally not a gate
because it introduces a second loss-summation order. The gradient arrays are
the registered bitwise gate and are exact.

SHA-256:

```text
3863cfab88ed8e627845450c7c0ab1de156e4e31ed4035ab0afb3d249efbb330  p38_fixed_lm_head_vjp_p38_2h_vjp3_20260819.result.json
bec0c1421416e132da1aa94369a8c970a2d22aa557911a8f07847332ab03276c  p38_fixed_lm_head_vjp_p38_2h_vjp3_20260819.raw.log
```

Claim ceiling: this admits the M4096 fixed-lm-head VJP construction only. It
does not replace the P38h 64-TPU actual-model backward-no-commit gate.
