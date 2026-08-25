# V1.P4.6 — Hybrid overflow-safe global-norm clipping

Status: complete through host and pinned-image admission; target optimizer
transaction not run.

## Objective

Repair the optimizer boundary exposed by G5b without changing loss, backward,
DP/TP reduction, accumulation order, or stock finite-range clipping. Enable the
repair only in the three strict optimized Phase4 full recipes and retain an
independent all-finite hard gate.

## Frozen numerical contract

For gradient tree `g`, let

```text
n_stock = sqrt(sum_i g_i^2)
m       = max_i abs(g_i)
n_scaled = m * sqrt(sum_i (g_i / m)^2)
```

In real arithmetic `n_stock == n_scaled`. In FP32 the first expression can
overflow while every `g_i` and the mathematical norm remain finite. The
hybrid selector is:

```text
use_fallback = all_finite(g) and not isfinite(n_stock)
n_selected   = n_scaled if use_fallback else n_stock
```

The stock transform output is returned byte-for-byte whenever `n_stock` is
finite. The fallback is never selected for a tree containing NaN or Inf. The
registered max norms remain GSM8K `1.0` and FrozenLake `100.0`; they are not
changed by this phase.

G5b target shape ledger remains unchanged: caller-global M4096, shard-local
M256, canonical-kernel M256, semantic valid rows from the existing masks, and
per-rank scheduler token capacity 256 for GSM8K DP16xTP4. FrozenLake remains
caller-global M2048, shard-local/canonical M256 at DP8xTP8. The clipping repair
does not change any row count or sharding.

## Flag contract

- Name: `CANON_P63_OVERFLOW_SAFE_CLIP`.
- Tier/parser/default: numerical; exact boolean; missing or `0` is off, `1` is
  on, empty/other values are fatal.
- Writers: the GSM8K and FrozenLake V1 high-performance full profiles only.
- Readers: the exact Python training process in the two recipe entrypoints and
  the segmented `PeftTrainer` commit/microgradient boundary.
- Admission: V1 high-performance full, commit enabled, strict alignment,
  P59 enabled, and exact GSM8K DP16xTP4 or FrozenLake DP8xTP8 profile.
- Sunset: retire or weld only after all three target first commits plus full
  horizons pass and an upstream generic clipping replacement is separately
  reviewed. No certification transfers to P58/DeepSWE.

## Gates

1. Unit algebra: normal finite trees produce byte-identical outputs to stock
   `optax.clip_by_global_norm`; finite-overflow output agrees with an FP64
   oracle; NaN/Inf never selects fallback and remains non-finite.
2. Integration: the P58/P62 diagnostics and neighboring stock profiles reject
   or omit the flag; exact max norm/profile/stage/arm delivery is required.
3. Runtime receipt: every optimizer transaction records all-finite,
   stock-norm finiteness, stable/selected norm, fallback bit, max norm, and a
   finite positive clip factor. Postflight rejects missing, malformed, or
   non-finite receipts.
4. Host: focused utility/trainer/profile/renderer/classifier negatives, V1,
   P57, P59, APC, flag registry, syntax, and diff hygiene pass.
5. Pinned image: complete V1 gate passes from the final runtime tree.
6. Target: launch the three uninterrupted full recipes together from one
   approved immutable SHA. Their first real commits are independent gates; any
   strict alignment failure kills only that recipe.

## Claim ceiling

Until Gate 5: `IMPLEMENTED / HOST OR EXACT-IMAGE CONSTRUCTION ONLY / TARGET NOT
RUN`. Gate 5 does not certify an optimizer transaction. Only each recipe's real
first commit can promote that target.

## Rollback

Disable `CANON_P63_OVERFLOW_SAFE_CLIP` by reverting its V1 profile opt-ins;
the library default remains stock clipping. Revert classifier/receipt and
implementation hunks as one scoped numerical CL. Preserve all G5/P62 evidence.

## Result log

- 2026-08-25: phase pre-registered from complete G5b evidence. Verified by
  DP16xTP4 target that all 16 backward groups and the final accumulator are
  finite; not verified by any optimizer commit because the diagnostic
  deliberately performed zero commits.
- 2026-08-25: implementation admitted on the isolated final runtime tree.
  Stock-finite F32/BF16 clipping remains byte-exact, finite naive-L2 overflow
  agrees with the FP64 oracle, and NaN/Inf never selects the fallback. The
  pinned trainer transaction observed `naive_norm=inf`, stable norm
  `1.2973823439541035e22`, clip factor `7.707828007371153e-23`, and a finite
  nonzero parameter update.
- Host gates: V1 45/45, P57 144/144, P59 37/37, APC 31/31, flags 372/372,
  Python/Bash syntax and `git diff --check` all pass. The first exact-image
  attempt was blocked before Docker by the managed sandbox and is preserved as
  `INCONCLUSIVE_INFRASTRUCTURE` under `evidence/v1_hp_p63_exact_image_20260825_r1/`.
- Complete pinned-image r2 exits zero with exactly one terminal
  `V1_HP_EXACT_IMAGE_PASS ... p63_clip=1 ... manifests=3`. Raw log SHA is
  `31126e623c7ad775614a3ce1ff89d3798d095482d0cbefc84a47ae0d0a2d6c44`;
  receipt and per-file runtime hashes are under
  `evidence/v1_hp_p63_exact_image_20260825_r2/`.
- Claim ceiling: `HOST PASS / EXACT_IMAGE PASS / TARGET OPTIMIZER COMMIT NOT
  RUN`. No commit, push, manifest render, JobSet, or TPU launch occurred.
