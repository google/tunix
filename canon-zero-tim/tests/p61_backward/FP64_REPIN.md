# Fail-closed fp64 criteria checker

`check_fp64_repin.py` checks already computed, independently sealed
`canon-p61-fp64-reference-v1` metrics. It does not calculate fp64 gradients or
approve an anchor re-pin. Capture provenance, complete leaf identity, G4, and
per-geometry training certification remain separate requirements.

The historical small-norm criteria are retained:

- Overall relative L2 and one-minus-cosine ratio at most 1.05.
- Per-kind relative L2 ratio at most 1.05; every group's at most 1.10.
- Material groups (reference energy share at least 1e-3) have cosine ratio at
  most 1.10; small groups still face relative-error and norm guards.
- Overall A/B relative L2 obeys the registered triangle bound, A/B
  one-minus-cosine is at most 1e-3, and every B/A group norm ratio is in
  [0.98, 1.02].

Before applying these criteria, validate all six producer fields, exact oracle
norm identity, control/candidate norm identity across comparison blocks, and
the actual leaf-to-group partition. Group and overall norms/cosines must agree
with the complete leaf statistics. Relative errors and norm ratios must agree
with their component norms; distance and cosine obey the vector identity.
Missing coverage, wrong dtype/schema, contradictory summaries and duplicate
keys fail closed, even if the supplied aggregate numbers would pass admission.

Report-consistency checks allow relative rounding error of 1e-12. Only
dimensionless cosine and normalized squared-distance identities also allow
absolute error of 1e-12; norms never have an absolute tolerance. These are
consistency tolerances between float64 summaries, not changes to the numerical
admission thresholds above or a substitute for bitwise full-gradient proof.

Primitive norms and cosine must be finite; NaN is never accepted. The only
allowed infinity is a producer-defined derived ratio with a zero reference
norm (`norm_ratio_error`, and `rel_l2` if the candidate is nonzero). The whole
zero-control/oracle case is inconclusive, never admissible. A positive error
against a zero baseline is not skipped; a zero candidate with a nonzero
control is rejected. Finite-reference ratio overflow fails closed.

```bash
JAX_PLATFORMS=cpu python canon-zero-tim/tests/p61_backward/check_fp64_repin.py \
  /sealed/fp64-metrics.json --expected-sha256 REVIEWED_SHA256 \
  --expected-rows 16 --expected-leaves 399 --expected-groups 147
```

Coverage above describes the historical 8B capsule, not all models. Inputs and
expected SHA must be reviewed before running; do not silently recompute the
expected digest after a failure. The default `--share-floor 0.001` is part of
the registered criterion, not a knob to tune after seeing results.

Exit 0 is `FP64_REPIN_CRITERIA_PASS`; 1 is numerical RED or invalid metrics;
2 is `INCONCLUSIVE_NO_SIGNAL`. The receipt explicitly says
`capture_and_g4_admission=NOT_EVALUATED`. This is an explicit new entry point;
historical external `repin_check_v2.py` scripts and old verdicts are not edited.

```bash
JAX_PLATFORMS=cpu python -m pytest -q canon-zero-tim/tests/p61_backward/test_check_fp64_repin.py
```

Tests include CLI exit-code controls, worse-error and scaled-gradient
rejections, zero signal, incomplete metrics and malformed sealed input.
Numerical positive/negative fixtures come from coherent vectors, rather than
manually edited summary fields. Separate corruption controls splice valid
subreports, retain stale summaries, and swap leaf assignments. Compatibility
tests execute the real producer's pure metrics/grouping/streaming functions.
