# P38.2j — P38s12a analysis, row-231 E0-lite, and clean P38s12f

- Status: active; local E0-lite complete; P38s12d/e invalid; clean target
  P38s12f not run.

## Finding

- Confirmed: evidence commit `23bb2a3c` contains an internally consistent
  capsule, serving archive, pre-alignment record, classification, and byte-zero
  log. Re-extraction and reclassification reproduce the committed core files.
- Confirmed: despite the directory/run label `p38s12b`, the command used
  `--max_concurrency=256` and all 32 prompts x eight generations. Account this
  run as **P38s12a analysis-level evidence**, not the concurrency-32 arm.
- Confirmed: A-B had 46 differing elements / 74 differing bytes among 44,818
  action tokens with `max_abs=0.1039161682`; B-C was exact. Nine rows were red,
  while the row cap of eight omitted row 255.
- Confirmed: selected source row 231 is capsule row index 3 and carries the
  run's largest observed absolute mismatch.
- Boundary: the outer run ended `rc=137`, the infrastructure bundle is
  incomplete, and `SHA256SUMS` incorrectly included its own stale digest.
  Therefore the run is analysis-level, not a formally admitted target result.

## E0-lite preregistration

E0-lite uses row 231 and the existing mask-derived R0/R1 plus canonical REF.
It is a bounded falsifier, not strict E0, because it does not restore exact
live scheduler/cache state.

| Result | Required observations | Decision |
|---|---|---|
| `E0_LITE_REPRODUCED` | captured A-B red, captured B-C exact, REF equals captured B over the complete action vector, and R0 equals captured A over the complete action vector | Promote row 231 to strict-E0 construction and first-divergence instrumentation; do not yet claim a repair. |
| `E0_LITE_ENVELOPE_NOT_REPRODUCED` | captured and REF prerequisites pass, but R0 does not equal captured A | Stop interpreting R0/R1 operator counterfactuals; the missing live envelope remains causal context. Use the joined request state to design the next exact observer. |
| `E0_LITE_PREREQUISITE_FAILED` | the selected source row is not red, B-C is not exact, or REF does not equal captured B | Reject the replay measurement and repair capsule/weights/reference identity before further numerical interpretation. |

## Local E0-lite result

- Verdict: `E0_LITE_ENVELOPE_NOT_REPRODUCED`.
- Source row 231 (capsule index 3) contained 566 scored action tokens.
  Captured A-B remained red at 19 elements / 35 bytes with
  `max_abs=0.10391616821289062`; captured B-C was exact.
- REF reproduced captured B and T-old exactly over all 566 action values.
  R0 and R1 were repeat-exact with each other, but each differed from captured
  A at 470 / 566 elements and by as much as `30.886138916015625`.
- The one-bit negative control was detected. The 399 mapped/live model leaves
  were bitwise equal over 8,190,735,360 elements. No backward or optimizer
  operation ran.
- Decision: do not interpret R0/R1 RoPE, RPA, page, or residual
  counterfactuals and do not start the first-divergence walk. The mask-derived
  replay is missing causal live-serving state. The next target remains the
  separately rendered P38s12b concurrency-32 discriminator.
- Durable summary and hashes are in
  `artifacts/p38_2j_row231_e0lite_0813.md`.

## Engineering hardening

1. P38 target diagnostics use explicit process exit 42 after fsynced evidence
   and a flushed terminal marker; outer postflight accepts only that controlled
   exit and still rejects missing markers/classification.
2. Capsule capacity is 16 rows, covering the nine-row P38s12a population with
   headroom while remaining bounded.
3. Every future P38 report records host-derived action-depth geometry. Target
   postflight requires `max_logical_kv_prefix_length >= 1686`.
4. The evidence sealer requires the full Kubernetes/Pathways bundle, excludes
   `SHA256SUMS` from its own manifest, and immediately runs `sha256sum -c`.

## Clean P38s12f

P38s12d failed the stale recipe geometry before rollout. P38s12e contains
duplicated P38s12d logs rather than a new source-pinned run. Neither is the
concurrency discriminator. P38s12f uses a fresh run id and requires semantic
provenance checks in addition to byte hashes.

Render one source twice with the same run id: baseline concurrency 256 and
candidate concurrency 32. `check_p38_intent_diff.py` must prove that the only
manifest differences are `--max_concurrency` and its attestation label. Apply
only the concurrency-32 candidate. Keep prompt order, 32 prompts, eight
generations, mini-batch four, DP16xTP4, prefix cache off, precision, kernels,
capture schema, weights, and source fixed.

Interpret only a depth-sufficient run. A red concurrency-32 result proves that
small concurrency is insufficient to remove the carrier. One exact result must
be repeated before claiming concurrency/churn is a necessary trigger. Neither
outcome identifies an operator.

## Exit gate

- Local: focused tests for controlled exit, depth negative control, 16-row
  capsule, renderer, intent-diff, E0-lite classifier, and evidence seal pass.
- One host: row 231 produces one complete E0-lite report with exact repeats,
  one-bit negative control, equal weights, no backward, and zero optimizer
  commits.
- Target: P38s12f passes intent-diff before apply, reaches logical KV 1686,
  returns the sealed full bundle, and exits through controlled code 42.

## Claim ceiling

This phase can identify whether a mask-derived local replay reconstructs row
231 and whether concurrency 32 is a necessary trigger. It cannot establish a
page-lifetime bug, stale KV, RoPE/residual/cast carrier, or zero-TIM repair.
The observed envelope miss blocks strict E0 and every operator-level causal
claim from this local replay.
P48 waits for its DP16 resources and remains a separate workstream.
