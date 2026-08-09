# P35 envelope discriminator plan

## Phase P35.1 — Contract observability

Status: completed

1. Add action-element counts, exact denominators, byte/element fractions and action-masked hashes
   to both pre-backward and four-boundary reports.
2. Preserve legacy `differing_bytes` fields and log lines for classifier compatibility.
3. Add positive and negative CPU gates, including signed zero and a full-array drift outside the
   action mask.
4. Publish a source-pinned handoff that corrects the r18 metric interpretation.

Exit gate: focused unit tests pass, negative controls are observed to fail closed, and the report
schema is documented.

Result: PASS. See `artifacts/p35_1_local_gate.md`.

## Phase P35.2 — Three-arm Pathways discriminator

Status: in progress

Produce all three value arms before backward in one source-pinned process:

- A: native serving rescore with dynamic packing.
- B: native serving rescore constrained to one sequence per DP rank and canonical local M256.
- C: current adapter rescore constrained to the same sequence and local M256.

Attest identical weights, selected action IDs, masks, positions and policy version. Record exact
element/byte counts and masked hashes for A-B and B-C.

Decision table:

| A vs B | B vs C | Classification |
|---|---|---|
| red | exact | packing/metadata carrier |
| exact | red | wrapper/program-context carrier |
| red | red | both carriers |
| exact | exact | r18 contract/configuration carrier removed; advance to THIRDPROG |

Exit gate: exactly one complete A/B/C row, expected measurement count verified, all producer
attestations present, and an injected-drift negative control is rejected.

## Phase P35.3 — Exact-input replay if B vs C is red

Status: pending

Capture one real B input contract in process, then invoke the direct `runner.model_fn` entry and
the adapter wrapper with identical leaves, IDs, positions, attention metadata and initialized
caches. Store hashes and selected target statistics, not full model weights or cache tensors.

Exit gate: the first semantic boundary is localized to raw logits, log-softmax/sampling tail or
an earlier hidden-state checkpoint without changing precision, shape, loss or reductions.

## Phase P35.4 — Actual-model THIRDPROG

Status: pending

Only after `S_prefill == T_old`, compare the actual Qwen forward-only value with the primal
returned by its real `value_and_grad` program. Do not transfer the generic way-count verdict to
the production model.

Exit gate: complete action distribution is bitwise exact, or the run remains red with the first
divergent actual-model boundary recorded.

## Commands

Local gate:

```bash
python3 -m pytest -q tests/rl/alignment_test.py
git diff --check
```

Target commands are intentionally withheld until P35.2 has a fail-closed producer, classifier
and dry-run-validated manifest. Do not rerun r18 unchanged.
