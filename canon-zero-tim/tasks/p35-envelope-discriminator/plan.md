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

Status: completed on target

Produce all three value arms before backward in one source-pinned process:

- A: native serving rescore with dynamic packing.
- B: native serving rescore constrained to one sequence per DP rank and canonical local M256.
- C: current adapter rescore constrained to the same sequence and local M256.

Select the exact C rank-strided group containing the current first A-C mismatch. Attest identical
weights, selected token IDs, masks, positions, policy version, mesh/device order, request
distribution and fresh-cache semantics. Record exact element/byte counts and masked hashes for
A-B, B-C and direct A-C.

Decision table:

| A vs B | B vs C | Classification |
|---|---|---|
| red | exact | packing/metadata carrier |
| exact | red | adapter-envelope carrier; exact-input replay is still required before naming program context |
| red | red | both carriers |
| exact | exact | reproduction failure/inconclusive; direct A-C must reproduce the known red |

Exit gate: exactly one complete A/B/C row, expected measurement count verified, all producer
attestations present, and an injected-drift negative control is rejected.

Result: PASS. r28 returned one complete source-pinned measurement:
`A==B`, `B!=C`, `A!=C`, classified as `adapter_envelope_carrier`.
See `debug_logs/p35_r28_gsm8k_envelope.json` and its classification artifact.

## Phase P35.3 — Exact-input replay if B vs C is red

Status: locally complete; target not run

Capture one real B input contract in process. The replay chain is:

- R0: captured B tensors, fresh cache, live leaves, direct model entry;
- R1: exact R0 tensors and direct entry, trainer-mapped leaves;
- R2: mapped leaves and direct entry, adapter-generated metadata/cache contract;
- R3: unchanged production adapter envelope replayed on the original complete batch;
- C: the original production adapter value.

This separates weight memory placement, metadata/cache construction and the outer `lax.map`
program. B/R0 and R3/C are hard anchors. Store exact reduced comparisons and compact selected
target stages, not full model weights, full-vocabulary rows or cache tensors.

Local exit gate: classifier/renderer negative controls, focused adapter replay, complete CPU gate,
both exact-image model installs and `git diff --check` pass. Result: PASS; see
`artifacts/p35_3_local_gate.md`.

Target exit gate: one Attempt-0 report reproduces B!=C, keeps both anchors and all repeats exact,
and classifies at least one of placement, metadata/cache construction or outer-program context.

## Phase P35.4 — Actual-model THIRDPROG

Status: pending

Only after `S_prefill == T_old`, compare the actual Qwen forward-only value with the primal
returned by its real `value_and_grad` program. Do not transfer the generic way-count verdict to
the production model.

Exit gate: complete action distribution is bitwise exact, or the run remains red with the first
divergent actual-model boundary recorded.

## Commands

Local gates:

```bash
sudo docker run --rm -v "$PWD:/workspace:ro" -w /workspace \
  -e PYTHONDONTWRITEBYTECODE=1 -e JAX_PLATFORMS=cpu \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  bash canon-zero-tim/tests/p33_workloads/run_cpu.sh
bash canon-zero-tim/tests/p33_workloads/run_exact_image.sh \
  tunix_frozenlake_image:vllm-tpu0.25.0
git diff --check
```

After a reviewed commit is pushed, render with
`canon-zero-tim/cluster/render_p35_jobset.py`, then require a server-side dry run before apply.
The exact operator commands are in `cluster/P35_ENVELOPE_HANDOFF.md`. Do not rerun r18 unchanged.
