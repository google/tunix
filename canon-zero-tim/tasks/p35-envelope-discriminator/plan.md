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

Status: locally complete; target r29 infrastructure-inconclusive

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

r29 result: NOT PASSED. The run entered the first captured/live replay but the IFRT proxy socket
closed before either report was written. The archived log proves two B records and one logical
`(4096, 151936)` float32 logits tensor per record; it does not prove OOM, host transfer or
autoscaler eviction. Replay must be bounded and instrumented before r30.

### Phase P35.3b — Bounded replay execution repair

Status: locally complete and published; target r30 infrastructure-inconclusive

Persist the completed P35.2 evidence before replay and serialize every captured record with
explicit begin/complete markers while preserving the original numerical program boundaries. A
fused target-only candidate was rejected after a CPU bitwise gate found 178/256 changed target
logprobs. Prove the serialized original path on CPU, exact image and one-host TP4 before
publishing an r30 source pin. See `phases/p35-3b-bounded-replay.md`.

### Phase P35.3c — First-record stage localization

Status: locally complete; implementation source pin `7484ab78`; no target run

r30 wrote the preliminary report but lost the IFRT service inside the first captured/live record.
The canonical target-logprob callable was already jitted, so the Python stack cannot identify an
unjitted scorer as the cause. Add a default-off probe that preserves every numerical callable and
waits after model, logits, sampling, canonical logprob, target gathers and compact output assembly.
Append one fsynced JSONL event per ready stage and stop after record 1 with an explicit
`NO_NUMERICAL_VERDICT` marker.

Exit gate: local classifier, renderer, postflight, CPU, exact-image and TP4 mechanics gates pass;
then one source-pinned r31 Attempt 0 mechanically identifies the last ready stage while archiving
the complete Pathways service and Kubernetes evidence. See
`phases/p35-3c-first-record-stage-probe.md`.

Local result: PASS. Focused classifier/renderer tests pass 14/14; the complete P33/P35 CPU
contract, both exact-image overlays and a real four-device v5p TP4 production-shape mechanics
test pass. The TP4 test uses a synthetic forward and proves array/stage mechanics only. See
`artifacts/p35_3c_local_gate.md`. Target r31 remains NOT RUN.

## Phase P35.4 — Actual-model THIRDPROG

Status: pending

Only after `S_prefill == T_old`, compare the actual Qwen forward-only value with the primal
returned by its real `value_and_grad` program. Do not transfer the generic way-count verdict to
the production model.

Exit gate: complete action distribution is bitwise exact, or the run remains red with the first
divergent actual-model boundary recorded.

## Phase P35.3a — Direct-attached one-host reproduction

Status: completed; local carrier not reproduced

Run one bounded DP1xTP4 diagnostic on the existing four-chip v5p host before r29. Match one target
DP rank's 16-trajectory local geometry, keep local M256, stop before backward and disable W&B.
This is a platform contrast, not a substitute for the 64-chip result.

Exit gate: either one complete fail-closed six-arm report or an explicit
`LOCAL_NOT_REPRODUCED` result from the known-red guard after A/C measurement. See
`phases/p35-3a-onehost-reproduction.md`.

Result: `LOCAL_NOT_REPRODUCED`. The DP1xTP4 direct-attached run found no bitwise A/C action
mismatch and stopped before B/replay as required. This narrows the known r28 carrier toward the
64-chip Pathways/multi-host envelope; it does not replace target r29.

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
