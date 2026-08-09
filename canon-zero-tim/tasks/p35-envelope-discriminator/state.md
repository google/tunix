# P35 envelope discriminator state

- Status: active; P35.2 target complete, P35.3 exact-input replay in progress
- Active phase: P35.3 exact-input replay
- Task directory: `canon-zero-tim/tasks/p35-envelope-discriminator/`
- Directory state: tracked
- Branch at bind: `codex/p34-scheduler-contract-0809`
- Reviewed implementation commit: `366ac2b1ff2806b48646a0188927e724bf569978`
- Updated: 2026-08-09 UTC

## Objective

Causally separate packing/metadata geometry from wrapper/program context at the first red
production boundary, `S_prefill != T_old`, on 64-chip Pathways. Do not use backward, optimizer,
reward or correlation to classify this boundary.

## Last verified facts

1. In both returned r18 workloads the action-only `S_decode == S_prefill` boundary is bitwise
   exact, while `S_prefill != T_old` is red.
2. The historical byte counts were divided by action-token counts. That is not a token mismatch
   rate. The valid byte fractions are 20.0% for GSM8K and 23.7% for FrozenLake; an element-level
   count was not recorded.
3. GSM8K r18 used a wrong serving scheduler contract: DP16 expanded per-rank `M=4096` into a
   global `M=65536`, while the adapter used global `M=4096`, local `M=256`.
4. FrozenLake r18 already used per-rank `16/256`, so its serving and adapter canonical local M
   were both 256. M mismatch alone cannot explain the FrozenLake red boundary.
5. The real `T_old` hot path is an outer JIT plus `lax.map`, with each 256-token group calling
   the complete `runner.model_fn`. It is not the P28 per-layer segmented backward tool.
6. The generic Pathways way-count probe proves that `jit(f)` and
   `jit(value_and_grad(f)).primal` may differ even in a replicated no-reduction arm. It is a
   platform mechanism clue and a future THIRDPROG risk, not a causal proof for the current pair
   of forward-only envelopes.
7. The same-session canonical Qwen operator probe is bitwise exact across its tested program
   contexts. Canonical operators remain useful, but they do not prove the whole model envelope.
8. GSM8K r19 corrected the scheduler contract to global M4096/local M256, but the action-only
   `S_prefill != T_old` result was effectively unchanged. M mismatch is therefore excluded as
   the load-bearing carrier for that boundary.
9. P35 attempt r21 completed rollout but failed before the three-arm producer while computing
   native reference logprobs: Splash query block 256 did not divide sequence length 1088
   (`1024 + 64`). It produced no P35 report or classification and is not a numerical target.
10. P35 attempt r24 confirmed that response 256 removes the Splash divisibility failure and ran
    the unchanged A rescore, but a diagnostic-only single-chunk assertion rejected the selected
    sequence group before B. The assertion confused one request per rank with one chunk per
    request. r24 produced no report or classification and is not a carrier verdict.
11. P35 attempt r25 stopped in the Pathways compilation service before A/B/C and emitted no P35
    report. It is an infrastructure interruption, not a numerical verdict.
12. P35 attempt r26 completed rollout, native arm A, reference logprobs and two arm-B metadata
    records, then stopped in exact weight attestation because one leaf was in host memory and the
    other was in device memory. JAX rejects a single `eq` with mixed memory-space input types.
13. A four-chip one-host v5p probe reproduced the same mixed-memory exception on JAX 0.10.2.
    Explicitly placing the host leaf into the device `NamedSharding` made equal values pass and a
    changed-value negative control fail. This validates the placement repair direction, not the
    target A/B/C result.
14. P35 attempt r28 completed one source-pinned A/B/C measurement on 64-chip Pathways. A versus B
    was bitwise exact at all 3,244 selected action elements, while B versus C and direct A versus C
    each differed at 1,529/3,244 elements and 3,106/12,976 bytes. The mechanical classifier returned
    `COMPLETE / adapter_envelope_carrier` and the injected one-bit negative control was observed.
15. r28 attested all 310 mapped/live leaves bitwise equal, but every pair crossed
    `pinned_host->device`. This excludes different weight bits, not memory placement as an executable
    variable. Likewise `metadata_B_matches_C` proves the selected sequence semantics, positions,
    request distribution and active page contract; it is not a byte-for-byte equality claim for the
    complete B and C metadata/cache tensors.

## Current hypothesis split

- H1, packing/metadata carrier: real scheduler packing, page/block tables, positions, cache
  ownership or request grouping differ from the adapter-generated envelope.
- H2, program-context carrier: identical tensor inputs differ because the serving wrapper and
  adapter wrapper compile/lower the same `runner.model_fn` in different program contexts.
- H3, both carriers: H1 and H2 are independently load-bearing.

- H1 is not load-bearing for the measured r28 group: changing native serving from dynamic A to
  grouped B left every selected action logprob bitwise exact.
- The remaining carrier is inside the B-serving versus C-adapter envelope. P35.3 must separate
  exact captured metadata/cache inputs, weight memory placement, adapter outer-program context and
  the processed-logprob tail before naming a kernel or compiler cause.

## Completed locally

- P35 schema v2 records A-B, B-C and direct A-C element/byte counts and masked hashes.
- The producer selects the exact rank-strided C group containing the current first A-C mismatch;
  it refuses a batch that does not reproduce the known red boundary.
- A is the unchanged native serving rescore. B reuses the native serving API with one selected
  request per DP rank. C is the existing canonical adapter value.
- Compact arm-labelled engine evidence records tokens, positions, sequence lengths, query starts,
  request distribution, active block tables, prefix-cache reset, cache contract and concrete mesh order. It does not dump
  hidden states, logits, model weights or cache contents.
- Trainer-anchor leaves are mapped to live engine leaves and compared bitwise on device. Checksums
  remain provenance only; signed zero and one-bit drift are rejected.
- The classifier rejects missing arms, red attestations, ineffective negative controls,
  count/hash inconsistency, bitwise-transitivity violations and an exact A-C pair that failed to
  reproduce the known production red.
- The runner accepts only diagnostic exit 1 plus exactly one stop marker, one immutable report and
  a `COMPLETE` classification. It rejects missing marker/report and every other exit code.
- The P35 response contract is uniquely 256 in the renderer, workload command, recipe and cluster
  preflight. A renderer-to-preflight integration test accepts 256 and rejects the known-bad 64.
- B and C retain canonical local M256 while admitting sequences that span multiple scheduler
  records. The metadata gate reconstructs every rank's complete token stream across records and
  checks contiguous positions, cumulative KV lengths, at most one active request per rank,
  request distribution, active page IDs and complete sequence coverage. Missing tail chunks are
  rejected.
- Pinned-image CPU gate PASS; qwen1p7b and qwen8b overlay installs each matched all 29 manifest
  entries and passed 10/10 prompt/decode chunk tests.
- `git diff --check`, Python AST checks and shell syntax checks PASS.
- Mixed host/device exact comparison now normalizes one leaf at a time into the existing device
  sharding. Both operand orders pass on a four-chip one-host v5p; signed-zero and one-bit negative
  controls fail as required. The adapter suite, complete CPU gate and both exact-image model gates
  pass.
- P35.3 is implemented behind `CANON_P35_EXACT_REPLAY=1`. Its six-arm chain is
  B -> R0(captured/live) -> R1(captured/mapped) -> R2(adapter metadata/direct) ->
  R3(production adapter repeat) -> C(original production value). B/R0 and R3/C are hard anchors;
  placement, metadata/cache construction and outer-program boundaries are classified separately.
- The replay repeats R0, R1 and R2, reports compact target-stage equality, refuses ineffective
  negative controls and known-red reproduction failures, and prints SHA-256 for every returned
  JSON. Full model weights, full logits and cache tensors are not serialized.
- Focused pinned-image replay tests PASS; the complete P33/P35 CPU gate and both exact-image model
  installs PASS. The complete adapter/envelope suite is 40 PASS/5 skipped. A real four-device
  one-host v5p TP4 smoke passed the replay and exact-equality controls (2 PASS, 35.90s). The
  default-off implementation is published as `366ac2b1`; no target P35.3 run or cloud-resource
  lifecycle change has occurred.

Evidence: `artifacts/p35_1_local_gate.md`, `artifacts/p35_2_local_gate.md` and
`artifacts/p35_3_local_gate.md`.

## Next action

Render one source-pinned r29 Attempt-0 JobSet from published commit `366ac2b1` exactly as recorded
in `cluster/P35_ENVELOPE_HANDOFF.md`. Copy all JSON, metadata and raw logs before deleting the
coordinator Pod; the state directory is on `/tmp`.

## Hard gates

- Missing expected measurement rows is `INCONCLUSIVE`, never PASS.
- Any red earlier gate makes downstream values from that run VOID.
- A target numerical conclusion requires all three arms A/B/C in one source-pinned run.
- Direct A-C must reproduce the known red boundary. Exact A-B plus exact B-C with red A-C is a
  transitivity failure and therefore `INCONCLUSIVE`, never a pass.
- Bitwise verdicts use `differing_elements == 0` and matching masked hashes. Correlation and
  `max_abs` are descriptive only.
- Target inputs must attest weights, mesh/device order, token IDs, validity/action masks,
  positions, attention metadata, page tables, cache initialization and canonical local M.

## Blockers

The P35.3 implementation has only local CPU, exact-image and bounded one-host TP4 evidence. The
published 64-chip launch remains an operator action.

## Rollback

Leave `CANON_P35_ENVELOPE` and `CANON_P35_EXACT_REPLAY` unset. The A path, training, backward,
optimizer, W&B and credentials remain unchanged. Preserve all earlier logs and handoffs.
