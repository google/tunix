# P35 envelope discriminator state

- Status: active; multi-chunk P35.2 repair locally implemented, r24 failed before measurement
- Active phase: P35.2 target admission
- Task directory: `canon-zero-tim/tasks/p35-envelope-discriminator/`
- Directory state: tracked
- Branch at bind: `codex/p34-scheduler-contract-0809`
- Reviewed base commit: `b2de4f16bf1a0d691ff027c7d74515ad911cc081`
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

## Current hypothesis split

- H1, packing/metadata carrier: real scheduler packing, page/block tables, positions, cache
  ownership or request grouping differ from the adapter-generated envelope.
- H2, program-context carrier: identical tensor inputs differ because the serving wrapper and
  adapter wrapper compile/lower the same `runner.model_fn` in different program contexts.
- H3, both carriers: H1 and H2 are independently load-bearing.

No hypothesis is green yet.

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

Evidence: `artifacts/p35_1_local_gate.md` and `artifacts/p35_2_local_gate.md`.

## Next action

Publish the reviewed multi-chunk repair, resolve and verify the resulting
`origin/yuxzhang/canon-zero-tim` SHA, render the one GSM8K envelope-short JobSet as run r25, run a
server-side Kubernetes dry run, then let the
operator launch Attempt 0. The target
must stop before backward and return the raw log, schema-v2 report, compact metadata records,
classification and SHA-256 values. Until that happens no carrier is classified.

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

The target JobSet must pin the reviewed SHA published on `yuxzhang/canon-zero-tim`; the operator
must resolve and verify that SHA before rendering. The 64-chip launch remains an operator action
on the GKE cluster.

## Rollback

Leave `CANON_P35_ENVELOPE` unset. The A path, training, backward, optimizer, W&B and credentials
remain unchanged. Preserve all r18/r19 logs and prior handoffs.
