# P35 envelope discriminator state

- Status: active
- Active phase: P35.2 three-arm producer
- Task directory: `canon-zero-tim/tasks/p35-envelope-discriminator/`
- Directory state: tracked
- Branch at bind: `codex/p34-scheduler-contract-0809`
- Base commit: `ad309a810e35121d7d25db67c32c2712d9f8e086`
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

## Current hypothesis split

- H1, packing/metadata carrier: real scheduler packing, page/block tables, positions, cache
  ownership or request grouping differ from the adapter-generated envelope.
- H2, program-context carrier: identical tensor inputs differ because the serving wrapper and
  adapter wrapper compile/lower the same `runner.model_fn` in different program contexts.
- H3, both carriers: H1 and H2 are independently load-bearing.

No hypothesis is green yet.

## Completed locally

- P35.1 report schema is additive and preserves the legacy byte field.
- Exact-image alignment tests: 13/13 PASS.
- Three-arm classifier tests: 5/5 PASS.
- Native serving grouped-rescore tests: 6/6 PASS, including the complete-group positive control
  and partial-group rejection.
- Negative controls cover one ULP, signed zero, masked-out full-array drift, missing arms, red
  attestations, unobserved injected drift and hash/count inconsistency.
- `git diff --check`, Python compilation and executable English-only scan PASS.

Evidence: `artifacts/p35_1_local_gate.md`.

## Next action

Wire the grouped native-rescore B arm into a default-off, pre-backward P35 producer. It must emit
the classifier schema, verify expected measurement count and stop before backward. The remaining
admission gaps are actual page/block metadata attestation and exact trainer-anchor versus engine
weight attestation. Do not ask the operator to launch the 64-chip run until both are fail-closed.

## Hard gates

- Missing expected measurement rows is `INCONCLUSIVE`, never PASS.
- Any red earlier gate makes downstream values from that run VOID.
- A target numerical conclusion requires all three arms A/B/C in one source-pinned run.
- Bitwise verdicts use `differing_elements == 0` and matching masked hashes. Correlation and
  `max_abs` are descriptive only.
- Target inputs must attest weights, mesh/device order, token IDs, validity/action masks,
  positions, attention metadata, page tables, cache initialization and canonical local M.

## Blockers

The local observability work is unblocked. The eventual 64-chip target run requires the user's
separate launch decision and an operator on the GKE cluster.

## Rollback

Keep `CANON_PRE_ALIGN_GATE` disabled and do not render the P35 target manifest. Local report-field
changes are additive; reverting them must preserve all r18 logs and prior handoffs.
