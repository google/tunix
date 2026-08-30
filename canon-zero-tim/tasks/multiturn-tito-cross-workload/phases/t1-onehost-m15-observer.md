# T1 — One-host M15 rendered-text observer

- Status: complete

## Motivation

The offline Qwen3-8B user-turn fixture remained token-exact, while real
DeepSWE drifted. Only a live M15 request can decide whether FrozenLake's actual
sampled assistant/EOS and environment strings cross a retokenization seam.

## Registered carrier

- direct-attached one-host v5p, Qwen3-8B, DP1xTP4;
- M15/main, three bounded diagnostic rounds, concurrency one;
- APC off, rendered-text model call unchanged, continuity mode `verify`;
- strict pre-alignment A/B/C, B full reset, all cached-token counts zero;
- `CANON_P38_PRECHECK_ONLY=1`, controlled exit, no backward, no optimizer
  commit, no eval/checkpoint, no W&B;
- production P45/M15 profiles and renderers remain selector-absent.

The carrier reuses `CANON_P38_ONEHOST_REHEARSAL=1` as its diagnostic identity.
No new environment selector is introduced. The continuity selector admits
`verify` only for this local identity; `exact` stays rejected here so the first
live verdict cannot silently repair the path it is supposed to observe.

## Gates

1. Host selector tests must admit only the exact DP1xTP4/APC-off identity and
   reject exact mode, APC-on, TP drift, production profile leakage, eval,
   checkpoint, and missing controlled-exit fields.
2. The runner must refuse a busy lane and a reused output label.
3. Runtime must return three round-complete markers and at least one later-turn
   continuity receipt per round.
4. Any B-C byte difference, malformed/missing continuity receipt, backward,
   optimizer commit, APC hit, or unexpected Docker exit is fatal.
5. `TOKEN_STREAM_DIFFERENT` is a valid scientific result in `verify` mode; it
   classifies the legacy transport seam and is not an alignment waiver.

## Decision table

| Live result | Next action |
|---|---|
| all verify receipts equal and A/B/C exact | leave production M15 non-TiTO; TiTO has no correctness benefit shown |
| any verify receipt differs while B-C exact | freeze the first red turn and run exact replay T3 |
| B-C red | stop; not a TiTO-specific experiment |
| no later-turn coverage | carrier inconclusive; adjust only workload coverage |

## Result

Attempt r4 completed all three numerical rounds: 17/17 live legacy prompt
reconstructions were token-identical, A-B and B-C were zero bytes in every
round, and no backward or optimizer transaction ran. It is not yet a formal
PASS because the preregistered explicit B-full-reset receipt was missing even
though the raw log showed three vLLM reset operations. The classifier failed
closed on that single missing receipt. A fresh r5 with the fail-closed marker
repair is required; r1-r4 remain immutable evidence.

Fresh r7 completed the repaired contract. It emitted 17/17
`TOKEN_STREAM_EQUAL` receipts, exactly three explicit B-full-reset/all-zero
cache receipts, and three strict alignment PASS rounds with zero A-B and B-C
bytes over 515, 766, and 748 action tokens. It exited through controlled code
42 before backward or optimizer commit. All manifest entries verify. By the
preregistered decision table, M15 production remains non-TiTO and no exact
replay is required. This result is one-host DP1xTP4 evidence, not DP8xTP8
certification.
