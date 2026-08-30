# Plan

## Outcome

Determine whether text rendering and re-tokenization actually changes later-turn
M15 prompts, whether exact token-in/token-out repairs that seam without moving
strict A/B/C, and whether the shared implementation preserves the already-proven
DeepSWE transport. Production DeepSWE remains TiTO; M15 full defaults to
non-TiTO and exposes exact only through an explicit target option. No phase may infer DP8xTP8 certification
from one-host evidence.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| T0 | Real-tokenizer transcript oracle for DeepSWE tool turns and FrozenLake user turns | exact per-turn hashes, role boundaries, first mismatch, and a poison negative that the oracle catches | complete |
| T1 | Isolated M15 one-host DP1xTP4 carrier and delivery chain | host positives plus production/P45/DeepSWE leakage negatives; no production renderer change | complete |
| T2 | Legacy rendered-text observer run | one real M15 trajectory with at least three nonterminal turns, per-turn `TOKEN_STREAM_*`, explicit B-full-reset receipt, no prompt override, no backward/commit | complete — r7 17/17 equal, 3/3 strict |
| T3 | Exact-token matched replay and strict alignment | same bounded carrier as r7; actual prompt equals reconstructed prompt; cross-arm prompt/trajectory hashes match; strict A=B=C; zero backward/commit | complete — r8 17/17 exact-equal, 3/3 strict, r7/r8 cross-arm MATCH |
| T4 | Shared-runtime DeepSWE regression control | rerun existing Qwen3-4B DP1xTP4 controlled carrier only if shared token helper/runtime changed | not required — no DeepSWE/token helper change; scoped rollout regressions pass |
| T5 | Paired performance attribution | correctness-green legacy/exact captures report prompt tokens, host tokenization, prefill/decode and compile separately | deferred — bounded wall time is 489s legacy versus 499s exact, but no component-level causal performance claim |
| T6 | Target decision | decision table records one-host TiTO health separately from production/DP8xTP8 admission | complete — one-host healthy; production remains selector-absent pending DP8xTP8 target |

## Decisions

- Confirmed: DeepSWE TiTO is a common transport invariant selected by the DeepSWE workload identity and has real DP1xTP4 evidence.
- Confirmed: M15 `verify|exact` currently admits only the production DP8xTP8 full identity, so it cannot honestly be used on one host without a dedicated diagnostic identity.
- Confirmed: both workloads use the Qwen parser, whose assistant-end updater appends zero tokens; FrozenLake differs by `enable_thinking=False` and user-role environment messages, while DeepSWE uses tool/user messages.
- Decision: the available direct-attached host and the existing certified rehearsal carrier expose four devices, so use M15 DP1xTP4 locally. This proves token transport and strict local alignment only; target TP8 remains explicitly unverified.
- Decision: compare live arms only internally. Cross-arm causal comparison uses a frozen trajectory/turn capsule because identical seeds do not guarantee identical sampling after prompt IDs diverge.
- Decision: diagnostics are strict, APC-off, max-concurrency one, zero backward, and zero optimizer commit. The production M15 full recipe remains untouched.
- Hypothesis: M15 may or may not reproduce DeepSWE retokenization drift; the legacy observer must decide before exact TiTO is described as a repair.
- Finding: a real 23-turn DeepSWE trajectory yields 10/11 later-turn drift receipts, beginning at turn 2/token 2242; a short synthetic FrozenLake user-only transcript yields 2/2 equality. The role geometry therefore matters and the synthetic M15 fixture cannot authorize production TiTO.
- Finding: live M15 r4 yields 17/17 later-turn equality across three rounds and strict A=B=C, but the formal classifier correctly rejected the run because its explicit B-full-reset marker was absent. The fail-closed observation is repaired and host-gated; fresh target evidence remains required.
- User decision: legacy equality is insufficient; certify M15 exact TiTO itself on the matched one-host carrier. This reopens T3/T6 without changing the production default or claiming DP8xTP8.
- Finding: exact r8 is healthy on the matched one-host carrier: 17/17 exact-token receipts, three strict A=B=C rounds, and zero backward/commit. Ordered prompt receipts and per-round token/action-mask hashes match legacy r7 exactly.
- Decision: this closes one-host TiTO correctness. It authorizes keeping an
  explicit exact M15 full option in the renderer, default off; using that
  option is itself the DP8xTP8 target admission boundary.
