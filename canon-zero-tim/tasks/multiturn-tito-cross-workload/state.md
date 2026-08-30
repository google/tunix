# State

- Status: one-host complete; production target deliberately unverified
- Objective: classify multi-turn token continuity independently for DeepSWE and FrozenLake M15, then prove the M15 legacy-observer and exact-token paths on one v5p host while keeping M15 full default off.
- Definition of done: a real-tokenizer transcript oracle covers both role geometries; a M15 DP1xTP4 zero-backward/zero-commit carrier returns per-turn prompt-token verdicts plus strict A/B/C; the DeepSWE control remains green if shared runtime changes. DP8xTP8 remains a separate target gate.
- Task directory: `canon-zero-tim/tasks/multiturn-tito-cross-workload`
- Directory state: tracked candidate; not ignored and uncommitted
- Current phase: T6 — target decision recorded
- Last verified fact: exact r8 is `EXACT_TOKEN_CONTINUITY_ALIGNMENT_PASS`: 17/17 receipts equal, three B-full-reset receipts, three strict A/B/C rounds all zero, and zero backward/optimizer. Its ordered prompt receipts and round token/action-mask hashes match legacy r7.
- Next action: keep M15 full selector absent by default. If the explicit exact
  option is selected later, treat that DP8xTP8 full as the target gate; do not
  inherit the one-host claim.
- Blockers: none. DP8xTP8 remains deliberately unverified and unauthorized, not a blocker to the leave-off decision.
- Key artifacts: `../p58-deepswe-native-zero-comparison/state.md`; `../v1-phase4-three-full-recipes/phases/v1-p4-16-m15-nontito-curve-first.md`
- Updated: 2026-08-30T19:35:19Z
