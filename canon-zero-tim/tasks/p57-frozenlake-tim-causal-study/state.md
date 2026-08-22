# State

- Status: active
- Objective: measure native/no-IS, native/token-IS, and complete zero-TIM/no-IS on both the original P45 FrozenLake workload and frozen M15 using signed full-training curves and final isolated evaluations.
- Definition of done: all six cells pass local/target arm receipts, complete their fixed horizons and final evaluations, and produce within-workload `is-mismatch`, `zero-mismatch`, and `zero-is` contrasts under the registered claim ceiling.
- Task directory: `canon-zero-tim/tasks/p57-frozenlake-tim-causal-study`
- Directory state: tracked
- Current phase: [P57.1b — three-arm baselines](phases/p57-1b-three-arm-baselines.md)
- Last verified fact: the five-tuple runtime repair is target-proven by native runs `n45a` (two committed steps) and `n15a` (one committed step, then continued rollout). Token-IS `i45a` passed rollout, pre-backward warnings, arm purity, and backward, then failed post-backward because `alignment.check_batch` exempted only P57 stock `mismatch` from canonical Engine Module C, while the registered stock `is` arm also intentionally has `CANON_ENGINE_MODULE_C=0`. `i45a` is `INCONCLUSIVE`. A local one-line scope repair admits both registered stock arms; the pinned-image full gate and focused positive/unknown-arm negative passed. No numerical or optimizer semantics changed.
- Next action: from the published repair SHA, confirm/package any old `i15a` state, render only the `is` wave using fresh IDs `i45b/i15b` and campaign `p57-native-is-c`, record both manifest hashes, then request separate launch approval. Do not relaunch native or launch deferred `zero`.
- Blockers: authoritative old `i15a` status; then target launch approval and two independent 64-chip slices for concurrent IS replacements. The repaired IS post-backward path is target `NOT RUN`.
- Key artifacts: [plan.md](plan.md); [active phase](phases/p57-1b-three-arm-baselines.md); [RUNBOOK.md](RUNBOOK.md); [HANDOFF.md](HANDOFF.md)
- Updated: 2026-08-22 UTC
