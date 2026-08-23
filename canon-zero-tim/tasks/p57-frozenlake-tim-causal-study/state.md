# State

- Status: active
- Objective: measure native/no-IS, native/token-IS, and complete zero-TIM/no-IS on both the original P45 FrozenLake workload and frozen M15 using signed full-training and isolated 50-step milestone curves.
- Definition of done: all six cells pass local/target arm receipts, complete their fixed horizons and ten registered isolated evaluations, and produce within-workload `is-mismatch`, `zero-mismatch`, and `zero-is` contrasts under the registered claim ceiling.
- Task directory: `canon-zero-tim/tasks/p57-frozenlake-tim-causal-study`
- Directory state: tracked
- Current phase: [P57.1b — three-arm baselines](phases/p57-1b-three-arm-baselines.md)
- Last verified fact: `n45c/n15c/i45c/i15c` all completed real Step-0 rollout and then failed in shared trajectory packaging before trainer alignment/backward because the policy-seeded FrozenLake environment task lacked the rendered `prompts` and replaced the prompt-bearing trajectory task. They are immutable `INCONCLUSIVE` attempts with no update/checkpoint. Locally, the compatibility repair preserves prompt-bearing DeepSWE environment input, merges FrozenLake trajectory prompts with durable policy metadata, and keeps missing-prompt fail-closed. P57 host tests pass 119/119; both P45/P57 and P58 pinned production-image gates exit zero. Training mathematics, IS treatment, numerical kernels, optimizer, and checkpoint policy are unchanged.
- Next action: review the local prompt-provenance repair. After separate commit/push approval, rerender both train waves and both 20-manifest eval schedules from the immutable repaired SHA using `n45d/n15d/i45d/i15d`, `fn/fi`, and fresh `*-450-b` namespaces. Run four step-0 evals before training; after separate approval run the four uninterrupted 450-step trains; only after durable completion run positive milestone evals. Do not launch deferred `zero` yet.
- Blockers: review and commit/push approval; explicit acceptance of the several-terabyte four-arm GCS milestone envelope; then separate step-0-eval, training, and positive-eval launch approvals. The repaired 450-update path is `TARGET NOT RUN`; earlier 200-update and `*c` attempts remain immutable evidence and are not resumable into the replacement campaign.
- Key artifacts: [plan.md](plan.md); [active phase](phases/p57-1b-three-arm-baselines.md); [RUNBOOK.md](RUNBOOK.md); [HANDOFF.md](HANDOFF.md)
- Updated: 2026-08-23 UTC
