# Plan

## Outcome

Prepare exactly three uninterrupted 64-chip Zero-TIM full trains. A full run is
the first target-topology certification; there is no separate short canary.
Every run stays strict and carries one warmed update-window XProf capture plus
the semantic Perfetto timeline.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| V1.P4.1 | Integrated default-off P56/P59/APC implementations | syntax, focused CPU, manifest | complete |
| V1.P4.2 | Three immutable manifests and intent verifier | exactly three renderer PASS records | complete |
| V1.P4.3 | Host/real-env/exact-image admission | all positive and negative markers | active (host/real-env green; image pending) |
| V1.P4.4 | Full target runs | complete horizons, zero ALIGN FAIL, profiles packaged | pending |

## Decisions

- Decision: P59 is accepted under the user's ordinary-JAX FP64 gradient-correctness policy; serial/update trajectory differences remain disclosed.
- Decision: APC is on only for the two multi-turn FrozenLake recipes; B rescore always resets the cache.
- Decision: the profiled update is excluded from steady-state performance means.
