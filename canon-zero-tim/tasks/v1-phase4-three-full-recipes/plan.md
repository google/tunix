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
| V1.P4.3 | Host/real-env/exact-image admission | all positive and negative markers | complete |
| V1.P4.4 | Attempt-3 RPA-local repair and direct full target reruns | publish exact-image-green repair; GSM8K first-commit admission, then complete horizons with zero ALIGN FAIL | active (host green; exact-image and post-fix target not run) |

## Decisions

- Decision: P59 is accepted under the user's ordinary-JAX FP64 gradient-correctness policy; serial/update trajectory differences remain disclosed.
- Decision: attempt-2 target evidence VETOES APC for M15/main. APC remains on only for P45; B rescore always resets the cache and the strict gate is unchanged.
- Decision: the profiled update is excluded from steady-state performance means.
- Decision: launch full-horizon jobs directly with no short canary. Start GSM8K first; after its first real optimizer commit passes strict alignment and the registered P59-local/fixed-head/optimizer receipts, keep it running and start P45 followed by M15 from the same exact source SHA.
