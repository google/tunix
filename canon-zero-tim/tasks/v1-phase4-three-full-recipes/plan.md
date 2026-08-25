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
| V1.P4.4 | Attempt-6 P59 staged-spec repair plus uniform APC-off/JAX-cache receipt hardening | host + pinned-image + one-host TPU mechanism | superseded before target publication; admission evidence preserved |
| V1.P4.5 | Attempt-7 first-red numerical localization for P59 grouped backward | pre-registered no-commit receipts identify the first bad boundary; matched DP2xTP2 carrier and target diagnostic remain fail-closed | active |

## Decisions

- Decision: P59 is accepted under the user's ordinary-JAX FP64 gradient-correctness policy; serial/update trajectory differences remain disclosed.
- Decision: attempt-2 target evidence VETOES APC for M15/main; user elected the same APC-off production policy for P45. All three full recipes are APC-off. B rescore always resets the cache and the strict gate is unchanged.
- Decision: all three manifests retain the P33 JAX persistent-cache bucket. Exact restore/save receipts are mandatory carrier evidence; miss/error remains a performance limitation, not a numerical verdict.
- Decision: the profiled update is excluded from steady-state performance means.
- Decision: launch all three full-horizon jobs in one wave with no short canary and no cross-recipe first-commit dependency. Each recipe independently passes strict alignment plus its registered P59-local/fixed-head/token/APC/optimizer receipts while the other healthy runs continue from the same exact source SHA.
- Decision correction (2026-08-25): max-scaled L2 is an overflow-safe observer, not an admitted optimizer repair. Attempt 7 did not establish that the finite gradient magnitude was legitimate. No full recipe may use stable clipping to turn an unexplained `norm=inf` into an optimizer transaction; first localize the earliest bad numerical boundary in a zero-commit carrier.
