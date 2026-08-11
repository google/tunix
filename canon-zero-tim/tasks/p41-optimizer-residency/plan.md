# Plan

## Outcome

Provide an explicit optimizer-placement switch for GSM8K and FrozenLake without
changing the default offload regime, precision, update arithmetic, gradient
order, or alignment policy.  Prove configuration behavior on CPU, then compare
one bounded GSM8K update on the existing DP1xTP4 v5p host.  FrozenLake receives
the same default-off switch but does not advance to full training in this phase.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| P41.1 | Explicit offload/resident contract in both recipes and P33 evidence | CPU tests accept exactly one placement and reject ambiguous or un-attested placement | passed |
| P41.2 | Bounded one-host GSM8K offload/resident comparison | Both arms execute one real update; parameter and optimizer evidence remains valid; resident arm does not OOM; elapsed/HBM are recorded | passed |
| P41.3 | Handoff and recommendation | State/log identify measured speed, peak HBM, claim boundary, and targeted rollback | passed |
| P41.4 | FrozenLake/Qwen3-8B resident capacity admission | One strict DP1xTP4 update reaches commit with device-resident optimizer state, finite nonzero gradients, exact alignment, valid state transition, and no OOM | not admitted |

## Decisions

- Confirmed: optimizer arithmetic already runs on TPU; offload adds synchronous optimizer-state H2D and D2H around each commit.
- Confirmed: the 1.7B optimizer is about 3.2 GiB per TP4 chip; the 8B optimizer is about 15.3 GiB per TP4 chip.
- Hypothesis: GSM8K device residency fits and shortens commit time; FrozenLake capacity is plausible but unproved during backward.
- Decision: introduce `CANON_OPT_STATE_RESIDENT`, default `0`; resident and offload are mutually exclusive actual placements.
- Decision: do not weaken any alignment, gradient, replica-equality, transaction, or precision gate.
- Confirmed: on DP1xTP4 Qwen3-1.7B, device residency is bitwise update-equivalent for the controlled one-update workload, increases measured peak HBM by 1.47 GiB per chip, and improves the optimizer transaction by 1.201x.
- Boundary: FrozenLake has the same switch and fail-closed placement attestation, but Qwen3-8B residency has not run through backward and remains unadmitted for production.
- Decision: P41.4 is resident-only. It answers capacity and transaction validity; it does not claim a FrozenLake speedup or replace the GSM8K pair comparison.
- Decision: FrozenLake alignment remains fully fail-closed. A serving-side A/B carrier is a numerical blocker, not an allowed warning in this admission.
- Boundary: a local DP1xTP4 pass does not admit DP16xTP4 Pathways or a multi-update production campaign.
- Result: `p41fl1` completed backward, resident commit, and weight sync without OOM; all four A/B/C records were exact and the aggregate gradient changed parameters. The pre-registered release gate still failed because only one of four stochastic FrozenLake microbatches had nonzero advantage/gradient.
- Capacity warning: the measured per-chip peak was 97,955,232,768 bytes against a 102,803,437,568-byte limit, leaving only 4,848,204,800 bytes (4.52 GiB). This is too thin to promote resident placement to the FrozenLake full-training default from a one-update canary.
- Rollback: leave `CANON_OPT_STATE_RESIDENT=0`; the existing pinned-host offload path remains the default.
