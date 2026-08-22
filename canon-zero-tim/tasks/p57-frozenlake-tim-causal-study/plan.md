# Plan

## Outcome

Run a causal paired study of trainer-inference mismatch (TIM) on dense
Qwen3-8B FrozenLake. Workload discovery uses the untreated high-throughput
serving regime (`CANON_P57_INFERENCE_REGIME=stock-fast`), with the complete
numerical zero-TIM bundle disabled, because it is faster and because inspecting
zero-TIM learning during benchmark selection would bias the later comparison.
One no-update temperature-0.7 calibration compares three mixed long-context
recipes: M10 (5x5–10x10, 10 turns), M15 (5x5–12x12, 15 turns), and M20
(5x5–15x15, 20 turns). The selected recipe then receives one complete stock
200-update curve. Only after that curve satisfies the frozen selection rule may
any zero-TIM learning outcome be observed.

The study tests a hypothesis; it is not structured to guarantee a positive
TIM effect. A valid null result means only that the frozen dense
FrozenLake/GSPO-RLOO recipe is robust to the measured mismatch dose.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| P57.0 | Local stock-fast calibration, data, renderer, and evaluator readiness | Host and pinned-image gates pass; rendered and resolved environments mechanically prove the zero-TIM bundle is off | complete |
| P57.1 | Stock-only workload calibration and full-curve selection | One no-update temperature-0.7 M10/M15/M20 calibration is complete; one preregistered mixed recipe then completes a stock 200-update curve ending in 60–70%; no zero-arm result is launched or inspected | active |
| P57.2 | Immutable recipe and treatment-dose admission | Main data split and full recipe are frozen before zero unblinding; bounded zero arm is exact and stock arm has finite reproducible A-B dose | pending |
| P57.3 | Paired one-seed operational pilot | Both frozen arms complete 50 updates with valid checkpoints/evaluations and unchanged treatment dose | pending |
| P57.4 | Paired multi-seed main campaign | Two arms x at least three paired seeds complete the fixed 200-update horizon and isolated evaluations | pending |
| P57.5 | Preregistered analysis and claim decision | Capability, stability, mechanism, and systems results are complete under the claim ceiling | pending |

Exactly one phase may be active. Expensive TPU launches require explicit user
approval. Advancing a phase requires its exit gate and a decision entry in
`log.md`.

## Discovery-before-unblinding invariant

- P57.1 renders exactly one arm: `mismatch`, with
  `CANON_P57_INFERENCE_REGIME=stock-fast` and the 12 presence-sensitive
  numerical switches absent. Calibration/evaluation keep every numerical and
  alignment gate zero; stock training keeps launch/checkpoint/telemetry and
  warning-only observation while its rollout/trainer treatment bundle remains
  zero. Processed-B is the sole observer-only exception: mismatch training
  installs a signed two-file runner/helper delta that is reached only by the
  post-rollout `S_prefill` request. It applies temperature/top-k/top-p and uses
  absolute request-history target IDs; it never supplies old logprobs or a
  gradient input. Calibration/evaluation retain the byte-identical stock runner.
- “Stock-fast” means the untreated rollout-A and trainer-C numerical programs.
  It does not mean the B measurement is absent or knowingly mislabeled. Shared
  nonnumerical infrastructure remains equal: image, model/TP overlay, DP8xTP8,
  vLLM capacity, sampling, resident placement, and datasets.
- P57 training deliberately disables token sampler importance sampling in both
  arms. `sampler_is=None`, `use_rollout_logps=True`, and the learner must prove
  that rollout A is `old_per_token_logps` and `sampler_is_weights` is absent.
  Standard GSPO policy-ratio clipping at epsilon 0.003/0.005 remains part of
  the shared base algorithm; TIM-aware TIS weights, substituting trainer B/C
  for A, and mismatch-conditioned filtering or reweighting are forbidden.
- P57.1 may inspect stock solve rates only. It may not launch, read, or use a
  zero-arm learning outcome.
- Calibration order is `m10`, `m15`, `m20`. All three are evaluated from the
  same immutable base weights at temperature 0.7 with eight generations per
  map, without trainer rescore, backward, optimizer updates, or checkpoint
  writes.
- Recipe selection requires stochastic solve, mixed-group/nonzero-advantage,
  context-length, and physical-cap receipts. The eligible recipe closest to 20%
  stochastic solve receives one full stock 200-update curve; ties prefer M15,
  then M10, then M20. There is no greedy or train-20 rejection stage.
- P57.1 uses one uninterrupted stock 0→200 training JobSet. There is no eval-0
  and no 50/100/150 evaluation pause. The discovery curve is the signed
  on-policy training trajectory; checkpointing remains recovery infrastructure.
- Ideal automatic freeze: the preregistered trailing update-200 on-policy solve
  statistic is 60–70%, with valid treatment-dose and trajectory-health receipts.
- The 55–75% band is a hard review guardrail, not an alternate automatic
  target. A result in 55–60% or 70–75% stops for user review. Outside it is
  rejected as floor/ceiling.
- The selected recipe uses a disjoint `selection` split for its stock curve;
  the unseen `main` split is signed before the first
  zero-arm outcome is unblinded. A task is never selected for a large observed
  zero-minus-stock gap.
- `p57cal6` selected M15 mechanically: 24.625% solve, 56% mixed groups,
  context max 7,403, completion max 6,223, and no cap hit. The original receipt
  is immutable; missing map provenance was derived into a new receipt by exact
  deterministic rematerialization and `group_id` join, then accepted by the
  unchanged classifier as `PASS / FREEZE_M15`.

## Paired-arm invariants (must be implemented and certified in P57.2)

- The causal treatment is the complete numerical zero-TIM bundle:
  - stock/mismatch: untreated stock-fast serving and corresponding untreated
    trainer numerical paths;
  - zero: the fully registered canonical forward/backward/serving bundle,
    including the fixed lm-head.
- Processed-B observation is shared instrumentation rather than treatment. It
  must be enabled in both training arms solely to compare A and B under the
  same temperature/top-k/top-p semantics; it is dormant during sampling and
  is not consumed by the loss. The mismatch arm uses the minimal observer
  runner delta; the zero arm may reach the equivalent facility through its
  canonical runner, but both must prove the same B semantics and target IDs.
- Both arms use rollout A as the old-policy denominator and have no sampler-TIS
  weights. The runtime purity receipt is required exactly once per training
  run; processed B and trainer C may be observed but may not enter the
  `TrainExample` old-logprob or sampler-weight fields.
- This design estimates the system-bundle effect. It cannot attribute a result
  to lm-head, RoPE, attention, or any individual kernel.
- Initial checkpoint, source/image/model digests, maps and order, topology,
  seeds, rollout sampling, optimizer, learning rate, objective, clipping,
  horizon, checkpoint schedule, and performance flags are equal across arms.
- Zero is fail-closed on any A-B, B-C, A-C, nonfinite, transaction, replica,
  or checkpoint-contract failure.
- Stock may continue through finite A-B only. B-C and every non-treatment
  correctness failure remain fatal.
- Training never evaluates in-process. Isolated evaluation consumes immutable
  base weights or a signed durable checkpoint and an immutable map set.
- P57.1 runs continuously to 200; LatestN(1) is only a recovery mechanism in
  that discovery phase. The later paired campaign may register its own equal
  evaluation schedule for both arms before unblinding.
- No warning-only run may be described as bitwise zero-TIM.
- The existing paired renderer is staging-only until P57.2 proves that every
  nonnumerical field is equal and every numerical treatment field differs as
  preregistered. P57.1 classification does not authorize it.

## Registered outcomes

- Primary: held-out solve-rate area under the curve through update 200.
- Secondary: held-out solve at update 200 and time-to-threshold.
- Stability: collapse incidence and paired-seed dispersion.
- Mechanism: A-B dose, B-C exactness, ratio tail, clip hits, gradient/update
  norm, effective/mixed groups, context/turn/completion length, invalid action.
- Systems: seconds/update, sampled tokens/second, evaluation/checkpoint cost,
  and fixed-input warm replay overhead.

## Claim ceiling

- Positive: measured finite TIM harms capability or stability under this exact
  frozen dense Qwen3-8B recipe.
- Null: this recipe is robust to this measured dose; not that TIM is harmless.
- Missing dose: `NO_TREATMENT`; no causal claim.
- Nonexact zero, B-C failure, floor/ceiling, excessive truncation, or incomplete
  receipts invalidate the affected experiment.
- Three paired seeds support a campaign-level result, not a universal claim.

## Rollback

P57 profiles, renderer, evaluator, materialized data contract, and classifiers
are additive and default off. Leaving P57 fields unset or reverting the P57
changes restores the pre-P57 P45 behavior.
