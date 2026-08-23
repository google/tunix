# Plan

## Outcome

Estimate the functional and systems effects of trainer-inference mismatch on
dense Qwen3-8B FrozenLake with a two-workload, three-treatment study. The easy
historical P45 workload tests ceiling masking; the selected long-context M15
workload tests a harder 24.625%-initial-solve regime. Within each workload the
three cells differ only in the registered numerical/sampler treatment:

1. native/no-IS (`TIM_ARM=mismatch`): stock-fast A/C, rollout A is old logprob,
   no TIS weights;
2. native/token-IS (`TIM_ARM=is`): the identical stock-fast A/C program,
   trainer C is old logprob and token TIS is present;
3. zero-TIM/no-IS (`TIM_ARM=zero`): the complete canonical numerical bundle,
   rollout A is old logprob, no TIS weights.

Processed B is shared observation only. Standard GSPO current/old ratio and
epsilon 0.003/0.005 clipping remain in all cells. The study tests a hypothesis;
it is not structured to guarantee a positive zero-TIM result.

Execution is staged by user decision. Queue the four native-program cells
first: P45 and M15 under native/no-IS and native/token-IS. The two Zero-TIM
cells remain part of the six-cell design but are deferred until a separate
launch decision after the first four runs are packaged.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| P57.0 | Stock-fast calibration and M15 selection | `p57cal6` classifier PASS/FREEZE_M15 | complete |
| P57.1 | Former selection-only native curve | Preserved historical evidence | superseded |
| P57.1b | Four native-program cells | CPU + exact-image gates; P45 and M15 native/no-IS plus native/token-IS complete full horizons | active |
| P57.2 | Deferred complete Zero-TIM pair | Separate user launch decision; P45 and M15 zero/no-IS cells complete with strict receipts | pending |
| P57.3 | In-process held-out evaluation and within-workload contrasts | Exact rollout-only evaluations at 0,50,...,300 and preregistered contrast table | pending |
| P57.4 | Replication/claim decision | Either stop at concept-study ceiling or run paired multi-seed campaign | pending |

Exactly one phase is active. Every TPU launch and every commit/push requires
separate user approval.

## Frozen workload contracts

| Field | P45 historical workload | P57 M15 workload |
|---|---|---|
| data | original deterministic generator, train seed 42/eval seed 123, side 2–9, p 0.60–0.85 | materialized `m15/main`, side 5–12, p 0.82 |
| turns | 5 | 15 |
| prompt/response cap | 4,096 / 2,048 | 4,096 / 8,192 |
| updates | 300 | 300 |
| topology | DP8xTP8 | DP8xTP8 |
| rollout rows | 32 prompts x 8 = 256 | 32 prompts x 8 = 256 |
| optimizer | AdamW 1e-6, b1 0.9, b2 0.95, wd 0, resident | same |
| objective | GSPO-token, RLOO, beta 0, epsilon 0.003/0.005 | same |
| sampling | temperature 0.7, top-p 1, top-k 0 | same |
| experiment/engine seed | CLI/data shuffle 42; vLLM global 0 | same |
| checkpoint | final step 300 only; `LatestN(1)`, no evidence milestones | same |

P57 `l0` is not used: it matches only the historical envelope, not the exact
P45 dataset identity. Absolute capability is never compared across P45 and
M15 as if workload were controlled.

## Treatment invariants

- Native/no-IS and native/IS use the same stock-fast numerical program and
  differ only in `sampler_is`, old-logprob identity, and TIS weights.
- Native/no-IS and zero/no-IS share sampler semantics; they differ by the
  complete registered zero-TIM bundle, not fixed lm-head alone.
- Native arms continue through finite A-B as warning-only. B-C, nonfinite,
  structural, replica, transaction, optimizer, and checkpoint failures remain
  fatal. Zero is strict on A-B-C.
- The observer-only processed-B helper never supplies old logprobs or gradients.
  The IS arm's trainer C is intentionally a training input and must be declared
  as such by its runtime receipt.
- Initial model/source/image, data/order, sampling, topology, optimizer,
  objective, horizon, checkpoint schedule, and nonnumerical infrastructure are
  held equal within each workload.
- The active 300-update primary cells run uninterrupted and write exactly one
  scheduled checkpoint at update 300. `LatestN(1)` retains that final actor and
  optimizer state. Historical P45 and M15-selection carriers keep their
  registered ten-update cadence; they are not members of the primary contrast.
- Primary dataset identity is fail-closed: P45 train/eval hashes are
  `ddc96fd9...`/`b10add7f...`; M15 `main` hashes are
  `ff1e659b...`/`8edb61cb...`. Full values and generator namespaces are in the
  runbook. Runtime startup and postflight both attest them.
- Every paired command pins experiment/data-shuffle seed 42 and the rollout
  engine global seed 0. Per-request stochastic seeding is unsupported in this
  backend, so equal seed configuration does not imply byte-identical
  temperature-0.7 trajectories across independent launches.
- The 300-update horizon is shared by all three treatments. The immediate
  queue contains only the four native-program cells, but the deferred zero-TIM
  cells must use the same 300-update horizon when separately authorized.
- Each training JobSet runs rollout-only held-out evaluation at updates
  0, 50, 100, 150, 200, 250, and 300. It uses the common temperature-0.7
  sampling recipe, 100 held-out prompts, and eight generations. Evaluation
  examples never enter the trainer or backward path.
- Step labels identify the update count installed in the rollout engine: 0 is
  the initial policy; at 50 the update-51 weights have not yet been synced; and
  300 is after the final update and rollout weight sync. The same signed
  100-row held-out set is reused at all seven points.
- The final point runs after update 300 and rollout weight sync. A fail-closed
  postflight classifier requires exactly seven points and 800 rewards per
  point. Checkpoint milestones are not used for the primary curve.

## Registered outcomes

- Primary: within-workload on-policy solve curve and rollout-only held-out
  solve curve.
- Secondary: area under the update-indexed solve curve and time-to-threshold.
- Stability/mechanism: nonfinite/collapse incidence, A-B dose, B-C exactness,
  TIS weight tail and clip fraction, ordinary GSPO ratio/clip tail, gradient and
  update norms, mixed/nonzero-advantage groups, turn/context/completion lengths.
- Systems: seconds/update, sampled tokens/second, rescore, backward/update,
  checkpoint, and final evaluation costs.
- Contrasts: `is - mismatch`, `zero - mismatch`, and `zero - is`, always within
  the same workload.

## Claim ceiling

- One valid curve per cell is concept evidence, not a stability theorem.
- A P45 null with an M15 positive is consistent with ceiling masking.
- A null on both workloads means only robustness to the measured mismatch dose.
- Missing A-B dose is `NO_TREATMENT`; nonexact zero, B-C failure, truncation,
  floor/ceiling, or incomplete receipts invalidates the affected contrast.
- General capability/stability claims require paired multi-seed replication and
  counterbalanced launch order.

## Rollback

All P57 paths are additive and default empty. Leaving P57 fields unset restores
the pre-study P45 behavior. Do not mutate the historical P45 renderer/profile;
the experiment uses the isolated P57 renderer/profile with explicit arm choice.
