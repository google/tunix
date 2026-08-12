# Plan

## Outcome

Restore periodic held-out FrozenLake evaluation for Qwen3-8B DP16xTP4 full
training. Evaluation must use the synchronized rollout policy, run exactly once
at each registered train step, cover every registered held-out prompt and
generation, report online W&B metrics, and never mutate optimizer, gradient
accumulator, training data, or training-step accounting.

This task does not change optimizer placement. Device-resident optimizer
admission remains governed by P41, whose Qwen3-8B result is NOT ADMITTED and
left only 4.52 GiB of HBM per chip. This task also does not claim that the
remaining decode-versus-prefill carrier is fixed; finite numerical alignment
records remain warning-only for the explicitly admitted full-training profile.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| P42.1 | Default-off evaluation-enabled FrozenLake workload, profile, renderer, preflight, and classifier contracts | Pinned `bash canon-zero-tim/tests/p33_workloads/run_cpu.sh` passes positive and one-fault negative controls | completed |
| P42.2 | Pinned-image schedule, reward-inventory, W&B-summary, and isolation contract | Pinned gate observes the once-per-policy-step schedule, rejects incomplete/nonfinite summaries, and proves the segmented path does not submit evaluation examples to the actor update | local pass; target isolation pending |
| P42.2b | Admit legitimate duplicate rank-gradient signatures in production while preserving the strict synthetic admission probe | Reducer and adapter suites pass on 64 forced CPU devices; the full pinned P33 gate passes; rank cadence, contribution count, fixed order, and replica equality remain hard gates | local pass; target retry pending |
| P42.3 | One admitted 64-chip full run with evaluation enabled | Step-0 evaluation inventory is complete, W&B logs the `frozenlake_eval/eval/*` summary, all 16 first-update reduction groups complete, training commits the next update, and all non-numerical safety gates remain green | attempted; failed before first reduction, retry pending |

## Signed evaluation contract

- Workload: Qwen3-8B FrozenLake, DP16xTP4, 32 prompts x 8 generations.
- Held-out source: the existing FrozenLake `test_dataset`; it must be nonempty
  and must not be substituted with the training iterator.
- Cadence: every 10 global optimizer updates, including a step-0 baseline.
- Coverage: `held_out_prompts * 8` rewards exactly once per scheduled step.
- Policy version: evaluation consumes the rollout weights registered for the
  scheduled pre-update policy step.
- Metrics: `frozenlake_eval/eval/reward`, `solve`, `n`, `wall_seconds`, and
  `policy_step` are written at a monotonic W&B train step.
- Isolation: evaluation uses `Mode.EVAL`; it cannot increment optimizer steps,
  mutate parameters, modify the gradient accumulator, consume a training
  prompt, or contaminate the training reward window.
- Cache discipline: evaluation starts with a reset evaluation request state;
  evaluation requests cannot reuse training prefix-cache state.
- Alignment policy: finite numerical mismatch may warn only in the explicitly
  admitted FrozenLake full-training profile. NaN/Inf, shape, metadata, weight,
  transaction, replica, HBM, and Pathways failures remain hard errors.

## Decisions

- Confirmed: the learner already guards against repeated evaluation at every
  gradient microbatch through `_last_eval_train_step` and has exact evaluation
  reward-inventory checks.
- Confirmed: the current P33 contract explicitly requires
  `CANON_P33_DISABLE_EVAL=1`, sets `periodic_evaluation=False`, and postflight
  requires one evaluation-disabled marker.
- Decision: introduce a separate default-off evaluation-enabled contract. Do
  not silently reinterpret or delete the existing evaluation-disabled profile.
- Decision: evaluation enablement and optimizer residency remain separate
  decisions, so either optimizer placement can be tested without changing the
  evaluation semantics.
- Confirmed by target attempt `p42e2`: the previously fixed DP16 geometry is
  active (`local_M=256`, `global_M=4096`), and step-0 evaluation completed all
  800 held-out trajectories. The run then stopped before the first fixed DP
  reduction because production required all 16 compact gradient signatures to
  be distinct.
- Decision: pairwise gradient-value uniqueness is a synthetic admission
  property, not a mathematical production requirement. FrozenLake uses binary
  rewards with RLOO, so an all-success or all-failure prompt group produces
  exact zero advantages and can legitimately duplicate a rank contribution.
  Production therefore reports signature multiplicity without rejecting it.
  Synthetic probes retain strict uniqueness. Rank cadence, exactly 16
  contributions, the registered fixed tree, and post-reduction replica
  equality remain fail-closed.

## Rollback

Leave the new evaluation admission flag at zero and continue rendering the
existing evaluation-disabled FrozenLake profile. Do not remove the P31 learner
path or rewrite historical P33 evidence.

## Operator handbook

The only target procedure for this phase is
`../../cluster/P42_FROZENLAKE_EVAL_RUNBOOK.md`. It records the exact manifest,
local gate, server-side dry-run, evidence inventory, W&B curves, stop
conditions, and rollback manifest. Target status remains NOT RUN until that
procedure returns a complete classified artifact.
