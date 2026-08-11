# P42.1 — restore the FrozenLake evaluation contract

- Status: local complete; target NOT RUN

## Finding

- Confirmed: `examples/frozenlake/train_frozenlake_qwen3.py` already accepts
  `--eval_every_n_steps`, builds `test_dataset`, and can pass it to the learner.
- Confirmed: `tunix/rl/agentic/agentic_rl_learner.py` contains the P31 evaluation
  inventory, once-per-global-step scheduling, reward accounting, and W&B metric
  plumbing.
- Confirmed: `tunix/rl/dp_workloads.py`, the FrozenLake profile,
  `render_p33_jobsets.py`, `00_env.sh`, and `90_run.sh` currently freeze or attest
  evaluation-disabled behavior.
- Hypothesis: restoring a default-off evaluation-enabled contract is sufficient;
  no new numerical model path is required.

## Execution

1. Add a FrozenLake-only, default-off evaluation admission variable. Reject it
   for GSM8K, diagnostic no-commit stages, missing held-out data, or an invalid
   cadence.
2. Render the full-training command with `CANON_P31_ENABLE_EVAL=1` and
   `--eval_every_n_steps=10`. Preserve the existing disabled profile as the
   rollback arm.
3. Replace the evaluation-disabled postflight expectation only for the admitted
   evaluation profile with exactly one enablement attestation and complete
   evaluation inventory records.
4. Add CPU tests for pre-update policy-step scheduling, repeated calls at one
   global step, empty/nonfinite summary input, incomplete reward coverage, and
   invalid or overlapping workload selection.
5. Run those contracts in the pinned image and statically retain the P28
   segmented isolation boundary: evaluation performs rollout and metric
   collection, but its examples are not submitted to the actor update. The
   first target must still prove that evaluation does not prevent the next
   committed update; no local test is relabeled as target state isolation.
6. On the first approved target full run, require the step-0 evaluation to
   complete and training to reach at least the next optimizer update. The same
   JobSet continues training; no separate target evaluation canary is required.

## Exit gate

- Command after implementation:
  `bash canon-zero-tim/tests/p33_workloads/run_cpu.sh` in the pinned image when
  host dependencies are incomplete.
- Pass: all schedule, coverage, isolation, W&B, and one-fault negative controls
  pass, followed by one pinned-image pass and one target evaluation record that
  does not stop the next training update.
- Fail: missing or duplicate evaluation, incomplete held-out coverage, stale
  policy version, optimizer/accumulator mutation, training-data consumption,
  non-monotonic W&B metrics, or any hard health-gate failure.

## Result

Implemented locally. The pinned-image P33 gate passed, including the separate
evaluation-enabled manifest, preflight/postflight selection, schedule helper,
finite W&B summary, exact 45-step inventory classifier, and negative controls.
The 64-chip target remains NOT RUN, so no evaluation-side state-isolation or
training-continuation claim has been made.
