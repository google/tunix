# P57 lessons

- Removing a canonical overlay is not only a flag change. Audit every later
  admission/proof call for an implicit adapter dependency; a stock route may
  share transport while being unable to share a canonical proof primitive.
- A no-update rollout must still synchronize actor weights. Treat successful
  `update_params` as transport evidence only; never upgrade it to live-weight
  equality without an independent adapter-backed comparison.
- Every stock-fast startup fix needs one behavioral test that runs the complete
  no-update sync branch in both regimes: stock must sync without attesting,
  canonical must sync and attest, and canonical inequality must fail closed.
- Behavioral fakes must preserve production ownership boundaries. Do not put a
  learner admission field on a fake cluster merely to simplify a test;
  represent `learner.should_sync_weights` and `learner.rl_cluster` separately.
- A workload-only target log can diagnose a traceback but cannot certify a
  run. Require wrapper markers from byte zero through terminal exit before
  classifying calibration evidence as complete.
- When a top-level router broadens a mode from calibration to train/eval, test
  every invoked leaf step under the same tuple matrix. Static proof that the
  entrypoint selects a step does not prove that the step's own guard admits it.
- A no-update evaluator that enters trainer rescore still inherits the trainer
  mesh contract. On DP8, a generation group of two is not a legal global row
  shape even if GRPO itself accepts two; gate generation count against the DP
  divisor before rendering.
