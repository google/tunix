# State

- Status: target attempt failed before the first DP reduction; local reducer-contract fix complete; target retry pending
- Objective: Restore periodic held-out evaluation to the canonical Qwen3-8B FrozenLake full-training profile without changing training updates, optimizer placement, or the warning-only alignment policy.
- Definition of done: The CPU and exact-image gates pass; a target run executes exactly one complete held-out evaluation at each scheduled step, logs complete W&B evaluation metrics, performs no evaluation-side optimizer mutation, and continues training.
- Task directory: `canon-zero-tim/tasks/p42-frozenlake-evaluation`
- Directory state: tracked on `yuxzhang/canon-zero-tim`; the current fix is workspace-local until publication is approved
- Current phase: [P42.2b — correct the production gradient-signature contract](phases/p42-2b-gradient-signature-contract.md), locally complete; P42.3 target retry pending
- Last verified fact: In target attempt `p42e2`, all 800 step-0 evaluation trajectories completed and the DP16 engine reported `local_M=256 global_M=4096`. The first reverse group reached `gradient_reducer_ready` and then stopped at the non-mathematical pairwise-signature-uniqueness assertion before reduction or optimizer commit. Locally, reducer tests passed 19/19, adapter tests passed 36/36, and the complete pinned P33 gate passed after production was changed to report legitimate duplicates while retaining all structural reduction gates.
- Next action: Publish only after approval, rerun the pinned gate at the exact published SHA, and retry the 64-chip evaluation manifest through `../../cluster/P42_FROZENLAKE_EVAL_RUNBOOK.md`. The first target proof must show all 16 reduction groups, replica-exact output, one optimizer commit, and the next training step.
- Blockers: The fix is not published and has not run on the 64-chip target. A separate approved target retry is required.
- Key artifacts: `../../debug_logs/p42_p42e2_frozenlake_eval.raw.log`; `../../cluster/P42_FROZENLAKE_EVAL_RUNBOOK.md`; `../../../tunix/rl/dp_training.py`; `../../../tunix/rl/canonical_qwen3_adapter.py`; `../../../tests/rl/dp_training_test.py`; `../../../tests/rl/canonical_qwen3_adapter_test.py`
- Updated: 2026-08-12 UTC
