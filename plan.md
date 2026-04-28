1. **Update `tunix/rl/algo_core.py`**:
   - Rename `agentic_rloo` to `rloo`.
2. **Update `tunix/rl/agentic/agentic_grpo_learner.py`**:
   - We already changed `algo_variant`, `advantage_estimator`, and `policy_loss_fn` to `"grpo"` in `GRPOConfig`. I will double check.
3. **Run tests**:
   - Re-run `pytest tests/rl/agentic/agentic_grpo_learner_test.py tests/rl/grpo/grpo_learner_test.py tests/rl/ppo/ppo_helpers_test.py` to ensure tests still pass.
4. **Pre-commit**: Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.
5. **Submit**: Once all tests pass, submit the change.
