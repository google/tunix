# FrozenLake M15 Zero-TIM Full Step 61 Empty Completion FunctionalMappingError Incident Report

**Incident ID**: `m15_step61_empty_completion_incident`  
**Workload**: `canon-p57-fl-zero-m15-mw21-6c701164` (64 TPU v5p, 16 worker pods + 1 head pod)  
**Execution Date**: 2026-09-01  
**Step Reached**: 61 / 300 (20.3%)  
**Solve Rate Reached**: 48.4%  
**Failure Point**: `tunix/rl/canonical_qwen3_adapter.py:7590` in `_p32_group_spec` called from `segmented_dp_grpo_value_and_grad` (line 8591 / 8642)

---

## 1. Incident Summary

JobSet `canon-p57-fl-zero-m15-mw21-6c701164` ran stably for **43+ hours**, completing 61 full multi-turn RL training steps on 64 TPU v5p devices with Solve Rate rising from initial baselines to **48.4%** (10.1m / step).

At Step 61, after completing the multi-turn rollout across 256 environments (up to 15 turns) and passing the pre-alignment gate with warnings (`frozenlake-full-alignment-warning-v1`), the learner crashed during reverse group specification generation before backward propagation.

```text
[rank0]: Traceback (most recent call last):
[rank0]:   File "<frozen runpy>", line 198, in _run_module_as_main
[rank0]:   File "<frozen runpy>", line 88, in _run_code
[rank0]:   File "/app/examples/frozenlake/train_frozenlake_qwen3.py", line 2129, in <module>
[rank0]:     grpo_trainer.train(
[rank0]:   File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 3999, in train
[rank0]:     segmented_result = self._run_p28_g6_update(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 1622, in _run_p28_g6_update
[rank0]:     result = adapter.segmented_dp_grpo_value_and_grad(
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/app/tunix/rl/canonical_qwen3_adapter.py", line 8590, in segmented_dp_grpo_value_and_grad
[rank0]:     specs = tuple(
[rank0]:             ^^^^^^
[rank0]:   File "/app/tunix/rl/canonical_qwen3_adapter.py", line 8591, in <genexpr>
[rank0]:     self._p32_group_spec(
[rank0]:   File "/app/tunix/rl/canonical_qwen3_adapter.py", line 7590, in _p32_group_spec
[rank0]:     raise FunctionalMappingError(
[rank0]: tunix.rl.canonical_qwen3_adapter.FunctionalMappingError: P32 grouped reverse requires nonempty prompt/completion on every rank: n=[4752, 5147, 4387, 6904, 9193, 6176, 1226, 7412] prompt=[1066, 1066, 1010, 1138, 1226, 1138, 1226, 1010] completion=[3686, 4081, 3377, 5766, 7967, 5038, 0, 6402]
```

---

## 2. Root Cause Analysis

1. **Failure Mechanism**:
   - In Step 61, the 256 global trajectories were partitioned across DP ranks (DP=8, 32 trajectories per rank, split into 32 microbatches with 8 local trajectories per rank).
   - In one of the 32 microbatch groups, all 8 trajectories assigned to DP Rank 6 had finished or terminated without generating any new action tokens in that segment (`completion_length = 0` on Rank 6, while other ranks had `[3686, 4081, 3377, 5766, 7967, 5038, 0, 6402]`).
2. **Hardcoded Admission Check**:
   - In `tunix/rl/canonical_qwen3_adapter.py`:
     ```python
     specs = tuple(
         self._p32_group_spec(
             grouped_inputs[0][index],
             grouped_inputs[1][index],
             grouped_inputs[2][index],
             grouped_inputs[3][index],
             algo_config.temperature,
             allow_empty_completion=p34,  # <-- Only enabled for DeepSWE (p34), False for FrozenLake
         )
         for index in range(contract.local_trajectories)
     )
     ```
   - Because `p34` is False for FrozenLake workloads, `allow_empty_completion` was False, causing `_p32_group_spec` to fail closed and raise `FunctionalMappingError`.
3. **Correct Semantics**:
   - Multi-turn RL environments naturally encounter trajectories where some ranks have 0 valid completion tokens for a given segmented step (masked with `action_mask=0`, contributing 0 to loss and gradient).
   - Both DeepSWE and multi-turn FrozenLake should admit `allow_empty_completion=True` (or enable it across all multi-turn agentic segmented DP runs).

---

## 3. Evidence Files

- `RAW_ERROR.log`: Pod log tail capturing Step 61 rescore, alignment gate pass, and rank0 traceback.
- `SHA256SUMS`: Cryptographic checksums of incident artifacts.
