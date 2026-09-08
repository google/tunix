# FrozenLake M15 Zero-TIM Full Training Crash Report (Step 40)

- **JobSet Name**: `canon-p57-fl-zero-m15-r09-dfd5e79f`
- **Pod**: `canon-p57-fl-zero-m15-r09-dfd5e79f-pathways-head-0-0-d629r`
- **Namespace**: `default`
- **Cluster**: Bodaborg GKE TPU Cluster (`cloud-tpu-shared-capacity`, `europe-west4-a`)
- **Hardware Topology**: 64 TPU v5p chips (`4x4x4` 3D Torus, 16 hosts in NAP pool `nap-ct5p-hightp-4t-1i75nqc8` + 1 Head node)
- **Model**: Qwen3-8B with 15-turn long-horizon interaction (P67 protocol)
- **Crash Timestamp**: `2026-09-06 10:58:59 UTC`
- **Total Runtime**: ~31 hours continuous training (0 pod restarts, 0 hardware faults)

---

## 1. Executive Summary & Key Milestones

The FrozenLake M15 Zero-TIM full reinforcement learning training ran continuously for 31 hours, progressing from **Step 0 through Step 40 / 300 (13.3%)**.

- **Solve Ratio (解决率)**:
  - Baseline (Step 0): **16.0%** (41 / 256)
  - Mid-stage (Step 6): **33.2%** (85 / 256)
  - Latest Committed Step 39 (Call 40): **39.1%** (100 / 256) — **All-time historic peak**
- **Optimizer & Gradient Stability**:
  - Update 15: `stable_norm = 0.2080`, `all_finite = 1`, `clip_factor = 1.0`
  - Update 38: `stable_norm = 0.1033`, `all_finite = 1`, `clip_factor = 1.0`
  - Update 39: `stable_norm = 0.0849`, `all_finite = 1`, `clip_factor = 1.0`
  - Step 40 weight synchronization committed: `[P28.G6] weight_sync_committed count=1 policy_version=40 global_steps=40`.

At Step 40 (Call 41 Rollout), multiple trajectories encountered turn generation timeouts (`MODEL_TIMEOUT`), causing an empty `request_ids` list. During post-rollout batch map processing, an overly strict assertion in `token_continuity.py` raised `ValueError: record-full row identity is malformed`, causing rank 0 to terminate.

---

## 2. Verbatim Error Traceback

```text
2026-09-06 10:58:56 - WARNING - [absl] [step_idx=0, pair_index=4, group_id=1299] trajectory clipped: MODEL_TIMEOUT
2026-09-06 10:58:56 - ERROR - [absl] [step_idx=0, pair_index=4, group_id=1299] model generation exceeded the 1799.9s turn deadline and was aborted
[CANON_P57_TOKEN_CONTINUITY_SUMMARY] workload=m15 trajectory_id=3deae7b6caf340208b46624b35a777b6 steps=0 expected_later_turns=0 receipts=0 verdict=UNEXERCISED
2026-09-06 10:58:58 - WARNING - [absl] [step_idx=12, pair_index=2, group_id=1285] trajectory clipped: MODEL_TIMEOUT
2026-09-06 10:58:58 - ERROR - [absl] [step_idx=12, pair_index=2, group_id=1285] model generation exceeded the 27.3s turn deadline and was aborted
[CANON_P57_TOKEN_CONTINUITY_SUMMARY] workload=m15 trajectory_id=b9ad6df39f1946759f095cc31488d339 steps=12 expected_later_turns=11 receipts=11 verdict=PASS
...
2026-09-06 10:58:59 - INFO - [absl] [DEEPSWE.ROLLOUT_DEADLINE] batch_complete prompt_groups=32 elapsed_secs=3767.1 deadline_secs=None
[rank0]: Traceback (most recent call last):
[rank0]:   File "<frozen runpy>", line 198, in _run_module_as_main
[rank0]:   File "<frozen runpy>", line 88, in _run_code
[rank0]:   File "/app/examples/frozenlake/train_frozenlake_qwen3.py", line 2282, in <module>
[rank0]:     grpo_trainer.train(
[rank0]:   File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 3949, in train
[rank0]:     train_examples = self._batch_to_train_example(
[rank0]:                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 3211, in _batch_to_train_example
[rank0]:     return self._process_results(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/app/tunix/rl/agentic/agentic_grpo_learner.py", line 748, in _process_results
[rank0]:     token_continuity.append_full_record_batch_map(row_identity)
[rank0]:   File "/app/tunix/rl/agentic/token_continuity.py", line 1913, in append_full_record_batch_map
[rank0]:     raise ValueError("record-full row identity is malformed")
[rank0]: ValueError: record-full row identity is malformed
```

---

## 3. Root Cause Analysis

In `tunix/rl/agentic/token_continuity.py`:
```python
trajectory_id = record.get("trajectory_id")
row = record.get("sequence_row")
step = record.get("policy_step")
request_ids = record.get("request_ids")
if (
    not isinstance(trajectory_id, str)
    or len(trajectory_id) != 32
    or any(character not in "0123456789abcdef" for character in trajectory_id)
    or type(row) is not int
    or row < 0
    or type(step) is not int
    or step < 0
    or not isinstance(request_ids, list)
    or not request_ids                       # <-- FAILING ASSERTION
    or any(
        not isinstance(request_id, str) or not request_id
        for request_id in request_ids
    )
    or len(set(request_ids)) != len(request_ids)
):
  raise ValueError("record-full row identity is malformed")
```

When trajectory `3deae7b6...` (group_id=1299, pair_index=4) timed out at Turn 0:
- `steps=0, receipts=0, verdict=UNEXERCISED`.
- `item.traj.get("p57_token_continuity_request_ids", ())` was empty `()`.
- `request_ids` in `row_identity` was `[]`.
- `not request_ids` evaluated to `True`, triggering the fatal `ValueError`.

---

## 4. Associated Log Files
- `head_pod_tail.log`: Complete available head pod log buffer (13,574 lines) capturing Steps 38-40.
- `crash_traceback.log`: Concentrated log window covering the timeouts, `batch_complete`, and python stack trace.
