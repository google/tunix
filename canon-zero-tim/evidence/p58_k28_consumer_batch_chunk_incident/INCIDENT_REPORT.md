# DeepSWE Qwen3-4B Zero-HP Full (K28) Consumer Batch Chunk Incident Report

**Incident ID**: `p58_k28_consumer_batch_chunk_incident`  
**Workload**: `canon-p58-ds4b-zero-hp-full-k28` (128 TPU v5p, 32 worker pods + 1 head pod)  
**Execution Date**: 2026-09-01  
**Source Commit**: `aff657c8e54c9b88d41571a3e68ea48569b172e9`  
**Step Reached**: Step 0 Completed 100% (First update committed, 6 SWE problems solved, Solve Rate 4.69%); Step 1 failed during consumer chunk processing  
**Failure Point**: `tunix/rl/deepswe_debug.py:1481` in `persist_batch` called from `tunix/rl/agentic/agentic_grpo_learner.py:1126` (`_process_results`) during Step 1 consumer batch execution  

---

## 1. Incident Summary & Key Accomplishments

JobSet `canon-p58-ds4b-zero-hp-full-k28` achieved major historical milestones for the DeepSWE RL system:

1. **Step 0 Complete End-to-End Success**:
   - Evaluated 116 real SWE-bench trajectories across multi-turn sandboxes.
   - Solved **6 real SWE problems** (Overall Solve Rate **4.69%**, Group 5 achieved **37.5%** [6/16]).
   - Rescore B pre-alignment check cleared across all 374,516 action tokens with **0 differing bytes** (`S_decode_vs_S_prefill = 0 B`, `S_prefill_vs_T_old = 0 B`).
   - 16-microbatch backward pass completed in ~1.3 seconds.
   - `[V1.FIRST_UPDATE]` and `[P63.STABLE_CLIP]` passed: `update=0 all_finite=1 stable_norm=0.0169 clip_factor=1.0`.
   - Parameter update committed: `parameter_changed_elements=3663318432`, `effective_learning_rate=1e-6`.
   - Global step 0 completed in 3769.3s with `train_reward=0.047 train_solve=0.047 n=128`.

2. **Step 1 Failure**:
   - Step 1 collected trajectories across 8 prompt groups up to Turn 38 (12.8k context tokens).
   - When the first micro-batch chunk of 2 prompt groups (32 trajectories) finished in the orchestrator, the consumer loop pulled the partial chunk and invoked `_batch_to_train_example` -> `_process_results` -> `deepswe_debug.persist_batch`.
   - `persist_batch` enforced `len(trajectories) == 128` and raised `ValueError: DeepSWE artifact batch requires exactly 128 trajectories, rewards, and advantages`.

---

## 2. Root Cause Analysis

1. **Consumer Chunk Execution vs Batch Persistence Contract**:
   - When `compute_logps_micro_batch_size > 1` (or `train_micro_batch_size = 8`), `_process_in_consumer = True` in `agentic_rl_learner.py:3458-3466`.
   - `_data_consumer_batch_generator` yields micro-batches from `train_data_queue`.
   - In `agentic_rl_learner.py:3793`, `self._batch_to_train_example` is called per micro-batch chunk.
   - `_batch_to_train_example` forwards the chunk to `_process_results`, which unconditionally invokes `deepswe_debug.persist_batch`.
   - `tunix/rl/deepswe_debug.py:1476-1484` strictly verifies:
     ```python
     if (
         len(trajectories) != expected_trajectories # 128
         or len(rewards) != expected_trajectories
         or len(advantages) != expected_trajectories
     ):
       raise ValueError(
           "DeepSWE artifact batch requires exactly "
           f"{expected_trajectories} trajectories, rewards, and advantages"
       )
     ```
   - Because `observed_trajectories=32 != 128`, `persist_batch` raised `ValueError`, causing the python runtime to crash.

2. **Teardown Side-Effect**:
   - Python exit code 1 triggered the wrapper `90_run.sh` postflight receipt check.
   - Because Step 1 crashed before VJP execution, `p38_fixed_vjp=0` caused `FATAL: fixed lm-head executable receipt contract failed`.

---

## 3. Resolution Plan (K29)

1. In `tunix/rl/agentic/agentic_grpo_learner.py` / `tunix/rl/deepswe_debug.py`:
   - Support micro-batch chunk accumulation before calling `persist_batch`, or only persist the batch manifest once all `global_trajectories` (128) for the step are processed.
2. Redeploy DeepSWE under **K29** on 128 TPU.

---

## 4. Evidence Files

- `RAW_ERROR.log`: Full runtime log showing Step 0 success, Step 1 rollout completion, and `persist_batch` exception.
- `SHA256SUMS`: Cryptographic checksums of incident artifacts.
