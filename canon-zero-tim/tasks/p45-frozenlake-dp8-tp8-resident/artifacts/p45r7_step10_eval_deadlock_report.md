# P45r7 Step 10 Evaluation Deadlock & W&B Metrics Stall Technical Report

**Date**: 2026-08-17  
**JobSet**: `canon-p45-fl-eval-p45r7-a94d6c0c`  
**Hardware Topology**: 64 TPU v5p (`DP8xTP8`, Concurrency 256)  
**Configuration**: Full Resident FrozenLake Training, Max Steps 450, `--eval_every_n_steps=10`  
**Status**: DEADLOCKED at Step 10 Evaluation boundary (14h+ stall since `2026-08-16 14:07:21 UTC`)  

---

## 1. Executive Summary

1. **Observations**:
   * FrozenLake training progressed smoothly through Step 10 (`train_steps_after: 11`), successfully persisting full weights, AdamW optimizer states (`mu`, `nu`), and reference model checkpoints to persistent storage (`/mnt/disks/tunix-data/frozenlake/checkpoints/`).
   * Upon reaching the Step 10 boundary with `--eval_every_n_steps=10`, the training loop invoked the evaluation routine on 100 held-out prompts (800 generations).
   * Evaluation processed the first 20 prompt groups (`[CANON_ALIGN_PRE_EVIDENCE]` records 1..20), after which the vLLM engine finished remaining requests (`Engine 000: Running: 0 reqs, Waiting: 0 reqs`).
   * The main Python trainer process (PID 355) entered a permanent blocking wait on `eval_examples = eval_future.result()` in `agentic_rl_learner.py:2425`.
   * No further steps were executed, and W&B stopped receiving updates after Step 10.

2. **Asset Integrity (Checkpoints Safe)**:
   * The complete Step 10 checkpoint is intact on PVC storage.

---

## 2. Root Cause Analysis

### A. Deadlock Sequence

```text
train_step == 10
  │
  ├─► Checkpoint saved to PVC (train_steps_after: 11)
  │
  └─► _should_run_eval() == True (eval_every_n_steps == 10)
        │
        ├─► eval_future = asyncio.run_coroutine_threadsafe(_eval_runner_async(eval_orchestrator), loop)
        │
        ├─► eval_examples = eval_future.result()   <--- [BLOCKED IN SYNCHRONOUS futex_wait]
        │
        └─► Inside _eval_runner_async:
              _orchestrator_producer -> orchestrator.yield_batches() -> group_queue_manager.get_batch()
              ▲
              │ (Waiting on _have_ready.wait())
              ▼
              rollout_orchestrator.run_producers_from_stream:
              asyncio.wait(active_tasks, return_when=FIRST_COMPLETED) hung because
              active worker tasks did not complete / unhandled async cancellation,
              preventing prepare_clear() from ever firing.
```

### B. Code-Level Flaws

1. **Synchronous `eval_future.result()` without Timeout**:
   In `tunix/rl/agentic/agentic_rl_learner.py`:
   ```python
   eval_future = asyncio.run_coroutine_threadsafe(
       _eval_runner_async(eval_orchestrator), self.loop
   )
   eval_examples = eval_future.result()  # ❌ No timeout; if any async child hangs, main thread deadlocks indefinitely
   ```
2. **Missing Rollout Batch Timeout in Eval**:
   `self.algo_config.rollout_batch_timeout` defaults to `None`, meaning `yield_batches` will never time out on incomplete groups.
3. **Queue Manager Drain Deadlock**:
   In `group_queue_manager.py:110`, `_get_one_ready_group` calls `await self._have_ready.wait()`. If producers fail to call `prepare_clear()`, the consumer coroutine hangs forever.

---

## 3. Actionable Runbooks for Incoming / Debugging Agent

### Option 1: Immediate Resume from Step 10 Checkpoint (Recommended)
To immediately resume training without risking in-training eval deadlocks:
1. Delete stalled JobSet:
   ```bash
   kubectl delete jobset canon-p45-fl-eval-p45r7-a94d6c0c
   ```
2. Render and launch with `--eval_every_n_steps=0` (or disabled training eval):
   ```bash
   python3 canon-zero-tim/cluster/render_p45_frozenlake_jobsets.py \
     --source-commit "$(git rev-parse HEAD)" \
     --run-id p45r8 \
     --output-dir /tmp/p45r8 \
     --restore-step 10
   kubectl apply -f /tmp/p45r8/jobset-p45-frozenlake.yaml
   ```

### Option 2: Code Hardening for Evaluation Coroutine
In `tunix/rl/agentic/agentic_rl_learner.py`:
- Wrap `eval_future.result(timeout=1800)` with a hard timeout and logging.
- Ensure `_eval_runner_async` cancels all pending orchestrator producer tasks in a `finally` block before exiting.
