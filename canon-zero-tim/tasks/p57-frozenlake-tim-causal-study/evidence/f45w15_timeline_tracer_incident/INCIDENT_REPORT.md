# Incident Report: FrozenLake P45 Full Wave 15 Step-1 Timeline Span Stack Underflow

- **JobSet Name**: `canon-p57-fl-zero-f45w15-799a0bd1`
- **Workload**: FrozenLake P45 Zero-TIM Full Wave 15 (Qwen3-8B, 64 TPU v5p, DP8xTP8)
- **Source SHA**: `799a0bd1ed5ecfd7a2f6e42eeaced82886fec76c`
- **Runtime Image**: `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server@sha256:3d9a6523f2262e6881c700542a9d89a4032103a79ee2851a72ca353d24ad5f95`
- **Failure Timestamp**: 2026-08-28T10:26:52Z
- **Progress Reached**: Step 0 100% completed & committed; failed during Step 1 Rollout
- **Failure Mode**: `EXPERIMENTAL_TRACER_SPAN_UNDERFLOW` (`ValueError: host-139531592390336: no more spans to end.`)

---

## 1. Executive Summary

1. **Step 0 Executed with Bitwise Exact Pre-Alignment and Stable Gradient Commit**:
   - **Pre-Alignment**: `S_decode_vs_S_prefill: 0 B` differing bytes (0 / 46,596 elements), `S_prefill_vs_T_old: 0 B` differing bytes (0 / 46,596 elements) (`verdict: PASS`).
   - **First Update Gate**: `[V1.FIRST_UPDATE] PASS step=0 stage=full optimizer_commits=1 lr=1e-06 max_grad_norm=1.0 naive_norm=0.5510 naive_norm_finite=True stable_norm=0.5510 fallback=0 clip_factor=1.0`.
   - **AdamW Commit**: Optimizer state and model weights successfully transitioned to Step 1.
2. **Step 1 Rollout Crash**:
   - During concurrent multi-threaded trajectory collection in Step 1 rollout, `self._perf_v2.span(...)` in `tunix/rl/rl_cluster.py:985` called into `tunix/perf/experimental/tracer.py:346` (`host_timeline.stop_span(end)`).
   - Because `Timeline.stop_span` encountered an empty span stack for thread/host `host-139531592390336`, it raised `ValueError: host-139531592390336: no more spans to end.`
   - This unhandled exception terminated the `_runner` worker coroutines in `RolloutOrchestrator`, caused `yield_batches` stream failure, stopped the trainer, and triggered orderly shutdown.

---

## 2. Complete Traceback (from `head_jax_tpu.log` Lines 20401–20485)

```python
2026-08-28 10:26:52 - ERROR - [absl] Caught exception inside model_call: host-139531592390336: no more spans to end.
Traceback (most recent call last):
  File "/app/tunix/rl/agentic/trajectory/trajectory_collect_engine.py", line 681, in _safe_model_call
    return self.model_call(
           ^^^^^^^^^^^^^^^^
  File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 2685, in _model_call
    result = self.rl_cluster.generate(
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/tunix/rl/rl_cluster.py", line 985, in generate
    ) as span, self._perf_v2.span(
               ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/app/tunix/perf/experimental/tracer.py", line 346, in span
    host_timeline.stop_span(end)
  File "/app/tunix/perf/experimental/timeline.py", line 236, in stop_span
    raise ValueError(f"{self.id}: no more spans to end.")
ValueError: host-139531592390336: no more spans to end.

2026-08-28 10:26:52 - ERROR - [absl] Fatal error in runner for pair 0: host-139531592390336: no more spans to end.
Traceback (most recent call last):
  File "/app/tunix/rl/agentic/pipeline/rollout_orchestrator.py", line 180, in _runner
    episode_count = await self._run_and_queue_one_episode(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/tunix/rl/agentic/pipeline/rollout_orchestrator.py", line 118, in _run_and_queue_one_episode
    traj = await self._collect_trajectory(agent, env, mode=collect_mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/tunix/rl/agentic/pipeline/rollout_orchestrator.py", line 104, in _collect_trajectory
    return await engine.collect(mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/tunix/rl/agentic/trajectory/trajectory_collect_engine.py", line 272, in collect
    done = await self._one_step()
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/app/tunix/rl/agentic/trajectory/trajectory_collect_engine.py", line 695, in _one_step
    rollout_output, _ = await self._run_with_timing(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/tunix/rl/agentic/trajectory/trajectory_collect_engine.py", line 184, in _run_with_timing
    result = await asyncio.wait_for(fut, timeout=timeout)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/asyncio/tasks.py", line 520, in wait_for
    return await fut
           ^^^^^^^^^
  File "/usr/local/lib/python3.12/concurrent/futures/thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/tunix/rl/agentic/trajectory/trajectory_collect_engine.py", line 681, in _safe_model_call
    return self.model_call(
           ^^^^^^^^^^^^^^^^
  File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 2685, in _model_call
    result = self.rl_cluster.generate(
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/tunix/rl/rl_cluster.py", line 985, in generate
    ) as span, self._perf_v2.span(
               ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/app/tunix/perf/experimental/tracer.py", line 346, in span
    host_timeline.stop_span(end)
  File "/app/tunix/perf/experimental/timeline.py", line 236, in stop_span
    raise ValueError(f"{self.id}: no more spans to end.")
ValueError: host-139531592390336: no more spans to end.
```

---

## 3. Root Cause Analysis

1. **Experimental Tracer Timeline Span Stack Concurrency Defect**:
   - When rollout uses async/multithreaded trajectory workers (`ThreadPoolExecutor` threads invoking `_safe_model_call` in parallel), multiple tasks concurrently execute context managers wrapped in `self._perf_v2.span(...)`.
   - In `tunix/perf/experimental/tracer.py` and `tunix/perf/experimental/timeline.py`, `HostTimeline` maintains an internal stack of open spans (`self._open_spans`).
   - If an async context switch, timeout cancellation, or cross-thread generator evaluation occurs, `stop_span` is called when `self._open_spans` is already empty or mismatched, raising `ValueError(f"{self.id}: no more spans to end.")`.
2. **Postflight Check Rejection**:
   - Because the training loop was halted in Step 1 Rollout (before Step 1 backward/VJP was reached), postflight script `90_run.sh` checked `p38_fixed_vjp=0` on shutdown and emitted `[P38.FIXED_LM_HEAD] RECEIPTS_FAIL endpoint=untied_lm_head ... vjp=0 reasons=missing_fixed_order_vjp`. This was a consequence of the early crash during rollout, not an algorithmic defect of Step 0.

---

## 4. Evidence Files in this Package

- `head_jax_tpu.log`: Full 3.19 MB stdout/stderr from `pathways-head` `jax-tpu` trainer container
- `head_pathways_proxy.log`: 1.52 MB IFRT proxy logs
- `head_pathways_rm.log`: 7.84 MB Pathways Resource Manager logs
- `worker_00.log` through `worker_15.log`: Complete logs across all 16 TPU v5p worker nodes (64 TPU chips)
- `RAW_ERROR.log`: Extracted traceback and shutdown sequence
- `SHA256SUMS`: Cryptographic checksums of all 20 evidence files
