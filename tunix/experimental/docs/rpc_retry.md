# Design Doc: Fault-Tolerant RPC & Retry Semantics in Tunix `remote_execution`

<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->

**Status**: Draft
**Component**: [`remote_execution.py`](https://github.com/google/tunix/tree/main/tunix/experimental/worker/remote_execution.py), [`distributed_rl_engine.py`](https://github.com/google/tunix/tree/main/tunix/experimental/orchestrator/distributed_rl_engine.py)

---

## 1. Background & Problem Statement

Tunix distributed RL training communicates across Orchestrator, Rollout Workers, Inference Workers, and Trainer Workers using [`GrpcRemoteActorHandle`](https://github.com/google/tunix/tree/main/tunix/experimental/worker/remote_execution.py#L581-L764) and [`GrpcRemoteExecutionServer`](https://github.com/google/tunix/tree/main/tunix/experimental/worker/remote_execution.py#L381-L490). All traffic is multiplexed over three generic RPC methods on `tunix.ExecutionService`:

1. `/tunix.ExecutionService/Execute` (`submit` / `asubmit`)
2. `/tunix.ExecutionService/DispatchTask` (`dispatch_task`)
3. `/tunix.ExecutionService/PollResponses` (`poll_responses`)

Currently, [`remote_execution.py`](https://github.com/google/tunix/tree/main/tunix/experimental/worker/remote_execution.py) has **no retry mechanism** and `wait_for_ready` is disabled (`False` by default in `grpc.aio`):

- Transient TCP resets, Kubernetes/Borg pod startup races, or brief subchannel disconnects immediately fail RPCs with `grpc.StatusCode.UNAVAILABLE`.
- A dropped `DispatchTask` or `PollResponses` RPC permanently loses trajectories and causes [`rl_program.py`](https://github.com/google/tunix/tree/main/tunix/experimental/orchestrator/rl_program.py#L488-L496) to stop responding indefinitely waiting for `_in_flight_rollouts` to reach `0`.

> [!WARNING]
> **Why a blanket gRPC `retryPolicy` is unsafe**: Simply enabling a channel-wide gRPC `service_config` `retryPolicy` across `tunix.ExecutionService` introduces severe correctness bugs because the three RPC endpoints have fundamentally different idempotency, payload-size, and state-mutation semantics.

---

## 2. Hazard Analysis of Blanket gRPC Retries in RL Training

If a network failure occurs **after** request bytes have reached the server (e.g., TCP connection drops while the server is executing or serializing the response), an automatic gRPC retry re-invokes the server handler:

| RPC Endpoint | Primary RL Callers | Payload Size | Failure Mode Under Blind In-Flight Retry |
| :--- | :--- | :--- | :--- |
| **`/tunix.ExecutionService/Execute`** | `TrainerWorker.fwd_bwd`<br>`TrainerWorker.update`<br>`per_token_logps`, `score`<br>`pre/post_weight_sync` | 1 KiB – 128 MiB+ (streamed) | **Silent training corruption**:<br>1. Retrying `fwd_bwd` accumulates gradients for the same microbatch twice.<br>2. Retrying `update` steps the optimizer (`opt_state`, step counter) twice.<br>3. Payloads > 256 KiB exceed gRPC's default `max_retry_buffer_size`, causing gRPC C-core to silently commit the stream and skip retries anyway. |
| **`/tunix.ExecutionService/DispatchTask`** | `RolloutWorker.generate` ([`distributed_rl_engine.py:L148`](https://github.com/google/tunix/tree/main/tunix/experimental/orchestrator/distributed_rl_engine.py#L148)) | 1 KiB – 64 KiB (`RolloutRequest`) | **Duplicate trajectory generation**:<br>[`RemoteExecutionServer.dispatch_task`](https://github.com/google/tunix/tree/main/tunix/experimental/worker/remote_execution.py#L241-L249) does not deduplicate `request_id`. Retrying spawns a duplicate vLLM generation task and enqueues a second `RolloutResponse`, skewing `_in_flight_rollouts` and group counts. |
| **`/tunix.ExecutionService/PollResponses`** | `DistributedRLEngine.poll_rollouts` ([`distributed_rl_engine.py:L305`](https://github.com/google/tunix/tree/main/tunix/experimental/orchestrator/distributed_rl_engine.py#L305)) | 64 B (request)<br>10 KiB – 10 MiB (response) | **Permanent trajectory loss & step hang**:<br>[`RemoteExecutionServer.poll_response`](https://github.com/google/tunix/tree/main/tunix/experimental/worker/remote_execution.py#L272-L283) executes a destructive `self._get_response_queue().get()`. If the connection drops while transmitting the popped response back to the orchestrator, the trajectory is already gone from the queue. Retrying `PollResponses` pulls the *next* trajectory, permanently losing the previous one and hanging `_in_flight_rollouts > 0` ([`rl_program.py:L493-495`](https://github.com/google/tunix/tree/main/tunix/experimental/orchestrator/rl_program.py#L493-L495)). |

---

## 3. Proposed Architecture

To achieve end-to-end fault tolerance without duplicate gradient updates or lost trajectories, we combine **transport-level connection readiness** with **protocol-level idempotency**:

```mermaid
sequenceDiagram
    participant Orch as Orchestrator (GrpcRemoteActorHandle)
    participant Srv as Worker (GrpcRemoteExecutionServer)

    Note over Orch,Srv: 1. All RPCs: wait_for_ready=True (queues on client if subchannel disconnected)
    
    rect rgb(235, 245, 255)
    Note over Orch,Srv: 2. Idempotent DispatchTask (Retryable via gRPC service_config)
    Orch->>Srv: DispatchTask(request_id="req_p1_g0_v1", method="generate")
    Srv->>Srv: Check _seen_task_ids["req_p1_g0_v1"] (deduplicate if retry)
    Srv-->>Orch: ACK ("req_p1_g0_v1")
    end

    rect rgb(240, 255, 240)
    Note over Orch,Srv: 3. Lossless PollResponses (Single-slot unacked buffer + ack_id)
    Orch->>Srv: PollResponses(timeout_s=50.0, ack_id="prev_req_id")
    Srv->>Srv: Clear _unacked_response if matches ack_id, else re-send _unacked_response
    Srv-->>Orch: ExecutionResponse(request_id="req_p1_g0_v1", result=TrajectoryItem)
    end

    rect rgb(255, 245, 235)
    Note over Orch,Srv: 4. Deduplicated Execute (Trainer fwd_bwd / update)
    Orch->>Srv: Execute(request_id="exec_uuid", method="fwd_bwd")
    Srv->>Srv: If "exec_uuid" inflight: await existing Task; if completed: return cached response
    Srv-->>Orch: ExecutionResponse(request_id="exec_uuid", result=metrics)
    end
```

---

### 3.1. Layer 1: `wait_for_ready=True` on All gRPC Calls

In [`GrpcRemoteActorHandle`](https://github.com/google/tunix/tree/main/tunix/experimental/worker/remote_execution.py#L669-L729), enable `wait_for_ready=True` on all RPC invocations (`Execute`, `DispatchTask`, `PollResponses`) and in the channel's `methodConfig`.

- **Behavior**: When a worker subchannel is in `CONNECTING` or `TRANSIENT_FAILURE` (e.g., during startup or immediately after an HTTP/2 keepalive ping timeout), gRPC holds the RPC in the client queue until the subchannel reconnects or `self._rpc_timeout_s` expires.
- **Safety**: Because `wait_for_ready` only buffers requests **before** any bytes are written to the TCP socket, the server is guaranteed not to have received the call yet. Combined with gRPC C-core's built-in *transparent retries* (which only retry when the stream is refused prior to application processing via HTTP/2 `REFUSED_STREAM`), this eliminates pre-dispatch connection errors with zero duplicate-execution risk and no message-size buffer limits.

---

### 3.2. Layer 2: Idempotent `DispatchTask` + Method-Scoped gRPC Retry Policy

1. **Propagate `request_id` from `DistributedRLEngine`**:
   In [`DistributedRLEngine.dispatch_rollout_requests`](https://github.com/google/tunix/tree/main/tunix/experimental/orchestrator/distributed_rl_engine.py#L148), pass `req.request_id` explicitly:
   ```python
   res = worker.dispatch_task(
       request_id=req.request_id, method_name="generate", requests=[req]
   )
   ```
2. **How gRPC `retryPolicy` Interacts with `_dispatched_request_ids` & Server-Side Failures**:
   When `worker.dispatch_task(request_id="req_p1_g0_v1", ...)` is called, the client serializes a single `ExecutionRequest` with `request_id="req_p1_g0_v1"`. If the TCP stream drops (`UNAVAILABLE`) *after* the server started the task but *before* the ACK arrived, gRPC's `retryPolicy` automatically re-sends the **exact same payload and `request_id`** within `0.1s–1.0s`.

   To handle both transport retries and server-side execution failures cleanly, `RemoteExecutionServer` tracks task lifecycle state (`IN_FLIGHT` vs. `COMPLETED`) rather than a permanent blind set:

   | Failure / Retry Scenario | Server State & Behavior |
   | :--- | :--- |
   | **1. Failure *during* `DispatchTask` handler** (e.g. deserialization error before `asyncio.create_task`) | `request_id` is **not** recorded in `_dispatched_request_ids`. Server returns `INTERNAL` / `UNKNOWN` (not `UNAVAILABLE`), so gRPC `retryPolicy` does not retry and fails fast to caller. |
   | **2. Network drop on ACK while task is `IN_FLIGHT`, `UNACKED`, or already `COMPLETED`** | gRPC `retryPolicy` retries `DispatchTask` with the same `request_id`. Server sees `request_id` in `_dispatched_request_ids` (a bounded FIFO `dict` of max `10,000` IDs $\approx 500\text{ KB}$) and immediately returns `request_id` (no-op ACK). Retaining successful IDs in the bounded FIFO even after `PollResponses` ACKs them prevents a late transport retry (`100ms` backoff) from re-running a fast task (`5ms`) that was already polled and ACKed. |
   | **3. Asynchronous execution fails *inside* `_run_and_enqueue`** (e.g. vLLM error or environment exception) | [`execute_request`](https://github.com/google/tunix/tree/main/tunix/experimental/worker/remote_execution.py#L365-L371) / [`RolloutWorker`](https://github.com/google/tunix/tree/main/tunix/experimental/worker/remote_execution.py#L372-L383) wraps the error into an `ExecutionResponse` / `RolloutResponse(status="ERROR")` and enqueues it to `_response_queue`. **Failed `request_id`s are immediately evicted from `_dispatched_request_ids`** so an intentional application-level retry of that `request_id` is allowed to re-run. |

3. **Method-Scoped gRPC `retryPolicy`**:
   Configure `_grpc_options()` with a `grpc.service_config` that enables automatic gRPC retries (`UNAVAILABLE`, up to 4 attempts with exponential backoff `0.1s` -> `1.0s`) specifically for `/tunix.ExecutionService/DispatchTask` (where payloads are small prompts well within the retry buffer).

---

### 3.3. Layer 3: Exactly-Once `PollResponses` (`_unacked_responses` Dict + Zero-Gap `_pop_and_stage` Waiter)

To preserve the existing `Optional[ExecutionResponse]` return type of `poll_response()` / `poll_responses()` while avoiding **lock-convoy deadlocks**, **inter-tick cancellation races**, and **dropped responses under overlapping polls**, Section 3.3 enforces three invariants:

1. **Zero-Tick Gap Between `queue.get()` and `_unacked_responses` (`_pop_and_stage`)**:
   - **Subtle `asyncio` Race**: If `waiter = asyncio.create_task(queue.get())` merely called `queue.get()` and waited for the outer `poll_response` coroutine to resume in a subsequent event-loop turn to insert `resp` into `self._unacked_responses`, a one-tick window would exist where `waiter.done()` is `True` (`resp` is removed from `queue`) while `self._unacked_responses` does not yet contain `resp`. If a retry Poll #2 arrived in that exact tick, Poll #2 would see an empty `_unacked_responses` and miss `resp`.
   - **Fix**: Define `waiter = asyncio.create_task(_pop_and_stage())`, where `_pop_and_stage()` executes `resp = await queue.get()` and **in the exact same synchronous coroutine step (with zero `await`s in between)** inserts `self._unacked_responses[resp.request_id] = resp`. In Python `asyncio`, a coroutine never yields without an `await`, making `queue.get()` + `_unacked_responses` insertion strictly atomic with respect to all other coroutines.

2. **Fine-Grained `self._state_lock` Released During Long-Poll Wait**:
   - `self._state_lock` is held only for $< 1\,\mu\text{s}$ to reconcile `ack_request_id`, check `self._unacked_responses`, and cancel any stale `self._active_poll_waiter` left behind by a disconnected poll. It is **never held across `await asyncio.wait_for(waiter, timeout=timeout_s)`**.

3. **Insertion-Ordered `_unacked_responses: dict[str, ExecutionResponse]`**:
   - Handles multiple unacknowledged responses (e.g., if Poll #1 popped `resp_1` as its TCP stream died and Poll #2 popped `resp_2`). When Poll #3 sends `ack_request_id="id_2"`, `"id_2"` is removed and `"id_1"` remains at the head of `_unacked_responses` for immediate re-delivery.

```python
async def poll_response(
    self,
    timeout_s: float = LONG_POLL_TIMEOUT_S,
    ack_request_id: Optional[str] = None,
) -> Optional[ExecutionResponse]:
  queue = self._get_response_queue()

  async def _pop_and_stage() -> ExecutionResponse:
    resp = await queue.get()
    # Zero awaits between queue.get() and _unacked_responses insertion:
    # impossible for any coroutine or cancellation to interleave here.
    if resp.request_id is None:
      resp.request_id = uuid.uuid4().hex
    self._unacked_responses[resp.request_id] = resp
    return resp

  # Phase A: Short critical section (< 1 us; no blocking await inside _state_lock)
  async with self._state_lock:
    # 1. Acknowledge the specific request_id confirmed by the client
    if ack_request_id is not None:
      self._unacked_responses.pop(ack_request_id, None)

    # 2. If any previously popped response is still unacked, re-deliver the
    # oldest one and rotate it to the back of the ordered dict.
    if self._unacked_responses:
      oldest_id = next(iter(self._unacked_responses))
      resp = self._unacked_responses.pop(oldest_id)
      self._unacked_responses[oldest_id] = resp
      return resp

    # 3. Preempt any stale/zombie _pop_and_stage() waiter from a disconnected poll
    if (
        self._active_poll_waiter is not None
        and not self._active_poll_waiter.done()
    ):
      self._active_poll_waiter.cancel()

    if timeout_s == 0.0:
      try:
        resp = queue.get_nowait()
      except asyncio.QueueEmpty:
        return None
      if resp.request_id is None:
        resp.request_id = uuid.uuid4().hex
      self._unacked_responses[resp.request_id] = resp
      return resp

    waiter = asyncio.create_task(_pop_and_stage())
    self._active_poll_waiter = waiter

  # Phase B: Await waiter OUTSIDE _state_lock so long-polls never block retries.
  # If waiter completed on the exact tick timeout/cancellation fired, _pop_and_stage()
  # has ALREADY placed resp into self._unacked_responses for the next poll!
  try:
    return await asyncio.wait_for(waiter, timeout=timeout_s)
  except (asyncio.TimeoutError, asyncio.CancelledError):
    if not waiter.done():
      waiter.cancel()
    return None
```

- **Client-side (`GrpcRemoteActorHandle.poll_responses`) — Causal Ordering & Duplicate Prevention**:
  - Guarded on the client handle by `async with self._client_poll_lock:` so concurrent client pollers on the same handle cannot send out-of-order `ack_request_id`s (note: unlike a server lock, a client-side lock never blocks retries because retries happen inside `_call_with_retry` while holding `_client_poll_lock`).
  - Upon receiving `resp`, always updates `self._last_polled_request_id = resp.request_id` **first**, and if `resp.request_id in self._seen_response_ids`, immediately loops to send `ack_request_id = resp.request_id` and fetch the next unseen response.

---

### 3.4. Layer 4: In-Flight Coalescing, Piggybacked `ack_execute_ids`, & Client Retry for `Execute` (`submit` / `asubmit`)

For synchronous/awaitable RPCs over `/tunix.ExecutionService/Execute` (`TrainerWorker.fwd_bwd`, `TrainerWorker.update`, `per_token_logps`), four invariants ensure lossless, zero-duplicate execution:

1. **Client-Side `_call_with_retry` for Payloads `> 256 KiB`**:
   - Because gRPC C-core's `retryPolicy` buffer (`per_rpc_retry_buffer_size`) is capped at 256 KiB, `GrpcRemoteActorHandle` wraps `Execute` and `PollResponses` in a Python `_call_with_retry` loop on `StatusCode.UNAVAILABLE` (up to 4 attempts, `0.1s` $\to$ `1.0s` backoff) that re-sends the **in-memory `ExecutionRequest` instance** (preserving `request.request_id` with 0 extra RAM).

2. **Isolating `Execute` Deduplication from Background `DispatchTask`**:
   - `execute_request` remains the raw method-execution helper called by `_run_and_enqueue`, while `_handle_execute` calls `execute_idempotent_request` so background vLLM rollouts never touch `_completed_executions`.

3. **Piggybacked `ack_execute_ids: tuple[str, ...]` Set + Bounded FIFO Cap (`_MAX_COMPLETED_EXECUTIONS = 16`)**:
   - **Why a set (`ack_execute_ids`) instead of a single `ack_execute_id`**: If two `asubmit()` calls (`exec_1` and `exec_2`) run concurrently via `asyncio.gather()` and both complete before `exec_3` is sent, storing a single `self._last_acked_execute_id` on the client would overwrite `"exec_1"` with `"exec_2"`, leaving `"exec_1"` unacked on the server. Using `self._pending_execute_acks: set[str]` on the client accumulates **all** completed `request_id`s since the last `Execute` call and clears them only after the next `Execute` RPC succeeds.
   - Furthermore, the server enforces a safety FIFO cap (`_MAX_COMPLETED_EXECUTIONS = 16`) on `self._completed_executions` so that even if a client disconnects permanently after its final step, server memory cannot leak.

```python
class RemoteExecutionServer:

  def __init__(self, ...):
    ...
    self._state_lock = asyncio.Lock()
    # Active in-flight Execute tasks: request_id -> asyncio.Task[ExecutionResponse]
    self._inflight_executions: dict[str, asyncio.Task[ExecutionResponse]] = {}
    # Completed unacknowledged Execute responses: request_id -> ExecutionResponse
    self._completed_executions: dict[str, ExecutionResponse] = {}

  async def execute_idempotent_request(
      self, req: ExecutionRequest
  ) -> ExecutionResponse:
    req_id = req.request_id

    async with self._state_lock:
      # 0. Evict all previously acknowledged Execute responses BEFORE running the new step
      for ack_id in req.ack_execute_ids:
        self._completed_executions.pop(ack_id, None)

      # Case 1 (Post-Execution Retry): Task already finished, reply dropped on wire
      if req_id is not None and req_id in self._completed_executions:
        return self._completed_executions[req_id]

      # Case 2 (In-Flight Coalescing): First attempt is STILL running on the TPU
      if req_id is not None and req_id in self._inflight_executions:
        task = self._inflight_executions[req_id]
      else:
        # Case 3 (First Attempt): Spawn shielded asyncio.Task
        task = asyncio.create_task(self.execute_request(req))
        if req_id is not None:
          self._inflight_executions[req_id] = task

          def _on_done(t: asyncio.Task[ExecutionResponse]) -> None:
            self._inflight_executions.pop(req_id, None)
            if not t.cancelled() and t.exception() is None:
              res = t.result()
              # Only cache successful responses so server-side errors can be retried
              if res.error_message is None:
                self._completed_executions[req_id] = res
                while len(self._completed_executions) > _MAX_COMPLETED_EXECUTIONS:
                  oldest = next(iter(self._completed_executions))
                  self._completed_executions.pop(oldest, None)

          task.add_done_callback(_on_done)

    # Await outside _state_lock; asyncio.shield prevents TCP drop from cancelling TPU execution
    return await asyncio.shield(task)
```

---

## 4. Proposed `_grpc_options()` Configuration

```python
_SERVICE_CONFIG = json.dumps({
    "methodConfig": [
        {
            # Default for all methods on tunix.ExecutionService:
            # Wait for subchannel readiness before sending bytes.
            "name": [{"service": "tunix.ExecutionService"}],
            "waitForReady": True,
        },
        {
            # DispatchTask carries small RolloutRequest payloads and is
            # idempotent by request_id on the server.
            "name": [{
                "service": "tunix.ExecutionService",
                "method": "DispatchTask",
            }],
            "waitForReady": True,
            "retryPolicy": {
                "maxAttempts": 4,
                "initialBackoff": "0.1s",
                "maxBackoff": "1.0s",
                "backoffMultiplier": 2.0,
                "retryableStatusCodes": ["UNAVAILABLE"],
            },
        },
    ]
})
```

---

## 5. Performance & Memory Overhead Analysis

| Component | Latency Overhead (Happy Path) | Memory Overhead | Key Trade-off / Nuance |
| :--- | :--- | :--- | :--- |
| **1. `wait_for_ready=True`** | **0 ns** (single C-core state check when channel is `READY`) | **0 B** in steady state | If a worker crashes *permanently*, calls wait up to `rpc_timeout_s` (60s) before raising `DEADLINE_EXCEEDED` instead of failing in < 1 ms with `UNAVAILABLE`. |
| **2. `DispatchTask` Deduplication + `retryPolicy`** | **< 1 µs** (`dict` lookup/insert of `request_id` string) | **< 5 MB** total across Client + Server | Bounded `10,000`-ID FIFO prevents late transport retries (`100ms` backoff) from re-running fast tasks (`5ms`) that were already polled and ACKed. |
| **3. `PollResponses` `_unacked_responses` + `_pop_and_stage`** | **< 1 µs** (zero-gap staging inside waiter; 0 extra RPC round-trips) | **~10–100 KB** per unacked response until next poll | Staging `resp` into `_unacked_responses` inside `_pop_and_stage()` with zero `await`s after `queue.get()` eliminates inter-tick races between `queue` and `_unacked_responses`. |
| **4. `Execute` In-Flight Coalescing + `ack_execute_ids` Buffer** | **< 1 µs** | **0 B** during next TPU execution (freed as soon as next `Execute` arrives with `ack_execute_ids`) | Piggybacking `ack_execute_ids` (as a set) frees all completed responses *before* the next TPU step begins and supports concurrent `asubmit()` completions without leaking entries. |

---

## 6. Summary of All Edge Cases Checked

| # | Edge Case / Race Scenario | How the Design Handles It |
| :--- | :--- | :--- |
| **1** | **Zombie Long-Poll Holding Lock / Stealing Queue Item**: Poll #1 disconnects mid-wait while sleeping on `queue.get()`. | `_state_lock` is released during `await asyncio.wait_for(waiter, timeout_s)`, and Poll #2 cancels `self._active_poll_waiter` before creating its own waiter. |
| **2** | **Inter-Tick Gap Between `queue.get()` and `_unacked_responses`**: `waiter` pops an item at Tick $T$, and Poll #2 arrives before `poll_response` resumes at Tick $T+1$. | `_pop_and_stage()` inserts `resp` into `self._unacked_responses` immediately after `resp = await queue.get()` with **zero `await`s in between**, making `queue.get()` + staging strictly atomic. |
| **3** | **Multiple Unacked Responses**: Poll #1 pops `resp_1` as its socket dies, while Retry Poll #2 pops `resp_2` and ACKs `"id_2"`. | `self._unacked_responses` is an insertion-ordered `dict[str, ExecutionResponse]`. ACKing `"id_2"` removes only `"id_2"`, and `"id_1"` is immediately re-delivered on Poll #3. |
| **4** | **Duplicate Poll Livelock**: Client receives `resp_1`, then a retry race causes the server to re-send `resp_1` which is already in client's `_seen_response_ids`. | Client still sets `self._last_polled_request_id = resp_1.request_id` and loops immediately to send the ACK, clearing `resp_1` from `_unacked_responses`. |
| **5** | **Late `DispatchTask` Retry Arriving *After* Fast Task Was Polled & ACKed**: Task finishes in `5ms`, is polled and ACKed at `20ms`, and gRPC's delayed `DispatchTask` retry arrives at `100ms`. | Successful `request_id`s are retained in the bounded `_dispatched_request_ids` FIFO (`10,000` max entries) even after `ack_request_id`, and are only evicted early if `_run_and_enqueue` failed (`error_message is not None`). |
| **6** | **`Execute` Payloads `> 256 KiB` Bypassing gRPC C-Core `retryPolicy`**: Training batches exceed gRPC's 256 KiB C-core retry buffer. | `GrpcRemoteActorHandle` implements a Python `_call_with_retry` wrapper on `StatusCode.UNAVAILABLE` reusing the in-memory `ExecutionRequest` (0 extra RAM). |
| **7** | **Background `DispatchTask` Clobbering `Execute` Cache**: `_run_and_enqueue` calls `execute_request()` in the background. | `Execute` deduplication lives in `execute_idempotent_request()` (called only by `_handle_execute`), completely isolated from `_run_and_enqueue()`. |
| **8** | **Multiple Concurrent `asubmit()` Calls Completing Before Next `Execute`**: Two `asubmit()` calls finish via `asyncio.gather()` before the next `Execute` call sends an ACK. | Client accumulates completed IDs in a set `self._pending_execute_acks: set[str]` (`ack_execute_ids: tuple[str, ...]`) rather than a single string, plus a server FIFO safety cap (`16`). |
