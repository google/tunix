# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Isolated performance benchmark for `remote_execution`.

Measures pure chunk serialization, zero-copy reassembly, end-to-end gRPC
streaming (`asubmit`), and concurrent multi-worker `dispatch_task` +
`poll_responses` throughput and event-loop responsiveness.

Run in CI mode (default ~128 MiB payloads, fast):
  pytest tests/experimental/worker/remote_execution_perf_test.py -s

Run at full MoE router-replay scale (>2.15 GiB payloads):
  pytest tests/experimental/worker/remote_execution_perf_test.py -s \
    --full_scale=true --payload_gb=2.15
"""

import asyncio
from collections.abc import AsyncIterator
import contextlib
import dataclasses
import json
import time
from typing import Any

from absl import flags
from absl import logging
from absl.testing import absltest
import numpy as np
import portpicker
from tunix.experimental.worker import remote_execution

_FULL_SCALE = flags.DEFINE_bool(
    "full_scale",
    False,
    "If True, runs full-scale (>2.15 GiB) payload benchmarks.",
)
_PAYLOAD_GB = flags.DEFINE_float(
    "payload_gb",
    2.15,
    "Target payload size in GiB when --full_scale=true.",
)
_REPORT_PATH = flags.DEFINE_string(
    "report_path",
    "",
    "Optional file path to write the JSON benchmark report.",
)


# ==============================================================================
# Event-Loop Lag Monitor & Telemetry
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class LoopLagMetrics:
  """Summary of asyncio event-loop scheduling lag during a workload."""

  p50_ms: float
  p95_ms: float
  p99_ms: float
  max_ms: float
  blocked_over_10ms_total_ms: float


class EventLoopLagMonitor:
  """Samples asyncio event-loop scheduling lag at a 1 ms interval."""

  def __init__(self, interval_s: float = 0.001):
    self._interval_s = interval_s
    self._lags_ms: list[float] = []
    self._running = False
    self._task: asyncio.Task[None] | None = None

  async def _loop(self) -> None:
    loop = asyncio.get_running_loop()
    next_wake = loop.time() + self._interval_s
    while self._running:
      await asyncio.sleep(self._interval_s)
      now = loop.time()
      lag_ms = max(0.0, (now - next_wake) * 1000.0)
      self._lags_ms.append(lag_ms)
      next_wake = now + self._interval_s

  def start(self) -> None:
    self._lags_ms.clear()
    self._running = True
    self._task = asyncio.create_task(self._loop())

  async def stop(self) -> LoopLagMetrics:
    self._running = False
    if self._task is not None:
      with contextlib.suppress(asyncio.CancelledError):
        await self._task
      self._task = None
    if not self._lags_ms:
      return LoopLagMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(self._lags_ms, dtype=np.float64)
    blocked = float(np.sum(arr[arr > 10.0]))
    return LoopLagMetrics(
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        max_ms=float(np.max(arr)),
        blocked_over_10ms_total_ms=blocked,
    )


@dataclasses.dataclass(frozen=True)
class PerfResult:
  """Metrics for a single benchmark workload."""

  workload: str
  payload_mib: float
  wall_ms: float
  throughput_gib_s: float
  chunk_count: int
  loop_lag: LoopLagMetrics
  extra: dict[str, float] = dataclasses.field(default_factory=dict)


# ==============================================================================
# Mock Worker Actor for gRPC Benchmarks
# ==============================================================================


class _EchoPayloadWorker:
  """Worker actor supporting large payload echo, generation, and fast pings."""

  def __init__(self, response_payload: dict[str, Any]):
    self._response_payload = response_payload

  async def echo_summary(self, payload: dict[str, Any]) -> dict[str, int]:
    routed = payload["routed_experts"]
    return {
        "nbytes": int(routed.nbytes),
        "first_val": int(routed.flat[0]),
        "last_val": int(routed.flat[-1]),
    }

  async def generate_large(self, prompt_id: int) -> dict[str, Any]:
    await asyncio.sleep(0.005)
    return {
        "prompt_id": prompt_id,
        "payload": self._response_payload,
    }

  async def ping(self, seq: int) -> int:
    return seq + 1


# ==============================================================================
# Payload Builders
# ==============================================================================


def _build_mixed_payload(
    target_bytes: int, num_buffers: int = 16
) -> dict[str, Any]:
  """Builds a multi-buffer payload with per-trajectory routed_experts arrays."""
  bytes_per_buf = max(4096, target_bytes // num_buffers)
  elem_count = bytes_per_buf // 2  # int16
  routed_list = []
  base = np.arange(elem_count, dtype=np.int16)
  for i in range(num_buffers):
    arr = base.copy()
    arr[0] = i
    arr[-1] = i + 100
    routed_list.append(arr)
  small_meta = [
      np.ones((256,), dtype=np.float32) * idx for idx in range(num_buffers)
  ]
  return {
      "trajectory_experts": routed_list,
      "logps": small_meta,
      "step": 42,
  }


def _build_single_buffer_payload(target_bytes: int) -> dict[str, Any]:
  """Builds a payload dominated by a single contiguous routed_experts tensor."""
  elem_count = max(2048, target_bytes // 2)
  routed = np.empty((elem_count,), dtype=np.int16)
  routed[0] = 17
  routed[-1] = 99
  routed[elem_count // 2] = 55
  small_meta = [np.zeros((1024,), dtype=np.float32) for _ in range(16)]
  return {
      "routed_experts": routed,
      "small_meta": small_meta,
      "step": 7,
  }


# ==============================================================================
# Formatting Helpers
# ==============================================================================


def _format_results_table(results: list[PerfResult]) -> str:
  """Formats a summary table of isolated remote_execution performance metrics."""
  lines = [
      "",
      "=" * 112,
      "REMOTE EXECUTION ISOLATED PERFORMANCE BENCHMARK",
      "=" * 112,
      (
          f"{'Workload':<36} | {'Size MiB':>8} | {'Wall ms':>9} | "
          f"{'GiB/s':>7} | {'Chunks':>6} | {'Lag p50 ms':>10} | "
          f"{'Lag p99 ms':>10} | {'Lag Max ms':>10} | {'Blocked>10ms':>12}"
      ),
      "-" * 112,
  ]
  for res in results:
    lines.append(
        f"{res.workload:<36} | {res.payload_mib:>8.1f} | "
        f"{res.wall_ms:>9.1f} | {res.throughput_gib_s:>7.2f} | "
        f"{res.chunk_count:>6d} | {res.loop_lag.p50_ms:>10.2f} | "
        f"{res.loop_lag.p99_ms:>10.2f} | {res.loop_lag.max_ms:>10.2f} | "
        f"{res.loop_lag.blocked_over_10ms_total_ms:>10.1f}ms"
    )
    if "ping_p95_ms" in res.extra:
      lines.append(
          f"  --> Concurrent Control Ping Latency: "
          f"p50={res.extra['ping_p50_ms']:.2f}ms, "
          f"p95={res.extra['ping_p95_ms']:.2f}ms, "
          f"max={res.extra['ping_max_ms']:.2f}ms "
          f"(n={int(res.extra['ping_count'])})"
      )
  lines.append("=" * 112)
  return "\n".join(lines)


# ==============================================================================
# Test Suite
# ==============================================================================


class RemoteExecutionPerfTest(absltest.TestCase):
  """Isolated performance benchmark for `remote_execution`."""

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls._results: list[PerfResult] = []

  @classmethod
  def tearDownClass(cls) -> None:
    if cls._results:
      table = _format_results_table(cls._results)
      logging.info("%s", table)
      print(table, flush=True)
      if _REPORT_PATH.value:
        records = [dataclasses.asdict(r) for r in cls._results]
        with open(_REPORT_PATH.value, "w", encoding="utf-8") as f:
          json.dump(records, f, indent=2)
    super().tearDownClass()

  def _target_bytes(self) -> int:
    if _FULL_SCALE.value:
      return int(_PAYLOAD_GB.value * (1024**3))
    return 128 * 1024 * 1024  # 128 MiB in fast CI mode

  def test_w1_pure_chunk_serialization(self) -> None:
    """Benchmarks pure chunk serialization (Single-Buffer W1A & Multi-Buffer W1B)."""
    target_bytes = self._target_bytes()
    single_req = remote_execution.ExecutionRequest(
        method_name="train_step",
        args=(_build_single_buffer_payload(target_bytes),),
        kwargs={},
    )
    num_multi_bufs = max(32, target_bytes // (4 * 1024 * 1024))
    multi_req = remote_execution.ExecutionRequest(
        method_name="train_step",
        args=(_build_mixed_payload(target_bytes, num_buffers=num_multi_bufs),),
        kwargs={},
    )

    async def _run_serialize(
        workload: str, req: remote_execution.ExecutionRequest
    ) -> PerfResult:
      monitor = EventLoopLagMonitor()
      monitor.start()
      t0 = time.perf_counter()
      total_bytes = 0
      chunk_count = 0
      async for chunk in req.serialize_async_chunks():
        total_bytes += len(chunk)
        chunk_count += 1
        await asyncio.sleep(0)
      wall_ms = (time.perf_counter() - t0) * 1000.0
      lag = await monitor.stop()
      gib = total_bytes / (1024**3)
      return PerfResult(
          workload=workload,
          payload_mib=total_bytes / (1024**2),
          wall_ms=wall_ms,
          throughput_gib_s=gib / (wall_ms / 1000.0),
          chunk_count=chunk_count,
          loop_lag=lag,
      )

    async def _main() -> list[PerfResult]:
      warmup_req = remote_execution.ExecutionRequest(
          method_name="warmup",
          args=(np.zeros((1024,), dtype=np.int16),),
          kwargs={},
      )
      async for _ in warmup_req.serialize_async_chunks():
        pass
      res_a = await _run_serialize("W1A_SingleBuffer_Serialize", single_req)
      res_b = await _run_serialize("W1B_MultiBuffer_Serialize", multi_req)
      return [res_a, res_b]

    results = asyncio.run(_main())
    self.__class__._results.extend(results)
    self.assertGreater(results[0].throughput_gib_s, 0.0)
    self.assertGreater(results[1].throughput_gib_s, 0.0)

  def test_w2_pure_chunk_reassembly_and_unpickle(self) -> None:
    """Benchmarks pure chunk reassembly & unpickling from in-memory chunks."""
    target_bytes = self._target_bytes()
    payload = _build_single_buffer_payload(target_bytes)
    req = remote_execution.ExecutionRequest(
        method_name="train_step", args=(payload,), kwargs={}
    )
    chunks = list(req.serialize_chunks())
    total_bytes = sum(len(c) for c in chunks)

    async def _iter_chunks(chunk_list: list[bytes]) -> AsyncIterator[bytes]:
      for c in chunk_list:
        await asyncio.sleep(0)
        yield c

    async def _main() -> PerfResult:
      monitor = EventLoopLagMonitor()
      monitor.start()
      t0 = time.perf_counter()
      restored = (
          await remote_execution.ExecutionRequest.deserialize_async_chunks(
              _iter_chunks(chunks)
          )
      )
      wall_ms = (time.perf_counter() - t0) * 1000.0
      lag = await monitor.stop()
      self.assertEqual(restored.args[0]["routed_experts"][0], 17)
      # Verify writability is preserved on reassembled NumPy arrays.
      restored.args[0]["routed_experts"][0] = 42
      self.assertEqual(restored.args[0]["routed_experts"][0], 42)
      gib = total_bytes / (1024**3)
      return PerfResult(
          workload="W2_Pure_Reassemble_Unpickle",
          payload_mib=total_bytes / (1024**2),
          wall_ms=wall_ms,
          throughput_gib_s=gib / (wall_ms / 1000.0),
          chunk_count=len(chunks),
          loop_lag=lag,
      )

    res = asyncio.run(_main())
    self.__class__._results.append(res)
    self.assertGreater(res.throughput_gib_s, 0.0)

  def test_w3_grpc_asubmit_large_payload_roundtrip(self) -> None:
    """Benchmarks end-to-end gRPC TCP loopback asubmit() for a large payload."""
    target_bytes = self._target_bytes()
    payload = _build_single_buffer_payload(target_bytes)
    payload_mib = payload["routed_experts"].nbytes / (1024**2)

    async def _main() -> PerfResult:
      worker = _EchoPayloadWorker(response_payload={})
      chunk_bytes = remote_execution._STREAM_CHUNK_BYTES
      port = portpicker.pick_unused_port()
      server = remote_execution.GrpcRemoteExecutionServer(
          worker, stream_chunk_bytes=chunk_bytes
      )
      await server.start_serving_async(port)
      handle = remote_execution.GrpcRemoteActorHandle(
          f"localhost:{port}",
          rpc_timeout_s=300.0,
          stream_chunk_bytes=chunk_bytes,
      )
      try:
        await handle.asubmit("ping", 1)
        monitor = EventLoopLagMonitor()
        monitor.start()
        t0 = time.perf_counter()
        summary = await handle.asubmit("echo_summary", payload)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        lag = await monitor.stop()
        self.assertEqual(summary["first_val"], 17)
        self.assertEqual(summary["last_val"], 99)
        gib = payload_mib / 1024.0
        est_chunks = (
            int(np.ceil(payload["routed_experts"].nbytes / chunk_bytes)) + 1
        )
        return PerfResult(
            workload="W3_gRPC_asubmit_LargePayload",
            payload_mib=payload_mib,
            wall_ms=wall_ms,
            throughput_gib_s=gib / (wall_ms / 1000.0),
            chunk_count=est_chunks,
            loop_lag=lag,
        )
      finally:
        await handle.close()
        await server.stop_serving(grace=0.0)

    res = asyncio.run(_main())
    self.__class__._results.append(res)
    self.assertGreater(res.throughput_gib_s, 0.0)

  def test_w4_grpc_concurrent_streaming_and_control_plane(self) -> None:
    """Benchmarks 4 concurrent dispatch_task/poll_responses streams + control pings."""
    num_workers = 4
    per_worker_bytes = self._target_bytes() // num_workers
    worker_payload = _build_single_buffer_payload(per_worker_bytes)
    total_mib = (worker_payload["routed_experts"].nbytes * num_workers) / (
        1024**2
    )

    async def _main() -> PerfResult:
      chunk_bytes = remote_execution._STREAM_CHUNK_BYTES
      servers: list[remote_execution.GrpcRemoteExecutionServer] = []
      handles: list[remote_execution.GrpcRemoteActorHandle] = []
      try:
        for _ in range(num_workers):
          port = portpicker.pick_unused_port()
          srv = remote_execution.GrpcRemoteExecutionServer(
              _EchoPayloadWorker(response_payload=worker_payload),
              stream_chunk_bytes=chunk_bytes,
          )
          await srv.start_serving_async(port)
          servers.append(srv)
          hdl = remote_execution.GrpcRemoteActorHandle(
              f"localhost:{port}",
              rpc_timeout_s=300.0,
              stream_chunk_bytes=chunk_bytes,
          )
          await hdl.asubmit("ping", 0)
          handles.append(hdl)

        ping_latencies_ms: list[float] = []
        stop_pings = False

        async def _control_pinger() -> None:
          seq = 0
          while not stop_pings:
            t_ping = time.perf_counter()
            res = await handles[seq % num_workers].asubmit("ping", seq)
            ping_latencies_ms.append((time.perf_counter() - t_ping) * 1000.0)
            self.assertEqual(res, seq + 1)
            seq += 1
            await asyncio.sleep(0.010)

        async def _dispatch_and_poll(
            idx: int, hdl: remote_execution.GrpcRemoteActorHandle
        ) -> dict[str, Any]:
          req_id = await hdl.dispatch_task(
              f"req_{idx}", "generate_large", idx
          )
          self.assertEqual(req_id, f"req_{idx}")
          resp = await hdl.poll_responses(timeout_s=300.0)
          assert resp is not None
          return resp.unwrap()

        monitor = EventLoopLagMonitor()
        monitor.start()
        ping_task = asyncio.create_task(_control_pinger())
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[_dispatch_and_poll(i, h) for i, h in enumerate(handles)]
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        stop_pings = True
        await ping_task
        lag = await monitor.stop()

        for i, r in enumerate(results):
          self.assertEqual(r["prompt_id"], i)
          self.assertEqual(r["payload"]["routed_experts"][0], 17)

        ping_arr = (
            np.asarray(ping_latencies_ms, dtype=np.float64)
            if ping_latencies_ms
            else np.zeros((1,), dtype=np.float64)
        )
        gib = total_mib / 1024.0
        est_chunks = num_workers * (
            int(np.ceil(worker_payload["routed_experts"].nbytes / chunk_bytes))
            + 1
        )
        return PerfResult(
            workload="W4_gRPC_4xConcurrent_Poll_And_Ping",
            payload_mib=total_mib,
            wall_ms=wall_ms,
            throughput_gib_s=gib / (wall_ms / 1000.0),
            chunk_count=est_chunks,
            loop_lag=lag,
            extra={
                "ping_p50_ms": float(np.percentile(ping_arr, 50)),
                "ping_p95_ms": float(np.percentile(ping_arr, 95)),
                "ping_max_ms": float(np.max(ping_arr)),
                "ping_count": float(len(ping_latencies_ms)),
            },
        )
      finally:
        for hdl in handles:
          await hdl.close()
        for srv in servers:
          await srv.stop_serving(grace=0.0)

    res = asyncio.run(_main())
    self.__class__._results.append(res)
    self.assertGreater(res.throughput_gib_s, 0.0)

  def test_w5_high_frequency_rollout_rpc_serde(self) -> None:
    """Benchmarks 512 rollout request + response async chunk serde cycles."""
    num_items = 512
    requests = [
        remote_execution.ExecutionRequest(
            request_id=f"req_prompt_{i}_g{i % 16}_v5",
            method_name="generate",
            args=(),
            kwargs={
                "prompt": f"Solve math problem #{i}",
                "prompt_id": f"prompt_{i}",
                "group_index": i % 16,
                "generation_kwargs": {
                    "temperature": 0.8,
                    "max_tokens": 2048,
                    "return_logprobs": True,
                },
            },
        )
        for i in range(num_items)
    ]
    responses = [
        remote_execution.ExecutionResponse(
            result={
                "prompt_id": f"prompt_{i}",
                "group_index": i % 16,
                "prompt_tokens": np.ones(256, dtype=np.int32),
                "conversation_tokens": np.ones(1536, dtype=np.int32),
                "conversation_masks": np.ones(1536, dtype=np.float32),
                "old_per_token_logps": np.zeros(1536, dtype=np.float32),
                "trajectory_reward": 1.0,
            },
            request_id=f"req_prompt_{i}_g{i % 16}_v5",
        )
        for i in range(num_items)
    ]

    async def _main() -> PerfResult:
      monitor = EventLoopLagMonitor()
      monitor.start()
      t0 = time.perf_counter()
      total_bytes = 0
      chunk_count = 0
      for req, resp in zip(requests, responses):
        req_chunks = []
        async for c in req.serialize_async_chunks():
          total_bytes += len(c)
          chunk_count += 1
          req_chunks.append(c)
        _ = remote_execution.ExecutionRequest.deserialize_chunks(req_chunks)

        resp_chunks = []
        async for c in resp.serialize_async_chunks():
          total_bytes += len(c)
          chunk_count += 1
          resp_chunks.append(c)
        _ = remote_execution.ExecutionResponse.deserialize_chunks(resp_chunks)

      wall_ms = (time.perf_counter() - t0) * 1000.0
      lag = await monitor.stop()
      gib = total_bytes / (1024**3)
      return PerfResult(
          workload="W5_512x_Rollout_ReqResp_Serde",
          payload_mib=total_bytes / (1024**2),
          wall_ms=wall_ms,
          throughput_gib_s=gib / (wall_ms / 1000.0),
          chunk_count=chunk_count,
          loop_lag=lag,
      )

    res = asyncio.run(_main())
    self.__class__._results.append(res)
    self.assertGreater(res.throughput_gib_s, 0.0)


if __name__ == "__main__":
  absltest.main()

