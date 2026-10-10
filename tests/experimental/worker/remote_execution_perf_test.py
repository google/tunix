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
from collections.abc import AsyncIterator, Callable
import contextlib
import dataclasses
import gc
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any

from absl import flags
from absl import logging
from absl.testing import absltest
import numpy as np
import portpicker
from tunix.experimental.worker import network
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
_WORKER_SERVER_PORT = flags.DEFINE_integer(
    "worker_server_port",
    0,
    "Internal flag: if > 0, runs an isolated gRPC worker server on this port.",
)
_WORKER_PAYLOAD_BYTES = flags.DEFINE_integer(
    "worker_payload_bytes",
    0,
    "Internal flag: payload size in bytes for the isolated gRPC worker server.",
)
_REMOTE_ROLLOUT_TARGETS = flags.DEFINE_string(
    "remote_rollout_targets",
    "",
    "Optional comma-separated host:port list of pre-launched remote Rollout "
    "Worker servers for cross-machine benchmarking.",
)
_REMOTE_TRAINER_TARGET = flags.DEFINE_string(
    "remote_trainer_target",
    "",
    "Optional host:port of a pre-launched remote Trainer Worker server for "
    "cross-machine benchmarking.",
)
_NUM_ITERATIONS = flags.DEFINE_integer(
    "num_iterations",
    0,
    "Number of timed iterations per workload (0 = 5 when --full_scale=true, "
    "1 in fast CI mode).",
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
class OrchestratorComputeMetrics:
  """Metrics for concurrent Orchestrator Python + NumPy compute load."""

  steps_completed: int
  steps_per_s: float
  idle_baseline_steps_per_s: float
  compute_retention_pct: float
  checksum: int


class OrchestratorComputeSimulator:
  """Simulates concurrent Orchestrator Python GIL + NumPy batch assembly load.

  Runs two concurrent workloads while bulk network transfers are active:
  1. A continuous background worker thread executing pure-Python trajectory
     dictionary construction and token-level reward grading (holding the GIL
     in bytecode loops) followed by NumPy GRPO group advantage normalization
     and `routed_experts` slice assembly into a staging buffer.
  2. A periodic coroutine on the `asyncio` event loop performing trajectory
     group bookkeeping every 5 ms.
  """

  def __init__(self) -> None:
    self._running = False
    self._thread: threading.Thread | None = None
    self._async_task: asyncio.Task[None] | None = None
    self._steps_completed = 0
    self._checksum = 0
    self._elapsed_s = 0.0
    # Pre-allocated staging buffers for simulated batch assembly:
    # 16 trajectories x 2048 tokens logps/rewards + (1024, 62, 8) int16 router.
    self._src_logps = np.linspace(
        -2.5, -0.1, 16 * 2048, dtype=np.float32
    ).reshape(16, 2048)
    self._src_routed = np.arange(
        1024 * 62 * 8, dtype=np.int16
    ).reshape(1024, 62, 8)
    self._staging_routed = np.empty((1024, 62, 8), dtype=np.int16)

  def _run_one_step(self, step_idx: int) -> int:
    # 1. Pure-Python bytecode loop (holds GIL): simulate parsing & reward
    # grading across 16 trajectories x 128 sampled tokens and building dicts.
    acc = step_idx + 1
    traj_records: list[dict[str, int | float]] = []
    for t_idx in range(16):
      token_hash = acc ^ (t_idx * 4099)
      for tok in range(128):
        token_hash = ((token_hash * 33) ^ (tok + t_idx)) & 0x7FFFFFFF
      traj_records.append({
          "prompt_id": t_idx // 4,
          "group_idx": t_idx % 4,
          "reward": float(token_hash % 1000) * 0.001,
      })
    reward_sum = sum(float(r["reward"]) for r in traj_records)

    # 2. NumPy array compute + memory copy (GRPO group advantage normalization
    # + routed_experts slice copy into batch assembly staging buffer).
    rewards = np.array(
        [float(r["reward"]) for r in traj_records], dtype=np.float32
    ).reshape(4, 4)
    mean = np.mean(rewards, axis=1, keepdims=True)
    std = np.std(rewards, axis=1, keepdims=True)
    advantages = ((rewards - mean) / (std + np.float32(1e-8))).reshape(16, 1)
    weighted_logps = self._src_logps * advantages
    np.copyto(self._staging_routed, self._src_routed)
    self._staging_routed[0, 0, 0] = np.int16(step_idx & 0x7FFF)
    return int(reward_sum) + int(weighted_logps[0, 0]) + int(
        self._staging_routed[0, 0, 0]
    )

  def _worker_loop(self) -> None:
    t0 = time.perf_counter()
    steps = 0
    csum = 0
    while self._running:
      csum ^= self._run_one_step(steps)
      steps += 1
    self._elapsed_s = max(1e-9, time.perf_counter() - t0)
    self._steps_completed = steps
    self._checksum = csum

  async def _event_loop_bookkeeper(self) -> None:
    step = 0
    while self._running:
      # Simulate lightweight trajectory queue bookkeeping on the event loop.
      queue_map = {f"group_{i}": (step + i) ^ 0x5A for i in range(32)}
      self._checksum ^= sum(queue_map.values())
      step += 1
      await asyncio.sleep(0.005)

  def calibrate_idle_baseline(self, duration_s: float = 0.25) -> float:
    """Measures standalone steps/s with no concurrent network transfer."""
    self._running = True
    t0 = time.perf_counter()
    steps = 0
    csum = 0
    while (time.perf_counter() - t0) < duration_s:
      csum ^= self._run_one_step(steps)
      steps += 1
    elapsed = max(1e-9, time.perf_counter() - t0)
    self._running = False
    self._checksum ^= csum
    return float(steps) / elapsed

  def start(self) -> None:
    self._steps_completed = 0
    self._elapsed_s = 0.0
    self._running = True
    self._async_task = asyncio.create_task(self._event_loop_bookkeeper())
    self._thread = threading.Thread(
        target=self._worker_loop, name="orch-compute-sim", daemon=True
    )
    self._thread.start()

  async def stop(
      self, idle_baseline_steps_per_s: float
  ) -> OrchestratorComputeMetrics:
    self._running = False
    if self._thread is not None:
      await asyncio.to_thread(self._thread.join, 5.0)
      self._thread = None
    if self._async_task is not None:
      with contextlib.suppress(asyncio.CancelledError):
        await self._async_task
      self._async_task = None
    steps_per_s = float(self._steps_completed) / max(1e-9, self._elapsed_s)
    retention_pct = (
        (steps_per_s / idle_baseline_steps_per_s) * 100.0
        if idle_baseline_steps_per_s > 0.0
        else 0.0
    )
    return OrchestratorComputeMetrics(
        steps_completed=self._steps_completed,
        steps_per_s=steps_per_s,
        idle_baseline_steps_per_s=idle_baseline_steps_per_s,
        compute_retention_pct=retention_pct,
        checksum=self._checksum,
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


def _payload_page_checksum(arr: np.ndarray) -> int:
  """Computes a fast strided checksum across every 4 KiB OS page of `arr`."""
  step = max(1, 2048 // max(1, arr.itemsize))
  return int(np.sum(arr.ravel()[::step], dtype=np.int64))


class _EchoPayloadWorker:
  """Worker actor supporting large payload echo, generation, and fast pings."""

  def __init__(self, response_payload: dict[str, Any]):
    self._response_payload = response_payload

  async def echo_summary(
      self, payload: dict[str, Any], verify_checksum: bool = False
  ) -> dict[str, int]:
    routed = payload["routed_experts"]
    total_nbytes = int(routed.nbytes)
    checksum = _payload_page_checksum(routed) if verify_checksum else 0
    shards = payload["trajectory_shards"]
    for shard in shards:
      total_nbytes += int(shard.nbytes)
      if verify_checksum:
        checksum += _payload_page_checksum(shard)
    small_meta = payload["small_meta"]
    meta_nbytes = sum(int(m.nbytes) for m in small_meta)
    meta_sum = int(sum(float(m[0]) + float(m[-1]) for m in small_meta))
    return {
        "nbytes": int(routed.nbytes),
        "total_nbytes": total_nbytes,
        "meta_nbytes": meta_nbytes,
        "meta_sum": meta_sum,
        "checksum": checksum,
        "first_val": int(routed.flat[0]),
        "mid_val": int(routed.flat[routed.size // 2]),
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
  """Builds a payload dominated by a single contiguous routed_experts tensor.

  Every 4 KiB OS page is explicitly written so the buffer is backed by real
  physical DRAM pages rather than the kernel's shared zero page.

  Args:
    target_bytes: Approximate target size in bytes for the main tensor.

  Returns:
    A dictionary payload containing `routed_experts` and `small_meta`.
  """
  elem_count = max(2048, target_bytes // 2)
  routed = np.empty((elem_count,), dtype=np.int16)
  # Fill in 32 MiB blocks from a pre-populated pattern so every physical page
  # is dirtied in DRAM without allocating a second 2.15 GiB temporary array.
  block_elems = min(elem_count, 16 * 1024 * 1024)
  pattern = (np.arange(block_elems, dtype=np.int16) & np.int16(0x7FFF)) + 1
  for offset in range(0, elem_count, block_elems):
    end = min(elem_count, offset + block_elems)
    routed[offset:end] = pattern[: end - offset]
  routed[0] = 17
  routed[-1] = 99
  routed[elem_count // 2] = 55
  small_meta = [
      np.full((1024,), idx + 1.0, dtype=np.float32) for idx in range(16)
  ]
  return {
      "routed_experts": routed,
      "trajectory_shards": (),
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
    if (
        "wall_min_ms" in res.extra
        and "iterations" in res.extra
        and res.extra["iterations"] > 1.0
    ):
      lines.append(
          f"  --> Multi-Run Stats (n={int(res.extra['iterations'])}): "
          f"median={res.wall_ms:.1f}ms ({res.throughput_gib_s:.2f} GiB/s), "
          f"p10={res.extra['wall_p10_ms']:.1f}ms, "
          f"p90={res.extra['wall_p90_ms']:.1f}ms, "
          f"min={res.extra['wall_min_ms']:.1f}ms, "
          f"max={res.extra['wall_max_ms']:.1f}ms"
      )
    if "ping_p95_ms" in res.extra:
      idle_str = ""
      if "idle_ping_p50_ms" in res.extra:
        idle_str = (
            f" [idle baseline: p50={res.extra['idle_ping_p50_ms']:.2f}ms, "
            f"p95={res.extra['idle_ping_p95_ms']:.2f}ms]"
        )
      lines.append(
          f"  --> Concurrent Control Ping Latency: "
          f"p50={res.extra['ping_p50_ms']:.2f}ms, "
          f"p95={res.extra['ping_p95_ms']:.2f}ms, "
          f"max={res.extra['ping_max_ms']:.2f}ms "
          f"(n={int(res.extra['ping_count'])}){idle_str}"
      )
    if "rollout_to_orch_ms" in res.extra:
      e2e_gibs = res.extra["e2e_batch_gib_s"]
      lines.append(
          f"  --> 3-Tier Hop Breakdown: "
          f"4xRollout->Orch={res.extra['rollout_to_orch_ms']:.1f}ms, "
          f"Orch->Trainer={res.extra['orch_to_trainer_ms']:.1f}ms "
          f"(E2E Batch Goodput={e2e_gibs:.2f} GiB/s, "
          f"2-Hop Network Goodput={res.throughput_gib_s:.2f} GiB/s)"
      )
    if "orch_compute_steps_per_s" in res.extra:
      lines.append(
          "  --> Concurrent Orchestrator Python+NumPy Load: "
          f"{res.extra['orch_compute_steps_per_s']:.1f} steps/s "
          f"(idle={res.extra['orch_idle_steps_per_s']:.1f} steps/s, "
          f"retention={res.extra['orch_compute_retention_pct']:.1f}%, "
          f"net_retention={res.extra['net_goodput_retention_pct']:.1f}%, "
          f"steps={int(res.extra['orch_compute_steps'])})"
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
    if not flags.FLAGS.is_parsed():
      flags.FLAGS.mark_as_parsed()
    if _WORKER_SERVER_PORT.value > 0:
      _run_isolated_worker_server(
          _WORKER_SERVER_PORT.value, _WORKER_PAYLOAD_BYTES.value
      )
      os._exit(0)
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

  def tearDown(self) -> None:
    gc.collect()
    super().tearDown()

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
      chunk_bytes = network._STREAM_CHUNK_BYTES
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
        _ = await handle.asubmit("echo_summary", payload)
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
      chunk_bytes = network._STREAM_CHUNK_BYTES
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

        await asyncio.gather(
            *[_dispatch_and_poll(i, h) for i, h in enumerate(handles)]
        )
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

  def test_w6_multiprocess_isolated_e2e(self) -> None:
    """Benchmarks multi-process E2E gRPC with Rollout, Orchestrator, and Trainer in isolated OS processes."""
    target_bytes = self._target_bytes()
    remote_rollout_list = [
        t.strip()
        for t in _REMOTE_ROLLOUT_TARGETS.value.split(",")
        if t.strip()
    ]
    remote_trainer_addr = _REMOTE_TRAINER_TARGET.value.strip()
    num_rollout_workers = (
        len(remote_rollout_list) if remote_rollout_list else 4
    )
    per_worker_bytes = target_bytes // num_rollout_workers
    gc.collect()

    procs: list[subprocess.Popen[str]] = []
    try:
      if bool(remote_rollout_list) != bool(remote_trainer_addr):
        raise ValueError(
            "Both --remote_rollout_targets and --remote_trainer_target must be "
            "provided together for cross-machine benchmarking."
        )
      if remote_rollout_list and remote_trainer_addr:
        rollout_targets = remote_rollout_list
        trainer_target = remote_trainer_addr
      else:
        # Spawn 4 isolated Rollout Worker processes (serving per-worker
        # trajectories) and 1 isolated Trainer Worker process (receiving
        # full-batch train_step) BEFORE allocating the Orchestrator's
        # multi-gigabyte payload tensors so OS fork() does not copy 2.7 GiB of
        # parent page tables into 5 children.
        rollout_ports = [
            portpicker.pick_unused_port() for _ in range(num_rollout_workers)
        ]
        trainer_port = portpicker.pick_unused_port()
        is_blaze_binary = (
            os.path.basename(sys.argv[0]) == "remote_execution_perf_test"
        )
        if is_blaze_binary:
          base_cmd = [sys.argv[0]]
          child_env = None
        else:
          base_cmd = [sys.executable, os.path.abspath(__file__)]
          child_env = {
              **os.environ,
              "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
          }
        for port in rollout_ports:
          proc = subprocess.Popen(
              base_cmd
              + [
                  f"--worker_server_port={port}",
                  f"--worker_payload_bytes={per_worker_bytes}",
              ],
              stdin=subprocess.PIPE,
              stdout=subprocess.PIPE,
              text=True,
              env=child_env,
          )
          procs.append(proc)
        trainer_proc = subprocess.Popen(
            base_cmd
            + [
                f"--worker_server_port={trainer_port}",
                "--worker_payload_bytes=0",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=child_env,
        )
        procs.append(trainer_proc)

        for proc in procs:
          assert proc.stdout is not None
          ready_line = proc.stdout.readline().strip()
          self.assertEqual(ready_line, "READY")

        rollout_targets = [f"127.0.0.2:{p}" for p in rollout_ports]
        trainer_target = f"127.0.0.2:{trainer_port}"

      single_payload = _build_single_buffer_payload(target_bytes)
      expected_single_nbytes = int(single_payload["routed_experts"].nbytes)
      expected_single_checksum = _payload_page_checksum(
          single_payload["routed_experts"]
      )
      expected_meta_nbytes = sum(
          int(m.nbytes) for m in single_payload["small_meta"]
      )
      expected_meta_sum = int(
          sum(float(m[0]) + float(m[-1]) for m in single_payload["small_meta"])
      )

      worker_payload = _build_single_buffer_payload(per_worker_bytes)
      expected_worker_nbytes = int(worker_payload["routed_experts"].nbytes)
      expected_worker_checksum = _payload_page_checksum(
          worker_payload["routed_experts"]
      )
      expected_total_conc_nbytes = expected_worker_nbytes * num_rollout_workers
      expected_total_conc_checksum = (
          expected_worker_checksum * num_rollout_workers
      )

      num_iters = _NUM_ITERATIONS.value
      if num_iters <= 0:
        num_iters = 5 if _FULL_SCALE.value else 1

      async def _main() -> list[PerfResult]:
        chunk_bytes = network._STREAM_CHUNK_BYTES
        rollout_handles: list[remote_execution.GrpcRemoteActorHandle] = []
        trainer_handle: remote_execution.GrpcRemoteActorHandle | None = None
        try:
          for target_addr in rollout_targets:
            hdl = remote_execution.GrpcRemoteActorHandle(
                target_addr,
                rpc_timeout_s=300.0,
                stream_chunk_bytes=chunk_bytes,
            )
            await hdl.asubmit("ping", 0)
            rollout_handles.append(hdl)

          trainer_handle = remote_execution.GrpcRemoteActorHandle(
              trainer_target,
              rpc_timeout_s=300.0,
              stream_chunk_bytes=chunk_bytes,
          )
          await trainer_handle.asubmit("ping", 0)

          # Measure idle control-plane ping baseline before data-plane load.
          idle_pings_ms: list[float] = []
          for seq in range(20):
            t_p = time.perf_counter()
            res_p = await rollout_handles[seq % num_rollout_workers].asubmit(
                "ping", seq
            )
            idle_pings_ms.append((time.perf_counter() - t_p) * 1000.0)
            self.assertEqual(res_p, seq + 1)
          idle_ping_arr = np.asarray(idle_pings_ms, dtype=np.float64)

          # --- W6A: Orchestrator -> Trainer Isolated Push (Large Batch) ---
          warm_summary = await trainer_handle.asubmit(
              "echo_summary", single_payload, verify_checksum=True
          )
          self.assertIsNotNone(trainer_handle._bulk_port)
          self.assertGreater(trainer_handle._bulk_port, 0)
          self.assertNotEmpty(trainer_handle._seen_bulk_ports)
          self.assertEqual(warm_summary["total_nbytes"], expected_single_nbytes)
          self.assertEqual(warm_summary["meta_nbytes"], expected_meta_nbytes)
          self.assertEqual(warm_summary["meta_sum"], expected_meta_sum)
          self.assertEqual(
              warm_summary["checksum"], expected_single_checksum
          )

          monitor_a = EventLoopLagMonitor()
          monitor_a.start()
          walls_a_ms: list[float] = []
          actual_a_nbytes = 0
          for _ in range(num_iters):
            t0_a = time.perf_counter()
            summary = await trainer_handle.asubmit(
                "echo_summary", single_payload
            )
            walls_a_ms.append((time.perf_counter() - t0_a) * 1000.0)
            self.assertEqual(summary["first_val"], 17)
            self.assertEqual(summary["mid_val"], 55)
            self.assertEqual(summary["last_val"], 99)
            self.assertEqual(summary["total_nbytes"], expected_single_nbytes)
            self.assertEqual(summary["meta_nbytes"], expected_meta_nbytes)
            self.assertEqual(summary["meta_sum"], expected_meta_sum)
            actual_a_nbytes = int(summary["total_nbytes"])
          lag_a = await monitor_a.stop()

          walls_a_arr = np.asarray(walls_a_ms, dtype=np.float64)
          wall_a_med_ms = float(np.median(walls_a_arr))
          actual_a_mib = actual_a_nbytes / (1024**2)
          gib_a = actual_a_nbytes / (1024**3)
          est_chunks_a = int(np.ceil(actual_a_nbytes / chunk_bytes)) + 1
          res_a = PerfResult(
              workload="W6A_MP_OrchToTrainer_Push",
              payload_mib=actual_a_mib,
              wall_ms=wall_a_med_ms,
              throughput_gib_s=gib_a / (wall_a_med_ms / 1000.0),
              chunk_count=est_chunks_a,
              loop_lag=lag_a,
              extra={
                  "iterations": float(num_iters),
                  "wall_p10_ms": float(np.percentile(walls_a_arr, 10)),
                  "wall_p90_ms": float(np.percentile(walls_a_arr, 90)),
                  "wall_min_ms": float(np.min(walls_a_arr)),
                  "wall_max_ms": float(np.max(walls_a_arr)),
              },
          )

          # --- W6B: 4x Rollout -> Orchestrator Concurrent Pull + Pings ---
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

          def _start_control_pinger() -> (
              tuple[list[float], Callable[[], None], asyncio.Task[None]]
          ):
            latencies_ms: list[float] = []
            stop_flag = False

            def _stop() -> None:
              nonlocal stop_flag
              stop_flag = True

            async def _pinger() -> None:
              seq = 0
              while not stop_flag:
                t_ping = time.perf_counter()
                res = await rollout_handles[seq % num_rollout_workers].asubmit(
                    "ping", seq
                )
                latencies_ms.append((time.perf_counter() - t_ping) * 1000.0)
                self.assertEqual(res, seq + 1)
                seq += 1
                await asyncio.sleep(0.010)

            return latencies_ms, _stop, asyncio.create_task(_pinger())

          async def _run_three_tier_steps() -> (
              tuple[list[float], list[float], list[float], int]
          ):
            assert trainer_handle is not None
            walls_ms: list[float] = []
            pulls_ms: list[float] = []
            pushes_ms: list[float] = []
            last_batch_nbytes = 0
            for step_idx in range(num_iters):
              t0 = time.perf_counter()
              rollout_batches = await asyncio.gather(
                  *[
                      _dispatch_and_poll(i, h)
                      for i, h in enumerate(rollout_handles)
                  ]
              )
              t1 = time.perf_counter()
              trainer_batch = {
                  "routed_experts": (
                      rollout_batches[0]["payload"]["routed_experts"]
                  ),
                  "trajectory_shards": [
                      b["payload"]["routed_experts"]
                      for b in rollout_batches[1:]
                  ],
                  "small_meta": rollout_batches[0]["payload"]["small_meta"],
                  "step": step_idx + 1,
              }
              train_summary = await trainer_handle.asubmit(
                  "echo_summary", trainer_batch
              )
              t2 = time.perf_counter()
              walls_ms.append((t2 - t0) * 1000.0)
              pulls_ms.append((t1 - t0) * 1000.0)
              pushes_ms.append((t2 - t1) * 1000.0)
              self.assertEqual(train_summary["first_val"], 17)
              self.assertEqual(train_summary["mid_val"], 55)
              self.assertEqual(train_summary["last_val"], 99)
              self.assertEqual(
                  train_summary["total_nbytes"], expected_total_conc_nbytes
              )
              self.assertEqual(
                  train_summary["meta_nbytes"], expected_meta_nbytes
              )
              self.assertEqual(train_summary["meta_sum"], expected_meta_sum)
              last_batch_nbytes = int(train_summary["total_nbytes"])
              del rollout_batches, trainer_batch
            return walls_ms, pulls_ms, pushes_ms, last_batch_nbytes

          warm_b = await asyncio.gather(
              *[_dispatch_and_poll(i, h) for i, h in enumerate(rollout_handles)]
          )
          for i, h in enumerate(rollout_handles):
            self.assertNotEmpty(h._seen_bulk_ports)
            arr_w = warm_b[i]["payload"]["routed_experts"]
            self.assertEqual(int(arr_w.nbytes), expected_worker_nbytes)
            self.assertEqual(
                _payload_page_checksum(arr_w), expected_worker_checksum
            )
          del warm_b

          monitor_b = EventLoopLagMonitor()
          monitor_b.start()
          ping_latencies_ms, stop_pings_b, ping_task = _start_control_pinger()
          walls_b_ms: list[float] = []
          actual_b_nbytes = 0
          last_results_b: list[dict[str, Any]] = []
          try:
            for iter_idx in range(num_iters):
              t0_b = time.perf_counter()
              results = await asyncio.gather(
                  *[
                      _dispatch_and_poll(i, h)
                      for i, h in enumerate(rollout_handles)
                  ]
              )
              walls_b_ms.append((time.perf_counter() - t0_b) * 1000.0)
              iter_b_nbytes = 0
              for i, r in enumerate(results):
                self.assertEqual(r["prompt_id"], i)
                arr_r = r["payload"]["routed_experts"]
                iter_b_nbytes += int(arr_r.nbytes)
                self.assertEqual(int(arr_r.nbytes), expected_worker_nbytes)
                self.assertEqual(int(arr_r[0]), 17)
                self.assertEqual(int(arr_r[arr_r.size // 2]), 55)
                self.assertEqual(int(arr_r[-1]), 99)
                meta_r = r["payload"]["small_meta"]
                self.assertEqual(
                    sum(int(m.nbytes) for m in meta_r), expected_meta_nbytes
                )
                self.assertEqual(
                    int(sum(float(m[0]) + float(m[-1]) for m in meta_r)),
                    expected_meta_sum,
                )
              self.assertEqual(iter_b_nbytes, expected_total_conc_nbytes)
              actual_b_nbytes = iter_b_nbytes
              if iter_idx == num_iters - 1:
                last_results_b = results
              else:
                del results
          finally:
            stop_pings_b()
            lag_b = await monitor_b.stop()
            await ping_task

          for r in last_results_b:
            self.assertEqual(
                _payload_page_checksum(r["payload"]["routed_experts"]),
                expected_worker_checksum,
            )
          del last_results_b

          walls_b_arr = np.asarray(walls_b_ms, dtype=np.float64)
          wall_b_med_ms = float(np.median(walls_b_arr))
          ping_arr = (
              np.asarray(ping_latencies_ms, dtype=np.float64)
              if ping_latencies_ms
              else np.zeros((1,), dtype=np.float64)
          )
          actual_b_mib = actual_b_nbytes / (1024**2)
          gib_b = actual_b_nbytes / (1024**3)
          est_chunks_b = num_rollout_workers * (
              int(np.ceil(expected_worker_nbytes / chunk_bytes)) + 1
          )
          res_b = PerfResult(
              workload="W6B_MP_4xRolloutToOrch_Pull",
              payload_mib=actual_b_mib,
              wall_ms=wall_b_med_ms,
              throughput_gib_s=gib_b / (wall_b_med_ms / 1000.0),
              chunk_count=est_chunks_b,
              loop_lag=lag_b,
              extra={
                  "iterations": float(num_iters),
                  "wall_p10_ms": float(np.percentile(walls_b_arr, 10)),
                  "wall_p90_ms": float(np.percentile(walls_b_arr, 90)),
                  "wall_min_ms": float(np.min(walls_b_arr)),
                  "wall_max_ms": float(np.max(walls_b_arr)),
                  "ping_p50_ms": float(np.percentile(ping_arr, 50)),
                  "ping_p95_ms": float(np.percentile(ping_arr, 95)),
                  "ping_max_ms": float(np.max(ping_arr)),
                  "ping_count": float(len(ping_latencies_ms)),
                  "idle_ping_p50_ms": float(np.percentile(idle_ping_arr, 50)),
                  "idle_ping_p95_ms": float(np.percentile(idle_ping_arr, 95)),
              },
          )

          # --- W6C: Full 3-Tier (4x Rollout -> Orchestrator -> Trainer) ---
          # Warm up the 4-buffer trainer_batch structure once before timing.
          warm_rollout = await asyncio.gather(
              *[_dispatch_and_poll(i, h) for i, h in enumerate(rollout_handles)]
          )
          warm_trainer_batch = {
              "routed_experts": warm_rollout[0]["payload"]["routed_experts"],
              "trajectory_shards": [
                  b["payload"]["routed_experts"] for b in warm_rollout[1:]
              ],
              "small_meta": warm_rollout[0]["payload"]["small_meta"],
              "step": 0,
          }
          warm_c_summary = await trainer_handle.asubmit(
              "echo_summary", warm_trainer_batch, verify_checksum=True
          )
          self.assertEqual(
              warm_c_summary["total_nbytes"], expected_total_conc_nbytes
          )
          self.assertEqual(warm_c_summary["meta_sum"], expected_meta_sum)
          self.assertEqual(
              warm_c_summary["checksum"], expected_total_conc_checksum
          )
          del warm_rollout, warm_trainer_batch

          monitor_c = EventLoopLagMonitor()
          monitor_c.start()
          try:
            walls_c_ms, pulls_c_ms, pushes_c_ms, actual_c_batch_nbytes = (
                await _run_three_tier_steps()
            )
          finally:
            lag_c = await monitor_c.stop()

          walls_c_arr = np.asarray(walls_c_ms, dtype=np.float64)
          wall_c_med_ms = float(np.median(walls_c_arr))
          pull_c_med_ms = float(np.median(np.asarray(pulls_c_ms)))
          push_c_med_ms = float(np.median(np.asarray(pushes_c_ms)))
          # Total network bytes moved across the 2 hops: 2.15 GiB (Rollout ->
          # Orch) + 2.15 GiB (Orch -> Trainer) = 4.30 GiB; end-to-end RL batch
          # size is 2.15 GiB.
          batch_mib = actual_c_batch_nbytes / (1024**2)
          batch_gib = actual_c_batch_nbytes / (1024**3)
          two_hop_mib = batch_mib * 2.0
          two_hop_gib = batch_gib * 2.0
          res_c = PerfResult(
              workload="W6C_MP_3Tier_Rollout_Orch_Trainer",
              payload_mib=two_hop_mib,
              wall_ms=wall_c_med_ms,
              throughput_gib_s=two_hop_gib / (wall_c_med_ms / 1000.0),
              chunk_count=est_chunks_b * 2,
              loop_lag=lag_c,
              extra={
                  "iterations": float(num_iters),
                  "wall_p10_ms": float(np.percentile(walls_c_arr, 10)),
                  "wall_p90_ms": float(np.percentile(walls_c_arr, 90)),
                  "wall_min_ms": float(np.min(walls_c_arr)),
                  "wall_max_ms": float(np.max(walls_c_arr)),
                  "rollout_to_orch_ms": pull_c_med_ms,
                  "orch_to_trainer_ms": push_c_med_ms,
                  "e2e_batch_gib_s": batch_gib / (wall_c_med_ms / 1000.0),
              },
          )

          # --- W6D: Full 3-Tier + Concurrent Orchestrator Python+NumPy Load ---
          compute_sim = OrchestratorComputeSimulator()
          idle_steps_per_s = compute_sim.calibrate_idle_baseline(
              duration_s=0.25 if _FULL_SCALE.value else 0.08
          )
          monitor_d = EventLoopLagMonitor()
          monitor_d.start()
          compute_sim.start()
          ping_latencies_d_ms, stop_pings_d, ping_task_d = (
              _start_control_pinger()
          )
          try:
            walls_d_ms, pulls_d_ms, pushes_d_ms, _ = (
                await _run_three_tier_steps()
            )
          finally:
            stop_pings_d()
            compute_metrics = await compute_sim.stop(idle_steps_per_s)
            lag_d = await monitor_d.stop()
            await ping_task_d
          self.assertGreater(compute_metrics.steps_completed, 0)

          walls_d_arr = np.asarray(walls_d_ms, dtype=np.float64)
          wall_d_med_ms = float(np.median(walls_d_arr))
          pull_d_med_ms = float(np.median(np.asarray(pulls_d_ms)))
          push_d_med_ms = float(np.median(np.asarray(pushes_d_ms)))
          ping_d_arr = (
              np.asarray(ping_latencies_d_ms, dtype=np.float64)
              if ping_latencies_d_ms
              else np.zeros((1,), dtype=np.float64)
          )
          goodput_d = two_hop_gib / (wall_d_med_ms / 1000.0)
          net_retention_pct = (
              (goodput_d / res_c.throughput_gib_s) * 100.0
              if res_c.throughput_gib_s > 0.0
              else 0.0
          )
          res_d = PerfResult(
              workload="W6D_MP_3Tier_With_Orch_Python_Load",
              payload_mib=two_hop_mib,
              wall_ms=wall_d_med_ms,
              throughput_gib_s=goodput_d,
              chunk_count=est_chunks_b * 2,
              loop_lag=lag_d,
              extra={
                  "iterations": float(num_iters),
                  "wall_p10_ms": float(np.percentile(walls_d_arr, 10)),
                  "wall_p90_ms": float(np.percentile(walls_d_arr, 90)),
                  "wall_min_ms": float(np.min(walls_d_arr)),
                  "wall_max_ms": float(np.max(walls_d_arr)),
                  "rollout_to_orch_ms": pull_d_med_ms,
                  "orch_to_trainer_ms": push_d_med_ms,
                  "e2e_batch_gib_s": batch_gib / (wall_d_med_ms / 1000.0),
                  "ping_p50_ms": float(np.percentile(ping_d_arr, 50)),
                  "ping_p95_ms": float(np.percentile(ping_d_arr, 95)),
                  "ping_max_ms": float(np.max(ping_d_arr)),
                  "ping_count": float(len(ping_latencies_d_ms)),
                  "idle_ping_p50_ms": float(np.percentile(idle_ping_arr, 50)),
                  "idle_ping_p95_ms": float(np.percentile(idle_ping_arr, 95)),
                  "orch_compute_steps": float(compute_metrics.steps_completed),
                  "orch_compute_steps_per_s": compute_metrics.steps_per_s,
                  "orch_idle_steps_per_s": (
                      compute_metrics.idle_baseline_steps_per_s
                  ),
                  "orch_compute_retention_pct": (
                      compute_metrics.compute_retention_pct
                  ),
                  "net_goodput_retention_pct": net_retention_pct,
              },
          )
          return [res_a, res_b, res_c, res_d]
        finally:
          for hdl in rollout_handles:
            await hdl.close()
          if trainer_handle is not None:
            await trainer_handle.close()

      results = asyncio.run(_main())
      self.__class__._results.extend(results)
      for r in results:
        self.assertGreater(r.throughput_gib_s, 0.0)
    finally:
      for proc in procs:
        if proc.stdin is not None:
          with contextlib.suppress(Exception):
            proc.stdin.close()
      for proc in procs:
        try:
          proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
          proc.kill()
        if proc.stdout is not None:
          with contextlib.suppress(Exception):
            proc.stdout.close()
        if proc.stderr is not None:
          with contextlib.suppress(Exception):
            proc.stderr.close()


def _run_isolated_worker_server(port: int, payload_bytes: int) -> None:
  """Runs an isolated gRPC worker server until stdin is closed."""
  payload = (
      _build_single_buffer_payload(payload_bytes)
      if payload_bytes > 0
      else {}
  )
  worker = _EchoPayloadWorker(response_payload=payload)

  async def _serve() -> None:
    server = remote_execution.GrpcRemoteExecutionServer(
        worker, stream_chunk_bytes=network._STREAM_CHUNK_BYTES
    )
    await server.start_serving_async(port)
    # Warm up worker serde buffers once before signaling READY.
    resp = remote_execution.ExecutionResponse(result={"payload": payload})
    async for _ in resp.serialize_async_chunks():
      pass
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _wait_stdin_eof() -> None:
      try:
        sys.stdin.read()
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      loop.call_soon_threadsafe(stop_event.set)

    threading.Thread(target=_wait_stdin_eof, daemon=True).start()
    print("READY", flush=True)
    await stop_event.wait()
    await server.stop_serving(grace=0.0)

  asyncio.run(_serve())


if __name__ == "__main__":
  absltest.main()

