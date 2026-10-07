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
r"""Isolated performance benchmark for `batch_assembly` and `packing`.

Measures throughput (trajectories/s, tokens/s), wall-clock latency, and
concurrent `asyncio` event-loop scheduling lag (GIL contention when packing in
`asyncio.to_thread`) across both the Pure-Python/NumPy backend and the C++
`nanobind` (`_packing_ext`, `nogil`) backend on the exact same Forge machine:
  - W1: `SequencePackedBatchAssembler` on standard dense RL rollouts.
  - W2: `SequencePackedBatchAssembler` with 64-token segment alignment.
  - W3: `SequencePackedBatchAssembler` with MoE `routed_experts` replay
        (`[T, num_layers, top_k]`).
  - W4: `PaddedBatchAssembler` rectangular 2D batch padding.

Run via pytest (always compile the C++ extension in release mode):
  pytest tunix/experimental/orchestrator/batch_assembly_perf_test.py
"""

import asyncio
import contextlib
import dataclasses
import json
import time
from unittest import mock

from absl import flags
from absl import logging
from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import batch_assembly
from tunix.rl import packing

_FULL_SCALE = flags.DEFINE_bool(
    "batch_assembly_full_scale",
    False,
    "If True, runs larger trajectory counts and MoE layer dimensions.",
)
_NUM_ITERATIONS = flags.DEFINE_integer(
    "batch_assembly_num_iterations",
    5,
    "Number of timed iterations per benchmark workload.",
)
_REPORT_PATH = flags.DEFINE_string(
    "batch_assembly_report_path",
    "",
    "Optional file path to write the JSON benchmark report.",
)


def _full_scale() -> bool:
  return _FULL_SCALE.value if flags.FLAGS.is_parsed() else False


def _num_iterations() -> int:
  return _NUM_ITERATIONS.value if flags.FLAGS.is_parsed() else 2


def _report_path() -> str:
  return _REPORT_PATH.value if flags.FLAGS.is_parsed() else ""


# ==============================================================================
# Event-Loop Lag Monitor (GIL Contention Telemetry)
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class LoopLagMetrics:
  """Summary of asyncio event-loop scheduling lag during background packing."""

  p50_ms: float
  p95_ms: float
  p99_ms: float
  max_ms: float
  blocked_over_5ms_total_ms: float


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
    blocked = float(np.sum(arr[arr > 5.0]))
    return LoopLagMetrics(
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        max_ms=float(np.max(arr)),
        blocked_over_5ms_total_ms=blocked,
    )


@dataclasses.dataclass(frozen=True)
class PackingPerfResult:
  """Metrics for a single batch assembly benchmark workload."""

  workload: str
  backend: str
  num_trajectories: int
  total_tokens: int
  num_batches: int
  mean_wall_ms: float
  p50_wall_ms: float
  min_wall_ms: float
  trajectories_per_sec: float
  million_tokens_per_sec: float
  loop_lag: LoopLagMetrics


# ==============================================================================
# Synthetic Rollout Payload Builders
# ==============================================================================


def _build_rollout_payloads(
    num_items: int,
    *,
    min_prompt_len: int = 64,
    max_prompt_len: int = 512,
    min_completion_len: int = 128,
    max_completion_len: int = 1536,
    with_moe_routing: bool = False,
    num_layers: int = 16,
    top_k: int = 4,
    seed: int = 42,
) -> tuple[list[datatypes.RLTrainerPayload], int]:
  """Builds realistic variable-length unbatched RLTrainerPayload items."""
  rng = np.random.default_rng(seed)
  payloads: list[datatypes.RLTrainerPayload] = []
  total_tokens = 0
  for i in range(num_items):
    p_len = int(rng.integers(min_prompt_len, max_prompt_len + 1))
    c_len = int(rng.integers(min_completion_len, max_completion_len + 1))
    seq_len = p_len + c_len
    total_tokens += seq_len

    prompt_ids = np.arange(1, p_len + 1, dtype=np.int32)
    completion_ids = np.arange(100, 100 + c_len, dtype=np.int32)
    prompt_mask = np.ones(p_len, dtype=np.float32)
    completion_mask = np.ones(c_len, dtype=np.float32)
    advantages = np.full(c_len, float((i % 7) - 3) * 0.25, dtype=np.float32)
    old_logps = np.full(c_len, -0.5, dtype=np.float32)
    ref_logps = np.full(c_len, -0.6, dtype=np.float32)

    routed_experts: np.ndarray | None = None
    if with_moe_routing:
      # Sequence-aligned [p + c - 1, num_layers, top_k] int16 routing tensor.
      routed_experts = np.broadcast_to(
          np.arange(top_k, dtype=np.int16),
          (seq_len - 1, num_layers, top_k),
      ).copy()

    payloads.append(
        datatypes.RLTrainerPayload(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            advantages=advantages,
            old_per_token_logps=old_logps,
            ref_per_token_logps=ref_logps,
            routed_experts=routed_experts,
            metadata={"trajectory_id": f"traj_{i}"},
        )
    )
  return payloads, total_tokens


def _format_results_table(results: list[PackingPerfResult]) -> str:
  """Formats a summary table of batch assembly performance metrics."""
  lines = [
      "",
      "=" * 134,
      (
          "BATCH ASSEMBLY & SEQUENCE PACKING PERFORMANCE BENCHMARK"
          " (Python vs C++ nogil)"
      ),
      "=" * 134,
      (
          f"{'Workload':<32} | {'Backend':<10} | {'Trajs':>5} | {'Tokens':>8} |"
          f" {'Batches':>7} | {'Mean ms':>8} | {'Min ms':>8} | {'Traj/s':>8} |"
          f" {'MTok/s':>7} | {'Lag p99':>8} | {'Lag Max':>8} |"
          f" {'Blocked>5ms':>11}"
      ),
      "-" * 134,
  ]
  for res in results:
    lines.append(
        f"{res.workload:<32} | {res.backend:<10} | {res.num_trajectories:>5d} |"
        f" {res.total_tokens:>8d} | {res.num_batches:>7d} |"
        f" {res.mean_wall_ms:>8.2f} | {res.min_wall_ms:>8.2f} |"
        f" {res.trajectories_per_sec:>8.0f} |"
        f" {res.million_tokens_per_sec:>7.2f} |"
        f" {res.loop_lag.p99_ms:>6.2f}ms |"
        f" {res.loop_lag.max_ms:>6.2f}ms |"
        f" {res.loop_lag.blocked_over_5ms_total_ms:>9.1f}ms"
    )
  lines.append("=" * 134)
  return "\n".join(lines)


# ==============================================================================
# Benchmark Suite
# ==============================================================================


class BatchAssemblyPerfTest(absltest.TestCase):
  """Performance benchmark for SequencePackedBatchAssembler & PaddedBatchAssembler."""

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls._results: list[PackingPerfResult] = []

  @classmethod
  def tearDownClass(cls) -> None:
    if cls._results:
      table = _format_results_table(cls._results)
      logging.info("%s", table)
      print(table, flush=True)
      report_path = _report_path()
      if report_path:
        records = [dataclasses.asdict(r) for r in cls._results]
        with open(report_path, "w", encoding="utf-8") as f:
          json.dump(records, f, indent=2)
    super().tearDownClass()

  def _run_assembler_workload(
      self,
      workload: str,
      backend: str,
      assembler: (
          batch_assembly.SequencePackedBatchAssembler
          | batch_assembly.PaddedBatchAssembler
      ),
      payloads: list[datatypes.RLTrainerPayload],
      total_tokens: int,
      *,
      use_cpp_ext: bool,
  ) -> PackingPerfResult:
    """Runs warmup + timed iterations inside asyncio.to_thread with lag monitoring."""
    num_iters = max(1, _num_iterations())
    ext_val = packing._packing_ext if use_cpp_ext else None  # pylint: disable=protected-access

    def _one_step() -> int:
      batches = list(assembler.feed(payloads))
      batches.extend(assembler.flush())
      return len(batches)

    with mock.patch.object(packing, "_packing_ext", ext_val):
      # Warmup pass.
      num_batches = _one_step()

      async def _measure() -> tuple[list[float], LoopLagMetrics]:
        monitor = EventLoopLagMonitor(interval_s=0.001)
        monitor.start()
        iter_ms: list[float] = []
        for _ in range(num_iters):
          t0 = time.perf_counter()
          count = await asyncio.to_thread(_one_step)
          iter_ms.append((time.perf_counter() - t0) * 1000.0)
          self.assertEqual(count, num_batches)
        lag = await monitor.stop()
        return iter_ms, lag

      iter_ms, lag = asyncio.run(_measure())

    arr = np.asarray(iter_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    p50_ms = float(np.percentile(arr, 50))
    min_ms = float(np.min(arr))
    mean_s = mean_ms / 1000.0
    res = PackingPerfResult(
        workload=workload,
        backend=backend,
        num_trajectories=len(payloads),
        total_tokens=total_tokens,
        num_batches=num_batches,
        mean_wall_ms=mean_ms,
        p50_wall_ms=p50_ms,
        min_wall_ms=min_ms,
        trajectories_per_sec=len(payloads) / mean_s,
        million_tokens_per_sec=(total_tokens / 1e6) / mean_s,
        loop_lag=lag,
    )
    self.__class__._results.append(res)
    return res

  def _run_pack_core_backend(
      self,
      pack_items: list[packing.PackItem],
      total_tokens: int,
      *,
      backend: str,
      use_cpp: bool,
  ) -> PackingPerfResult:
    """Runs warmup + timed iterations for `packing.pack_core`."""
    num_iters = max(1, _num_iterations())
    ext_val = packing._packing_ext if use_cpp else None  # pylint: disable=protected-access

    def _one_pack() -> int:
      chunks = packing.pack_core(
          pack_items,
          budget=8192,
          pack_size=8,
          pad_id=0,
          segment_alignment_boundary=64,
      )
      return len(chunks)

    with mock.patch.object(packing, "_packing_ext", ext_val):
      num_batches = _one_pack()

      async def _measure() -> tuple[list[float], LoopLagMetrics]:
        monitor = EventLoopLagMonitor(interval_s=0.001)
        monitor.start()
        iter_ms: list[float] = []
        for _ in range(num_iters):
          t0 = time.perf_counter()
          count = await asyncio.to_thread(_one_pack)
          iter_ms.append((time.perf_counter() - t0) * 1000.0)
          self.assertEqual(count, num_batches)
        lag = await monitor.stop()
        return iter_ms, lag

      iter_ms, lag = asyncio.run(_measure())

    arr = np.asarray(iter_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    p50_ms = float(np.percentile(arr, 50))
    min_ms = float(np.min(arr))
    mean_s = mean_ms / 1000.0
    res = PackingPerfResult(
        workload="W0_PackCore_Isolated_Align64",
        backend=backend,
        num_trajectories=len(pack_items),
        total_tokens=total_tokens,
        num_batches=num_batches,
        mean_wall_ms=mean_ms,
        p50_wall_ms=p50_ms,
        min_wall_ms=min_ms,
        trajectories_per_sec=len(pack_items) / mean_s,
        million_tokens_per_sec=(total_tokens / 1e6) / mean_s,
        loop_lag=lag,
    )
    self.__class__._results.append(res)
    return res

  def test_w0_pack_core_isolated(self) -> None:
    """Benchmarks pure `packing.pack_core` (FFD + buffer population) in isolation."""
    num_items = 1024 if _full_scale() else 512
    payloads, total_tokens = _build_rollout_payloads(
        num_items, with_moe_routing=False
    )
    pack_items = [batch_assembly.to_pack_item(p) for p in payloads]

    for backend, use_cpp in (("Python", False), ("C++ nogil", True)):
      res = self._run_pack_core_backend(
          pack_items,
          total_tokens,
          backend=backend,
          use_cpp=use_cpp,
      )
      self.assertGreater(res.million_tokens_per_sec, 0.0)

  def test_w1_sequence_packed_dense_rollouts(self) -> None:
    """Benchmarks FFD sequence packing on 512 dense variable-length rollouts."""
    num_items = 1024 if _full_scale() else 512
    payloads, total_tokens = _build_rollout_payloads(
        num_items, with_moe_routing=False
    )
    for backend, use_cpp in (("Python", False), ("C++ nogil", True)):
      assembler = batch_assembly.SequencePackedBatchAssembler(
          batch_size=8,
          num_generations=8,
          mini_batch_size=num_items // 8,
          max_packed_len=8192,
          pad_id=0,
          segment_alignment_boundary=1,
      )
      res = self._run_assembler_workload(
          "W1_SeqPacked_Dense_Unaligned",
          backend,
          assembler,
          payloads,
          total_tokens,
          use_cpp_ext=use_cpp,
      )
      self.assertGreater(res.num_batches, 0)
      self.assertGreater(res.million_tokens_per_sec, 0.0)

  def test_w2_sequence_packed_aligned_64(self) -> None:
    """Benchmarks FFD sequence packing with 64-token hardware alignment."""
    num_items = 1024 if _full_scale() else 512
    payloads, total_tokens = _build_rollout_payloads(
        num_items, with_moe_routing=False
    )
    for backend, use_cpp in (("Python", False), ("C++ nogil", True)):
      assembler = batch_assembly.SequencePackedBatchAssembler(
          batch_size=8,
          num_generations=8,
          mini_batch_size=num_items // 8,
          max_packed_len=8192,
          pad_id=0,
          segment_alignment_boundary=64,
      )
      res = self._run_assembler_workload(
          "W2_SeqPacked_Dense_Align64",
          backend,
          assembler,
          payloads,
          total_tokens,
          use_cpp_ext=use_cpp,
      )
      self.assertGreater(res.num_batches, 0)
      self.assertGreater(res.million_tokens_per_sec, 0.0)

  def test_w3_sequence_packed_moe_router_replay(self) -> None:
    """Benchmarks sequence packing with 3D MoE routed_experts replay tensors."""
    num_items = 256 if _full_scale() else 128
    num_layers = 32 if _full_scale() else 16
    payloads, total_tokens = _build_rollout_payloads(
        num_items,
        with_moe_routing=True,
        num_layers=num_layers,
        top_k=4,
    )
    for backend, use_cpp in (("Python", False), ("C++ nogil", True)):
      assembler = batch_assembly.SequencePackedBatchAssembler(
          batch_size=8,
          num_generations=8,
          mini_batch_size=num_items // 8,
          max_packed_len=8192,
          pad_id=0,
          segment_alignment_boundary=64,
      )
      res = self._run_assembler_workload(
          f"W3_SeqPacked_MoE_L{num_layers}_K4",
          backend,
          assembler,
          payloads,
          total_tokens,
          use_cpp_ext=use_cpp,
      )
      self.assertGreater(res.num_batches, 0)
      self.assertGreater(res.million_tokens_per_sec, 0.0)

  def test_w4_padded_batch_assembler(self) -> None:
    """Benchmarks PaddedBatchAssembler rectangular 2D padding."""
    num_items = 1024 if _full_scale() else 512
    payloads, total_tokens = _build_rollout_payloads(
        num_items, with_moe_routing=False
    )
    for backend, use_cpp in (("Python", False), ("C++ nogil", True)):
      assembler = batch_assembly.PaddedBatchAssembler(
          batch_size=16,
          max_prompt_length=512,
          max_response_length=1536,
          pad_id=0,
          num_generations=8,
          mini_batch_size=num_items // 8,
      )
      res = self._run_assembler_workload(
          "W4_PaddedBatchAssembler_2D",
          backend,
          assembler,
          payloads,
          total_tokens,
          use_cpp_ext=use_cpp,
      )
      self.assertGreater(res.num_batches, 0)
      self.assertGreater(res.million_tokens_per_sec, 0.0)


if __name__ == "__main__":
  absltest.main()
