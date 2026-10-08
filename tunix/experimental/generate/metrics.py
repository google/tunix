# Copyright 2026 The Tunix Authors.
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

"""vLLM-style performance metrics and periodic status logging for LLMEngine."""

from collections.abc import Callable, Hashable, Sequence
import dataclasses
import threading
import time

from absl import logging
import jax
import numpy as np
from tunix.experimental.generate import request as request_lib

PerfMetricValueT = tuple[
    jax.typing.ArrayLike | str,
    Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike] | None,
]

# The most batch starts kept for batches that have not completed yet. The
# oldest is dropped beyond this, so a batch that never completes, e.g. because
# its sampling call failed, cannot hold on to its start forever.
_MAX_PENDING_BATCH_STARTS = 1024


@dataclasses.dataclass(frozen=True, kw_only=True)
class EngineMetricsSnapshot:
  """Immutable snapshot of engine, scheduler, request, and batch metrics."""

  # Step and token counts. `prompt_tokens` counts the tokens of prefill steps:
  # not prefix cache hits, nor a prompt's last token when it is the only one
  # left to compute, which runs as a decode. A preempted prompt counts again.
  num_active_steps: int
  num_prefill_steps: int
  prompt_tokens: int
  generation_tokens: int

  # 1. Submitted rollout batch completion times.
  completed_batches: int
  last_batch_completion_time_s: float
  avg_batch_completion_time_s: float

  # 2. Rollout batch work over wall-clock time, from each batch's start to its
  # completion. The throughputs divide the prefill and generated tokens by that
  # time, both summed over the batches. Only batches whose start was recorded
  # count. A batch counts all the engine's work while it is in flight,
  # including the work of any other batches or eval rollouts in flight at the
  # same time.
  num_timed_batches: int
  avg_batch_requests: float
  avg_prefill_throughput_tok_per_s: float
  avg_generation_throughput_tok_per_s: float

  # 3. Engine step and scheduler durations.
  avg_schedule_duration_ms: float
  avg_engine_step_duration_ms: float

  # 4. Request-specific lifecycle latencies (over completed requests).
  completed_requests: int
  avg_request_queue_time_s: float
  avg_request_ttft_s: float
  avg_request_prefill_time_s: float
  avg_request_decode_time_s: float
  avg_request_e2e_latency_s: float
  max_request_e2e_latency_s: float
  # Decode time per output token after the first, over the requests that
  # generated more than one.
  avg_request_time_per_output_token_ms: float

  # 5. Live vLLM-style system gauges. The request counts and KV cache usage are
  # read when the snapshot is taken.
  num_running_reqs: int
  num_waiting_reqs: int
  kv_cache_usage_pct: float
  prefix_cache_hit_rate_pct: float
  num_preemptions: int

  # 6. Gauges sampled at the end of every engine step, over the window.
  avg_num_running_reqs: float
  avg_kv_cache_usage_pct: float
  max_kv_cache_usage_pct: float

  def to_perf_metrics(self) -> dict[str, PerfMetricValueT]:
    """Converts the snapshot into a metric dict for `RLCluster.buffer_metrics`.

    Totals and engine step averages are always included. Averages over requests
    or batches, including the token throughputs, are only included when the
    snapshot has some, so that a window without any does not log a misleading
    0. The live KV cache usage is left out: a training step's metrics are
    flushed once its rollouts finish, when the KV cache is about empty again.
    Its average and peak over the engine steps stand in for it.

    Returns:
      The metrics, by name, with the op that aggregates repeated values.
    """
    metrics: dict[str, PerfMetricValueT] = {
        "rollout/avg_schedule_duration_ms": (
            self.avg_schedule_duration_ms,
            np.mean,
        ),
        "rollout/avg_engine_step_duration_ms": (
            self.avg_engine_step_duration_ms,
            np.mean,
        ),
        "rollout/prefix_cache_hit_rate_pct": (
            self.prefix_cache_hit_rate_pct,
            np.mean,
        ),
        "rollout/num_preemptions": (
            float(self.num_preemptions),
            np.sum,
        ),
        "rollout/completed_requests": (
            float(self.completed_requests),
            np.sum,
        ),
        "rollout/completed_batches": (
            float(self.completed_batches),
            np.sum,
        ),
        "rollout/avg_num_running_reqs": (
            self.avg_num_running_reqs,
            np.mean,
        ),
        "rollout/avg_kv_cache_usage_pct": (
            self.avg_kv_cache_usage_pct,
            np.mean,
        ),
        "rollout/max_kv_cache_usage_pct": (
            self.max_kv_cache_usage_pct,
            np.max,
        ),
    }
    if self.completed_requests > 0:
      metrics.update({
          "rollout/avg_request_queue_time_s": (
              self.avg_request_queue_time_s,
              np.mean,
          ),
          "rollout/avg_request_ttft_s": (
              self.avg_request_ttft_s,
              np.mean,
          ),
          "rollout/avg_request_prefill_time_s": (
              self.avg_request_prefill_time_s,
              np.mean,
          ),
          "rollout/avg_request_decode_time_s": (
              self.avg_request_decode_time_s,
              np.mean,
          ),
          "rollout/avg_request_e2e_latency_s": (
              self.avg_request_e2e_latency_s,
              np.mean,
          ),
          "rollout/max_request_e2e_latency_s": (
              self.max_request_e2e_latency_s,
              np.max,
          ),
          "rollout/avg_request_time_per_output_token_ms": (
              self.avg_request_time_per_output_token_ms,
              np.mean,
          ),
      })
    if self.completed_batches > 0:
      metrics.update({
          "rollout/last_batch_completion_time_s": (
              self.last_batch_completion_time_s,
              np.mean,
          ),
          "rollout/avg_batch_completion_time_s": (
              self.avg_batch_completion_time_s,
              np.mean,
          ),
      })
    if self.num_timed_batches > 0:
      metrics.update({
          "rollout/avg_generation_throughput_tok_per_s": (
              self.avg_generation_throughput_tok_per_s,
              np.mean,
          ),
          "rollout/avg_prefill_throughput_tok_per_s": (
              self.avg_prefill_throughput_tok_per_s,
              np.mean,
          ),
          "rollout/avg_batch_requests": (
              self.avg_batch_requests,
              np.mean,
          ),
      })
    return metrics


@dataclasses.dataclass
class _MetricsWindow:
  """Mutable accumulator for a single time window or cumulative run."""

  num_active_steps: int = 0
  num_prefill_steps: int = 0
  active_step_time_s: float = 0.0
  schedule_time_s: float = 0.0
  num_schedules: int = 0
  # When the first active step started, by `time.perf_counter()`.
  first_step_start_s: float | None = None

  prompt_tokens: int = 0
  generation_tokens: int = 0

  # Sums of the gauges sampled at the end of every active step, and the peak
  # KV cache usage.
  running_reqs_sum: int = 0
  kv_cache_usage_sum: float = 0.0
  max_kv_cache_usage: float = 0.0

  completed_requests: int = 0
  request_queue_time_s: float = 0.0
  request_ttft_s: float = 0.0
  request_prefill_time_s: float = 0.0
  request_decode_time_s: float = 0.0
  request_e2e_latency_s: float = 0.0
  max_request_e2e_latency_s: float = 0.0
  num_tpot_requests: int = 0
  request_tpot_s: float = 0.0

  completed_batches: int = 0
  total_batch_completion_time_s: float = 0.0
  last_batch_completion_time_s: float = 0.0

  # Batches whose start was recorded, and the engine's work between each one's
  # start and completion.
  num_timed_batches: int = 0
  timed_batch_time_s: float = 0.0
  batch_requests: int = 0
  batch_prompt_tokens: int = 0
  batch_generation_tokens: int = 0

  prefix_cache_queries: int = 0
  prefix_cache_hits: int = 0
  num_preemptions: int = 0

  def reset(self) -> None:
    for field in dataclasses.fields(self):
      setattr(self, field.name, field.default)

  def add_finished_request(self, req: request_lib.RequestState) -> None:
    """Adds a finished request's lifecycle latencies."""
    num_generated = len(req.token_ids) - req.prompt_length
    self.completed_requests += 1
    self.request_queue_time_s += req.queue_time_s
    if req.finished_time is None:
      return

    e2e = max(0.0, req.finished_time - req.arrival_time)
    self.request_e2e_latency_s += e2e
    self.max_request_e2e_latency_s = max(self.max_request_e2e_latency_s, e2e)
    if req.first_token_time is None:
      return

    decode_t = max(0.0, req.finished_time - req.first_token_time)
    self.request_ttft_s += max(0.0, req.first_token_time - req.arrival_time)
    self.request_decode_time_s += decode_t
    if num_generated > 1:
      self.num_tpot_requests += 1
      self.request_tpot_s += decode_t / (num_generated - 1)
    if req.first_scheduled_time is not None:
      self.request_prefill_time_s += max(
          0.0, req.first_token_time - req.first_scheduled_time
      )


@dataclasses.dataclass(frozen=True)
class _EngineWork:
  """Counts of the requests and tokens the engine completed."""

  completed_requests: int
  prompt_tokens: int
  generation_tokens: int

  def __sub__(self, other: "_EngineWork") -> "_EngineWork":
    return _EngineWork(
        completed_requests=self.completed_requests - other.completed_requests,
        prompt_tokens=self.prompt_tokens - other.prompt_tokens,
        generation_tokens=self.generation_tokens - other.generation_tokens,
    )


class MetricsCollector:
  """Collects engine/request/batch metrics and emits vLLM-style logs."""

  def __init__(
      self,
      *,
      log_stats_interval_s: float = 10.0,
      engine_index: int = 0,
  ):
    if log_stats_interval_s < 0.0:
      raise ValueError(
          f"log_stats_interval_s must be >= 0, got {log_stats_interval_s}."
      )
    self._log_stats_interval_s = log_stats_interval_s
    self._engine_index = engine_index
    self._lock = threading.Lock()

    self._cumulative = _MetricsWindow()
    self._log_window = _MetricsWindow()
    self._step_window = _MetricsWindow()
    # The engine's cumulative work at the start of each batch not yet
    # completed, by batch id, oldest first.
    self._batch_starts: dict[Hashable, _EngineWork] = {}

    self._last_log_time: float = time.perf_counter()

  @property
  def log_stats_interval_s(self) -> float:
    return self._log_stats_interval_s

  @property
  def _windows(self) -> tuple[_MetricsWindow, ...]:
    return (self._cumulative, self._log_window, self._step_window)

  def _cumulative_work_locked(self) -> _EngineWork:
    return _EngineWork(
        completed_requests=self._cumulative.completed_requests,
        prompt_tokens=self._cumulative.prompt_tokens,
        generation_tokens=self._cumulative.generation_tokens,
    )

  def record_schedule(self, duration_s: float, num_preemptions: int = 0) -> None:
    """Records the host CPU duration of a non-empty `schedule_step` call."""
    with self._lock:
      for window in self._windows:
        window.schedule_time_s += duration_s
        window.num_schedules += 1
        window.num_preemptions += num_preemptions

  def record_step(
      self,
      *,
      step_duration_s: float,
      num_prompt_tokens: int,
      num_generation_tokens: int,
      finished_requests: Sequence[request_lib.RequestState],
      prefix_cache_queries: int = 0,
      prefix_cache_hits: int = 0,
      num_running_reqs: int = 0,
      kv_cache_usage_fraction: float = 0.0,
  ) -> None:
    """Records an active `LLMEngine.step()` execution and finished requests.

    Args:
      step_duration_s: How long the step took.
      num_prompt_tokens: The prompt tokens the step prefilled.
      num_generation_tokens: The tokens the step generated.
      finished_requests: The requests that finished in the step.
      prefix_cache_queries: The pages the step looked up in the prefix cache.
      prefix_cache_hits: The pages the step found in the prefix cache.
      num_running_reqs: The requests in the step's batch.
      kv_cache_usage_fraction: The fraction of the KV cache in use after the
        step.
    """
    step_start_s = time.perf_counter() - step_duration_s
    with self._lock:
      for window in self._windows:
        if window.first_step_start_s is None:
          window.first_step_start_s = step_start_s
        window.num_active_steps += 1
        window.active_step_time_s += step_duration_s
        window.prompt_tokens += num_prompt_tokens
        window.generation_tokens += num_generation_tokens
        window.prefix_cache_queries += prefix_cache_queries
        window.prefix_cache_hits += prefix_cache_hits
        window.running_reqs_sum += num_running_reqs
        window.kv_cache_usage_sum += kv_cache_usage_fraction
        window.max_kv_cache_usage = max(
            window.max_kv_cache_usage, kv_cache_usage_fraction
        )
        if num_prompt_tokens > 0:
          window.num_prefill_steps += 1

        for req in finished_requests:
          window.add_finished_request(req)

  def record_batch_start(self, batch_id: Hashable | None = None) -> None:
    """Marks the start of a rollout batch.

    `record_batch_completion` with the same `batch_id` then also records the
    requests and tokens the engine completed in between, so that the batch's
    throughput is measured over its wall-clock time.

    Args:
      batch_id: Tells apart batches that may be in flight at the same time.
    """
    with self._lock:
      # Re-inserted, so that the starts stay ordered oldest first.
      self._batch_starts.pop(batch_id, None)
      if len(self._batch_starts) >= _MAX_PENDING_BATCH_STARTS:
        del self._batch_starts[next(iter(self._batch_starts))]
      self._batch_starts[batch_id] = self._cumulative_work_locked()

  def record_batch_completion(
      self, duration_s: float, batch_id: Hashable | None = None
  ) -> None:
    """Records the completion time of a submitted rollout batch of prompts.

    Args:
      duration_s: How long the batch took, from its start to its completion.
      batch_id: The id the batch's start was recorded with. Without a recorded
        start, only the batch's duration is recorded.
    """
    with self._lock:
      start = self._batch_starts.pop(batch_id, None)
      work = None if start is None else self._cumulative_work_locked() - start
      for window in self._windows:
        window.completed_batches += 1
        window.last_batch_completion_time_s = duration_s
        window.total_batch_completion_time_s += duration_s
        if work is not None:
          window.num_timed_batches += 1
          window.timed_batch_time_s += duration_s
          window.batch_requests += work.completed_requests
          window.batch_prompt_tokens += work.prompt_tokens
          window.batch_generation_tokens += work.generation_tokens

    batch_str = "" if batch_id is None else f" {batch_id}"
    if work is None:
      logging.info(
          "Engine %03d: Rollout batch%s finished in %.2f s.",
          self._engine_index,
          batch_str,
          duration_s,
      )
      return
    logging.info(
        "Engine %03d: Rollout batch%s finished in %.2f s: %d requests, "
        "Prefill throughput: %.1f tokens/s, "
        "Generation throughput: %.1f tokens/s.",
        self._engine_index,
        batch_str,
        duration_s,
        work.completed_requests,
        work.prompt_tokens / duration_s if duration_s > 0.0 else 0.0,
        work.generation_tokens / duration_s if duration_s > 0.0 else 0.0,
    )

  def _build_snapshot_locked(
      self,
      window: _MetricsWindow,
      *,
      num_running_reqs: int,
      num_waiting_reqs: int,
      kv_cache_usage_fraction: float,
  ) -> EngineMetricsSnapshot:
    avg_sched_ms = (
        (window.schedule_time_s / window.num_schedules) * 1000.0
        if window.num_schedules > 0
        else 0.0
    )
    n_steps = window.num_active_steps
    avg_step_ms = (
        (window.active_step_time_s / n_steps) * 1000.0 if n_steps > 0 else 0.0
    )
    avg_running = window.running_reqs_sum / n_steps if n_steps > 0 else 0.0
    avg_kv_pct = (
        (window.kv_cache_usage_sum / n_steps) * 100.0 if n_steps > 0 else 0.0
    )

    n_req = window.completed_requests
    avg_queue_s = window.request_queue_time_s / n_req if n_req > 0 else 0.0
    avg_ttft_s = window.request_ttft_s / n_req if n_req > 0 else 0.0
    avg_prefill_s = window.request_prefill_time_s / n_req if n_req > 0 else 0.0
    avg_decode_s = window.request_decode_time_s / n_req if n_req > 0 else 0.0
    avg_e2e_s = window.request_e2e_latency_s / n_req if n_req > 0 else 0.0
    avg_tpot_ms = (
        (window.request_tpot_s / window.num_tpot_requests) * 1000.0
        if window.num_tpot_requests > 0
        else 0.0
    )

    prefix_queries = (
        window.prefix_cache_queries
        if window.prefix_cache_queries > 0
        else self._cumulative.prefix_cache_queries
    )
    prefix_hits = (
        window.prefix_cache_hits
        if window.prefix_cache_queries > 0
        else self._cumulative.prefix_cache_hits
    )
    prefix_hit_pct = (
        (prefix_hits / prefix_queries) * 100.0 if prefix_queries > 0 else 0.0
    )
    avg_batch_s = (
        window.total_batch_completion_time_s / window.completed_batches
        if window.completed_batches > 0
        else 0.0
    )

    n_timed = window.num_timed_batches
    timed_s = window.timed_batch_time_s
    return EngineMetricsSnapshot(
        num_active_steps=n_steps,
        num_prefill_steps=window.num_prefill_steps,
        prompt_tokens=window.prompt_tokens,
        generation_tokens=window.generation_tokens,
        completed_batches=window.completed_batches,
        last_batch_completion_time_s=window.last_batch_completion_time_s,
        avg_batch_completion_time_s=avg_batch_s,
        num_timed_batches=n_timed,
        avg_batch_requests=(
            window.batch_requests / n_timed if n_timed > 0 else 0.0
        ),
        avg_prefill_throughput_tok_per_s=(
            window.batch_prompt_tokens / timed_s if timed_s > 0.0 else 0.0
        ),
        avg_generation_throughput_tok_per_s=(
            window.batch_generation_tokens / timed_s if timed_s > 0.0 else 0.0
        ),
        avg_schedule_duration_ms=avg_sched_ms,
        avg_engine_step_duration_ms=avg_step_ms,
        completed_requests=n_req,
        avg_request_queue_time_s=avg_queue_s,
        avg_request_ttft_s=avg_ttft_s,
        avg_request_prefill_time_s=avg_prefill_s,
        avg_request_decode_time_s=avg_decode_s,
        avg_request_e2e_latency_s=avg_e2e_s,
        max_request_e2e_latency_s=window.max_request_e2e_latency_s,
        avg_request_time_per_output_token_ms=avg_tpot_ms,
        num_running_reqs=num_running_reqs,
        num_waiting_reqs=num_waiting_reqs,
        kv_cache_usage_pct=kv_cache_usage_fraction * 100.0,
        prefix_cache_hit_rate_pct=prefix_hit_pct,
        num_preemptions=window.num_preemptions,
        avg_num_running_reqs=avg_running,
        avg_kv_cache_usage_pct=avg_kv_pct,
        max_kv_cache_usage_pct=window.max_kv_cache_usage * 100.0,
    )

  def snapshot(
      self,
      *,
      num_running_reqs: int = 0,
      num_waiting_reqs: int = 0,
      kv_cache_usage_fraction: float = 0.0,
  ) -> EngineMetricsSnapshot:
    """Returns a cumulative snapshot of all metrics recorded so far."""
    with self._lock:
      return self._build_snapshot_locked(
          self._cumulative,
          num_running_reqs=num_running_reqs,
          num_waiting_reqs=num_waiting_reqs,
          kv_cache_usage_fraction=kv_cache_usage_fraction,
      )

  def flush_step_snapshot(
      self,
      *,
      num_running_reqs: int = 0,
      num_waiting_reqs: int = 0,
      kv_cache_usage_fraction: float = 0.0,
  ) -> EngineMetricsSnapshot:
    """Returns the snapshot for the current training step and resets the step window."""
    with self._lock:
      snap = self._build_snapshot_locked(
          self._step_window,
          num_running_reqs=num_running_reqs,
          num_waiting_reqs=num_waiting_reqs,
          kv_cache_usage_fraction=kv_cache_usage_fraction,
      )
      self._step_window.reset()
      return snap

  def maybe_log_stats(
      self,
      *,
      num_running_reqs: int,
      num_waiting_reqs: int,
      kv_cache_usage_fraction: float,
      force: bool = False,
  ) -> bool:
    """Logs a vLLM-style periodic status line if `log_stats_interval_s` elapsed.

    The line's throughputs are over the wall-clock time since the first engine
    step after the previous line started.
    """
    if self._log_stats_interval_s <= 0.0 and not force:
      return False
    now = time.perf_counter()
    with self._lock:
      if not force and (now - self._last_log_time) < self._log_stats_interval_s:
        return False
      if self._log_window.num_active_steps == 0 and not force:
        self._last_log_time = now
        return False
      snap = self._build_snapshot_locked(
          self._log_window,
          num_running_reqs=num_running_reqs,
          num_waiting_reqs=num_waiting_reqs,
          kv_cache_usage_fraction=kv_cache_usage_fraction,
      )
      start_s = self._log_window.first_step_start_s
      wall_s = now - start_s if start_s is not None else 0.0
      prefill_tput = snap.prompt_tokens / wall_s if wall_s > 0.0 else 0.0
      gen_tput = snap.generation_tokens / wall_s if wall_s > 0.0 else 0.0
      self._log_window.reset()
      self._last_log_time = now

    batch_str = ""
    if snap.completed_batches > 0:
      batch_str = (
          f", Last batch: {snap.last_batch_completion_time_s:.2f} s"
          f", Avg batch: {snap.avg_batch_completion_time_s:.2f} s"
          f" (n={snap.completed_batches})"
      )
    logging.info(
        "Engine %03d: Avg prefill throughput: %.1f tokens/s, "
        "Avg generation throughput: %.1f tokens/s, "
        "Running: %d reqs, Waiting: %d reqs, "
        "KV cache usage: %.1f%%, Prefix cache hit rate: %.1f%%, "
        "Avg step: %.2f ms, Avg sched: %.2f ms, "
        "Avg req wait: %.3f s, Avg req e2e: %.3f s%s",
        self._engine_index,
        prefill_tput,
        gen_tput,
        snap.num_running_reqs,
        snap.num_waiting_reqs,
        snap.kv_cache_usage_pct,
        snap.prefix_cache_hit_rate_pct,
        snap.avg_engine_step_duration_ms,
        snap.avg_schedule_duration_ms,
        snap.avg_request_queue_time_s,
        snap.avg_request_e2e_latency_s,
        batch_str,
    )
    return True
