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

from collections.abc import Callable, Sequence
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class EngineMetricsSnapshot:
  """Immutable snapshot of engine, scheduler, request, and batch metrics."""

  # Step and token counts.
  num_active_steps: int
  num_prefill_steps: int
  prompt_tokens: int
  generation_tokens: int

  # 1. Submitted rollout batch completion times.
  completed_batches: int
  last_batch_completion_time_s: float
  avg_batch_completion_time_s: float

  # 2. Token throughputs (excluding paused / idle time).
  avg_prompt_throughput_tok_per_s: float
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

  # 5. Live vLLM-style system gauges.
  num_running_reqs: int
  num_waiting_reqs: int
  kv_cache_usage_pct: float
  prefix_cache_hit_rate_pct: float
  num_preemptions: int

  def to_perf_metrics(self) -> dict[str, PerfMetricValueT]:
    """Converts the snapshot into a metric dict for `RLCluster.buffer_metrics`."""
    metrics: dict[str, PerfMetricValueT] = {
        "rollout/avg_generation_throughput_tok_per_s": (
            self.avg_generation_throughput_tok_per_s,
            np.mean,
        ),
        "rollout/avg_prefill_throughput_tok_per_s": (
            self.avg_prefill_throughput_tok_per_s,
            np.mean,
        ),
        "rollout/avg_prompt_throughput_tok_per_s": (
            self.avg_prompt_throughput_tok_per_s,
            np.mean,
        ),
        "rollout/avg_schedule_duration_ms": (
            self.avg_schedule_duration_ms,
            np.mean,
        ),
        "rollout/avg_engine_step_duration_ms": (
            self.avg_engine_step_duration_ms,
            np.mean,
        ),
        "rollout/kv_cache_usage_pct": (
            self.kv_cache_usage_pct,
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
    return metrics


@dataclasses.dataclass
class _MetricsWindow:
  """Mutable accumulator for a single time window or cumulative run."""

  num_active_steps: int = 0
  num_prefill_steps: int = 0
  active_step_time_s: float = 0.0
  prefill_step_time_s: float = 0.0
  schedule_time_s: float = 0.0
  num_schedules: int = 0

  prompt_tokens: int = 0
  generation_tokens: int = 0

  completed_requests: int = 0
  request_queue_time_s: float = 0.0
  request_ttft_s: float = 0.0
  request_prefill_time_s: float = 0.0
  request_decode_time_s: float = 0.0
  request_e2e_latency_s: float = 0.0

  completed_batches: int = 0
  total_batch_completion_time_s: float = 0.0
  last_batch_completion_time_s: float = 0.0

  prefix_cache_queries: int = 0
  prefix_cache_hits: int = 0
  num_preemptions: int = 0

  def reset(self) -> None:
    self.num_active_steps = 0
    self.num_prefill_steps = 0
    self.active_step_time_s = 0.0
    self.prefill_step_time_s = 0.0
    self.schedule_time_s = 0.0
    self.num_schedules = 0
    self.prompt_tokens = 0
    self.generation_tokens = 0
    self.completed_requests = 0
    self.request_queue_time_s = 0.0
    self.request_ttft_s = 0.0
    self.request_prefill_time_s = 0.0
    self.request_decode_time_s = 0.0
    self.request_e2e_latency_s = 0.0
    self.completed_batches = 0
    self.total_batch_completion_time_s = 0.0
    self.last_batch_completion_time_s = 0.0
    self.prefix_cache_queries = 0
    self.prefix_cache_hits = 0
    self.num_preemptions = 0


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

    self._last_log_time: float = time.perf_counter()

  @property
  def log_stats_interval_s(self) -> float:
    return self._log_stats_interval_s

  def record_schedule(self, duration_s: float, num_preemptions: int = 0) -> None:
    """Records the host CPU duration of a non-empty `schedule_step` call."""
    with self._lock:
      for window in (self._cumulative, self._log_window, self._step_window):
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
  ) -> None:
    """Records an active `LLMEngine.step()` execution and finished requests."""
    with self._lock:
      for window in (self._cumulative, self._log_window, self._step_window):
        window.num_active_steps += 1
        window.active_step_time_s += step_duration_s
        window.prompt_tokens += num_prompt_tokens
        window.generation_tokens += num_generation_tokens
        window.prefix_cache_queries += prefix_cache_queries
        window.prefix_cache_hits += prefix_cache_hits
        if num_prompt_tokens > 0:
          window.num_prefill_steps += 1
          window.prefill_step_time_s += step_duration_s

        for req in finished_requests:
          window.completed_requests += 1
          window.request_queue_time_s += req.queue_time_s
          if req.finished_time is not None:
            e2e = max(0.0, req.finished_time - req.arrival_time)
            window.request_e2e_latency_s += e2e
            if req.first_token_time is not None:
              ttft = max(0.0, req.first_token_time - req.arrival_time)
              decode_t = max(0.0, req.finished_time - req.first_token_time)
              window.request_ttft_s += ttft
              window.request_decode_time_s += decode_t
              if req.first_scheduled_time is not None:
                prefill_t = max(
                    0.0, req.first_token_time - req.first_scheduled_time
                )
                window.request_prefill_time_s += prefill_t

  def record_batch_completion(self, duration_s: float) -> None:
    """Records the completion time of a submitted rollout batch of prompts."""
    with self._lock:
      for window in (self._cumulative, self._log_window, self._step_window):
        window.completed_batches += 1
        window.last_batch_completion_time_s = duration_s
        window.total_batch_completion_time_s += duration_s

  def _build_snapshot_locked(
      self,
      window: _MetricsWindow,
      *,
      num_running_reqs: int,
      num_waiting_reqs: int,
      kv_cache_usage_fraction: float,
  ) -> EngineMetricsSnapshot:
    avg_prompt_tput = (
        window.prompt_tokens / window.active_step_time_s
        if window.active_step_time_s > 0.0
        else 0.0
    )
    avg_prefill_tput = (
        window.prompt_tokens / window.prefill_step_time_s
        if window.prefill_step_time_s > 0.0
        else 0.0
    )
    avg_gen_tput = (
        window.generation_tokens / window.active_step_time_s
        if window.active_step_time_s > 0.0
        else 0.0
    )
    avg_sched_ms = (
        (window.schedule_time_s / window.num_schedules) * 1000.0
        if window.num_schedules > 0
        else 0.0
    )
    avg_step_ms = (
        (window.active_step_time_s / window.num_active_steps) * 1000.0
        if window.num_active_steps > 0
        else 0.0
    )
    n_req = window.completed_requests
    avg_queue_s = window.request_queue_time_s / n_req if n_req > 0 else 0.0
    avg_ttft_s = window.request_ttft_s / n_req if n_req > 0 else 0.0
    avg_prefill_s = window.request_prefill_time_s / n_req if n_req > 0 else 0.0
    avg_decode_s = window.request_decode_time_s / n_req if n_req > 0 else 0.0
    avg_e2e_s = window.request_e2e_latency_s / n_req if n_req > 0 else 0.0

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
    return EngineMetricsSnapshot(
        num_active_steps=window.num_active_steps,
        num_prefill_steps=window.num_prefill_steps,
        prompt_tokens=window.prompt_tokens,
        generation_tokens=window.generation_tokens,
        completed_batches=window.completed_batches,
        last_batch_completion_time_s=window.last_batch_completion_time_s,
        avg_batch_completion_time_s=avg_batch_s,
        avg_prompt_throughput_tok_per_s=avg_prompt_tput,
        avg_prefill_throughput_tok_per_s=avg_prefill_tput,
        avg_generation_throughput_tok_per_s=avg_gen_tput,
        avg_schedule_duration_ms=avg_sched_ms,
        avg_engine_step_duration_ms=avg_step_ms,
        completed_requests=n_req,
        avg_request_queue_time_s=avg_queue_s,
        avg_request_ttft_s=avg_ttft_s,
        avg_request_prefill_time_s=avg_prefill_s,
        avg_request_decode_time_s=avg_decode_s,
        avg_request_e2e_latency_s=avg_e2e_s,
        num_running_reqs=num_running_reqs,
        num_waiting_reqs=num_waiting_reqs,
        kv_cache_usage_pct=kv_cache_usage_fraction * 100.0,
        prefix_cache_hit_rate_pct=prefix_hit_pct,
        num_preemptions=window.num_preemptions,
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
    """Logs a vLLM-style periodic status line if `log_stats_interval_s` elapsed."""
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
        "Engine %03d: Avg prompt throughput: %.1f tokens/s, "
        "Avg prefill throughput: %.1f tokens/s, "
        "Avg generation throughput: %.1f tokens/s, "
        "Running: %d reqs, Waiting: %d reqs, "
        "KV cache usage: %.1f%%, Prefix cache hit rate: %.1f%%, "
        "Avg step: %.2f ms, Avg sched: %.2f ms, "
        "Avg req wait: %.3f s%s",
        self._engine_index,
        snap.avg_prompt_throughput_tok_per_s,
        snap.avg_prefill_throughput_tok_per_s,
        snap.avg_generation_throughput_tok_per_s,
        snap.num_running_reqs,
        snap.num_waiting_reqs,
        snap.kv_cache_usage_pct,
        snap.prefix_cache_hit_rate_pct,
        snap.avg_engine_step_duration_ms,
        snap.avg_schedule_duration_ms,
        snap.avg_request_queue_time_s,
        batch_str,
    )
    return True
