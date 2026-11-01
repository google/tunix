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

"""A vLLM stat logger that emits `engine_stats` events.

Registered through `AsyncLLM.from_engine_args(stat_loggers=[...])` only when
`tunix.experimental.events` is on (see `vllm_sampler_v2.RLVllmSampler.start`);
vLLM keeps its default logging stat logger next to it. `record()` runs on the
engine's output-handler path, so it only adds counters; `log()` runs from
`do_log_stats()` (every `VLLM_LOG_STATS_INTERVAL`, the sampler's stats loop)
and emits one event with the interval's rates. The stats it reads
(`SchedulerStats`, `IterationStats`) are not a stable vLLM interface, so every
read is a `getattr` with a default and a failure is counted, not raised.
"""

from __future__ import annotations

import time
from typing import Any

from tunix.experimental import events
from vllm.v1.metrics.loggers import StatLoggerBase


class EventsStatLogger(StatLoggerBase):
  """Per-engine stat logger; `vllm_config` is unused."""

  def __init__(self, vllm_config: Any, engine_index: int = 0):
    del vllm_config
    self.engine_index = engine_index
    self._reset(time.monotonic())
    self._running = 0
    self._waiting = 0
    self._kv_usage = 0.0

  def _reset(self, now: float) -> None:
    self._t0 = now
    self._prompt_tokens = 0
    self._gen_tokens = 0
    self._prefix_queries = 0
    self._prefix_hits = 0

  def record(self, scheduler_stats: Any, iteration_stats: Any,
             mm_cache_stats: Any = None, engine_idx: int = 0) -> None:
    del mm_cache_stats, engine_idx
    try:
      if scheduler_stats is not None:
        self._running = int(getattr(scheduler_stats, "num_running_reqs", 0))
        self._waiting = int(getattr(scheduler_stats, "num_waiting_reqs", 0))
        self._kv_usage = float(getattr(scheduler_stats, "kv_cache_usage", 0.0))
        pc = getattr(scheduler_stats, "prefix_cache_stats", None)
        if pc is not None:
          self._prefix_queries += int(getattr(pc, "queries", 0))
          self._prefix_hits += int(getattr(pc, "hits", 0))
      if iteration_stats is not None:
        pts = getattr(iteration_stats, "prompt_token_stats", None)
        self._prompt_tokens += int(getattr(pts, "computed", 0) or 0)
        self._gen_tokens += int(
            getattr(iteration_stats, "num_generation_tokens", 0) or 0)
    except Exception as e:  # pylint: disable=broad-except
      events._count_error("stat_logger", e)  # pylint: disable=protected-access

  def log(self) -> None:
    try:
      now = time.monotonic()
      dt = now - self._t0
      if events.ON:
        events.emit(
            "engine_stats",
            engine_idx=self.engine_index,
            running=self._running,
            waiting=self._waiting,
            kv_usage=self._kv_usage,
            prefix_hit=(self._prefix_hits / self._prefix_queries
                        if self._prefix_queries else None),
            prompt_tok_s=self._prompt_tokens / dt if dt > 0 else None,
            gen_tok_s=self._gen_tokens / dt if dt > 0 else None,
            interval_s=dt,
        )
      self._reset(now)
    except Exception as e:  # pylint: disable=broad-except
      events._count_error("stat_logger", e)  # pylint: disable=protected-access

  def log_engine_initialized(self) -> None:
    pass
