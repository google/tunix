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

"""Continuous batching engine.

`LLMEngine` ties the v2 sampler together: each step, the `Scheduler` picks a
batch of requests, the `KVCacheManager` maps them onto KV cache pages, and the
`ModelRunner` runs the model over the batch and samples the next tokens.
"""

import dataclasses

from flax import nnx
from flax.nnx import graph
from tunix.experimental.generate import kv_cache_manager as kv_cache_manager_lib
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import scheduler as scheduler_lib
from tunix.generate import utils


def _instantiate_components(
    transformer: nnx.Module,
    cache_config: kv_cache_manager_lib.CacheConfig,
    scheduler_config: scheduler_lib.SchedulerConfig,
    model_runner_config: model_runner_lib.ModelRunnerConfig,
) -> tuple[
    model_runner_lib.ModelRunner,
    kv_cache_manager_lib.KVCacheManager,
    scheduler_lib.Scheduler,
]:
  """Validates the configs and builds the engine's components from them.

  Args:
    transformer: The model to sample from. Its `init_kv_cache` builds the KV
      cache manager for its layers.
    cache_config: The KV cache config.
    scheduler_config: The scheduler config.
    model_runner_config: The model runner config.

  Returns:
    A tuple of (the model runner, the KV cache manager, the scheduler).

  Raises:
    ValueError: If the scheduler and model runner disagree on
      `num_scheduler_steps`.
  """
  if (
      scheduler_config.num_scheduler_steps
      != model_runner_config.num_scheduler_steps
  ):
    raise ValueError(
        'scheduler_config.num_scheduler_steps'
        f' ({scheduler_config.num_scheduler_steps}) must equal'
        ' model_runner_config.num_scheduler_steps'
        f' ({model_runner_config.num_scheduler_steps}).'
    )

  kv_cache_manager = transformer.init_kv_cache(cache_config)  # pyrefly: ignore[missing-attribute]

  # A sliding window layer must hold the KV of every token in its window while
  # it processes a prefill chunk, so a longer chunk size is truncated to the
  # largest power of 2 within the smallest window.
  window_sizes = [
      g.window_size
      for g in kv_cache_manager.cache_geometries.values()
      if g.window_size is not None
  ]
  if window_sizes:
    min_window_size = min(window_sizes)
    if scheduler_config.max_chunked_prefill_length > min_window_size:
      scheduler_config = dataclasses.replace(
          scheduler_config,
          max_chunked_prefill_length=1 << (min_window_size.bit_length() - 1),
      )

  model_runner = model_runner_lib.ModelRunner(transformer, model_runner_config)
  scheduler = scheduler_lib.Scheduler(
      config=scheduler_config, kv_cache_manager=kv_cache_manager
  )
  return model_runner, kv_cache_manager, scheduler


class LLMEngine:
  """A continuous batching engine."""

  def __init__(
      self,
      transformer: nnx.Module,
      cache_config: kv_cache_manager_lib.CacheConfig,
      scheduler_config: scheduler_lib.SchedulerConfig,
      model_runner_config: model_runner_lib.ModelRunnerConfig,
      max_model_len: int,
  ):
    """Initializes the engine.

    Args:
      transformer: The model to sample from. Its `init_kv_cache` builds the KV
        cache manager for its layers.
      cache_config: The KV cache config.
      scheduler_config: The scheduler config.
      model_runner_config: The model runner config.
      max_model_len: The maximum number of tokens a sequence may hold (prompt +
        generated).

    Raises:
      ValueError: If `max_model_len` is not positive, or the scheduler and
        model runner disagree on `num_scheduler_steps`.
    """
    if max_model_len <= 0:
      raise ValueError(f'max_model_len must be positive, got {max_model_len}.')

    self._model_runner, self._kv_cache_manager, self._scheduler = (
        _instantiate_components(
            transformer,
            cache_config,
            scheduler_config,
            model_runner_config,
        )
    )
    self._scheduler_config = self._scheduler._config  # pylint: disable=protected-access
    self._model_runner_config = model_runner_config

    # The maximum number of tokens a sequence may hold (prompt + generated).
    self._max_model_len = max_model_len
    # The maximum number of pages a sequence may hold in the KV cache.
    self._max_pages_per_seq = utils.cdiv(
        max_model_len + scheduler_config.num_scheduler_steps,
        cache_config.page_size,
    )
    num_device_pages = self._kv_cache_manager.num_device_pages
    if self._max_pages_per_seq > num_device_pages:
      raise ValueError(
          f'A sequence of max_model_len={max_model_len} with'
          f' num_scheduler_steps={scheduler_config.num_scheduler_steps} and'
          f' page_size={cache_config.page_size} needs up to'
          f' {self._max_pages_per_seq} pages, but the KV cache only has'
          f' {num_device_pages} device pages.'
      )

  def model_def_and_state(
      self,
  ) -> tuple[graph.GraphDef[nnx.Module], list[nnx.Variable]]:
    return self._model_runner.model_def_and_state()
