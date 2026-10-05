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
import time
from typing import Any

from flax import nnx
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jaxtyping
import numpy as np
from tunix.experimental.generate import kv_cache_manager as kv_cache_manager_lib
from tunix.experimental.generate import metrics as metrics_lib
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import scheduler as scheduler_lib
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.generate import tokenizer_adapter
from tunix.generate import utils
from tunix.models import paged_attention


_FINISH_REASONS: dict[request_lib.RequestStatus, request_lib.FinishReason] = {
    request_lib.RequestStatus.FINISHED_EOS: 'stop',
    request_lib.RequestStatus.FINISHED_LENGTH: 'length',
}


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
          max_chunked_prefill_length=min_window_size,
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
      tokenizer: tokenizer_adapter.TokenizerAdapter,
      cache_config: kv_cache_manager_lib.CacheConfig,
      scheduler_config: scheduler_lib.SchedulerConfig,
      model_runner_config: model_runner_lib.ModelRunnerConfig,
      max_model_len: int,
      *,
      log_stats_interval_s: float = 10.0,
  ):
    """Initializes the engine.

    Args:
      transformer: The model to sample from. Its `init_kv_cache` builds the KV
        cache manager for its layers.
      tokenizer: The tokenizer for text prompts.
      cache_config: The KV cache config.
      scheduler_config: The scheduler config.
      model_runner_config: The model runner config.
      max_model_len: The maximum number of tokens a sequence may hold (prompt +
        generated).
      log_stats_interval_s: Interval in seconds between periodic vLLM-style
        stats log lines. 0 disables periodic logging.

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
    self._tokenizer = tokenizer
    self._scheduler_config = scheduler_config
    self._model_runner_config = model_runner_config

    # The maximum number of tokens a sequence may hold (prompt + generated).
    self._max_model_len = max_model_len
    # The maximum number of pages a sequence may hold in the KV cache.
    self._max_pages_per_seq = utils.cdiv(
        max_model_len + scheduler_config.num_scheduler_steps,
        cache_config.page_size,
    )

    # Requests submitted since the last step, in submission order.
    self._new_requests: list[request_lib.RequestState] = []
    # Requests that are neither finished nor aborted, by request id.
    self._requests: dict[str, request_lib.RequestState] = {}
    # The batch scheduled during the previous step's forward pass, ready to run.
    self._scheduled_batch: (
        tuple[tuple[request_lib.RequestState, ...], tuple[int, int, int]] | None
    ) = None
    self._metrics = metrics_lib.MetricsCollector(
        log_stats_interval_s=log_stats_interval_s
    )

  def model_def_and_state(
      self,
  ) -> tuple[graph.GraphDef[nnx.Module], list[nnx.Variable]]:
    return self._model_runner.model_def_and_state()

  @property
  def tokenizer(self) -> tokenizer_adapter.TokenizerAdapter:
    return self._tokenizer

  @property
  def transformer_state(self) -> statelib.State:
    """The weights the engine runs, which every step reads in place."""
    return self._model_runner.transformer_state

  @property
  def mesh(self) -> jax.sharding.Mesh:
    return self._model_runner_config.mesh

  @property
  def metrics(self) -> metrics_lib.MetricsCollector:
    return self._metrics

  def get_metrics(self) -> metrics_lib.EngineMetricsSnapshot:
    """Returns a cumulative snapshot of engine performance metrics."""
    num_waiting = self._scheduler.num_pending_requests + sum(
        not req.is_done for req in self._new_requests
    )
    return self._metrics.snapshot(
        num_running_reqs=self._scheduler.num_running_requests,
        num_waiting_reqs=num_waiting,
        kv_cache_usage_fraction=self._kv_cache_manager.kv_cache_usage_fraction,
    )

  def flush_step_metrics(self) -> metrics_lib.EngineMetricsSnapshot:
    """Returns and resets the current training-step metrics window."""
    num_waiting = self._scheduler.num_pending_requests + sum(
        not req.is_done for req in self._new_requests
    )
    return self._metrics.flush_step_snapshot(
        num_running_reqs=self._scheduler.num_running_requests,
        num_waiting_reqs=num_waiting,
        kv_cache_usage_fraction=self._kv_cache_manager.kv_cache_usage_fraction,
    )

  def do_log_stats(self, *, force: bool = False) -> bool:
    """Logs periodic vLLM-style stats if `log_stats_interval_s` has elapsed."""
    num_waiting = self._scheduler.num_pending_requests + sum(
        not req.is_done for req in self._new_requests
    )
    return self._metrics.maybe_log_stats(
        num_running_reqs=self._scheduler.num_running_requests,
        num_waiting_reqs=num_waiting,
        kv_cache_usage_fraction=self._kv_cache_manager.kv_cache_usage_fraction,
        force=force,
    )

  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: tuple[Any, ...] | None = None,
  ) -> None:
    """Syncs new weights into the model the engine runs.

    Every in-flight request is preempted: the KV it holds was computed with
    the previous weights, so it is discarded, and the request re-prefills its
    prompt and the tokens sampled so far with the new weights. Tokens already
    sampled are kept. The prefix cache is cleared for the same reason.

    Args:
      updated_weights: The new weights, either all of the model's parameters
        or its LoRA parameters alone.
      filter_types: The variable types to update. All of them when None.
    """
    # Only the weight update can fail, so running it first leaves the engine
    # usable if it does.
    self._model_runner.update_params(updated_weights, filter_types)
    self.reset_kv_caches()

  def reset_kv_caches(self) -> None:
    """Discards all KV, for after the weights change in place.

    Every in-flight request is preempted, and re-prefills its prompt and the
    tokens sampled so far. The prefix cache is cleared.
    """
    self._scheduled_batch = None
    # Preempting releases every request's pages first, so that the reset
    # frees every page, including the ones the prefix cache holds.
    self._scheduler.preempt_all()
    self._scheduler.pop_num_preemptions()
    self._kv_cache_manager.reset_kv_caches()

  def add_request(
      self,
      request: sampler_lib.SamplingRequest,
      *,
      arrival_time: float | None = None,
  ) -> None:
    """Submits a request, to be scheduled from the next step on.

    Requests are checked against the engine's limits here, since a request
    that fails later would fail the whole batch it was scheduled in.

    Args:
      request: The request to run. Its prompt is either text, which the
        engine tokenizes, or a 1-D integer array of token ids.
      arrival_time: Optional monotonic timestamp when the request was submitted.

    Raises:
      ValueError: If the request id is already running, the request has no
        sampling params or asks for one the engine does not support, the
        prompt is empty, or the prompt plus `max_tokens` exceeds
        `max_model_len`.
      TypeError: If the prompt is neither text nor a 1-D integer array.
    """
    request_id = request.request_id
    if request_id in self._requests:
      raise ValueError(f'Request {request_id} is already running.')
    sampling_params = request.sampling_params
    if sampling_params is None:
      raise ValueError(f'Request {request_id} has no sampling_params.')
    self._check_sampling_params(request_id, sampling_params)

    prompt_token_ids = self._tokenize(request_id, request.prompt)
    if not prompt_token_ids:
      raise ValueError(f'Request {request_id} has an empty prompt.')
    max_len = len(prompt_token_ids) + sampling_params.max_tokens
    if max_len > self._max_model_len:
      raise ValueError(
          f'Request {request_id} has a prompt of {len(prompt_token_ids)}'
          f' tokens and generates up to {sampling_params.max_tokens} tokens,'
          f' {max_len} in total, over max_model_len={self._max_model_len}.'
      )

    req = request_lib.RequestState(
        request_id,
        prompt_token_ids,
        sampling_params,
        arrival_time=arrival_time,
    )
    self._requests[request_id] = req
    self._new_requests.append(req)

  def _tokenize(self, request_id: str, prompt: object) -> list[int]:
    """Returns the token ids of `prompt`, text or a 1-D integer array."""
    if isinstance(prompt, str):
      return self._tokenizer.encode(prompt)
    if (
        isinstance(prompt, np.ndarray)
        and prompt.ndim == 1
        and np.issubdtype(prompt.dtype, np.integer)
    ):
      return prompt.tolist()
    raise TypeError(
        f'Request {request_id} has a prompt of type {type(prompt)}; expected'
        ' text or a 1-D integer array of token ids.'
    )

  def _check_sampling_params(
      self, request_id: str, params: sampler_lib.SamplingParams
  ) -> None:
    """Raises a ValueError if the engine cannot sample with `params`."""
    if params.max_tokens < 1:
      raise ValueError(
          f'Request {request_id} has max_tokens={params.max_tokens}; it must'
          ' be positive.'
      )
    if params.temperature < 0.0:
      raise ValueError(
          f'Request {request_id} has temperature={params.temperature}; it'
          ' must be non-negative.'
      )
    if params.top_p is not None and not 0.0 < params.top_p <= 1.0:
      raise ValueError(
          f'Request {request_id} has top_p={params.top_p}; it must be in'
          ' (0, 1].'
      )
    if params.top_k is not None:
      if params.top_k < 1:
        raise ValueError(
            f'Request {request_id} has top_k={params.top_k}; it must be'
            ' positive.'
        )
      max_top_k = self._model_runner_config.max_top_k
      if max_top_k != -1 and params.top_k > max_top_k:
        raise ValueError(
            f'Request {request_id} has top_k={params.top_k}, over'
            f' max_top_k={max_top_k}.'
        )
    if params.seed is not None and not 0 <= params.seed < 1 << 32:
      raise ValueError(
          f'Request {request_id} has seed={params.seed}; it must be in'
          ' [0, 2**32).'
      )
    unsupported = {
        'beam_size': params.beam_size is not None,
        'return_routed_experts': params.return_routed_experts,
        'routed_experts_prompt_start': params.routed_experts_prompt_start != 0,
    }
    for name, is_set in unsupported.items():
      if is_set:
        raise ValueError(
            f'Request {request_id} sets {name}, which the engine does not'
            ' support.'
        )
    # The model runner config fixes the outputs of every step, so a request
    # must ask for exactly those.
    config = self._model_runner_config
    if params.return_logprobs != config.return_logprobs:
      raise ValueError(
          f'Request {request_id} has return_logprobs={params.return_logprobs},'
          f' but the engine has return_logprobs={config.return_logprobs}.'
      )
    if params.return_logits != config.return_logits:
      raise ValueError(
          f'Request {request_id} has return_logits={params.return_logits},'
          f' but the engine has return_logits={config.return_logits}.'
      )

  def abort_request(self, request_id: str) -> None:
    """Aborts a request. Has no effect if it is unknown or already done.

    A request the scheduler already holds keeps its KV cache pages until the
    next step, which releases them.

    Args:
      request_id: The id of the request to abort.
    """
    if request_id not in self._requests:
      return
    self._scheduler.abort_request(self._requests.pop(request_id))

  def has_unfinished_requests(self) -> bool:
    """Whether stepping the engine has work left to do."""
    return (
        any(not req.is_done for req in self._new_requests)
        or self._scheduler.num_active_requests > 0
    )

  def step(self) -> list[request_lib.RequestOutput]:
    """Runs one step of the continuous batch.

    Returns:
      The outputs of the requests that finished during this step.
    """
    step_start = time.perf_counter()
    # Requests aborted before they reach the scheduler hold no pages, so they
    # are dropped here.
    new_requests = [req for req in self._new_requests if not req.is_done]
    self._new_requests = []

    if self._scheduled_batch is not None:
      scheduled, distribution = self._scheduler.drop_completed(
          *self._scheduled_batch
      )
      self._scheduled_batch = None
    else:
      scheduled, distribution = (), (0, 0, 0)

    if not scheduled:
      sched_start = time.perf_counter()
      scheduled, distribution = self._scheduler.schedule_step(new_requests)
      sched_duration_s = time.perf_counter() - sched_start
      new_requests = []
      if not scheduled:
        return []
      self._metrics.record_schedule(
          sched_duration_s,
          num_preemptions=self._scheduler.pop_num_preemptions(),
      )

    tokens, metadata = self._prepare_batch(scheduled, distribution)
    i, _, k = distribution
    num_prompt_tokens = int(np.sum(metadata.query_lens[i:k]))

    generated_tokens, logits, logprobs, pages = (
        self._model_runner.execute_step(
            cache=self._kv_cache_manager.get_physical_pages(),
            tokens=tokens,
            metadata=metadata,
            sampling_params=tuple(req.sampling_params for req in scheduled),
        )
    )

    # JAX functions cannot have side effects, so the model runner returns the
    # updated pages rather than writing them in place. They must be written back
    # to the KV cache manager here.
    self._kv_cache_manager.update_device_pool(pages)

    # Schedule the next step while the current forward pass runs on device.
    coloc_sched_start = time.perf_counter()
    next_scheduled, next_distribution = self._scheduler.schedule_step(
        new_requests
    )
    coloc_sched_duration_s = time.perf_counter() - coloc_sched_start

    generated_tokens, logits, logprobs = jax.device_get(
        (generated_tokens, logits, logprobs)
    )

    # The runner pads its outputs to `max_num_seqs` rows. Drop the
    # padding.
    n = len(scheduled)
    finished = self._scheduler.update_from_output(
        scheduled,
        generated_tokens[:n],
        None if logits is None else logits[:n],
        None if logprobs is None else logprobs[:n],
    )
    drop_start = time.perf_counter()
    next_scheduled, next_distribution = self._scheduler.drop_completed(
        next_scheduled, next_distribution
    )
    coloc_sched_duration_s += time.perf_counter() - drop_start
    if next_scheduled:
      self._scheduled_batch = (next_scheduled, next_distribution)
      self._metrics.record_schedule(
          coloc_sched_duration_s,
          num_preemptions=self._scheduler.pop_num_preemptions(),
      )

    step_duration_s = time.perf_counter() - step_start
    prefix_queries, prefix_hits = (
        self._kv_cache_manager.pop_prefix_cache_stats()
    )
    self._metrics.record_step(
        step_duration_s=step_duration_s,
        num_prompt_tokens=num_prompt_tokens,
        num_generation_tokens=self._scheduler.last_num_generated_tokens,
        finished_requests=finished,
        prefix_cache_queries=prefix_queries,
        prefix_cache_hits=prefix_hits,
    )
    self.do_log_stats()

    for req in finished:
      del self._requests[req.request_id]
    return [self._make_output(req) for req in finished]

  def _make_output(
      self, req: request_lib.RequestState
  ) -> request_lib.RequestOutput:
    """Returns the output of the finished request `req`."""
    generated = req.token_ids[req.prompt_length :]
    return request_lib.RequestOutput(
        request_id=req.request_id,
        text=self._tokenizer.decode(generated),
        prompt_token_ids=np.asarray(
            req.token_ids[: req.prompt_length], dtype=np.int32
        ),
        token_ids=np.asarray(generated, dtype=np.int32),
        logprobs=(
            np.asarray(req.logprobs, dtype=np.float32)
            if self._model_runner_config.return_logprobs
            else None
        ),
        logits=(
            np.asarray(req.logits)
            if self._model_runner_config.return_logits
            else None
        ),
        finish_reason=_FINISH_REASONS[req.status],
    )

  def _prepare_batch(
      self,
      scheduled: tuple[request_lib.RequestState, ...],
      distribution: tuple[int, int, int],
  ) -> tuple[np.ndarray, paged_attention.RPAMetadata]:
    """Packs the scheduled requests into the model runner's inputs.

    Every row buffer spans `max_num_seqs` rows and every page table
    `max_pages_per_seq` pages, so their shapes never change.

    Args:
      scheduled: The scheduled requests, ordered as decodes, chunked prefills
        and full prefills.
      distribution: The row layout returned by the scheduler.

    Returns:
      A tuple of (the packed tokens each row runs this step, the ragged
      execution metadata).
    """
    num_rows = self._scheduler_config.max_num_seqs
    query_lens = np.zeros((num_rows,), dtype=np.int32)
    kv_lens = np.zeros((num_rows,), dtype=np.int32)
    page_indices = {
        cache_name: np.full(
            (num_rows, self._max_pages_per_seq), -1, dtype=np.int32
        )
        for cache_name in self._kv_cache_manager.cache_names
    }

    batch_tokens: list[int] = []
    for row, req in enumerate(scheduled):
      start = req.num_computed_tokens - req.num_in_flight_tokens

      if req.is_chunked_prefill:
        end = min(
            len(req.token_ids), start + self._scheduler.chunked_prefill_length
        )
        req.num_computed_tokens = end
      else:
        end = len(req.token_ids)
        req.num_computed_tokens = (
            end + self._scheduler_config.num_scheduler_steps - 1
        )
      req.num_in_flight_tokens = req.num_computed_tokens - start

      batch_tokens.extend(req.token_ids[start:end])
      query_lens[row] = end - start
      kv_lens[row] = end
      for cache_name, idxs in self._kv_cache_manager.get_page_idxs(req).items():
        page_indices[cache_name][row, : len(idxs)] = idxs

    # Pad the tokens to a power of 2 to limit jax recompilation in the model
    # runner.
    num_tokens = min(
        utils.next_power_of_2(len(batch_tokens)),
        self._scheduler_config.max_num_batched_tokens,
    )
    tokens = np.zeros((num_tokens,), dtype=np.int32)
    tokens[: len(batch_tokens)] = batch_tokens

    metadata = paged_attention.RPAMetadata(
        page_indices={
            cache_name: idxs for cache_name, idxs in page_indices.items()
        },
        kv_lens=kv_lens,
        query_lens=query_lens,
        distribution=np.asarray(distribution, dtype=np.int32),
        chunk_prefill_size=self._scheduler.chunked_prefill_length,
    )
    return tokens, metadata
