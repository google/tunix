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
import itertools
from typing import Any

from flax import nnx
from flax.nnx import graph
import jax
import jaxtyping
import numpy as np
from tunix.experimental.generate import kv_cache_manager as kv_cache_manager_lib
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import scheduler as scheduler_lib
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.generate import utils


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
          max_chunked_prefill_length=1 << (min_window_size.bit_length() - 1),
      )

  model_runner = model_runner_lib.ModelRunner(transformer, model_runner_config)
  scheduler = scheduler_lib.Scheduler(
      config=scheduler_config, kv_cache_manager=kv_cache_manager
  )
  return model_runner, kv_cache_manager, scheduler


def _next_power_of_2(x: int) -> int:
  """Returns the next power of 2 that is not smaller than x."""
  x = int(x)
  if x <= 1:
    return 1
  return 1 << (x - 1).bit_length()


class LLMEngine:
  """A continuous batching engine."""

  def __init__(
      self,
      transformer: nnx.Module,
      cache_config: kv_cache_manager_lib.CacheConfig,
      scheduler_config: scheduler_lib.SchedulerConfig,
      model_runner_config: model_runner_lib.ModelRunnerConfig,
      max_model_len: int,
      eos_token_ids: frozenset[int] = frozenset(),
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
      eos_token_ids: Token ids that finish a request once sampled.

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
    self._eos_token_ids = eos_token_ids
    self._replicated_sharding = jax.sharding.NamedSharding(
        model_runner_config.mesh, jax.sharding.PartitionSpec()
    )

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

  def model_def_and_state(
      self,
  ) -> tuple[graph.GraphDef[nnx.Module], list[nnx.Variable]]:
    return self._model_runner.model_def_and_state()

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
    self._scheduler.preempt_all()
    self._kv_cache_manager.reset_kv_caches()

  def add_request(self, request: request_lib.Request) -> None:
    """Submits a request, to be scheduled from the next step on."""
    request_id = request.request_id
    if request_id in self._requests:
      raise ValueError(f'Request {request_id} is already running.')
    sampling_params = request.sampling_params
    self._check_sampling_params(request_id, sampling_params)

    prompt_token_ids = request.prompt_token_ids
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
        request_id, prompt_token_ids, sampling_params
    )
    self._requests[request_id] = req
    self._new_requests.append(req)

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
      max_top_k = self._model_runner_config.max_top_k
      if max_top_k > 0 and params.top_k > max_top_k:
        raise ValueError(
            f'Request {request_id} has top_k={params.top_k}, over'
            f' max_top_k={max_top_k}.'
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
    new_requests = [req for req in self._new_requests if not req.is_done]
    self._new_requests = []

    scheduled, distribution = self._scheduler.schedule_step(new_requests)
    if not scheduled:
      return []

    tokens, metadata = self._prepare_batch(scheduled, distribution)
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

    generated_tokens, logits, logprobs = jax.device_get(
        (generated_tokens, logits, logprobs)
    )

    # The runner pads its outputs to `max_num_seqs` rows. Drop the
    # padding.
    n = len(scheduled)
    finished = self._update_from_output(
        scheduled,
        distribution,
        generated_tokens[:n],
        None if logits is None else logits[:n],
        None if logprobs is None else logprobs[:n],
    )

    for req in finished:
      del self._requests[req.request_id]
    return [self._make_output(req) for req in finished]

  def _update_from_output(
      self,
      scheduled: tuple[request_lib.RequestState, ...],
      distribution: tuple[int, int, int],
      generated_tokens: np.ndarray,
      logits: np.ndarray | None = None,
      logprobs: np.ndarray | None = None,
  ) -> list[request_lib.RequestState]:
    """Updates request state based on a sampler output."""
    i, j, k = distribution
    completed_reqs = []
    for row in itertools.chain(range(i), range(j, k)):
      req = scheduled[row]
      if req.is_done:
        req.num_in_flight_tokens = 0
        continue

      self._update_req_from_output(
          req,
          generated_tokens[row],
          logits[row] if logits is not None else None,
          logprobs[row] if logprobs is not None else None,
      )

      if req.is_done:
        completed_reqs.append(req)

    return completed_reqs

  def _update_req_from_output(
      self,
      req: request_lib.RequestState,
      generated_tokens: np.ndarray,
      logits: np.ndarray | None = None,
      logprobs: np.ndarray | None = None,
  ) -> None:
    """Updates request state based on a sampler output."""
    req.num_in_flight_tokens = 0
    if req.is_done:
      return

    # Gather the new tokens generated for this request. The model runner
    # does not check for EOS and may produce garbage tokens.
    new_tokens = []
    for new_token in generated_tokens:
      token_id = int(new_token)
      new_tokens.append(token_id)
      if token_id in self._eos_token_ids:
        req.status = request_lib.RequestStatus.FINISHED_EOS
        break

    # Truncate the new tokens if the request has reached its max length.
    n_generated = (
        len(req.token_ids) + len(new_tokens) - req.prompt_length
    )
    n_overflow = n_generated - req.sampling_params.max_tokens
    if n_overflow > 0:
      new_tokens = new_tokens[:-n_overflow]
      req.status = request_lib.RequestStatus.FINISHED_LENGTH
    elif n_overflow == 0 and not req.is_done:
      req.status = request_lib.RequestStatus.FINISHED_LENGTH

    # Add the new tokens to the request.
    num_added = len(new_tokens)
    req.token_ids.extend(new_tokens)
    if logprobs is not None:
      req.logprobs.extend(logprobs[:num_added])
    if logits is not None:
      req.logits.extend(logits[:num_added])

  def _make_output(
      self, req: request_lib.RequestState
  ) -> request_lib.RequestOutput:
    """Returns the output of the finished request `req`."""
    generated = req.token_ids[req.prompt_length :]
    return request_lib.RequestOutput(
        request_id=req.request_id,
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
  ) -> tuple[jax.Array, model_runner_lib.RPAMetadata]:
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
      start = req.num_computed_tokens

      if req.is_chunked_prefill:
        end = start + self._scheduler.chunked_prefill_length
        req.num_computed_tokens = end
        req.num_in_flight_tokens = 0
      else:
        end = len(req.token_ids)
        req.num_in_flight_tokens = self._scheduler_config.num_scheduler_steps
        req.num_computed_tokens = end + req.num_in_flight_tokens - 1

      batch_tokens.extend(req.token_ids[start:end])
      query_lens[row] = end - start
      kv_lens[row] = end
      for cache_name, idxs in self._kv_cache_manager.get_page_idxs(req).items():
        page_indices[cache_name][row, : len(idxs)] = idxs

    # Pad the tokens to a power of 2 to limit jax recompilation in the model
    # runner.
    num_tokens = min(
        _next_power_of_2(len(batch_tokens)),
        self._scheduler_config.max_num_batched_tokens,
    )
    tokens = np.zeros((num_tokens,), dtype=np.int32)
    tokens[: len(batch_tokens)] = batch_tokens

    metadata = model_runner_lib.RPAMetadata(
        page_indices={
            cache_name: table for cache_name, table in page_indices.items()
        },
        kv_lens=kv_lens,
        query_lens=query_lens,
        distribution=np.asarray(distribution, dtype=np.int32),
        chunk_prefill_size=self._scheduler.chunked_prefill_length,
    )
    device_tokens = jax.device_put(tokens, self._replicated_sharding)
    return device_tokens, metadata
