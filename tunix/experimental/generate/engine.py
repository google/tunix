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


@dataclasses.dataclass(frozen=True)
class _InFlightStep:
  """A model runner step currently executing on the device."""

  scheduled: tuple[request_lib.RequestState, ...]
  distribution: tuple[int, int, int]
  num_prompt_tokens: int
  generated_tokens: jax.Array
  logits: jax.Array | None
  logprobs: jax.Array | None


@jax.jit
def _write_decode_tokens(
    tokens: jax.Array,
    prev_generated_tokens: jax.Array,
    target_idxs: jax.Array,
) -> jax.Array:
  """Writes the last token sampled by each surviving row into `tokens`."""
  return tokens.at[target_idxs].set(
      prev_generated_tokens[:, -1], mode='drop'
  )


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
      tokenizer: tokenizer_adapter.TokenizerAdapter,
      cache_config: kv_cache_manager_lib.CacheConfig,
      scheduler_config: scheduler_lib.SchedulerConfig,
      model_runner_config: model_runner_lib.ModelRunnerConfig,
      max_model_len: int,
      eos_token_ids: frozenset[int] = frozenset(),
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
      eos_token_ids: Token ids that finish a request once sampled.
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
    # The step launched during the previous step's forward pass.
    self._in_flight_step: _InFlightStep | None = None
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
    self._in_flight_step = None
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
      input_ids = self._tokenizer.encode(prompt)
      bos_tok = [self._tokenizer.bos_id()] if self._tokenizer.bos_id() else []
      return self._tokenizer.dedup_bos_ids(bos_tok + input_ids)
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

    if self._in_flight_step is not None and any(
        not req.is_done for req in self._in_flight_step.scheduled
    ):
      in_flight = self._in_flight_step
      self._in_flight_step = None
    else:
      self._in_flight_step = None
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
      tokens, target_idxs, metadata = self._prepare_batch(
          scheduled, distribution
      )
      in_flight = self._execute_batch(
          scheduled, distribution, tokens, target_idxs, metadata
      )

    # Schedule the next step, load its prefill token buffer onto the device,
    # write its decode tokens from `in_flight.generated_tokens`, and launch it
    # while the current forward pass runs on device.
    coloc_sched_start = time.perf_counter()
    next_scheduled, next_distribution = self._scheduler.schedule_step(
        new_requests
    )
    next_scheduled, next_distribution = self._drop_completed(
        next_scheduled, next_distribution
    )
    coloc_sched_duration_s = time.perf_counter() - coloc_sched_start
    if next_scheduled:
      self._metrics.record_schedule(
          coloc_sched_duration_s,
          num_preemptions=self._scheduler.pop_num_preemptions(),
      )
      next_tokens, target_idxs, next_metadata = self._prepare_batch(
          next_scheduled,
          next_distribution,
          prev_scheduled=in_flight.scheduled,
      )
      self._in_flight_step = self._execute_batch(
          next_scheduled,
          next_distribution,
          next_tokens,
          target_idxs,
          next_metadata,
          prev_generated_tokens=in_flight.generated_tokens,
      )

    generated_tokens, logits, logprobs = jax.device_get(
        (in_flight.generated_tokens, in_flight.logits, in_flight.logprobs)
    )

    # The runner pads its outputs to `max_num_seqs` rows. Drop the
    # padding.
    n = len(in_flight.scheduled)
    num_running = sum(not req.is_done for req in in_flight.scheduled)
    finished, num_generated_tokens = self._update_from_output(
        in_flight.scheduled,
        in_flight.distribution,
        generated_tokens[:n],
        None if logits is None else logits[:n],
        None if logprobs is None else logprobs[:n],
    )
    self._scheduler.drop_completed((), (0, 0, 0))
    if self._in_flight_step is not None and all(
        req.is_done for req in self._in_flight_step.scheduled
    ):
      self._in_flight_step = None

    step_duration_s = time.perf_counter() - step_start
    prefix_queries, prefix_hits = (
        self._kv_cache_manager.pop_prefix_cache_stats()
    )
    self._metrics.record_step(
        step_duration_s=step_duration_s,
        num_prompt_tokens=in_flight.num_prompt_tokens,
        num_generation_tokens=num_generated_tokens,
        finished_requests=finished,
        prefix_cache_queries=prefix_queries,
        prefix_cache_hits=prefix_hits,
        num_running_reqs=num_running,
        kv_cache_usage_fraction=self._kv_cache_manager.kv_cache_usage_fraction,
    )
    self.do_log_stats()

    for req in finished:
      del self._requests[req.request_id]
    return [self._make_output(req) for req in finished]

  def _drop_completed(
      self,
      scheduled: tuple[request_lib.RequestState, ...],
      distribution: tuple[int, int, int],
  ) -> tuple[tuple[request_lib.RequestState, ...], tuple[int, int, int]]:
    """Drops finished, aborted, or max-length-in-flight requests."""
    scheduled, distribution = self._scheduler.drop_completed(
        scheduled, distribution
    )
    i, j, k = distribution
    keep = (
        lambda r: r.num_total_tokens - r.prompt_length
        < r.sampling_params.max_tokens
    )
    decodes = [r for r in scheduled[:i] if keep(r)]
    chunked = [r for r in scheduled[i:j] if keep(r)]
    prefills = [r for r in scheduled[j:k] if keep(r)]

    new_i = len(decodes)
    new_j = new_i + len(chunked)
    new_k = new_j + len(prefills)
    return tuple(decodes + chunked + prefills), (new_i, new_j, new_k)

  def _execute_batch(
      self,
      scheduled: tuple[request_lib.RequestState, ...],
      distribution: tuple[int, int, int],
      tokens: jax.Array,
      target_idxs: jax.Array,
      metadata: paged_attention.RPAMetadata,
      prev_generated_tokens: jax.Array | None = None,
  ) -> _InFlightStep:
    """Writes any in-flight decode tokens on device and launches the runner."""
    i, _, k = distribution
    num_prompt_tokens = int(np.sum(metadata.query_lens[i:k]))
    if prev_generated_tokens is not None:
      tokens = _write_decode_tokens(tokens, prev_generated_tokens, target_idxs)
    self._kv_cache_manager.wait_for_transfers()
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
    generated_tokens.copy_to_host_async()
    if logits is not None:
      logits.copy_to_host_async()
    if logprobs is not None:
      logprobs.copy_to_host_async()
    return _InFlightStep(
        scheduled=scheduled,
        distribution=distribution,
        num_prompt_tokens=num_prompt_tokens,
        generated_tokens=generated_tokens,
        logits=logits,
        logprobs=logprobs,
    )

  def _update_from_output(
      self,
      scheduled: tuple[request_lib.RequestState, ...],
      distribution: tuple[int, int, int],
      generated_tokens: np.ndarray,
      logits: np.ndarray | None = None,
      logprobs: np.ndarray | None = None,
  ) -> tuple[list[request_lib.RequestState], int]:
    """Updates request state based on a sampler output."""
    i, j, k = distribution
    now = time.perf_counter()
    completed_reqs = []
    num_generated_tokens = 0
    for row in itertools.chain(range(i), range(j, k)):
      req = scheduled[row]
      if req.is_done:
        req.num_in_flight_tokens = 0
        continue

      num_generated_tokens += self._update_req_from_output(
          req,
          generated_tokens[row],
          logits[row] if logits is not None else None,
          logprobs[row] if logprobs is not None else None,
          now=now,
      )

      if req.is_done:
        req.finished_time = now
        completed_reqs.append(req)

    return completed_reqs, num_generated_tokens

  def _update_req_from_output(
      self,
      req: request_lib.RequestState,
      generated_tokens: np.ndarray,
      logits: np.ndarray | None = None,
      logprobs: np.ndarray | None = None,
      *,
      now: float | None = None,
  ) -> int:
    """Updates request state based on a sampler output."""
    req.num_in_flight_tokens = max(
        0, req.num_in_flight_tokens - self._scheduler_config.num_scheduler_steps
    )
    if req.is_done:
      return 0

    was_running = req.status == request_lib.RequestStatus.RUNNING

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
    if num_added > 0 and req.first_token_time is None:
      req.first_token_time = time.perf_counter() if now is None else now
    req.token_ids.extend(new_tokens)
    if logprobs is not None:
      req.logprobs.extend(logprobs[:num_added])
    if logits is not None:
      req.logits.extend(logits[:num_added])

    if req.is_done:
      req.num_in_flight_tokens = 0
      if was_running:
        req.num_computed_tokens = len(req.token_ids) - 1

    return num_added

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
      prev_scheduled: tuple[request_lib.RequestState, ...] = (),
  ) -> tuple[jax.Array, jax.Array, paged_attention.RPAMetadata]:
    """Packs the scheduled requests into the model runner's inputs.

    Every row buffer spans `max_num_seqs` rows and every page table
    `max_pages_per_seq` pages, so their shapes never change.

    Args:
      scheduled: The scheduled requests, ordered as decodes, chunked prefills
        and full prefills.
      distribution: The row layout returned by the scheduler.
      prev_scheduled: The requests running in the in-flight step, if any.

    Returns:
      A tuple of (the device token buffer with prefill tokens loaded and blank
      slots for in-flight decodes, the target indices mapping each row of the
      previous step to its decode slot or out of bounds, the ragged execution
      metadata).
    """
    num_rows = self._scheduler_config.max_num_seqs
    query_lens = np.zeros((num_rows,), dtype=np.int32)
    kv_lens = np.zeros((num_rows,), dtype=np.int32)
    page_indices: dict[str, np.ndarray] = {}

    prev_row_by_id = {
        req.request_id: prev_row for prev_row, req in enumerate(prev_scheduled)
    }
    prev_to_decode_row: dict[int, int] = {}

    batch_tokens: list[int] = []
    for row, req in enumerate(scheduled):
      start = req.num_computed_tokens

      if req.is_chunked_prefill:
        end = start + self._scheduler.chunked_prefill_length
        req.num_computed_tokens = end
        req.num_in_flight_tokens = 0
        batch_tokens.extend(req.token_ids[start:end])
      else:
        end = req.num_total_tokens
        if req.num_in_flight_tokens > 0:
          batch_tokens.append(0)
          prev_to_decode_row[prev_row_by_id[req.request_id]] = row
        else:
          batch_tokens.extend(req.token_ids[start:end])
        req.num_in_flight_tokens += self._scheduler_config.num_scheduler_steps
        req.num_computed_tokens = (
            end + self._scheduler_config.num_scheduler_steps - 1
        )

      query_lens[row] = end - start
      kv_lens[row] = end
      row_groups: dict[int, np.ndarray] = {}
      for cache_name, idxs in self._kv_cache_manager.get_page_idxs(req).items():
        group_table = row_groups.get(id(idxs))
        if group_table is None:
          group_table = page_indices.get(cache_name)
          if group_table is None:
            group_table = np.full(
                (num_rows, self._max_pages_per_seq), -1, dtype=np.int32
            )
          group_table[row, : len(idxs)] = idxs
          row_groups[id(idxs)] = group_table
        page_indices[cache_name] = group_table

    if not page_indices:
      empty_table = np.full(
          (num_rows, self._max_pages_per_seq), -1, dtype=np.int32
      )
      page_indices = {
          cache_name: empty_table
          for cache_name in self._kv_cache_manager.cache_names
      }

    # Pad the tokens to a power of 2 to limit jax recompilation in the model
    # runner.
    num_tokens = min(
        _next_power_of_2(len(batch_tokens)),
        self._scheduler_config.max_num_batched_tokens,
    )
    tokens = np.zeros((num_tokens,), dtype=np.int32)
    tokens[: len(batch_tokens)] = batch_tokens

    target_idxs = np.full((num_rows,), num_tokens, dtype=np.int32)
    for prev_row, decode_row in prev_to_decode_row.items():
      target_idxs[prev_row] = decode_row

    metadata = paged_attention.RPAMetadata(
        page_indices={
            cache_name: table for cache_name, table in page_indices.items()
        },
        kv_lens=kv_lens,
        query_lens=query_lens,
        distribution=np.asarray(distribution, dtype=np.int32),
        chunk_prefill_size=self._scheduler.chunked_prefill_length,
    )
    device_tokens, device_target_idxs = jax.device_put(
        (tokens, target_idxs), self._replicated_sharding
    )
    return device_tokens, device_target_idxs, metadata
