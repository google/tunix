"""Core orchestration engine for continuous batching execution."""

import dataclasses
import time
from typing import Any, Optional, Tuple
from absl import logging
from flax import nnx
from flax.nnx import graph
from flax.nnx import statelib
import jax
from jax import numpy as jnp
import jaxtyping
import numpy as np
from tunix.experimental.generate import kv_cache_manager as kv_manager_lib
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import scheduler as scheduler_lib
from tunix.experimental.generate import utils
from tunix.generate import utils as generate_utils


# PERF-DEBUG: how often `step` emits its throughput line, in seconds.
_PERF_LOG_INTERVAL_S = 5.0

@dataclasses.dataclass(frozen=True, kw_only=True)
class EngineConfig:
  """Configuration for the Tunix LLMEngine."""

  # The maximum length of a prompt.
  max_prompt_length: int = 128

  # --- Cache configs ---
  # How many bytes to allocate on the TPU for KV Cache storage.
  # This is a total over all rollout TPU devices.
  max_tpu_bytes: int = 1 * 1024**3
  # How many bytes to allocate on the CPU for KV Cache storage.
  # 0 if cpu offloading is disabled.
  max_cpu_bytes: int = 0
  # How many KV values to store per page.
  page_size: int = 16
  # Whether or not to enable prefix caching.
  enable_prefix_caching: bool = True
  # The dtype to cache KV values in.
  dtype: jnp.dtype

  # --- Scheduling configs ---
  # The maximum number of requests that can be sampled in one step.
  max_num_seqs: int = 32
  # The maximum number of tokens to compute KV values for in one step.
  max_num_batch_tokens: int = 4096
  # The chunked prefill size.
  chunked_prefill_length: int | None = None
  # The maximum number of tokens to generate for a request.
  max_tokens_to_generate: int = 512

  # --- Model runner configs ---
  # The sampling mode to use, one of "greedy", "top_p", or "beam_search".
  sampling_mode: str
  # For top_p, it contains "top_p", "top_k", and temperature.
  # For beam search, it contains "beam_size".
  sampling_parameters: dict[str, float | int] = dataclasses.field(
      default_factory=dict
  )
  # Forbidden token ids.
  forbidden_token_ids: tuple[int, ...] | None = None
  # Whether or not return logprobs.
  return_logprobs: bool = False
  # Whether or not to return logits.
  return_logits: bool = False
  # Whether to return the prompt as part of the output sample.
  echo: bool = False
  # The number of decode steps to execute per sampling step.
  num_decode_steps: int = 1

  # --- Sharding configs ---
  # The mesh axis the KV cache is data-parallel over, if any.
  dp_axis: str | None = None
  # The mesh axis the KV cache is tensor-parallel over, if any.
  tp_axis: str | None = None
  # The size of `dp_axis` in the mesh.
  dp_size: int = 1
  # The mesh the model is sharded over. Required when an axis is set, since
  # pages are allocated outside any mesh context.
  mesh: jax.sharding.Mesh | None = None

  def __post_init__(self):
    # Echoing returns the prompt's own logits and logprobs, which are never
    # computed for the pages a prefix cache hit reuses. Echoing therefore turns
    # the prefix cache off rather than return holes.
    if self.echo and self.enable_prefix_caching:
      object.__setattr__(self, 'enable_prefix_caching', False)

  @property
  def model_runner_config(self) -> model_runner_lib.ModelRunnerConfig:
    return model_runner_lib.ModelRunnerConfig(
        sampling_mode=self.sampling_mode,
        sampling_parameters=self.sampling_parameters,
        forbidden_token_ids=self.forbidden_token_ids,
        return_logprobs=self.return_logprobs,
        return_logits=self.return_logits,
        echo=self.echo,
        num_decode_steps=self.num_decode_steps,
        mesh=self.mesh,
    )

  @property
  def scheduler_config(self) -> scheduler_lib.SchedulerConfig:
    chunked_prefill_length = (
        self.chunked_prefill_length
        if self.chunked_prefill_length is not None
        else min(1024, self.max_num_batch_tokens)
    )
    return scheduler_lib.SchedulerConfig(
        max_num_batch_tokens=self.max_num_batch_tokens,
        max_seqs_per_batch=self.max_num_seqs,
        max_tokens_to_generate=self.max_tokens_to_generate,
        chunked_prefill_length=chunked_prefill_length,
        num_decode_steps=self.num_decode_steps,
    )

  @property
  def cache_config(self) -> kv_manager_lib.CacheConfig:
    return kv_manager_lib.CacheConfig(
        max_tpu_bytes=self.max_tpu_bytes,
        max_cpu_bytes=self.max_cpu_bytes,
        page_size=self.page_size,
        enable_prefix_caching=self.enable_prefix_caching,
        dtype=self.dtype,
        dp_axis=self.dp_axis,
        tp_axis=self.tp_axis,
        dp_size=self.dp_size,
        mesh=self.mesh,
    )


class LLMEngine:
  """Core Continuous Batching Engine orchestration layer.

  The engine runs on token ids alone: callers tokenize their own prompts and
  hand it `Request`s, which keeps text handling, and the prompt length bound
  that comes with it, in the layer that owns the tokenizer.
  """

  def __init__(
      self,
      transformer: 'nnx.Module',
      engine_config: EngineConfig,
  ):
    self._engine_config = engine_config

    self._new_requests: list[request_lib.Request] = []
    self._max_seq_len = (
        engine_config.max_prompt_length
        + engine_config.max_tokens_to_generate
        + engine_config.num_decode_steps
    )

    self._model_runner = model_runner_lib.ModelRunner(
        transformer=transformer, config=engine_config.model_runner_config
    )

    self._kv_cache_manager = transformer.init_kv_cache(  # pytype: disable=attribute-error
        engine_config.cache_config
    )
    window_sizes = [
        g.window_size
        for g in self._kv_cache_manager.cache_geometries.values()
        if g.window_size is not None
    ]
    min_window_size = min(window_sizes) if window_sizes else None
    if engine_config.chunked_prefill_length is None:
      chunked_prefill_length = (
          min_window_size
          if min_window_size is not None
          else min(1024, engine_config.max_num_batch_tokens)
      )
      engine_config = dataclasses.replace(
          engine_config, chunked_prefill_length=chunked_prefill_length
      )
      self._engine_config = engine_config
    elif (
        min_window_size is not None
        and engine_config.chunked_prefill_length > min_window_size
    ):
      raise ValueError(
          'chunked_prefill_length must be less than or equal to the minimum '
          f'sliding window size ({min_window_size}), got '
          f'{engine_config.chunked_prefill_length}.'
      )

    self.scheduler = scheduler_lib.Scheduler(
        config=self._engine_config.scheduler_config,
        kv_cache_manager=self._kv_cache_manager,
    )

    # PERF-DEBUG: temporary throughput counters. Delete along with the block at
    # the end of `step`, `log_perf` and the counters in KVCacheManager.
    self._perf = {
        'prefill_tokens': 0,
        'prefill_s': 0.0,
        'decode_tokens': 0,
        'decode_s': 0.0,
    }
    self._perf_prev = {
        'prefill_tokens': 0,
        'prefill_s': 0.0,
        'decode_tokens': 0,
        'decode_s': 0.0,
        'cached_tokens': 0,
        'prompt_tokens': 0,
    }
    self._perf_last_log_s = time.perf_counter()

  @property
  def transformer(self) -> nnx.Module:
    return self._model_runner.transformer

  @property
  def transformer_state(self) -> statelib.State:
    return self._model_runner.transformer_state

  @property
  def dtype(self) -> jnp.dtype:
    return self._model_runner.dtype

  def model_def_and_state(self) -> tuple[graph.GraphDef[Any], list[Any]]:
    return self._model_runner.model_def_and_state()

  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Syncs new weights into the model the engine runs.

    Any in-flight request is preempted: the KV it has already computed was
    produced by the previous weights, so it is discarded and the request
    re-prefills its full context (prompt plus the tokens sampled so far) under
    the new weights. Tokens already sampled are kept.

    Args:
      updated_weights: The new weights.
      filter_types: The variable types to update. All of them when None.
    """
    # Update the weights first: this is the only step that can fail, and
    # leaving the scheduler and the cache untouched on failure keeps the engine
    # usable. `preempt_all` and `reset_kv_caches` cannot raise.
    self._model_runner.update_params(updated_weights, filter_types)
    # Preempt before resetting so the cache manager has already released every
    # per-request page allocation by the time the global reset runs.
    self.scheduler.preempt_all()
    self._kv_cache_manager.reset_kv_caches()

  @property
  def engine_config(self) -> EngineConfig:
    return self._engine_config

  @property
  def model_runner_config(self) -> model_runner_lib.ModelRunnerConfig:
    return self._model_runner.config

  def log_perf(self) -> None:
    """PERF-DEBUG: logs throughput and prefix cache hit rate since last log.

    Throughput is bucketed per step: a step that prefills any token counts its
    whole wall time, and only its prefill tokens, towards prefill. A pure
    decode step counts towards decode. Tokens a decode rides along with in a
    prefill step are left out of both, so neither rate is inflated.
    """
    rate = lambda n, s: n / s if s > 0 else 0.0
    perf = self._perf
    cached = self._kv_cache_manager.perf_cached_tokens
    prompt = self._kv_cache_manager.perf_prompt_tokens

    d_prefill_tokens = perf['prefill_tokens'] - self._perf_prev['prefill_tokens']
    d_prefill_s = perf['prefill_s'] - self._perf_prev['prefill_s']
    d_decode_tokens = perf['decode_tokens'] - self._perf_prev['decode_tokens']
    d_decode_s = perf['decode_s'] - self._perf_prev['decode_s']
    d_cached = cached - self._perf_prev['cached_tokens']
    d_prompt = prompt - self._perf_prev['prompt_tokens']

    self._perf_prev = {
        'prefill_tokens': perf['prefill_tokens'],
        'prefill_s': perf['prefill_s'],
        'decode_tokens': perf['decode_tokens'],
        'decode_s': perf['decode_s'],
        'cached_tokens': cached,
        'prompt_tokens': prompt,
    }

    logging.info(
        '[PERF-DEBUG] engine: prefill %.1f tok/s (%d tok in %.2fs) | decode'
        ' %.1f tok/s (%d tok in %.2fs) | prefix hit rate %.1f%% (%d/%d prompt'
        ' tokens)',
        rate(d_prefill_tokens, d_prefill_s),
        d_prefill_tokens,
        d_prefill_s,
        rate(d_decode_tokens, d_decode_s),
        d_decode_tokens,
        d_decode_s,
        100.0 * rate(d_cached, d_prompt),
        d_cached,
        d_prompt,
    )

  def add_request(self, req: request_lib.Request) -> None:
    """Submits a pre-tokenized request to the engine.

    Args:
      req: The request to run. Callers are responsible for tokenization, so that
        the engine operates purely on token ids.

    Raises:
      ValueError: If a request with the same id is already pending, or if the
        request cannot fit the per-request page table.
    """
    if any(r.request_id == req.request_id for r in self._new_requests):
      raise ValueError(f'Request {req.request_id} is already pending.')

    # The sampler truncates the prompts it tokenizes, and pads them to a width
    # derived from `max_prompt_length`, so an over-long prompt reaching the
    # engine would leave that output ragged.
    max_prompt_length = self._engine_config.max_prompt_length
    if req.prompt_length > max_prompt_length:
      raise ValueError(
          f'Request {req.request_id} has a prompt of {req.prompt_length} '
          f'tokens, over max_prompt_length={max_prompt_length}. Truncate the '
          'prompt or raise max_prompt_length.'
      )

    # Every step writes a request's pages into a row sized off `_max_seq_len`,
    # so a request that outgrows that row has to be rejected here rather than
    # overflow the row mid-step.
    max_tokens = (
        req.max_tokens_to_generate
        if req.max_tokens_to_generate is not None
        else self._engine_config.max_tokens_to_generate
    )
    page_size = self._engine_config.page_size
    n_pages_needed = utils.cdiv(
        req.prompt_length + max_tokens + self._engine_config.num_decode_steps,
        page_size,
    )
    n_pages_per_seq = utils.cdiv(self._max_seq_len, page_size)
    if n_pages_needed > n_pages_per_seq:
      raise ValueError(
          f'Request {req.request_id} needs {n_pages_needed} pages, more than '
          f'the {n_pages_per_seq} pages a request is sized for. It generates '
          f'up to {max_tokens} tokens on top of a prompt of '
          f'{req.prompt_length} tokens; lower max_tokens_to_generate or raise '
          'max_prompt_length.'
      )

    self._new_requests.append(req)

  def abort_request(self, request_id: str) -> bool:
    """Withdraws a request, freeing any KV pages it holds.

    Args:
      request_id: The id of the request to withdraw.

    Returns:
      True if the request was still in flight, False if the engine does not
      know about it (e.g. it already finished).
    """
    for i, req in enumerate(self._new_requests):
      if req.request_id == request_id:
        del self._new_requests[i]
        return True
    return self.scheduler.abort_request(request_id)

  def has_unfinished_requests(self) -> bool:
    return (
        len(self._new_requests) > 0 or self.scheduler.num_active_requests > 0
    )

  def step(self) -> list[request_lib.Request]:
    """One physical iteration of the continuous batch engine.

    Returns:
      The requests that finished during this iteration.
    """
    ordered_reqs, distribution_list = self.scheduler.schedule_step(
        self._new_requests
    )
    self._new_requests.clear()
    if not ordered_reqs:
      return []

    max_n_batch_tokens = self._engine_config.max_num_batch_tokens
    max_n_seqs = self._engine_config.max_num_seqs
    max_n_pages_per_seq = utils.cdiv(
        self._max_seq_len, self._engine_config.page_size
    )

    # Initalize ragged inputs for the attention kernel.
    query_lens = np.zeros(max_n_seqs, dtype=np.int32)
    kv_lens = np.zeros(max_n_seqs, dtype=np.int32)
    page_indices: dict[str, jax.Array | np.ndarray] = {
        cache_name: np.full(
            (max_n_seqs, max_n_pages_per_seq), -1, dtype=np.int32)
        for cache_name in self._kv_cache_manager.caches
    }
    distribution = np.array(distribution_list, dtype=np.int32)

    # --- Load ragged inputs for the attention kernel ---
    all_scheduled_tokens = []
    total_n_batch_tokens = 0
    for i, req in enumerate(ordered_reqs):
      q_len = req.num_in_flight_tokens
      n_completed = req.num_completed_tokens
      scheduled_tokens = req.token_ids[n_completed : n_completed + q_len]

      kv_len = n_completed + q_len
      all_scheduled_tokens.extend(scheduled_tokens)

      kv_lens[i] = kv_len
      query_lens[i] = q_len

      phys_idxs = self._kv_cache_manager.get_page_idxs(req.request_id)
      for cache_name, idxs in phys_idxs.items():
        page_indices[cache_name][i, : len(idxs)] = idxs

      total_n_batch_tokens += q_len

    # The buffer size for the input tokens is padded to the next power of 2
    # to avoid XLA-recompilation when executing the model runner.
    if len(all_scheduled_tokens) < max_n_batch_tokens:
      next_power_of_2 = generate_utils.next_power_of_2(total_n_batch_tokens)
      buffer_size = min(
          max_n_batch_tokens, next_power_of_2
      )
    else:
      buffer_size = max_n_batch_tokens

    # Load tokens into the input buffer.
    tokens = np.zeros(buffer_size, dtype=np.int32)
    tokens[:len(all_scheduled_tokens)] = all_scheduled_tokens

    # --- Run the model over the scheduled batch ---
    metadata = model_runner_lib.RPAMetadata(
        page_indices=page_indices,
        query_lens=query_lens,
        kv_lens=kv_lens,
        distribution=distribution,
        chunk_prefill_size=self._engine_config.chunked_prefill_length,
    )
    perf_start = time.perf_counter()  # PERF-DEBUG
    gen_tokens, logits, logp, next_cache = self._model_runner.execute_step(
        cache=self._kv_cache_manager.get_physical_pages(),
        tokens=tokens,
        metadata=metadata,
    )

    # Since JAX expects no-side effects,
    # pages are updated outside of the model runner.
    self._kv_cache_manager.update_tpu_pool(next_cache)

    finished_reqs = self.scheduler.update_from_output(
        gen_tokens,
        logits,
        logp,
    )

    # --- PERF-DEBUG: delete this block. ---
    # `update_from_output` reads the generated tokens, so the dispatch above
    # has landed by the time it returns and the elapsed time is real.
    perf_elapsed = time.perf_counter() - perf_start

    # Decodes sort first and contribute exactly one token each to the batch,
    # so whatever is left over is prefill.
    n_decode_seqs = distribution_list[0]
    n_prefill_tokens = total_n_batch_tokens - n_decode_seqs
    n_decodes = self.scheduler.last_num_generated_tokens
    if n_prefill_tokens > 0:
      self._perf['prefill_tokens'] += n_prefill_tokens
      self._perf['prefill_s'] += perf_elapsed
    else:
      self._perf['decode_tokens'] += n_decodes
      self._perf['decode_s'] += perf_elapsed

    if time.perf_counter() - self._perf_last_log_s >= _PERF_LOG_INTERVAL_S:
      self._perf_last_log_s = time.perf_counter()
      self.log_perf()
    # --- End PERF-DEBUG. ---

    return finished_reqs

