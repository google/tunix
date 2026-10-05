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

"""Sampler backed by the continuous batching engine."""

import asyncio
from collections.abc import Sequence
import itertools
import threading
import time
from typing import Any

from flax import nnx
from flax.nnx import statelib
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from tunix.experimental.generate import driver as driver_lib
from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import metrics as metrics_lib
from tunix.experimental.generate import request as request_lib
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.generate import base_sampler
from tunix.generate import tokenizer_adapter
from tunix.generate import utils


def _to_response(
    output: request_lib.RequestOutput,
) -> sampler_lib.SamplingResponse:
  return sampler_lib.SamplingResponse(
      request_id=output.request_id,
      text=output.text,
      prompt_token_ids=output.prompt_token_ids,
      token_ids=output.token_ids,
      logprobs=output.logprobs,
      finish_reason=output.finish_reason,
  )


class Sampler(base_sampler.BaseSampler):
  """Samples from an `LLMEngine` running in this process.

  In server mode, a `VanillaInProcessDriver` runs the engine on a background
  thread, so concurrent calls share a continuous batch. Otherwise, each call
  steps the engine itself until its requests finish, and concurrent calls take
  turns.
  """

  def __init__(
      self,
      engine: engine_lib.LLMEngine,
      *,
      server_mode: bool = False,
      poll_interval_s: float = 0.004,
      submission_threshold: int = 0,
      submission_timeout_s: float = 0.0,
  ):
    """Initializes the sampler, starting the engine loop in server mode.

    Args:
      engine: The engine to sample from. The sampler owns it once constructed.
      server_mode: Whether to run the engine on a `VanillaInProcessDriver`.
      poll_interval_s: How long the background loop waits for new work before
        looking again, in seconds. Only used when `server_mode` is True.
      submission_threshold: Only hand queued requests to the engine once this
        many have accumulated. 0 submits them as soon as they arrive. Only used
        when `server_mode` is True.
      submission_timeout_s: Submit a partial batch anyway once this many
        seconds have elapsed since the first request of the current window
        arrived. 0 disables the timeout, so a partial batch waits until the
        threshold is reached. Only used when `server_mode` is True.
    """
    self._engine = engine
    # Serializes offline calls and weight changes. In server mode, the driver
    # serializes them instead.
    self._engine_lock = threading.Lock()
    self._request_counter = itertools.count()
    self._driver: driver_lib.VanillaInProcessDriver | None = None
    if server_mode:
      self._driver = driver_lib.VanillaInProcessDriver(
          engine,
          poll_interval_s=poll_interval_s,
          submission_threshold=submission_threshold,
          submission_timeout_s=submission_timeout_s,
      )
      self._driver.start()

  @property
  def _tokenizer(self) -> tokenizer_adapter.TokenizerAdapter:
    return self._engine.tokenizer

  @property
  def mesh(self) -> jax.sharding.Mesh:
    return self._engine.mesh

  def tokenize(self, input_string: str) -> list[int]:
    return self._tokenizer.encode(input_string)

  def stop(self) -> None:
    """Stops the engine loop."""
    if self._driver is not None:
      self._driver.shutdown()

  # --- Inference ---
  def __call__(
      self,
      input_strings: str | Sequence[str] | None = None,
      max_generation_steps: int = 8,
      max_prompt_length: int | None = None,
      temperature: float = 0.0,
      top_p: float | None = None,
      top_k: int | None = None,
      beam_size: int | None = None,
      seed: int | None = None,
      multi_sampling: int = 1,
      return_logits: bool = False,
      echo: bool = False,
      pad_output: bool = False,
      return_logprobs: bool = False,
      prompt_token_ids: Sequence[Sequence[int] | np.ndarray] | None = None,
  ) -> base_sampler.SamplerOutput:
    """Samples a completion for each prompt, blocking until all finish.

    Args:
      input_strings: The text prompt, or prompts, to sample from. Mutually
        exclusive with `prompt_token_ids`.
      max_generation_steps: Maximum number of tokens to generate per prompt.
      max_prompt_length: Target width for `padded_prompt_tokens`. Rounded up to
        a power of 2 when None or smaller than the longest prompt.
      temperature: Sampling temperature. 0.0 selects greedy decoding.
      top_p: Top-p (nucleus) sampling threshold.
      top_k: Top-k sampling threshold.
      beam_size: Beam width for beam search.
      seed: Per-request random seed.
      multi_sampling: Number of independent samples to generate per prompt.
      return_logits: Whether to return per-step logits.
      echo: Whether to prepend the prompt to the generated tokens.
      pad_output: Whether to right-pad generated tokens to
        `max_generation_steps` (or `max_prompt_length + max_generation_steps`
        when `echo` is True).
      return_logprobs: Whether to return per-token log-probabilities.
      prompt_token_ids: Pre-tokenized prompts, used instead of `input_strings`.

    Returns:
      A `SamplerOutput` with the completions in the order the prompts were
      given.
    """
    if echo:
      raise ValueError('echo=True is not supported.')
    if multi_sampling < 1:
      raise ValueError(
          f'multi_sampling={multi_sampling}; it must be positive.'
      )
    if seed is not None and not 0 <= seed < 1 << 32:
      raise ValueError(f'seed={seed}; it must be in [0, 2**32).')

    if prompt_token_ids is not None and input_strings is not None:
      raise ValueError(
          'Cannot specify both input_strings and prompt_token_ids.'
      )
    if prompt_token_ids is not None:
      prompts: list[str | np.ndarray] = [
          np.asarray(ids, dtype=np.int32) for ids in prompt_token_ids
      ]
    elif input_strings is not None:
      prompts = (
          [input_strings]
          if isinstance(input_strings, str)
          else list(input_strings)
      )
    else:
      raise ValueError(
          'Either input_strings or prompt_token_ids must be provided.'
      )

    requests = [
        sampler_lib.SamplingRequest(
            request_id=str(next(self._request_counter)),
            prompt=prompt,
            sampling_params=sampler_lib.SamplingParams(
                max_tokens=max_generation_steps,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=(
                    None
                    if seed is None
                    else int(np.uint32(seed) + np.uint32(sample_idx))
                ),
                beam_size=beam_size,
                return_logprobs=return_logprobs,
                return_logits=return_logits,
            ),
        )
        for prompt in prompts
        for sample_idx in range(multi_sampling)
    ]

    start_time = time.perf_counter()
    if self._driver is not None:
      outputs = self._generate_server_mode(requests)
      if len(requests) > 1:
        self.record_batch_completion(time.perf_counter() - start_time)
    else:
      outputs = self._generate_offline(requests)
      if requests:
        self.record_batch_completion(time.perf_counter() - start_time)

    padded_prompt_tokens, prompt_lengths, padded_prompt_len = (
        utils.left_pad_prompt_tokens(
            [output.prompt_token_ids for output in outputs],
            max_prompt_length,
            self._tokenizer.pad_id(),
        )
    )

    tokens = [output.token_ids for output in outputs]
    logits = (
        [output.logits for output in outputs if output.logits is not None]
        if return_logits
        else None
    )
    logprobs = (
        [output.logprobs for output in outputs if output.logprobs is not None]
        if return_logprobs
        else None
    )

    if pad_output:
      target_len = (
          padded_prompt_len + max_generation_steps
          if echo
          else max_generation_steps
      )
      pad_id = self._tokenizer.pad_id()
      tokens = [
          utils.pad_to_length(t, target_len, pad_value=pad_id) for t in tokens
      ]
      if logits is not None:
        logits = [utils.pad_to_length(l, target_len) for l in logits]
      if logprobs is not None:
        logprobs = [utils.pad_to_length(lp, target_len) for lp in logprobs]

    return base_sampler.SamplerOutput(
        text=[output.text for output in outputs],
        tokens=tokens,
        padded_prompt_tokens=padded_prompt_tokens,
        prompt_lengths=prompt_lengths,
        logprobs=logprobs,  # pyrefly: ignore[bad-argument-type]
        logits=(
            [jnp.asarray(l) for l in logits] if logits is not None else None
        ),
    )

  def generate(
      self, requests: Sequence[sampler_lib.SamplingRequest]
  ) -> list[sampler_lib.SamplingResponse]:
    """Samples a completion for each request, blocking until all finish.

    Args:
      requests: The requests to sample.

    Returns:
      The response to each request, in order.
    """
    requests = list(requests)
    start_time = time.perf_counter()
    if self._driver is not None:
      outputs = self._generate_server_mode(requests)
      if len(requests) > 1:
        self.record_batch_completion(time.perf_counter() - start_time)
    else:
      outputs = self._generate_offline(requests)
      if requests:
        self.record_batch_completion(time.perf_counter() - start_time)
    return [_to_response(output) for output in outputs]

  async def sample(
      self,
      sampling_requests: (
          sampler_lib.SamplingRequest | Sequence[sampler_lib.SamplingRequest]
      ),
  ) -> sampler_lib.SamplingResponse | list[sampler_lib.SamplingResponse]:
    """Samples a completion for each request, like `generate` but async.

    Outside server mode, cancelling a call does not stop its requests, which
    run to completion on a worker thread.

    Args:
      sampling_requests: The request, or requests, to sample.

    Returns:
      The response to each request, in order, or a single response if given a
      single request.
    """
    is_single = isinstance(sampling_requests, sampler_lib.SamplingRequest)
    requests = [sampling_requests] if is_single else list(sampling_requests)

    if self._driver is not None:
      start_time = time.perf_counter()
      outputs = await self._sample_on_driver(self._driver, requests)
      if not is_single and requests:
        self.record_batch_completion(time.perf_counter() - start_time)
      responses = [_to_response(output) for output in outputs]
    else:
      responses = await asyncio.to_thread(self.generate, requests)

    return responses[0] if is_single else responses

  async def _sample_on_driver(
      self,
      driver: driver_lib.VanillaInProcessDriver,
      requests: list[sampler_lib.SamplingRequest],
  ) -> list[request_lib.RequestOutput]:
    request_futures = driver.submit_requests(requests)

    # Shielded, so that cancelling this call withdraws the requests through
    # the driver below rather than cancelling the futures under it.
    gathered = asyncio.gather(*map(asyncio.wrap_future, request_futures))
    try:
      return await asyncio.shield(gathered)
    except asyncio.CancelledError:
      for request in requests:
        driver.cancel(request.request_id)
      raise

  def _generate_server_mode(
      self, requests: list[sampler_lib.SamplingRequest]
  ) -> list[request_lib.RequestOutput]:
    assert self._driver is not None
    request_futures = self._driver.submit_requests(requests)
    return [future.result() for future in request_futures]

  def _generate_offline(
      self, requests: list[sampler_lib.SamplingRequest]
  ) -> list[request_lib.RequestOutput]:
    with self._engine_lock:
      for request in requests:
        self._engine.add_request(request)

      outputs: dict[str, request_lib.RequestOutput] = {}
      while self._engine.has_unfinished_requests():
        for output in self._engine.step():
          outputs[output.request_id] = output

    return [outputs[request.request_id] for request in requests]

  # --- Metrics ---
  def get_metrics(self) -> metrics_lib.EngineMetricsSnapshot:
    """Returns a cumulative metrics snapshot from the underlying engine."""
    if self._driver is not None:
      return self._driver.get_metrics()
    with self._engine_lock:
      return self._engine.get_metrics()

  def flush_step_metrics(self) -> metrics_lib.EngineMetricsSnapshot:
    """Flushes and returns per-training-step metrics from the engine."""
    if self._driver is not None:
      return self._driver.flush_step_metrics()
    with self._engine_lock:
      return self._engine.flush_step_metrics()

  def record_batch_completion(self, duration_s: float) -> None:
    """Records a completed rollout batch duration on the engine's metrics."""
    if self._driver is not None:
      self._driver.record_batch_completion(duration_s)
    else:
      self._engine.metrics.record_batch_completion(duration_s)

  # --- Weights ---
  @property
  def transformer(self) -> nnx.Module:
    graphdef, state = self._engine.model_def_and_state()
    return nnx.merge(graphdef, *state)

  @property
  def transformer_state(self) -> statelib.State:
    """The weights the engine runs, which every step reads in place."""
    return self._engine.transformer_state

  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: tuple[Any, ...] | None = None,
  ) -> None:
    """Syncs new weights into the engine, between its steps."""
    if self._driver is not None:
      self._driver.update_params(updated_weights, filter_types)
    else:
      with self._engine_lock:
        self._engine.update_params(updated_weights, filter_types)

  def reinitialize_cache(self) -> None:
    """Discards the engine's KV, for after the weights change in place."""
    if self._driver is not None:
      self._driver.reset_kv_caches()
    else:
      with self._engine_lock:
        self._engine.reset_kv_caches()
