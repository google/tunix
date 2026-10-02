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
import threading
from typing import Any

from flax.nnx import statelib
import jax
import jaxtyping
from tunix.experimental.generate import driver as driver_lib
from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import request as request_lib
from tunix.experimental.rollout import sampler as sampler_lib


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


class Sampler:
  """Samples from an `LLMEngine` running in this process.

  In server mode, a `VanillaInProcessDriver` runs the engine on a background
  thread, so the requests of concurrent `sample` calls share a continuous
  batch. Otherwise, each `sample` call steps the engine itself until its
  requests finish, and concurrent calls take turns.
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
      poll_interval_s: See `VanillaInProcessDriver`. Server mode only.
      submission_threshold: See `VanillaInProcessDriver`. Server mode only.
      submission_timeout_s: See `VanillaInProcessDriver`. Server mode only.
    """
    self._engine = engine
    # Serializes offline `sample` calls and weight changes. In server mode,
    # the driver serializes them instead.
    self._engine_lock = threading.Lock()
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
  def mesh(self) -> jax.sharding.Mesh:
    return self._engine.mesh

  def stop(self) -> None:
    """Stops the engine loop, failing every request still pending."""
    if self._driver is not None:
      self._driver.shutdown()

  # --- Inference ---
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
    if self._driver is not None:
      request_futures = self._driver.submit_requests(requests)
      outputs = [future.result() for future in request_futures]
    else:
      outputs = self._generate_offline(requests)
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
      outputs = await self._sample_on_driver(self._driver, requests)
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

  def _generate_offline(
      self, requests: list[sampler_lib.SamplingRequest]
  ) -> list[request_lib.RequestOutput]:
    """Steps the engine until every request finishes."""
    with self._engine_lock:
      for request in requests:
        self._engine.add_request(request)

      outputs = {}
      while self._engine.has_unfinished_requests():
        for output in self._engine.step():
          outputs[output.request_id] = output

    return [outputs[request.request_id] for request in requests]

  # --- Weights ---
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

  def delete_cache(self) -> None:
    # The KV cache stays allocated. `reinitialize_cache` discards its contents.
    pass

  def reinitialize_cache(self) -> None:
    """Discards the engine's KV, for after the weights change in place."""
    if self._driver is not None:
      self._driver.reset_kv_caches()
    else:
      with self._engine_lock:
        self._engine.reset_kv_caches()
