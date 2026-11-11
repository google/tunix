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

`LLMEngine` ties the v2 sampler together: each step, the `DPCoordinator`
schedules a batch of requests and maps them onto KV cache pages, and the
`ModelRunner` runs the model over the batch and samples the next tokens.
"""

import dataclasses
from typing import Any

from flax import nnx
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jaxtyping
import numpy as np

from tunix.common import configs as common_configs
from tunix.experimental.generate import dp_coordinator as dp_coordinator_lib
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import utils
from tunix.models import paged_attention


@dataclasses.dataclass(frozen=True)
class _InFlightStep:
  """A model runner step currently executing on the device."""

  scheduled: tuple[request_lib.RequestState, ...]
  distribution: tuple[int, int, int]
  generated_tokens: jax.Array
  logits: jax.Array | None
  logprobs: jax.Array | None


class LLMEngine:
  """A continuous batching engine."""

  def __init__(
      self,
      transformer: nnx.Module,
      rollout_config: common_configs.RolloutConfig,
      mesh: jax.sharding.Mesh,
      max_model_len: int,
  ):
    """Initializes the engine.

    Args:
      transformer: The model to sample from.
      rollout_config: The engine's rollout configuration.
      mesh: The device mesh the model runs on.
      max_model_len: The maximum number of tokens a sequence may hold (prompt +
        generated).

    Raises:
      ValueError: If `max_model_len` is not positive, or if `rollout_config`
        settings are invalid.
    """
    cache_geometries = transformer.cache_geometries()  # pyrefly: ignore[missing-attribute]
    dtype = transformer.config.dtype  # pyrefly: ignore[missing-attribute]

    self._rollout_config = rollout_config
    cache_config = utils.to_cache_config(rollout_config, dtype, mesh)
    self._scheduler_config = utils.to_scheduler_config(
        rollout_config, cache_geometries
    )
    self._model_runner_config = utils.to_model_runner_config(
        rollout_config, mesh
    )

    self._model_runner = model_runner_lib.ModelRunner(
        transformer, self._model_runner_config
    )

    eos_token_ids = frozenset(rollout_config.eos_tokens or ())
    self._dp_coordinator = dp_coordinator_lib.DPCoordinator(
        cache_config=cache_config,
        scheduler_config=self._scheduler_config,
        cache_geometries=cache_geometries,
        mesh=mesh,
        max_model_len=max_model_len,
        eos_token_ids=eos_token_ids,
    )

    # The step launched during the previous step's forward pass.
    self._in_flight_step: _InFlightStep | None = None

  def model_def_and_state(
      self,
  ) -> tuple[graph.GraphDef[nnx.Module], list[nnx.Variable]]:
    return self._model_runner.model_def_and_state()

  @property
  def transformer_state(self) -> statelib.State:
    """The weights the engine runs, which every step reads in place."""
    return self._model_runner.transformer_state

  @property
  def mesh(self) -> jax.sharding.Mesh:
    return self._model_runner_config.mesh

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
    tokens sampled so far. The prefix cache is cleared. If the device KV page
    pools were deleted, they are reallocated.
    """
    self._in_flight_step = None
    self._dp_coordinator.reset_kv_caches()

  def delete_kv_caches(self) -> list[request_lib.EngineResponse]:
    """Drains any in-flight step, preempts requests, and deletes device KV buffers."""
    outputs: list[request_lib.EngineResponse] = []
    if self._in_flight_step is not None:
      in_flight = self._in_flight_step
      self._in_flight_step = None
      outputs = self._consume_in_flight_step(in_flight)
    self._dp_coordinator.delete_kv_caches()
    return outputs

  def add_request(self, request: request_lib.EngineRequest) -> None:
    """Submits a request, to be scheduled from the next step on."""
    utils.verify_sampling_params(
        request.request_id, request.sampling_params, self._rollout_config
    )
    self._dp_coordinator.add_request(request)

  def abort_request(self, request_id: str) -> None:
    """Aborts a request. Has no effect if it is unknown or already done.

    A request still queued in the DP coordinator is removed immediately. A
    request the scheduler already holds keeps its KV cache pages until the next
    step, which releases them.

    Args:
      request_id: The id of the request to abort.
    """
    self._dp_coordinator.abort_request(request_id)

  def has_unfinished_requests(self) -> bool:
    """Whether stepping the engine has work left to do."""
    return self._dp_coordinator.has_unfinished_requests()

  def step(self) -> list[request_lib.EngineResponse]:
    """Runs one step of the continuous batch.

    Returns:
      The outputs of the requests that finished during this step.
    """
    if self._in_flight_step is not None and any(
        not req.is_done for req in self._in_flight_step.scheduled
    ):
      in_flight = self._in_flight_step
      self._in_flight_step = None
    else:
      self._in_flight_step = None
      scheduled, distribution = self._dp_coordinator.schedule_step()
      if not scheduled:
        return []
      tokens, target_idxs, metadata = self._prepare_batch(
          scheduled, distribution
      )
      in_flight = self._execute_batch(
          scheduled, distribution, tokens, target_idxs, metadata
      )

    # Schedule the next step, load its prefill token buffer onto the device,
    # write its decode tokens from `in_flight.generated_tokens`, and launch it
    # while the current forward pass runs on device.
    next_scheduled, next_distribution = self._dp_coordinator.schedule_step()
    if next_scheduled:
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

    return self._consume_in_flight_step(in_flight)

  def _consume_in_flight_step(
      self, in_flight: _InFlightStep
  ) -> list[request_lib.EngineResponse]:
    """Blocks on `in_flight`, updates request states, and returns finished outputs."""
    generated_tokens, logits, logprobs = jax.device_get(
        (in_flight.generated_tokens, in_flight.logits, in_flight.logprobs)
    )

    n = len(in_flight.scheduled)
    finished = self._update_from_output(
        in_flight.scheduled,
        in_flight.distribution,
        generated_tokens[:n],
        None if logits is None else logits[:n],
        None if logprobs is None else logprobs[:n],
    )
    if self._in_flight_step is not None and all(
        req.is_done for req in self._in_flight_step.scheduled
    ):
      self._in_flight_step = None

    return [req.to_response() for req in finished]

  def _execute_batch(
      self,
      scheduled: tuple[request_lib.RequestState, ...],
      distribution: tuple[int, int, int],
      tokens: jax.Array,
      target_idxs: jax.Array,
      metadata: paged_attention.PagedAttnMetadata,
      prev_generated_tokens: jax.Array | None = None,
  ) -> _InFlightStep:
    """Writes any in-flight decode tokens on device and launches the runner."""
    if prev_generated_tokens is not None:
      tokens = utils.jit_scatter_last_column(
          tokens, prev_generated_tokens, target_idxs
      )
    generated_tokens, logits, logprobs, pages = (
        self._model_runner.execute_step(
            cache=self._dp_coordinator.get_physical_pages(),
            tokens=tokens,
            metadata=metadata,
            sampling_params=tuple(req.sampling_params for req in scheduled),
        )
    )
    # JAX functions cannot have side effects, so the model runner returns the
    # updated pages rather than writing them in place. They must be written back
    # to the DP coordinator here.
    self._dp_coordinator.update_device_pool(pages)
    generated_tokens.copy_to_host_async()
    if logits is not None:
      logits.copy_to_host_async()
    if logprobs is not None:
      logprobs.copy_to_host_async()
    return _InFlightStep(
        scheduled=scheduled,
        distribution=distribution,
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
  ) -> list[request_lib.RequestState]:
    """Updates request state based on a sampler output."""
    return self._dp_coordinator.update_from_output(
        scheduled, distribution, generated_tokens, logits, logprobs
    )

  def _prepare_batch(
      self,
      scheduled: tuple[request_lib.RequestState, ...],
      distribution: tuple[int, int, int],
      prev_scheduled: tuple[request_lib.RequestState, ...] = (),
  ) -> tuple[jax.Array, jax.Array, paged_attention.PagedAttnMetadata]:
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
    batch_tokens, prev_to_decode_row, query_lens, kv_lens, page_indices = (
        self._dp_coordinator.prepare_batch(scheduled, prev_scheduled)
    )

    # Pad the tokens to a power of 2 to limit jax recompilation in the model
    # runner.
    num_tokens = min(
        utils.next_power_of_2(len(batch_tokens)),
        self._scheduler_config.max_num_batched_tokens,
    )
    tokens = np.zeros((1, num_tokens), dtype=np.int32)
    tokens[0, : len(batch_tokens)] = batch_tokens

    target_idxs = np.full((num_rows,), num_tokens, dtype=np.int32)
    for prev_row, decode_row in prev_to_decode_row.items():
      target_idxs[prev_row] = decode_row

    metadata = paged_attention.PagedAttnMetadata(
        page_indices={
            cache_name: table for cache_name, table in page_indices.items()
        },
        kv_lens=kv_lens,
        query_lens=query_lens,
        distribution=np.asarray((distribution,), dtype=np.int32),
        decode_only=bool(
            distribution[0] == distribution[2] and distribution[2] > 0
        ),
    )
    replicated_sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec()
    )
    device_tokens, device_target_idxs = jax.device_put(
        (tokens, target_idxs), replicated_sharding
    )
    return device_tokens, device_target_idxs, metadata
