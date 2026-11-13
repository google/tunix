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

`LLMEngine` ties the v2 sampler together: each step, each DP rank's
`DPCoordinator` schedules a batch of requests and maps them onto KV cache pages,
and the `ModelRunner` runs the model over the combined batch and samples the
next tokens.
"""

from collections.abc import Sequence
import dataclasses
import itertools
import math
from typing import Any

from flax import nnx
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np

from tunix.common import configs as common_configs
from tunix.common import routing
from tunix.experimental.generate import dp_coordinator as dp_coordinator_lib
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import utils
from tunix.models import paged_attention


@dataclasses.dataclass(frozen=True)
class ScheduledBatch:
  """The requests one engine step runs: each DP rank's rows, back to back."""

  # Rank 0's requests, then rank 1's, and so on. Each rank's requests are
  # ordered `[decodes, chunked prefills, full prefills]`.
  requests: tuple[request_lib.RequestState, ...]

  # The (i, j, k) distribution for each rank. [0, i) contains decode tokens,
  # [i, j) contains mixed (prefill and decode) tokens, and [j, k)
  # contains prefill tokens (i.e. chunked prefill tokens).
  distributions: tuple[tuple[int, int, int], ...]

  @classmethod
  def concat(
      cls, ranks: Sequence[dp_coordinator_lib.RankBatch]
  ) -> 'ScheduledBatch':
    """Returns the batch of each rank's `(requests, distribution)`."""
    return cls(
        requests=tuple(
            itertools.chain.from_iterable(reqs for reqs, _ in ranks)
        ),
        distributions=tuple(distribution for _, distribution in ranks),
    )

  def split(self) -> list[dp_coordinator_lib.RankBatch]:
    """Returns each rank's `(requests, distribution)`."""
    ranks = []
    start = 0
    for distribution in self.distributions:
      end = start + distribution[2]
      ranks.append((self.requests[start:end], distribution))
      start = end
    return ranks


@dataclasses.dataclass(frozen=True)
class _InFlightStep:
  """A model runner step currently executing on the device."""

  batch: ScheduledBatch
  generated_tokens: jax.Array
  logits: jax.Array | None
  logprobs: jax.Array | None
  num_steps: int


def _pad_batch_token_count(
    num_tokens: int, min_bucket: int, max_tokens: int
) -> int:
  """Pads `num_tokens` onto power-of-two shape buckets above 256 to limit padding waste."""
  target = max(num_tokens, min_bucket)
  if target <= min_bucket:
    return min(min_bucket, max_tokens)
  if max_tokens <= 256:
    return min(utils.next_power_of_2(target), max_tokens)
  bucket = max(min_bucket, 256)
  while bucket < target and bucket < max_tokens:
    bucket *= 2
  return min(bucket, max_tokens)


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
    self._dp_coordinators = tuple(
        dp_coordinator_lib.DPCoordinator(
            cache_config=cache_config,
            scheduler_config=self._scheduler_config,
            cache_geometries=cache_geometries,
            mesh=rank_mesh,
            max_model_len=max_model_len,
            eos_token_ids=eos_token_ids,
        )
        for rank_mesh in utils.dp_ranks(mesh)
    )

    # The most rows an engine step holds, across every rank.
    self._max_num_rows = (
        len(self._dp_coordinators) * self._scheduler_config.max_num_seqs
    )

    # The DP rank of each active request, by request id.
    self._request_to_rank: dict[str, int] = {}
    # The step launched during the previous step's forward pass.
    self._in_flight_step: _InFlightStep | None = None
    # Uniform device placeholder passed on cold steps so `execute_step` keeps a
    # single JIT trace signature across cold and pipelined steps.
    dp_axes = utils.kv_cache_dp_axes(mesh)
    num_dp_shards = math.prod(mesh.shape[axis] for axis in dp_axes)
    row_spec = (
        jax.sharding.PartitionSpec(dp_axes)
        if num_dp_shards > 1 and self._max_num_rows % num_dp_shards == 0
        else jax.sharding.PartitionSpec()
    )
    self._empty_prev_generated_tokens: jax.Array = jax.device_put(
        jnp.zeros(
            (
                self._max_num_rows,
                self._scheduler_config.num_scheduler_steps,
            ),
            dtype=jnp.int32,
        ),
        jax.sharding.NamedSharding(mesh, row_spec),
    )

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

  @property
  def kv_cache_group_page_counts(
      self,
  ) -> tuple[tuple[int | None, int, int], ...]:
    """Per-group `(window_size, num_caches, num_device_pages)` summary."""
    return self._dp_coordinators[0].group_device_page_counts

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
    self._model_runner.update_params(updated_weights, filter_types)
    self.reset_kv_caches()

  def reset_kv_caches(self) -> None:
    """Discards all KV, for after the weights change in place.

    Every in-flight request is preempted, and re-prefills its prompt and the
    tokens sampled so far. The prefix cache is cleared. If the device KV page
    pools were deleted, they are reallocated.
    """
    self._in_flight_step = None
    for coordinator in self._dp_coordinators:
      coordinator.reset_kv_caches()

  def delete_kv_caches(self) -> list[request_lib.EngineResponse]:
    """Drains any in-flight step, preempts requests, and deletes device KV buffers."""
    outputs: list[request_lib.EngineResponse] = []

    if self._in_flight_step is not None:
      in_flight = self._in_flight_step
      self._in_flight_step = None
      outputs = self._consume_in_flight_step(in_flight)

    for coordinator in self._dp_coordinators:
      coordinator.delete_kv_caches()

    return outputs

  def add_request(self, request: request_lib.EngineRequest) -> None:
    """Submits a request, to be scheduled from the next step on."""
    request_id = request.request_id
    if request_id in self._request_to_rank:
      raise ValueError(f'Request {request_id} is already running.')

    utils.verify_sampling_params(
        request_id, request.sampling_params, self._rollout_config
    )

    dp_rank = self._route(request)
    self._dp_coordinators[dp_rank].add_request(request)
    self._request_to_rank[request_id] = dp_rank

  def _route(self, request: request_lib.EngineRequest) -> int:
    """Returns the DP rank `request` runs in.

    Routing only reads the request, so every host routes the same way.

    Args:
      request: The request to route.

    Raises:
      ValueError: If `request`'s route key is a negative integer.
    """
    key = (
        request.route_key
        if request.route_key is not None
        else request.request_id
    )
    return routing.stable_route_hash(key) % len(self._dp_coordinators)

  def abort_request(self, request_id: str) -> None:
    """Aborts a request. Has no effect if it is unknown or already done.

    A request still queued in its DP coordinator is removed immediately. A
    request the scheduler already holds keeps its KV cache pages until the next
    step, which releases them.

    Args:
      request_id: The id of the request to abort.
    """
    dp_rank = self._request_to_rank.pop(request_id, None)
    if dp_rank is None:
      return
    self._dp_coordinators[dp_rank].abort_request(request_id)

  def has_unfinished_requests(self) -> bool:
    """Whether stepping the engine has work left to do."""
    return any(
        coordinator.has_unfinished_requests()
        for coordinator in self._dp_coordinators
    )

  def _schedule_step(self) -> ScheduledBatch:
    """Schedules every DP rank's next step."""
    return ScheduledBatch.concat([
        coordinator.schedule_step() for coordinator in self._dp_coordinators
    ])

  def _step_num_tokens(self, batch: ScheduledBatch) -> int:
    """Returns forward steps to run for `batch` (`1` when chunked prefilling)."""
    if any(j > i for i, j, _ in batch.distributions):
      return 1
    return self._scheduler_config.num_scheduler_steps

  def step(self) -> list[request_lib.EngineResponse]:
    """Runs one step of the continuous batch.

    Returns:
      The outputs of the requests that finished during this step.
    """
    if self._in_flight_step is not None and any(
        not req.is_done for req in self._in_flight_step.batch.requests
    ):
      in_flight = self._in_flight_step
      self._in_flight_step = None
    else:
      self._in_flight_step = None
      batch = self._schedule_step()
      if not batch.requests:
        return []
      step_tokens = self._step_num_tokens(batch)
      tokens, target_idxs, metadata = self._prepare_batch(
          batch, step_tokens=step_tokens
      )
      in_flight = self._execute_batch(
          batch, tokens, target_idxs, metadata, step_tokens=step_tokens
      )

    # Schedule the next step, pack its prefill tokens and scatter indices on
    # host, and launch it while the current forward pass runs on device.
    next_batch = self._schedule_step()
    if next_batch.requests:
      next_step_tokens = self._step_num_tokens(next_batch)
      next_tokens, target_idxs, next_metadata = self._prepare_batch(
          next_batch, prev_batch=in_flight.batch, step_tokens=next_step_tokens
      )
      self._in_flight_step = self._execute_batch(
          next_batch,
          next_tokens,
          target_idxs,
          next_metadata,
          prev_generated_tokens=in_flight.generated_tokens,
          step_tokens=next_step_tokens,
      )

    return self._consume_in_flight_step(in_flight)

  def _consume_in_flight_step(
      self, in_flight: _InFlightStep
  ) -> list[request_lib.EngineResponse]:
    """Blocks on `in_flight`, updates request states, and returns finished outputs."""
    generated_tokens, logits, logprobs = jax.device_get(
        (in_flight.generated_tokens, in_flight.logits, in_flight.logprobs)
    )

    finished = self._update_from_output(
        in_flight.batch,
        generated_tokens,
        logits,
        logprobs,
        num_steps=in_flight.num_steps,
    )
    if self._in_flight_step is not None and all(
        req.is_done for req in self._in_flight_step.batch.requests
    ):
      self._in_flight_step = None

    for req in finished:
      del self._request_to_rank[req.request_id]
    return [req.to_response() for req in finished]

  def _execute_batch(
      self,
      batch: ScheduledBatch,
      tokens: np.ndarray,
      target_idxs: np.ndarray,
      metadata: paged_attention.PagedAttnMetadata,
      prev_generated_tokens: jax.Array | None = None,
      step_tokens: int | None = None,
  ) -> _InFlightStep:
    """Launches the model runner with optional in-flight decode scatter."""
    num_steps = (
        self._scheduler_config.num_scheduler_steps
        if step_tokens is None
        else step_tokens
    )
    effective_prev = (
        self._empty_prev_generated_tokens
        if prev_generated_tokens is None
        else prev_generated_tokens
    )
    rank_pages = [
        coordinator.get_physical_pages()
        for coordinator in self._dp_coordinators
    ]
    generated_tokens, logits, logprobs, pages = (
        self._model_runner.execute_step(
            cache=utils.stitch_physical_pages(self.mesh, rank_pages),
            tokens=tokens,
            metadata=metadata,
            sampling_params=tuple(
                req.sampling_params for req in batch.requests
            ),
            prev_generated_tokens=effective_prev,
            target_idxs=target_idxs,
            max_steps=num_steps,
        )
    )

    # JAX functions cannot have side effects, so the model runner returns the
    # updated pages rather than writing them in place. Write each rank's
    # updated device shards back to its DP coordinator.
    for coordinator, updated_rank_pages in zip(
        self._dp_coordinators,
        utils.split_physical_pages(pages, rank_pages),
    ):
      coordinator.update_device_pool(updated_rank_pages)

    generated_tokens.copy_to_host_async()
    if logits is not None:
      logits.copy_to_host_async()
    if logprobs is not None:
      logprobs.copy_to_host_async()

    return _InFlightStep(
        batch=batch,
        generated_tokens=generated_tokens,
        logits=logits,
        logprobs=logprobs,
        num_steps=num_steps,
    )

  def _update_from_output(
      self,
      batch: ScheduledBatch,
      generated_tokens: np.ndarray,
      logits: np.ndarray | None = None,
      logprobs: np.ndarray | None = None,
      num_steps: int | None = None,
  ) -> list[request_lib.RequestState]:
    """Updates request state across all DP ranks based on sampler output."""
    rows_per_rank = self._scheduler_config.max_num_seqs

    completed_reqs: list[request_lib.RequestState] = []
    for r, (coordinator, (scheduled, distribution)) in enumerate(
        zip(self._dp_coordinators, batch.split())
    ):
      row_slice = slice(r * rows_per_rank, (r + 1) * rows_per_rank)
      finished = coordinator.update_from_output(
          scheduled,
          distribution,
          generated_tokens[row_slice],
          None if logits is None else logits[row_slice],
          None if logprobs is None else logprobs[row_slice],
          num_steps=num_steps,
      )
      completed_reqs.extend(finished)

    return completed_reqs

  def _prepare_batch(
      self,
      batch: ScheduledBatch,
      prev_batch: ScheduledBatch | None = None,
      step_tokens: int | None = None,
  ) -> tuple[np.ndarray, np.ndarray, paged_attention.PagedAttnMetadata]:
    """Packs each DP rank's scheduled requests into the model runner's host inputs.

    Every row buffer spans `max_num_rows` rows and every page table
    `max_pages_per_seq` pages, so their shapes never change. Caches sharing a
    KV geometry alias the same backing table so the model runner transfers one
    table per group instead of one per layer.

    Args:
      batch: The scheduled requests of every DP rank.
      prev_batch: The requests running in the in-flight step, if any.
      step_tokens: Number of forward steps executed in this batch.

    Returns:
      A tuple of (the host token buffer with prefill tokens loaded and blank
      slots for in-flight decodes, the host target indices mapping each row of
      the previous step to its decode slot or out of bounds, the ragged
      execution metadata).
    """
    num_rows = self._max_num_rows
    rows_per_rank = self._scheduler_config.max_num_seqs
    num_ranks = len(self._dp_coordinators)

    prev_ranks = (
        prev_batch.split()
        if prev_batch is not None
        else [((), (0, 0, 0))] * num_ranks
    )
    rank_batches = [
        coordinator.prepare_batch(
            scheduled, prev_scheduled, step_tokens=step_tokens
        )
        for coordinator, (scheduled, _), (prev_scheduled, _) in zip(
            self._dp_coordinators, batch.split(), prev_ranks
        )
    ]
    (
        rank_batch_tokens,
        prev_to_decode_rows,
        rank_query_lens,
        rank_kv_lens,
        rank_page_indices,
    ) = zip(*rank_batches)

    is_decode_only = bool(
        all(i == k for i, _, k in batch.distributions)
        and any(k > 0 for _, _, k in batch.distributions)
    )
    if is_decode_only:
      # Keep exact `max_num_seqs` width on pure-decode batches so non-pow2
      # `max_num_seqs` (e.g. 40) stay on `_jit_decode_only_loop` without
      # padding MLP/LM-head compute up to `next_power_of_2`.
      tokens_per_rank = self._scheduler_config.max_num_seqs
    else:
      max_rank_tokens = max((len(t) for t in rank_batch_tokens), default=0)
      min_bucket = utils.next_power_of_2(self._scheduler_config.max_num_seqs)
      tokens_per_rank = _pad_batch_token_count(
          max_rank_tokens,
          min_bucket,
          self._scheduler_config.max_num_batched_tokens,
      )
    num_tokens = num_ranks * tokens_per_rank
    tokens = np.zeros((num_ranks, tokens_per_rank), dtype=np.int32)
    for r, rank_tokens in enumerate(rank_batch_tokens):
      tokens[r, : len(rank_tokens)] = rank_tokens

    target_idxs = np.full((num_rows,), num_tokens, dtype=np.int32)
    for r, prev_to_decode_row in enumerate(prev_to_decode_rows):
      for prev_row, decode_row in prev_to_decode_row.items():
        target_idxs[r * rows_per_rank + prev_row] = (
            r * tokens_per_rank + decode_row
        )

    cache_groups = self._dp_coordinators[0].cache_groups
    page_indices: dict[str, jax.Array | np.ndarray] = {}
    for group_names in cache_groups:
      rep_name = group_names[0]
      table = (
          rank_page_indices[0][rep_name]
          if num_ranks == 1
          else np.concatenate([p[rep_name] for p in rank_page_indices], axis=0)
      )
      for cache_name in group_names:
        page_indices[cache_name] = table

    metadata = paged_attention.PagedAttnMetadata(
        page_indices=page_indices,
        kv_lens=np.concatenate(rank_kv_lens, axis=0),
        query_lens=np.concatenate(rank_query_lens, axis=0),
        distribution=np.asarray(batch.distributions, dtype=np.int32),
        decode_only=is_decode_only,
    )

    return tokens, target_idxs, metadata
