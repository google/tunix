# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orchestrator Relay coordinator for two-phase MoE `routed_experts` staging."""

import asyncio
from collections.abc import (
    AsyncIterator,
    Callable,
    Mapping,
    Sequence,
)
import math
import threading
import time
from typing import cast

from absl import logging
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import rl_engine_interface
from tunix.dist_rl.orchestrator import step_telemetry


def validate_batch_config(batch_config: batch_assembly.BatchConfig) -> None:
  """Validates explicit prompt and response budgets for Orchestrator Relay.

  Args:
    batch_config: Orchestrator batch configuration.

  Raises:
    ValueError: If `max_response_length` or `max_prompt_length` is missing or
      non-positive.
  """
  if (
      batch_config.max_response_length is None
      or batch_config.max_response_length <= 0
  ):
    raise ValueError(
        "BatchConfig.max_response_length must be positive when"
        " return_routed_experts=True."
    )
  if (
      batch_config.max_prompt_length is None
      or batch_config.max_prompt_length <= 0
  ):
    raise ValueError(
        "BatchConfig.max_prompt_length must be positive when"
        " return_routed_experts=True."
    )


def min_retained_policy_version(
    next_policy_version: int,
    *,
    max_staleness: int,
    max_version_span: int | None,
) -> int:
  """Computes the tight `2 * S + span + 2` retention bound for zombie scans.

  With max staleness `S = max_staleness` and cross-item version span
  `span = max_version_span or 0`:
  - When `sync_weights` publishes `V_new = next_policy_version`, the previous
    active trainer step was `V_new - 1`, so any rollout still admitted to the
    queue or assembler can have version down to `(V_new - 1) - S`.
  - A multi-turn / cross-item group spanning `span` versions, or an in-flight
    straggler started at `V_new - 1 - S` that finishes after up to `S + 1`
    additional sync rounds (dispatch + weight-sync overlap), can hold staged
    entries down to `V_new - (2 * S + span + 2)`.
  - Entries strictly below `V_new - (2 * S + span + 2)` can no longer be
    admitted or materialized by any live batch and are safe to evict.

  Args:
    next_policy_version: Policy version being published by `sync_weights`.
    max_staleness: Maximum allowed off-policy staleness (`S`).
    max_version_span: Optional cross-turn/cross-item policy version span limit.

  Returns:
    Minimum `staged_policy_version` to retain on rollout workers; entries
    strictly below this version are guaranteed unreferenced zombies.
  """
  return next_policy_version - (2 * max_staleness + (max_version_span or 0) + 2)


class RelayCoordinator:
  """Manages staged `RoutedExpertsRef` fetch, packing, release, and teardown.

  Note: Because staged `routed_experts` arrays live in rollout worker RAM,
  Orchestrator Relay does not preserve rollout-worker fault tolerance: evicting
  a rollout worker with pending refs will fail `fetch_routed_experts`.
  """

  def __init__(self, *, staging_overcommit_factor: float = 1.0) -> None:
    if (
        isinstance(staging_overcommit_factor, bool)
        or not isinstance(staging_overcommit_factor, (int, float))
        or not math.isfinite(staging_overcommit_factor)
        or staging_overcommit_factor < 1.0
    ):
      raise ValueError(
          "staging_overcommit_factor must be a finite float >= 1.0, got"
          f" {staging_overcommit_factor!r}."
      )
    self._engine: rl_engine_interface.AbstractRLEngine | None = None
    self._staging_overcommit_factor = float(staging_overcommit_factor)
    self._pending_releases: set[asyncio.Task[None]] = set()

  def bind_engine(
      self,
      engine: rl_engine_interface.AbstractRLEngine,
      *,
      assembler: batch_assembly.BatchAssembler[datatypes.RLTrainerPayload],
      encode_mode: datatypes.RoutedExpertsEncodeMode,
      num_experts: int | None,
      batch_config: batch_assembly.BatchConfig,
      max_staleness: int,
      full_batch_size: int,
      num_generations: int,
  ) -> None:
    """Binds `engine`, validates pool staging liveness, and sets routing spec.

    Args:
      engine: Active `AbstractRLEngine` backing rollout and trainer workers.
      assembler: Two-phase `BatchAssembler` to configure with the resolved spec.
      encode_mode: Requested MoE routing wire encoding mode (`"auto"`,
        `"int16"`, `"uint8"`, or `"uint10"`).
      num_experts: Total routed experts per MoE layer when `encode_mode="auto"`.
      batch_config: Validated prompt/response token budgets.
      max_staleness: Maximum off-policy staleness (`S`).
      full_batch_size: Number of prompt groups per optimizer step.
      num_generations: Rollouts generated per prompt (`G`).
    """
    validate_batch_config(batch_config)
    assert batch_config.max_prompt_length is not None
    assert batch_config.max_response_length is not None
    required_slots = (max_staleness + 2) * full_batch_size * num_generations
    spec = engine.routed_experts_spec(
        encode_mode=encode_mode,
        num_experts=num_experts,
    )
    bytes_per_token = datatypes.routed_experts_bytes_per_token(
        spec.num_layers, spec.top_k, spec.encoding
    )
    worst_case_bytes = (
        batch_config.max_prompt_length + batch_config.max_response_length
    ) * bytes_per_token
    engine.configure_staging_liveness(
        worst_case_bytes=worst_case_bytes,
        required_slots=required_slots,
        overcommit_factor=self._staging_overcommit_factor,
    )
    assembler.set_routed_experts_spec(spec)
    self._engine = engine

  def _on_release_done(self, done_task: asyncio.Task[None]) -> None:
    """Discards cleanly finished release tasks while keeping failed ones."""
    if not done_task.cancelled() and done_task.exception() is None:
      self._pending_releases.discard(done_task)

  def enqueue_release(self, refs: Sequence[datatypes.RoutedExpertsRef]) -> None:
    """Schedules asynchronous release of `refs` on the bound engine."""
    if not refs or self._engine is None:
      return
    task = asyncio.create_task(self._engine.release_routed_experts(tuple(refs)))
    self._pending_releases.add(task)
    task.add_done_callback(self._on_release_done)

  async def flush_releases(self) -> None:
    """Awaits in-flight release RPCs and raises the first failure, if any."""
    if not self._pending_releases:
      return
    tasks = tuple(self._pending_releases)
    self._pending_releases.clear()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    first_err: BaseException | None = None
    for res in results:
      if isinstance(res, BaseException) and not isinstance(
          res, asyncio.CancelledError
      ):
        if first_err is None:
          first_err = res
        else:
          logging.warning("Additional release_routed_experts failure: %s", res)
    if first_err is not None:
      raise first_err

  def release_dropped_refs(
      self,
      items: Sequence[datatypes.RLTrainerPayload | datatypes.TrajectoryItem],
  ) -> None:
    """Releases staged refs owned by dropped or filtered items."""
    self.enqueue_release([
        it.routed_experts_ref
        for it in items
        if it.routed_experts_ref is not None
    ])

  def release_unreferenced_critique_refs(
      self,
      group: Sequence[datatypes.TrajectoryItem],
      payloads: Sequence[datatypes.RLTrainerPayload],
  ) -> None:
    """Releases refs cleared by critique (degenerate or invalid rows)."""
    self.enqueue_release([
        src.routed_experts_ref
        for src, p in zip(group, payloads, strict=True)
        if src.routed_experts_ref is not None and p.routed_experts_ref is None
    ])

  async def iter_planned_microbatches(
      self,
      *,
      assembler: batch_assembly.BatchAssembler[datatypes.RLTrainerPayload],
      lock: threading.Lock,
      payloads: list[datatypes.RLTrainerPayload] | None,
      step_times: step_telemetry.StepTimes,
  ) -> AsyncIterator[tuple[batch_assembly.AssembledBatch, float]]:
    """Runs two-phase `ingest -> plan -> fetch -> pack -> release` per chunk.

    Args:
      assembler: Two-phase `BatchAssembler` configured with `RoutedExpertsSpec`.
      lock: Mutex serializing `assembler` access across background threads.
      payloads: One group's payloads (or `None` to flush buffered tails).
      step_times: Step telemetry accumulator for `routed_experts_fetch_time`.

    Yields:
      `(mb, pack_dt)` where `mb` is the packed `AssembledBatch` and `pack_dt`
      is CPU planning + packing time in seconds.
    """
    assert self._engine is not None
    is_tail_flush = payloads is None

    def _ingest() -> bool:
      with lock:
        return True if is_tail_flush else assembler.ingest(payloads)

    drain_all = await asyncio.to_thread(_ingest)
    del payloads, _ingest

    def _plan_next() -> batch_assembly.BatchPlan | None:
      with lock:
        plan = assembler.plan_next_batch(drain_all=drain_all)
        if plan is None and is_tail_flush:
          assembler.reset()
        return plan

    while True:
      plan_start = time.perf_counter()
      # Cancellation while awaiting _plan_next only occurs on terminal paths;
      # any popped plan's staged refs are reclaimed by teardown's clear_staged.
      plan = await asyncio.to_thread(_plan_next)
      plan_dt = time.perf_counter() - plan_start
      if plan is None:
        break
      fetched: dict[datatypes.RoutedExpertsRef, np.ndarray] = {}
      try:
        if plan.refs:
          fetch_start = time.monotonic()
          fetched = await self._engine.fetch_routed_experts(plan.refs)
          step_times.routed_experts_fetch_time += time.monotonic() - fetch_start
        pack_start = time.perf_counter()

        def _pack(
            p: batch_assembly.BatchPlan = plan,
            f: dict[datatypes.RoutedExpertsRef, np.ndarray] = fetched,
        ) -> batch_assembly.AssembledBatch:
          with lock:
            return assembler.pack_planned_batch(p, f)

        ready = [await asyncio.to_thread(_pack)]
        pack_dt = plan_dt + (time.perf_counter() - pack_start)
        del fetched, _pack
      finally:
        if plan.refs:
          self.enqueue_release(plan.refs)
      del plan
      yield ready.pop(), pack_dt

  async def log_staging_metrics(
      self,
      log_step: int,
      log_fn: Callable[[str, float, int], None],
  ) -> None:
    """Queries RolloutWorkers and logs aggregated staging buffer watermarks."""
    if self._engine is None:
      return
    worker_metrics = await self._engine.get_metrics(role=datatypes.Role.ROLLOUT)
    if not worker_metrics:
      return
    dicts = cast(Sequence[Mapping[str, float]], worker_metrics)
    used_vals = [float(m.get("staging/used_bytes", 0.0)) for m in dicts]
    reserved_vals = [float(m.get("staging/reserved_bytes", 0.0)) for m in dicts]
    zombie_vals = [
        float(m.get("staging/zombie_evictions_total", 0.0)) for m in dicts
    ]
    for name, value in (
        ("rollout/staging_used_bytes", float(np.sum(used_vals))),
        ("rollout/staging_used_bytes_max", float(np.max(used_vals))),
        ("rollout/staging_reserved_bytes", float(np.sum(reserved_vals))),
        (
            "rollout/staging_zombie_evictions_total",
            float(np.sum(zombie_vals)),
        ),
    ):
      log_fn(name, value, log_step)

  async def teardown(
      self,
      *,
      suppress_errors: bool,
  ) -> None:
    """Flushes pending releases and clears staged routing on active workers."""
    if self._engine is None:
      return
    first_err: BaseException | None = None
    for label, step_fn in (
        ("release_routed_experts", self.flush_releases),
        (
            "clear_all_staged_routed_experts",
            self._engine.clear_all_staged_routed_experts,
        ),
    ):
      try:
        await step_fn()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        if suppress_errors or first_err is not None:
          logging.warning(
              "Suppressing %s error during teardown: %s", label, exc
          )
        else:
          first_err = exc
    if first_err is not None:
      raise first_err
