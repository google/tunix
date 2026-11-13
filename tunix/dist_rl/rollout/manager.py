# Copyright 2026 Google LLC
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

"""Rollout Manager concurrency controller and Raiden KV migration orchestrator."""

import asyncio
from collections.abc import Mapping
import functools
import logging
import time
import types
from typing import Any, Callable, Dict, Optional, Sequence, Union
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import timeouts as timeouts_lib
from tunix.dist_rl.rollout import collector as collector_lib
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.trajectory import base_store as store
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.worker import staging_buffer as staging_buffer_lib
from tunix.dist_rl.worker import traffic_controller as traffic_controller_lib
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.rollout import base_rollout

TrajectoryOrError = Union[
    datatypes.TrajectoryItem,
    trajectory_lib.TrajectoryError,
]


class RolloutManager:
  """Internal trajectory and concurrency control core of RolloutWorker.

  Manages active TrajectoryCollectorEngine tasks, out-of-order completion
  streaming, and straggler KV-cache migration across SamplerServer slices.
  """

  def __init__(
      self,
      sampler: sampler_lib.Sampler,
      tokenizer: Any,
      chat_parser: Any,
      env_factory: Callable[..., base_environment.BaseTaskEnv],
      agent_factory: Callable[..., base_agent.ConversationAgentBase],
      env_config: Mapping[str, Any] = types.MappingProxyType({}),
      agent_config: Mapping[str, Any] = types.MappingProxyType({}),
      config: Optional[base_rollout.RolloutConfig] = None,
      max_concurrency: int = 64,
      drain_timeout_s: float = timeouts_lib.DEFAULT_DRAIN_TIMEOUT_S,
      trajectory_store: Optional[store.TrajectoryWriter] = None,
      staging_buffer: Optional[staging_buffer_lib.RolloutStagingBuffer] = None,
  ):
    """Initializes the RolloutManager.

    Args:
      sampler: Pre-constructed Sampler instance.
      tokenizer: Tokenizer for prompt/response encoding.
      chat_parser: Chat parser for conversation templating.
      env_factory: Factory callable producing environment instances.
      agent_factory: Factory callable producing agent instances.
      env_config: Default environment constructor kwargs. A request's
        `metadata["env_config"]` entries are shallow-merged on top of these.
      agent_config: Default agent constructor kwargs. A request's
        `metadata["agent_config"]` replaces these entirely when present.
      config: RolloutConfig configuration options.
      max_concurrency: Maximum number of concurrent episodes.
      drain_timeout_s: Soft drain window (seconds) in `pre_weight_sync` when
        `partial_rollout` is False. Waits up to `drain_timeout_s` for in-flight
        trajectories to finish naturally before `pause_all()` pauses remaining
        stragglers.
      trajectory_store: Optional TrajectoryWriter for persisting rollout steps
        and episode metadata.
      staging_buffer: Optional bounded staging buffer for MoE `routed_experts`.
    """
    self._drain_timeout_s = timeouts_lib.validate_positive_finite(
        drain_timeout_s, "drain_timeout_s"
    )
    self.config = config
    self._partial_rollout: bool = (
        bool(getattr(config, "partial_rollout", False))
        if config is not None
        else False
    )
    self._store_routed_experts: bool = (
        bool(getattr(config, "store_routed_experts", False))
        if config is not None
        else False
    )
    self._routed_experts_encode_mode: datatypes.RoutedExpertsEncodeMode = (
        config.routed_experts_encode_mode if config is not None else "auto"
    )
    self._routed_experts_num_experts: int | None = (
        config.routed_experts_num_experts if config is not None else None
    )
    raw_max_prompt = config.max_prompt_length if config is not None else None
    if raw_max_prompt is not None and (
        isinstance(raw_max_prompt, bool) or not isinstance(raw_max_prompt, int)
    ):
      raise TypeError(
          "RolloutConfig.max_prompt_length must be an int or None, got"
          f" {raw_max_prompt!r}."
      )
    self._max_prompt_length: int | None = raw_max_prompt
    self._staging_buffer: staging_buffer_lib.RolloutStagingBuffer | None = (
        staging_buffer
    )
    self._staging_bytes_per_token: int | None = None
    self._policy_version: int = 0
    self.sampler = sampler
    self.env_factory = env_factory
    self.agent_factory = agent_factory
    self.env_config: Mapping[str, Any] = types.MappingProxyType(
        dict(env_config)
    )
    self.agent_config: Mapping[str, Any] = types.MappingProxyType(
        dict(agent_config)
    )
    self.max_concurrency = max_concurrency
    self.tokenizer = tokenizer
    self.chat_parser = chat_parser
    self.trajectory_store = trajectory_store
    # `RolloutConfig.eos_tokens` is the stop set the sampler runs with, and it
    # overrides the tokenizer's own EOS. Collectors need it to tell a rollout
    # that stopped on its own from one that exhausted its budget.
    self.eos_ids = getattr(config, "eos_tokens", None) if config else None

    self._active_collectors: Dict[
        str, collector_lib.TrajectoryCollectorEngine
    ] = {}
    self._active_tasks: Dict[str, asyncio.Task[Any]] = {}
    self._traffic_inst = None

  @property
  def drain_timeout_s(self) -> float:
    return self._drain_timeout_s

  @property
  def partial_rollout(self) -> bool:
    return self._partial_rollout

  @property
  def active_tasks(self) -> tuple[asyncio.Task[Any], ...]:
    return tuple(self._active_tasks.values())

  @property
  def _traffic(self) -> traffic_controller_lib.TrafficController:
    if self._traffic_inst is None:
      self._traffic_inst = traffic_controller_lib.TrafficController()
    return self._traffic_inst

  def _release_item_ref(self, item: TrajectoryOrError) -> None:
    """Releases any staged `routed_experts_ref` held by an orphaned item."""
    if (
        self._staging_buffer is not None
        and isinstance(item, datatypes.TrajectoryItem)
        and item.routed_experts_ref is not None
    ):
      self._staging_buffer.release_staged([item.routed_experts_ref.ref_id])

  def _on_episode_task_done(
      self,
      slot_key: str,
      collector: collector_lib.TrajectoryCollectorEngine,
      reservation: staging_buffer_lib.StagingReservation | None,
      future: asyncio.Future[TrajectoryOrError],
      done_task: asyncio.Task[Any],
  ) -> None:
    """Reclaims uncommitted staging quota and active tracking on task exit."""
    if reservation is not None:
      reservation.cancel()
    if self._active_tasks.get(slot_key) is done_task:
      self._active_tasks.pop(slot_key, None)
    if self._active_collectors.get(slot_key) is collector:
      self._active_collectors.pop(slot_key, None)
    if done_task.cancelled() and not future.done():
      future.cancel()

  async def _reserve_for_request(
      self, request: datatypes.RolloutRequest
  ) -> staging_buffer_lib.StagingReservation | None:
    """Reserves worst-case staging bytes when `request.stage_routed_experts` is set."""
    if not request.stage_routed_experts:
      return None
    if self._staging_buffer is None:
      raise RuntimeError(
          "RolloutRequest(stage_routed_experts=True) requires initialized MoE"
          " staging on RolloutManager."
      )
    if self._staging_bytes_per_token is None:
      shape = self.sampler.routed_experts_shape()
      if shape is None:
        raise RuntimeError(
            "RolloutRequest(stage_routed_experts=True) requires a sampler with"
            " MoE routed_experts capture enabled."
        )
      encoding = datatypes.resolve_routed_experts_wire_encoding(
          self._routed_experts_encode_mode, self._routed_experts_num_experts
      )
      self._staging_bytes_per_token = datatypes.routed_experts_bytes_per_token(
          shape[0], shape[1], encoding
      )
    max_resp = request.max_response_length
    if (
        isinstance(max_resp, bool)
        or not isinstance(max_resp, int)
        or max_resp <= 0
    ):
      raise ValueError(
          "RolloutRequest(stage_routed_experts=True) requires positive"
          f" request.max_response_length, got {max_resp!r}."
      )
    if self._max_prompt_length is None or self._max_prompt_length <= 0:
      raise ValueError(
          "RolloutRequest(stage_routed_experts=True) requires positive"
          f" RolloutConfig.max_prompt_length, got {self._max_prompt_length!r}."
      )
    worst_case_bytes = (
        self._max_prompt_length + max_resp
    ) * self._staging_bytes_per_token
    return await self._staging_buffer.reserve(
        worst_case_bytes,
        is_paused_fn=lambda: not self._traffic.is_admission_open(),
    )

  def _stamp_target_policy_version(
      self, request: datatypes.RolloutRequest
  ) -> None:
    """Advances `request.target_policy_version` to the active worker version."""
    sampler_version = getattr(self.sampler, "policy_version", None)
    effective_version = self._policy_version
    if (
        isinstance(sampler_version, int)
        and not isinstance(sampler_version, bool)
        and sampler_version > effective_version
    ):
      effective_version = sampler_version
    if effective_version > int(request.target_policy_version or 0):
      request.target_policy_version = effective_version

  def _build_collector(
      self, traj_id: str, request: datatypes.RolloutRequest
  ) -> collector_lib.TrajectoryCollectorEngine:
    """Instantiates environment, agent, and `TrajectoryCollectorEngine`."""
    env_config = dict(self.env_config)
    env_config.update(request.metadata.get("env_config", {}))
    env_config.setdefault("group_index", request.group_index)
    env_config.setdefault("policy_version", request.target_policy_version)
    env_client = self.env_factory(**env_config)
    agent = self.agent_factory(
        **request.metadata.get("agent_config", self.agent_config)
    )
    collector_kwargs: dict[str, Any] = {
        "routed_experts_encode_mode": self._routed_experts_encode_mode,
        "routed_experts_num_experts": self._routed_experts_num_experts,
    }
    if request.stage_routed_experts:
      collector_kwargs["max_prompt_length"] = self._max_prompt_length
    if self.trajectory_store is not None:
      collector_kwargs["trajectory_store"] = self.trajectory_store
    if self._store_routed_experts:
      collector_kwargs["store_routed_experts"] = True
    return collector_lib.TrajectoryCollectorEngine(
        traj_id=traj_id,
        request=request,
        sampler=self.sampler,
        env_client=env_client,
        agent=agent,
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
        eos_ids=self.eos_ids,
        partial_rollout=self._partial_rollout,
        policy_version_fn=lambda: self._policy_version,
        **collector_kwargs,
    )

  async def _generate_one(
      self,
      request: datatypes.RolloutRequest,
      on_complete: Optional[Callable[[TrajectoryOrError], None]] = None,
  ) -> TrajectoryOrError:
    """Spawns an async task running the multi-turn episode loop concurrently."""
    if self._traffic.state == datatypes.WorkerState.STOPPED:
      raise traffic_controller_lib.AdmissionClosedError(
          "rollout worker is stopped"
      )
    reservation = await self._reserve_for_request(request)
    try:
      await self._traffic.wait_for_admission()
      if self._traffic.state == datatypes.WorkerState.STOPPED:
        raise traffic_controller_lib.AdmissionClosedError(
            "rollout worker is stopped"
        )
      if reservation is not None and not reservation.attach_to_task():
        raise traffic_controller_lib.AdmissionClosedError(
            "rollout staging buffer is clearing or reservation was revoked"
        )
      self._stamp_target_policy_version(request)
      loop = asyncio.get_running_loop()
      future: asyncio.Future[TrajectoryOrError] = loop.create_future()

      def _resolve(result: TrajectoryOrError) -> None:
        if future.done():
          self._release_item_ref(result)
          return
        if on_complete:
          on_complete(result)
        future.set_result(result)

      traj_id = request.traj_id
      collector = self._build_collector(traj_id, request)
      slot_key = traj_id
      if (
          traj_id in self._active_tasks
          or traj_id in self._active_collectors
      ):
        slot_key = f"{traj_id}#{id(collector):x}"
      self._active_collectors[slot_key] = collector
      owned_reservation, reservation = reservation, None
      task = asyncio.create_task(
          self._run_and_enqueue(
              collector,
              request,
              owned_reservation,
              _resolve,
              slot_key=slot_key,
          )
      )
      task.add_done_callback(
          functools.partial(
              self._on_episode_task_done,
              slot_key,
              collector,
              owned_reservation,
              future,
          )
      )
      self._active_tasks[slot_key] = task
      self._traffic.track(task)
    except BaseException:
      if reservation is not None:
        reservation.cancel()
      raise

    try:
      return await future
    except BaseException:
      collector.cancel()
      task.cancel()
      try:
        await task
      except BaseException:  # pylint: disable=broad-exception-caught
        pass
      if owned_reservation is not None:
        owned_reservation.cancel()
      if (
          future.done()
          and not future.cancelled()
          and future.exception() is None
      ):
        self._release_item_ref(future.result())
      raise

  async def generate(
      self,
      requests: datatypes.RolloutRequest | Sequence[datatypes.RolloutRequest],
      on_complete: Optional[Callable[[TrajectoryOrError], None]] = None,
  ) -> TrajectoryOrError | Sequence[TrajectoryOrError]:
    """Dispatches 1 or N requests concurrently to the internal Collector Engine pool."""
    if isinstance(requests, datatypes.RolloutRequest):
      return await self._generate_one(requests, on_complete=on_complete)
    tasks = [
        asyncio.create_task(self._generate_one(req, on_complete=on_complete))
        for req in requests
    ]
    try:
      return await asyncio.gather(*tasks)
    except BaseException:
      for t in tasks:
        if not t.done():
          t.cancel()
      results = await asyncio.gather(*tasks, return_exceptions=True)
      for res in results:
        if not isinstance(res, BaseException):
          self._release_item_ref(res)
      raise

  async def _run_and_enqueue(
      self,
      collector: collector_lib.TrajectoryCollectorEngine,
      request: datatypes.RolloutRequest,
      reservation: staging_buffer_lib.StagingReservation | None,
      resolve_cb: Callable[[TrajectoryOrError], None],
      *,
      slot_key: str,
  ) -> None:
    """Runs episode loop, removes active tracking, and resolves callbacks/streams."""
    run_start = time.perf_counter()
    trajectory: TrajectoryOrError
    try:
      item = await collector.run_episode()
      if isinstance(item.traj, dict):
        item.traj["execution_time_s"] = time.perf_counter() - run_start
      if reservation is not None:
        assert self._staging_buffer is not None
        self._staging_buffer.stage_trajectory_item(
            reservation, item, traj_id=collector.traj_id
        )
      trajectory = item
    except Exception as e:  # pylint: disable=broad-exception-caught
      if reservation is not None:
        reservation.cancel()
      error_metadata = dict(request.metadata or {})
      error_metadata.pop("env_config", None)
      error_metadata["prompt_id"] = request.prompt_id
      error_metadata["group_index"] = request.group_index
      error_metadata["policy_version"] = int(request.target_policy_version or 0)
      trajectory = trajectory_lib.TrajectoryError(
          trajectory_id=collector.traj_id,
          prompt_id=request.prompt_id,
          error_message=str(e),
          error_type=type(e).__name__,
          metadata=error_metadata,
      )
    finally:
      if reservation is not None:
        reservation.cancel()
      if self._active_collectors.get(slot_key) is collector:
        self._active_collectors.pop(slot_key, None)
      if self._active_tasks.get(slot_key) is asyncio.current_task():
        self._active_tasks.pop(slot_key, None)

    resolve_cb(trajectory)

  async def migrate_straggler(
      self,
      trajectory_id: str,
      source_server_id: str,
      target_server_id: str,
  ) -> bool:
    """Migrates an active long-tail trajectory using Raiden P2P KV transfer."""
    collector = self._active_collectors.get(trajectory_id)
    if not collector or collector.is_done:
      return False
    if not collector.is_paused:
      raise RuntimeError(
          f"Collector [{trajectory_id}] must be paused before KV migration."
      )

    token_ids = collector.get_accumulated_token_ids()
    return await self.sampler.migrate_kv_cache(
        route_key=trajectory_id,
        source_server_id=source_server_id,
        target_server_id=target_server_id,
        token_ids=token_ids,
    )

  def pause_all(self) -> None:
    for collector in self._active_collectors.values():
      collector.pause()

  def resume_all(self) -> None:
    for collector in self._active_collectors.values():
      collector.resume()

  def cancel_all(self) -> None:
    for collector in self._active_collectors.values():
      collector.cancel()
    for task in self._active_tasks.values():
      task.cancel()

  async def pre_weight_sync(
      self,
      sync_request: sampler_lib.WeightSyncRequest,
      **kwargs,
  ) -> Any:
    """Phase 3 Barrier 1: Closes admission and drains or pauses in-flight work."""
    t0 = time.monotonic()
    self._traffic.transition_to_syncing()
    in_flight_before = len(self._active_collectors)
    if not self._partial_rollout:
      await self._traffic.drain(self._drain_timeout_s)
    self.pause_all()
    res = await self.sampler.pre_weight_sync(sync_request, **kwargs)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    logging.info(
        "[RolloutManager] pre_weight_sync(partial_rollout=%s,"
        " in_flight_before=%d, in_flight_after=%d) completed in %.2f ms",
        self._partial_rollout,
        in_flight_before,
        len(self._active_collectors),
        elapsed_ms,
    )
    return res

  async def weight_sync(
      self,
      sync_request: sampler_lib.WeightSyncRequest,
      **kwargs,
  ) -> int:
    """Phase 3 Barrier 2: Installs the staged weights.

    Returns the round's policy version. `post_weight_sync` records it once the
    weights go live.
    """
    await self.sampler.weight_sync(sync_request, **kwargs)
    return sync_request.policy_version

  async def post_weight_sync(
      self,
      sync_request: sampler_lib.WeightSyncRequest,
      **kwargs,
  ) -> Any:
    """Phase 3 Barrier 3: Finalizes policy weight update and resumes collectors."""
    self._policy_version = sync_request.policy_version

    res = await self.sampler.post_weight_sync(sync_request, **kwargs)
    if self._staging_buffer is not None:
      self._staging_buffer.on_post_weight_sync(
          self._policy_version,
          min_retained_policy_version=sync_request.min_retained_policy_version,
      )

    self.resume_all()
    self._traffic.reopen()
    logging.info(
        "[RolloutManager] post_weight_sync(partial_rollout=%s,"
        " resumed_collectors=%d) completed",
        self._partial_rollout,
        len(self._active_collectors),
    )
    return res

  async def abort_weight_sync(
      self,
      sync_request: sampler_lib.WeightSyncRequest,
      **kwargs,
  ) -> Any:
    """Discards the round, delegates to the sampler, and resumes serving."""
    res = await self.sampler.abort_weight_sync(sync_request, **kwargs)
    # TODO(tunix-dev): It might be better to fail hard if weight sync failed
    # right now instead of letting it proceed silently, otherwise it may mess
    # up with the policy version.
    self.resume_all()
    self.reopen_admission()
    return res

  def reopen_admission(self) -> bool:
    """Reopens rollout admission after an aborted round."""
    return self._traffic.reopen()

  async def bind_weight_sync(self, **kwargs) -> Any:
    """Binds the sampler's destination-side transport for this round."""
    return await self.sampler.bind_weight_sync(**kwargs)

  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    """Returns the sampler's transport metadata for weight sync registration."""
    if self.sampler:
      return await self.sampler.get_weight_sync_metadata(**kwargs)
    return []

  def get_target_state(self) -> Any:
    """Returns the sampler target-state skeleton used for trainer conversion."""
    if self.sampler is None:
      raise RuntimeError("RolloutManager has no sampler configured.")
    return self.sampler.get_target_state()
