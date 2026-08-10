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
import logging
from typing import Any, AsyncIterator, Callable, Dict, Optional, Sequence, Union
from tunix.experimental.common import datatypes
from tunix.experimental.rl.agentic import registry
from tunix.experimental.rollout import collector as collector_lib
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.rollout import vanilla_sampler_adapter
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl.rollout import base_rollout

TrajectoryOrError = Union[
    trajectory_lib.Trajectory, trajectory_lib.TrajectoryError
]


class RolloutManager:
  """Internal trajectory and concurrency control core of RolloutWorker.

  Manages active TrajectoryCollectorEngine tasks, out-of-order completion
  streaming, and straggler KV-cache migration across SamplerServer slices.
  """

  def __init__(
      self,
      config: Optional[base_rollout.RolloutConfig] = None,
      sampler: Optional[sampler_lib.Sampler] = None,
      env_pool: Any = None,
      agent_factory: Optional[Callable[[], Any]] = None,
      max_concurrency: int = 64,
      tokenizer: Any = None,
      chat_parser: Any = None,
  ):
    self.config = config
    if sampler is None:
      sampler_type = getattr(config, "sampler_type", "vanilla")
      if sampler_type == "vllm":
        raise NotImplementedError(
            "vLLM sampler is not implemented yet. Use 'legacy_vllm' or"
            " 'vanilla'."
        )
      elif sampler_type == "legacy_vllm":
        from tunix.experimental.rollout import legacy_vllm_sampler_adapter  # pylint: disable=g-import-not-at-top

        sampler = legacy_vllm_sampler_adapter.LegacyVllmSamplerAdapter(  # pyrefly: ignore[bad-instantiation]
            server_id="legacy_vllm_sampler",
        )
      elif sampler_type == "vanilla":
        sampler = vanilla_sampler_adapter.VanillaSamplerAdapter(  # pyrefly: ignore[bad-instantiation]
            server_id="vanilla_sampler",
        )
      else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")
      sampler.initialize()

    if not isinstance(sampler, sampler_lib.Sampler):
      raise TypeError(
          f"Expected object implementing Sampler Protocol, got {type(sampler)}"
      )
    self.sampler = sampler
    self.env_pool = env_pool
    self.agent_factory = agent_factory
    self.max_concurrency = max_concurrency
    self.tokenizer = tokenizer
    self.chat_parser = chat_parser
    if self.tokenizer is None or self.chat_parser is None:
      raise ValueError(
          "RolloutManager requires valid tokenizer and chat_parser arguments"
          " (none can be None)."
      )

    self._active_collectors: Dict[
        str, collector_lib.TrajectoryCollectorEngine
    ] = {}
    self._active_tasks: Dict[str, asyncio.Task[Any]] = {}
    self._completed_queue: asyncio.Queue[TrajectoryOrError] = asyncio.Queue()
    # Weight-sync admission gate. The lock makes "check gate + register
    # task" (in _generate_one) atomic against "close gate + snapshot" (in
    # pre_weight_sync); without it a request can slip between the check and
    # the snapshot and run while the KV cache is being freed.
    self._admission_open = asyncio.Event()
    self._admission_open.set()
    self._admission_lock = asyncio.Lock()
    self._closed = False

  async def _generate_one(
      self,
      request: datatypes.RolloutRequest,
      on_complete: Optional[Callable[[TrajectoryOrError], None]] = None,
  ) -> TrajectoryOrError:
    """Spawns an async task running the multi-turn episode loop concurrently."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[TrajectoryOrError] = loop.create_future()

    def _resolve(result: TrajectoryOrError) -> None:
      # Settle the caller FIRST: a raising on_complete must never strand
      # generate() forever.
      if not future.done():
        future.set_result(result)
      if on_complete:
        try:
          on_complete(result)
        except Exception:  # pylint: disable=broad-exception-caught
          logging.exception("on_complete callback raised; result delivered")

    traj_id = request.traj_id

    # Cheap early reject BEFORE acquiring an env or building a collector;
    # the authoritative (atomic) check below re-verifies under the lock.
    if self._closed or not self._admission_open.is_set():
      raise RuntimeError(
          "weight sync in progress or manager closed; admission is closed"
      )

    env_name = getattr(self.config, "env_name", "")
    if env_name and registry.ENV_REGISTRY.contains(env_name):
      env_cls = registry.ENV_REGISTRY.get(env_name)
      env_config = request.metadata.get(
          "env_config", getattr(self.config, "env_config", {})
      )
      env_client = env_cls(**env_config)
    elif self.env_pool and hasattr(self.env_pool, "acquire_env"):
      env_client = self.env_pool.acquire_env(request.metadata.get("env_config"))
    else:
      env_client = None

    agent_name = getattr(self.config, "agent_name", "")
    if agent_name and registry.AGENT_REGISTRY.contains(agent_name):
      agent_cls = registry.AGENT_REGISTRY.get(agent_name)
      agent_config = getattr(self.config, "agent_config", {})
      agent = agent_cls(**agent_config)
    elif self.agent_factory and callable(self.agent_factory):
      agent = self.agent_factory()
    else:
      agent = None

    collector = collector_lib.TrajectoryCollectorEngine(
        traj_id=traj_id,
        request=request,
        sampler=self.sampler,
        env_client=env_client,
        agent=agent,
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
    )

    # Gate check and task registration are ATOMIC against pre_weight_sync's
    # close-and-snapshot; the episode itself runs outside the lock.
    def _release_env():
      if (
          self.env_pool
          and hasattr(self.env_pool, "release_env")
          and env_client is not None
      ):
        self.env_pool.release_env(env_client)

    async with self._admission_lock:
      if self._closed or not self._admission_open.is_set():
        _release_env()  # the env was acquired above; do not leak it
        raise RuntimeError(
            "weight sync in progress or manager closed; admission is closed"
            " until post/abort"
        )
      existing = self._active_tasks.get(traj_id)
      if existing is not None and not existing.done():
        _release_env()
        raise ValueError(
            f"traj_id {traj_id!r} already has an active task; overwriting it"
            " would hide a running collector from the weight-sync drain"
        )
      self._active_collectors[traj_id] = collector
      task = asyncio.create_task(
          self._run_and_enqueue(collector, request, _resolve)
      )

      def _settle_if_never_ran(finished: asyncio.Task) -> None:
        """Covers the window where the task is cancelled BEFORE its body

        runs: the coroutine's own CancelledError handler never executes, so
        nothing would clean the maps or settle the caller.
        """
        # Identity-checked, not a blind pop: the caller is woken by
        # set_result BEFORE this callback runs, so it may already have
        # registered a NEW episode under the same traj_id. Removing that one
        # would hide a live collector from the weight-sync drain.
        if self._active_tasks.get(traj_id) is finished:
          del self._active_tasks[traj_id]
        if self._active_collectors.get(traj_id) is collector:
          del self._active_collectors[traj_id]
        if future.done():
          return
        cancelled = trajectory_lib.TrajectoryError(
            trajectory_id=traj_id,
            prompt_id=request.prompt_id,
            error_message="cancelled before the episode started",
            error_type="CancelledError",
        )
        _resolve(cancelled)
        # The stream is a second consumer, not a mirror of the future: a
        # result that never reaches the queue makes as_completed_stream()
        # wait forever for an episode that is already over. put_nowait
        # because a done-callback cannot await, and the queue is unbounded.
        self._completed_queue.put_nowait(cancelled)

      task.add_done_callback(_settle_if_never_ran)
      self._active_tasks[traj_id] = task

    return await future

  async def generate(
      self,
      requests: (
          datatypes.RolloutRequest
          | Sequence[datatypes.RolloutRequest]
          | Any
          | Sequence[Any]
      ),
      on_complete: Optional[Callable[[TrajectoryOrError], None]] = None,
  ) -> TrajectoryOrError | Sequence[TrajectoryOrError] | Any:
    """Dispatches 1 or N requests concurrently to the internal Collector Engine pool."""
    if isinstance(requests, (list, tuple)):
      tasks = [
          asyncio.create_task(self._generate_one(req, on_complete=on_complete))
          for req in requests
      ]
      return await asyncio.gather(*tasks)
    return await self._generate_one(requests, on_complete=on_complete)  # pyrefly: ignore[bad-argument-type]

  async def _run_and_enqueue(
      self,
      collector: collector_lib.TrajectoryCollectorEngine,
      request: datatypes.RolloutRequest,
      resolve_cb: Callable[[TrajectoryOrError], None],
  ) -> None:
    """Runs episode loop, removes active tracking, and resolves callbacks/streams."""
    try:
      trajectory: TrajectoryOrError = await collector.run_episode()
    except asyncio.CancelledError:
      # cancel_all()/stop() cancels this task; the caller's future must
      # STILL be resolved or the original generate() hangs forever.
      trajectory = trajectory_lib.TrajectoryError(
          trajectory_id=collector.traj_id,
          prompt_id=request.prompt_id,
          error_message="cancelled (manager stopping)",
          error_type="CancelledError",
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      trajectory = trajectory_lib.TrajectoryError(
          trajectory_id=collector.traj_id,
          prompt_id=request.prompt_id,
          error_message=str(e),
          error_type=type(e).__name__,
      )
    finally:
      # Identity-checked for the same reason as the task's done-callback: a
      # traj_id re-used by a caller this episode already woke belongs to the
      # new episode, and the finished one must not remove it.
      traj_id = collector.traj_id
      task = asyncio.current_task()
      if self._active_tasks.get(traj_id) is task:
        del self._active_tasks[traj_id]
      if self._active_collectors.get(traj_id) is collector:
        del self._active_collectors[traj_id]
      try:
        if (
            self.env_pool
            and hasattr(self.env_pool, "release_env")
            and collector.env is not None
        ):
          self.env_pool.release_env(collector.env)
      except Exception:  # pylint: disable=broad-exception-caught
        # env release must never strand the caller's future below.
        pass

    # Resolve the caller FIRST (synchronous, cannot be interrupted by a
    # pending cancellation), then feed the stream.
    try:
      resolve_cb(trajectory)
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    await self._completed_queue.put(trajectory)

  async def pop_next_completed(self) -> TrajectoryOrError:
    """Pull-based stream: yields whichever trajectory finishes first out-of-order."""
    return await self._completed_queue.get()

  async def as_completed_stream(
      self,
  ) -> AsyncIterator[TrajectoryOrError]:
    """Async generator yielding completed trajectories strictly out-of-order."""
    # TODO(lancewang): Add termination condition to prevent hangs when stream
    # is exhausted.
    while True:
      yield await self.pop_next_completed()

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
    sampler = self.sampler
    if not sampler:
      return False
    return await sampler.migrate_kv_cache(
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
    # Gate first, cancellation second, and every cancel best-effort: a
    # collector that raises on cancel() must not be able to leave a stopped
    # manager admitting requests.
    self._closed = True  # the gate never reopens after stop
    self._admission_open.clear()
    for collector in list(self._active_collectors.values()):
      try:
        collector.cancel()
      except Exception:  # pylint: disable=broad-except
        logging.exception("collector.cancel() raised during stop")
    for task in list(self._active_tasks.values()):
      task.cancel()

  def reopen_admission(self) -> None:
    """Resumes collectors and admits traffic again after a weight-sync round.

    Deliberately NOT done by post/abort: admission may only reopen once the
    caller has positively confirmed the round reached a serving terminal.
    Reopening as soon as the sampler call returns admits requests over a
    rollback that may still turn out to have failed.
    """
    self.resume_all()
    if not self._closed:  # a stopped manager never reopens
      self._admission_open.set()

  async def pre_weight_sync(
      self, sync_request: sampler_lib.WeightSyncRequest | Any = None, **kwargs
  ) -> Any:
    """Closes admission, drains in-flight collectors to a barrier, then

    quiesces the sampler. The gate reopens only in post/abort.
    """
    async with self._admission_lock:  # close gate + snapshot, atomically
      self._admission_open.clear()
      active = [t for t in self._active_tasks.values() if not t.done()]
    if active:
      _, pending = await asyncio.wait(
          active, timeout=kwargs.pop("drain_timeout_s", 120.0)
      )
      if pending:
        # FAIL CLOSED, and deliberately WITHOUT cancelling. Cancelling is
        # now survivable (_run_and_enqueue resolves the caller on
        # CancelledError), so this is a product choice rather than a
        # necessity: a straggler is a request a client is waiting on, and
        # killing it to make a weight sync fit is the wrong trade. Failing
        # pre lets the coordinator abort; admission reopens on the OLD
        # weights and the stragglers finish normally.
        raise RuntimeError(
            f"{len(pending)} collector task(s) still running after the"
            " drain timeout; refusing to quiesce over live requests"
        )
    # pause AFTER the drain, deliberately: today collector.pause() is a
    # flag, but if it ever truly suspends execution, pausing first would
    # make the drain above deadlock by construction.
    self.pause_all()
    sampler = self.sampler
    if sampler is None:
      return None
    return await sampler.pre_weight_sync(sync_request, **kwargs)

  async def weight_sync(
      self, sync_request: sampler_lib.WeightSyncRequest | Any = None, **kwargs
  ) -> Any:
    """Phase 3 Barrier 2: Executes weight synchronization and resumes collectors."""
    completed_version = getattr(sync_request, "policy_version", 0)
    sampler = self.sampler
    if sampler is not None:
      res = await sampler.weight_sync(sync_request, **kwargs)
      if res is not None:
        completed_version = res
    return completed_version

  async def post_weight_sync(
      self, sync_request: sampler_lib.WeightSyncRequest | Any = None, **kwargs
  ) -> Any:
    """Finalizes the publish.

    Admission stays closed until the caller confirms the publish is serving and
    calls reopen_admission().
    """
    res = None
    sampler = self.sampler
    if sampler is not None:
      res = await sampler.post_weight_sync(sync_request, **kwargs)
    return res

  @staticmethod
  def _weight_sync_capable(
      sampler: Any,
  ) -> sampler_lib.WeightSyncCapableSampler:
    """Returns the sampler as a capable one, or fails diagnosably.

    Checked on EVERY capability entry point, not just bind: a sampler with
    only some of the three methods would otherwise pass bind, quiesce on
    pre, and then strand itself when abort turned out to be missing.

    It RETURNS the sampler rather than just asserting, because the callers
    need the narrowed type: `Sampler` and the adapters deliberately do not
    carry these three methods, so a checker looking at the field's declared
    type is right to reject the calls until the capability is established
    here.
    """
    if not isinstance(sampler, sampler_lib.WeightSyncCapableSampler):
      raise TypeError(
          f"sampler {type(sampler).__name__} does not implement the"
          " weight-sync capability (bind/abort/get_round)"
      )
    return sampler

  async def bind_weight_sync(self, **kwargs) -> Any:
    """Binds the sampler's transport synchronizer. Idempotent, no downtime."""
    sampler = self.sampler
    if sampler is None:
      return None
    return await self._weight_sync_capable(sampler).bind_weight_sync(**kwargs)

  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    """Transport metadata for this sampler, one entry per physical host."""
    sampler = self.sampler
    if sampler is None:
      return None
    return await sampler.get_weight_sync_metadata(**kwargs)

  async def abort_weight_sync(
      self, sync_request: sampler_lib.WeightSyncRequest | Any = None, **kwargs
  ) -> Any:
    """Rolls the sampler back. Admission stays closed.

    The sampler's abort returning is not proof the rollback happened; only
    the round report is, and that is checked one level up. Reopening here
    would admit requests over a rollback that still has to be confirmed, so
    the caller reopens via reopen_admission() once it is.
    """
    res = None
    sampler = self.sampler
    if sampler is not None:
      res = await self._weight_sync_capable(sampler).abort_weight_sync(
          sync_request, **kwargs
      )
    return res

  async def get_weight_sync_round(self, **kwargs) -> Any:
    sampler = self.sampler
    if sampler is None:
      return None
    return await self._weight_sync_capable(sampler).get_weight_sync_round(
        **kwargs
    )
