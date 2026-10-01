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

"""Unit tests for RolloutFaultToleranceManager, DistributedRLEngine failover, and ClusterOrchestrator rejoin."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any
from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.orchestrator import fault_tolerance
from tunix.experimental.orchestrator import orchestrator
from tunix.experimental.worker import remote_execution


class FakeRolloutActor(remote_execution.ActorHandle):
  """Controllable fake rollout ActorHandle for fault tolerance testing."""

  def __init__(self, worker_id: str):
    self.worker_id = worker_id
    self.target_address = f"localhost:{worker_id}"
    self.dispatched_requests: list[datatypes.RolloutRequest] = []
    self.queued_responses: list[Any] = []
    self.fail_dispatch: Exception | None = None
    self.fail_poll: Exception | None = None
    self.fail_heartbeat: Exception | None = None
    self.target_state: dict[str, Any] = {"weights": 1}
    self.lifecycle_calls: list[str] = []

  def submit(
      self, method_name: str | None = None, *args: Any, **kwargs: Any
  ) -> Any:
    del args, kwargs
    if method_name is not None:
      self.lifecycle_calls.append(method_name)
    return None

  async def asubmit(
      self, method_name: str | None = None, *args: Any, **kwargs: Any
  ) -> Any:
    del args, kwargs
    if method_name == "heartbeat":
      if self.fail_heartbeat is not None:
        raise self.fail_heartbeat
      return True
    if method_name == "get_target_state":
      if self.fail_dispatch is not None:
        raise self.fail_dispatch
      return self.target_state
    if method_name == "set_target_state":
      return True
    return None

  async def dispatch_task(
      self,
      request_id: str | None = None,
      method_name: str | None = None,
      *args: Any,
      **kwargs: Any,
  ) -> str:
    del method_name, args
    if self.fail_dispatch is not None:
      raise self.fail_dispatch
    requests = kwargs.get("requests", [])
    self.dispatched_requests.extend(requests)
    return request_id or "task_ok"

  async def poll_responses(
      self, timeout_s: float = 0.0
  ) -> remote_execution.ExecutionResponse | list[Any] | None:
    del timeout_s
    if self.fail_poll is not None:
      raise self.fail_poll
    if not self.queued_responses:
      return None
    out = list(self.queued_responses)
    self.queued_responses.clear()
    return out


def _make_response(
    prompt_id: str,
    group_index: int = 0,
    status: datatypes.TrajectoryStatus = datatypes.TrajectoryStatus.SUCCEEDED,
    worker_id: str = "w0",
) -> datatypes.RolloutResponse:
  item = datatypes.TrajectoryItem(
      prompt_id=prompt_id,
      group_index=group_index,
      traj={
          "status": status,
          "trajectory_reward": 1.0,
          "prompt_tokens": np.array([1, 2], dtype=np.int32),
          "conversation_tokens": np.array([1, 2, 3], dtype=np.int32),
          "conversation_masks": np.array([0.0, 0.0, 1.0], dtype=np.float32),
          "old_logprobs": np.array([0.0, 0.0, -0.1], dtype=np.float32),
      },
      metadata={"worker_id": worker_id},
  )
  return datatypes.RolloutResponse(
      request_id=f"req_{prompt_id}_g{group_index}",
      payload=item,
      error=None,
  )


@dataclasses.dataclass
class _FakeSyncResult:
  policy_version: int
  workers: list[Any]


class _FakeWeightSyncCoordinator:

  def __init__(self, synced_workers_fn):
    self._synced_workers_fn = synced_workers_fn
    self.sync_calls: list[int] = []

  async def sync(self, policy_version: int) -> _FakeSyncResult:
    self.sync_calls.append(policy_version)
    return _FakeSyncResult(
        policy_version=policy_version,
        workers=list(self._synced_workers_fn()),
    )


class RolloutFaultToleranceTest(absltest.TestCase):

  def test_capacity_gated_retry_queue_respects_max_inflight_per_worker(self):
    """Orphaned trajectories wait in _retry_queue until healthy workers free slots."""
    w0 = FakeRolloutActor("w0")
    w1 = FakeRolloutActor("w1")
    trainer = FakeRolloutActor("trainer")
    cfg = fault_tolerance.RolloutFaultToleranceConfig(
        max_inflight_per_worker=2,
        max_trajectory_retries=3,
    )
    engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[w0, w1],
        trainer_workers={datatypes.Role.ACTOR: trainer},
        fault_tolerance_config=cfg,
    )
    ft = engine.fault_tolerance_manager

    prompts = [{"prompt_id": f"p{i}", "prompt": f"q{i}"} for i in range(4)]
    asyncio.run(engine.dispatch_rollouts(prompts, num_generations=1))

    # Both workers have 2 in-flight trajectories (at max_inflight_per_worker=2).
    self.assertEqual(ft.inflight_count(w0), 2)
    self.assertEqual(ft.inflight_count(w1), 2)
    self.assertEqual(ft.retry_queue_size, 0)
    self.assertEqual(ft.pending_queue_size, 0)

    # Simulate w0 crashing on poll_rollouts; w1 returns 0 completions so far.
    w0.fail_poll = RuntimeError("gRPC UNAVAILABLE: w0 died")
    completed = asyncio.run(engine.poll_rollouts(timeout_s=0.01))
    self.assertEqual(completed, [])

    # w0 is evicted from the pool; its 2 trajectories are queued in _retry_queue
    # and NOT immediately pushed to w1 because w1 is already at capacity (2/2).
    self.assertNotIn(w0, engine._rollout_pool)
    self.assertEqual(
        ft.worker_status(w0), fault_tolerance.WorkerHealthStatus.FAILED
    )
    self.assertEqual(ft.inflight_count(w1), 2)
    self.assertEqual(ft.retry_queue_size, 2)
    self.assertEqual(ft.pending_queue_size, 2)
    self.assertLen(w1.dispatched_requests, 2)

    # Now 1 trajectory completes on w1 -> frees 1 slot (1/2) -> 1 retry drains!
    first_w1_req = w1.dispatched_requests[0]
    w1.queued_responses.append(
        _make_response(
            first_w1_req.prompt_id, first_w1_req.group_index, worker_id="w1"
        )
    )
    completed_step1 = asyncio.run(engine.poll_rollouts(timeout_s=0.01))
    self.assertLen(completed_step1, 1)
    self.assertEqual(completed_step1[0].traj_id, first_w1_req.traj_id)
    self.assertEqual(ft.retry_queue_size, 1)
    self.assertEqual(ft.inflight_count(w1), 2)
    self.assertLen(w1.dispatched_requests, 3)

    # Next trajectory completes on w1 -> frees 1 slot -> last retry drains!
    second_w1_req = w1.dispatched_requests[1]
    w1.queued_responses.append(
        _make_response(
            second_w1_req.prompt_id, second_w1_req.group_index, worker_id="w1"
        )
    )
    completed_step2 = asyncio.run(engine.poll_rollouts(timeout_s=0.01))
    self.assertLen(completed_step2, 1)
    self.assertEqual(ft.retry_queue_size, 0)
    self.assertEqual(ft.inflight_count(w1), 2)
    self.assertLen(w1.dispatched_requests, 4)

    # Complete the remaining 2 retried trajectories on w1.
    for req in w1.dispatched_requests[2:]:
      w1.queued_responses.append(
          _make_response(req.prompt_id, req.group_index, worker_id="w1")
      )
    completed_step3 = asyncio.run(engine.poll_rollouts(timeout_s=0.01))
    self.assertLen(completed_step3, 2)
    self.assertEqual(ft.inflight_count(), 0)

  def test_sticky_past_trajectory_assignments_preserved_across_failure_and_rejoin(
      self,
  ):
    """Past trajectory assignments stay pinned to surviving workers when pool membership changes."""
    w0 = FakeRolloutActor("w0")
    w1 = FakeRolloutActor("w1")
    w2 = FakeRolloutActor("w2")
    pool = remote_execution.RoutingActorPool([w0, w1, w2])
    ft = fault_tolerance.RolloutFaultToleranceManager(pool)

    reqs = [
        datatypes.RolloutRequest(
            request_id=f"r{i}",
            prompt=f"p{i}",
            prompt_id=f"p{i}",
            group_index=0,
        )
        for i in range(6)
    ]
    initial_assignments: dict[str, FakeRolloutActor] = {}
    for req in reqs:
      chosen = ft.select_worker(req)
      self.assertIsNotNone(chosen)
      ft.record_dispatch(req, chosen)  # pytype: disable=wrong-arg-types
      initial_assignments[req.traj_id] = chosen  # pytype: disable=container-type-mismatch

    # Complete the initial attempts so inflight counts return to 0, while
    # past trajectory placements remain recorded in the pool.
    for w in (w0, w1, w2):
      items = [
          _make_response(r.prompt_id, r.group_index, worker_id=w.worker_id).payload
          for r in reqs
          if initial_assignments[r.traj_id] is w
      ]
      ft.filter_and_complete_responses(w, items)  # pytype: disable=wrong-arg-types

    # Evict w0: trajectories previously placed on w1 or w2 MUST stay on w1/w2,
    # while trajectories previously placed on w0 are reassigned to w1 or w2.
    ft.evict_rollout_worker(w0, reason="w0 crashed")
    for req in reqs:
      prev = initial_assignments[req.traj_id]
      current = ft.select_worker(req)
      if prev in (w1, w2):
        self.assertIs(current, prev)
      else:
        self.assertIn(current, (w1, w2))

    # Rejoin w0 and promote after weight sync: trajectories on w1/w2 still do
    # not reshuffle back to w0.
    rejoined_w0 = FakeRolloutActor("w0")
    ft.on_worker_rejoined("w0", rejoined_w0)
    ft.on_weight_sync_completed([rejoined_w0, w1, w2])
    for req in reqs:
      prev = initial_assignments[req.traj_id]
      if prev in (w1, w2):
        self.assertIs(ft.select_worker(req), prev)

  def test_rc1_zombie_worker_duplicate_completion_is_dropped(self):
    """Late completion from an evicted zombie worker is deduplicated (RC-1)."""
    w0 = FakeRolloutActor("w0")
    w1 = FakeRolloutActor("w1")
    trainer = FakeRolloutActor("trainer")
    engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[w0, w1],
        trainer_workers={datatypes.Role.ACTOR: trainer},
    )
    ft = engine.fault_tolerance_manager

    asyncio.run(
        engine.dispatch_rollouts([{"prompt_id": "p_zombie", "prompt": "hi"}])
    )
    initial_worker = w0 if w0.dispatched_requests else w1
    backup_worker = w1 if initial_worker is w0 else w0
    req = initial_worker.dispatched_requests[0]

    # Evict initial_worker and drain retry to backup_worker
    ft.evict_rollout_worker(initial_worker, reason="simulated timeout")
    asyncio.run(ft.drain_retry_queue())
    self.assertLen(backup_worker.dispatched_requests, 1)

    # Backup worker finishes attempt #2
    backup_item = _make_response(
        req.prompt_id, req.group_index, worker_id=backup_worker.worker_id
    ).payload
    assert backup_item is not None
    accepted_first = ft.filter_and_complete_responses(
        backup_worker, [backup_item]
    )
    self.assertLen(accepted_first, 1)

    # Zombie initial_worker belatedly returns attempt #1 -> must be dropped!
    zombie_item = _make_response(
        req.prompt_id, req.group_index, worker_id=initial_worker.worker_id
    ).payload
    assert zombie_item is not None
    accepted_zombie = ft.filter_and_complete_responses(
        initial_worker, [zombie_item]
    )
    self.assertEqual(accepted_zombie, [])

  def test_suspect_temporarily_down_recovery_and_eviction_threshold(self):
    """1..2 missed heartbeats mark SUSPECT ('temporarily down'); 3rd miss evicts."""
    w0 = FakeRolloutActor("w0")
    w1 = FakeRolloutActor("w1")
    pool = remote_execution.RoutingActorPool([w0, w1])
    cfg = fault_tolerance.RolloutFaultToleranceConfig(
        max_consecutive_heartbeat_misses=3
    )
    ft = fault_tolerance.RolloutFaultToleranceManager(pool, config=cfg)

    req = datatypes.RolloutRequest(
        request_id="r1", prompt="p", prompt_id="p1", group_index=0
    )
    ft.record_dispatch(req, w0)

    # Miss 1 & 2 -> SUSPECT (still in pool, in-flight preserved, but deprioritized)
    w0.fail_heartbeat = TimeoutError("miss 1")
    asyncio.run(ft.run_heartbeat_once())
    self.assertEqual(
        ft.worker_status(w0), fault_tolerance.WorkerHealthStatus.SUSPECT
    )
    self.assertIn(w0, pool)
    self.assertEqual(ft.inflight_count(w0), 1)

    asyncio.run(ft.run_heartbeat_once())
    self.assertEqual(
        ft.worker_status(w0), fault_tolerance.WorkerHealthStatus.SUSPECT
    )
    self.assertIn(w0, pool)

    # While w0 is SUSPECT and w1 is HEALTHY, select_worker routes new work to w1
    new_req = datatypes.RolloutRequest(
        request_id="r2", prompt="p2", prompt_id="p2", group_index=0
    )
    self.assertIs(ft.select_worker(new_req), w1)

    # Recovery on heartbeat #3 -> returns to HEALTHY without evicting in-flight!
    w0.fail_heartbeat = None
    asyncio.run(ft.run_heartbeat_once())
    self.assertEqual(
        ft.worker_status(w0), fault_tolerance.WorkerHealthStatus.HEALTHY
    )
    self.assertEqual(ft.inflight_count(w0), 1)
    self.assertEqual(ft.retry_queue_size, 0)

    # Now 3 consecutive misses -> FAILED + evicted + in-flight moved to _retry_queue
    w0.fail_heartbeat = TimeoutError("dead")
    for _ in range(3):
      asyncio.run(ft.run_heartbeat_once())
    self.assertEqual(
        ft.worker_status(w0), fault_tolerance.WorkerHealthStatus.FAILED
    )
    self.assertNotIn(w0, pool)
    self.assertEqual(ft.inflight_count(w0), 0)
    self.assertEqual(ft.retry_queue_size, 1)

  def test_inflight_timeout_retries_then_emits_terminal_failed_item(self):
    """Trajectory exceeding max_trajectory_inflight_s retries then returns FAILED."""
    now = [100.0]
    w0 = FakeRolloutActor("w0")
    pool = remote_execution.RoutingActorPool([w0])
    cfg = fault_tolerance.RolloutFaultToleranceConfig(
        max_trajectory_inflight_s=30.0,
        max_trajectory_retries=2,
    )
    ft = fault_tolerance.RolloutFaultToleranceManager(
        pool, config=cfg, clock=lambda: now[0]
    )
    req = datatypes.RolloutRequest(
        request_id="r_slow", prompt="p", prompt_id="p_slow", group_index=0
    )
    ft.record_dispatch(req, w0)

    # Advance past 30s -> attempt 1 times out and enters _retry_queue
    now[0] = 135.0
    terminal = ft.check_inflight_timeouts()
    self.assertEqual(terminal, [])
    self.assertEqual(ft.retry_queue_size, 1)

    # Drain attempt 2 onto w0
    asyncio.run(ft.drain_retry_queue())
    self.assertEqual(ft.retry_queue_size, 0)
    self.assertEqual(ft.inflight_count(w0), 1)

    # Advance another 35s -> attempt 2 hits max_trajectory_retries=2 -> terminal FAILED!
    now[0] = 170.0
    terminal = ft.check_inflight_timeouts()
    self.assertLen(terminal, 1)
    self.assertEqual(terminal[0].traj_id, "traj_p_slow_g0")
    self.assertEqual(
        terminal[0].traj["status"], datatypes.TrajectoryStatus.FAILED
    )

  def test_zero_healthy_workers_fails_immediately_by_default(self):
    """By default (max_zero_worker_wait_s=0.0), 0 rollout workers fails the job immediately."""
    w0 = FakeRolloutActor("w0")
    trainer = FakeRolloutActor("trainer")
    engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[w0],
        trainer_workers={datatypes.Role.ACTOR: trainer},
    )
    asyncio.run(
        engine.dispatch_rollouts([{"prompt_id": "p0", "prompt": "hello"}])
    )
    w0.fail_poll = RuntimeError("last worker crashed")
    with self.assertRaises(fault_tolerance.NoHealthyRolloutWorkersError):
      asyncio.run(engine.poll_rollouts(timeout_s=0.01))

  def test_zero_healthy_workers_grace_period_and_timeout(self):
    """0 healthy workers waits up to max_zero_worker_wait_s before raising when configured."""
    now = [10.0]
    w0 = FakeRolloutActor("w0")
    pool = remote_execution.RoutingActorPool([w0])
    cfg = fault_tolerance.RolloutFaultToleranceConfig(
        max_zero_worker_wait_s=60.0
    )
    ft = fault_tolerance.RolloutFaultToleranceManager(
        pool, config=cfg, clock=lambda: now[0]
    )
    ft.evict_rollout_worker(w0, reason="crash")

    now[0] = 50.0
    ft.check_zero_worker_timeout()  # 40s < 60s -> no error

    now[0] = 75.0
    with self.assertRaises(fault_tolerance.NoHealthyRolloutWorkersError):
      ft.check_zero_worker_timeout()

  def test_pending_rejoin_promoted_only_after_weight_sync(self):
    """Restarted worker stays in PENDING_REJOIN until sync_weights completes."""
    w0 = FakeRolloutActor("w0")
    w1 = FakeRolloutActor("w1")
    trainer = FakeRolloutActor("trainer")
    rejoined_w0 = FakeRolloutActor("w0")

    synced_workers = [w1]
    coordinator = _FakeWeightSyncCoordinator(lambda: synced_workers)
    engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[w0, w1],
        trainer_workers={datatypes.Role.ACTOR: trainer},
        weight_sync_coordinator=coordinator,
    )
    ft = engine.fault_tolerance_manager

    # Evict w0 and stage rejoined_w0
    ft.evict_rollout_worker(w0, reason="pod restart")
    self.assertLen(engine._rollout_pool, 1)

    ft.register_rejoining_worker(rejoined_w0, worker_id="w0")
    self.assertEqual(
        ft.worker_status("w0"),
        fault_tolerance.WorkerHealthStatus.PENDING_REJOIN,
    )
    # Must NOT be in RoutingActorPool before sync_weights()
    self.assertNotIn(rejoined_w0, engine._rollout_pool)

    # Run sync_weights() including rejoined_w0 -> promoted to HEALTHY in pool!
    synced_workers.append(rejoined_w0)
    version = asyncio.run(engine.sync_weights(policy_version=2))
    self.assertEqual(version, 2)
    self.assertIn(rejoined_w0, engine._rollout_pool)
    self.assertEqual(
        ft.worker_status("w0"),
        fault_tolerance.WorkerHealthStatus.HEALTHY,
    )

  def test_cluster_orchestrator_rollout_rejoin_waits_for_next_weight_sync(self):
    """ClusterOrchestrator.register_worker_handle stages rejoined worker until sync_weights."""
    w0 = FakeRolloutActor("w0")
    w1 = FakeRolloutActor("w1")
    trainer = FakeRolloutActor("trainer")

    orch = orchestrator.ClusterOrchestrator()
    orch.register_worker_handle("trainer", [datatypes.Role.ACTOR], trainer)
    orch.register_worker_handle("w0", [datatypes.Role.ROLLOUT], w0)
    orch.register_worker_handle("w1", [datatypes.Role.ROLLOUT], w1)
    orch.bring_up_workers()
    assert orch.engine is not None

    synced_workers: list[Any] = [w1]
    orch.engine._weight_sync_coordinator = _FakeWeightSyncCoordinator(
        lambda: synced_workers
    )
    ft = orch.engine.fault_tolerance_manager

    # Simulate w0 failing during rollout execution
    ft.evict_rollout_worker(w0, reason="worker w0 died")
    self.assertLen(orch.engine.rollout_pool, 1)

    # w0 restarts and re-registers with ClusterOrchestrator
    rejoined_w0 = FakeRolloutActor("w0")
    orch.register_worker_handle("w0", [datatypes.Role.ROLLOUT], rejoined_w0)

    # Lifecycle bring-up ran on rejoined_w0, and it is staged in PENDING_REJOIN
    self.assertEqual(
        rejoined_w0.lifecycle_calls, ["initialize", "compile", "start"]
    )
    self.assertEqual(
        ft.worker_status("w0"),
        fault_tolerance.WorkerHealthStatus.PENDING_REJOIN,
    )
    self.assertNotIn(rejoined_w0, orch.engine.rollout_pool)

    # On next sync_weights(), rejoined_w0 receives weights and rejoins the pool
    synced_workers.append(rejoined_w0)
    asyncio.run(orch.engine.sync_weights(policy_version=1))
    self.assertIn(rejoined_w0, orch.engine.rollout_pool)
    self.assertEqual(
        ft.worker_status("w0"),
        fault_tolerance.WorkerHealthStatus.HEALTHY,
    )

  def test_mid_dispatch_rpc_exception_transparently_fails_over(self):
    """If worker.dispatch_task raises during dispatch_rollout_requests, it fails over."""
    w0 = FakeRolloutActor("w0")
    w1 = FakeRolloutActor("w1")
    trainer = FakeRolloutActor("trainer")
    w0.fail_dispatch = RuntimeError("Connection reset by peer")

    engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[w0, w1],
        trainer_workers={datatypes.Role.ACTOR: trainer},
    )
    prompts = [{"prompt_id": f"p{i}", "prompt": f"q{i}"} for i in range(4)]
    req_ids = asyncio.run(engine.dispatch_rollouts(prompts, num_generations=1))
    self.assertLen(req_ids, 4)
    self.assertNotIn(w0, engine._rollout_pool)
    self.assertLen(w1.dispatched_requests, 4)

  def test_multi_timeout_on_same_worker_does_not_double_emit_or_requeue(self):
    """Simultaneous timeouts on one worker must not both emit FAILED and re-queue."""
    now = [100.0]
    w0 = FakeRolloutActor("w0")
    pool = remote_execution.RoutingActorPool([w0])
    cfg = fault_tolerance.RolloutFaultToleranceConfig(
        max_trajectory_inflight_s=30.0,
        max_trajectory_retries=1,
    )
    ft = fault_tolerance.RolloutFaultToleranceManager(
        pool, config=cfg, clock=lambda: now[0]
    )
    for i in range(3):
      ft.record_dispatch(
          datatypes.RolloutRequest(
              request_id=f"r_{i}", prompt="p", prompt_id=f"p_{i}", group_index=0
          ),
          w0,
      )

    now[0] = 140.0
    terminal = ft.check_inflight_timeouts()
    self.assertLen(terminal, 3)
    self.assertEqual(ft.retry_queue_size, 0)
    self.assertEqual(ft.inflight_count(), 0)
    self.assertNotIn(w0, pool)

  def test_worker_eviction_does_not_burn_trajectory_retry_budget(self):
    """Worker crashes/evictions increment attempt_id (RC-1) without burning max_trajectory_retries."""
    now = [100.0]
    w0 = FakeRolloutActor("w0")
    w1 = FakeRolloutActor("w1")
    w2 = FakeRolloutActor("w2")
    pool = remote_execution.RoutingActorPool([w0, w1, w2])
    cfg = fault_tolerance.RolloutFaultToleranceConfig(
        max_trajectory_inflight_s=30.0,
        max_trajectory_retries=2,
    )
    ft = fault_tolerance.RolloutFaultToleranceManager(
        pool, config=cfg, clock=lambda: now[0]
    )
    req = datatypes.RolloutRequest(
        request_id="r_evict", prompt="p", prompt_id="p_evict", group_index=0
    )

    # Dispatch #1 on w0 -> w0 crashes (evicted, NOT a trajectory failure)
    ft.record_dispatch(req, w0)
    ft.evict_rollout_worker(w0, reason="crash 1")
    asyncio.run(ft.drain_retry_queue())

    # Dispatch #2 on w1 -> w1 also crashes (evicted, still 0 trajectory failures!)
    ft.evict_rollout_worker(w1, reason="crash 2")
    asyncio.run(ft.drain_retry_queue())
    self.assertEqual(ft.inflight_count(w2), 1)

    # Now on w2, trajectory actually times out once (failure 1/2 < max_trajectory_retries=2).
    # Because worker evictions did not burn retry budget, it must re-queue instead of failing!
    now[0] = 140.0
    terminal = ft.check_inflight_timeouts()
    self.assertEqual(terminal, [])
    self.assertEqual(ft.retry_queue_size, 1)

  def test_poll_rollouts_isolates_unwrap_errors_and_admission_closed(self):
    """ExecutionResponse.unwrap() errors on one worker must not drop other workers' items."""

    class _FailingUnwrapResponse:

      def __init__(self, exc: Exception):
        self._exc = exc

      def unwrap(self):
        raise self._exc

    w0 = FakeRolloutActor("w0")
    w1 = FakeRolloutActor("w1")
    trainer = FakeRolloutActor("trainer")
    engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[w0, w1],
        trainer_workers={datatypes.Role.ACTOR: trainer},
    )
    ft = engine.fault_tolerance_manager

    req0 = datatypes.RolloutRequest(
        request_id="r0", prompt="p0", prompt_id="p0", group_index=0
    )
    req1 = datatypes.RolloutRequest(
        request_id="r1", prompt="p1", prompt_id="p1", group_index=0
    )
    ft.record_dispatch(req0, w0)
    ft.record_dispatch(req1, w1)

    # w0 returns AdmissionClosedError inside ExecutionResponse.unwrap(); w1 returns a valid item
    w0.queued_responses.append(
        _FailingUnwrapResponse(
            RuntimeError("RemoteExecutionError [AdmissionClosedError]: closed")
        )
    )
    w1.queued_responses.append(_make_response("p1", 0, worker_id="w1"))

    completed = asyncio.run(engine.poll_rollouts(timeout_s=0.01))
    # w1's item is preserved and returned; w0 is NOT evicted on AdmissionClosedError,
    # and req0 was re-queued and immediately re-drained!
    self.assertLen(completed, 1)
    self.assertEqual(completed[0].traj_id, "traj_p1_g0")
    self.assertIn(w0, engine._rollout_pool)
    self.assertEqual(ft.inflight_count(), 1)

  def test_on_weight_sync_completed_ignores_failed_worker_round_reports(self):
    """Rejoined worker is NOT promoted if its WorkerRoundReport failed or was not committed."""

    @dataclasses.dataclass
    class _FakeReport:
      worker_id: str
      phase: str = "committed"
      error: str = ""
      needs_restart: bool = False

    w0 = FakeRolloutActor("w0")
    rejoined_w1 = FakeRolloutActor("w1")
    pool = remote_execution.RoutingActorPool([w0])
    ft = fault_tolerance.RolloutFaultToleranceManager(pool)
    ft.register_rejoining_worker(rejoined_w1, worker_id="w1")

    # Report shows w1 failed weight sync -> must stay PENDING_REJOIN
    promoted = ft.on_weight_sync_completed([
        _FakeReport(worker_id="w0", phase="committed"),
        _FakeReport(worker_id="w1", phase="failed", error="timeout"),
    ])
    self.assertEqual(promoted, [])
    self.assertNotIn(rejoined_w1, pool)
    self.assertEqual(
        ft.worker_status("w1"),
        fault_tolerance.WorkerHealthStatus.PENDING_REJOIN,
    )

    # Next round succeeds -> promoted to HEALTHY
    promoted_ok = ft.on_weight_sync_completed([
        _FakeReport(worker_id="w0", phase="committed"),
        _FakeReport(worker_id="w1", phase="committed"),
    ])
    self.assertEqual(promoted_ok, ["w1"])
    self.assertIn(rejoined_w1, pool)

  def test_on_weight_sync_completed_empty_sequence_does_not_promote(self):
    """Passing an empty sequence [] or mapping {} does not promote pending rejoin workers."""
    w0 = FakeRolloutActor("w0")
    rejoined_w1 = FakeRolloutActor("w1")
    pool = remote_execution.RoutingActorPool([w0])
    ft = fault_tolerance.RolloutFaultToleranceManager(pool)
    ft.register_rejoining_worker(rejoined_w1, worker_id="w1")

    self.assertEqual(ft.on_weight_sync_completed([]), [])
    self.assertNotIn(rejoined_w1, pool)
    self.assertEqual(ft.on_weight_sync_completed({}), [])
    self.assertNotIn(rejoined_w1, pool)

  def test_handle_dispatch_failure_evicts_worker_and_requeues_front(self):
    """Public handle_dispatch_failure rolls back _inflight, evicts worker, and queues at front."""
    w0 = FakeRolloutActor("w0")
    w1 = FakeRolloutActor("w1")
    pool = remote_execution.RoutingActorPool([w0, w1])
    ft = fault_tolerance.RolloutFaultToleranceManager(pool)

    req_existing = datatypes.RolloutRequest(
        request_id="r_existing",
        prompt="p_existing",
        prompt_id="p_existing",
        group_index=0,
    )
    ft.enqueue_retry(req_existing)

    req_failed = datatypes.RolloutRequest(
        request_id="r_failed",
        prompt="p_failed",
        prompt_id="p_failed",
        group_index=0,
    )
    ft.record_dispatch(req_failed, w0)
    self.assertEqual(ft.inflight_count(w0), 1)

    ft.handle_dispatch_failure(req_failed, w0, reason="connection_reset")
    self.assertEqual(ft.inflight_count(w0), 0)
    self.assertNotIn(w0, pool)
    self.assertEqual(
        ft.worker_status(w0), fault_tolerance.WorkerHealthStatus.FAILED
    )
    self.assertEqual(ft.retry_queue_size, 2)

  def test_poll_rollouts_zero_active_with_pending_rejoin_triggers_immediate_weight_sync(
      self,
  ):
    """When 0 active workers remain and a rejoined worker is pending, poll_rollouts triggers immediate weight sync."""

    @dataclasses.dataclass
    class _FakeSyncResult:
      policy_version: int
      workers: list[Any] = dataclasses.field(default_factory=list)

    class _FakeCoordinator:

      def __init__(self):
        self.synced_versions: list[int] = []

      async def sync(self, policy_version: int, **kwargs):
        del kwargs
        self.synced_versions.append(policy_version)
        return _FakeSyncResult(policy_version=policy_version)

    w0 = FakeRolloutActor("w0")
    trainer = FakeRolloutActor("trainer")
    coord = _FakeCoordinator()
    engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[w0],
        trainer_workers={datatypes.Role.ACTOR: trainer},
        weight_sync_coordinator=coord,
    )
    ft = engine.fault_tolerance_manager

    req0 = datatypes.RolloutRequest(
        request_id="r0", prompt="p0", prompt_id="p0", group_index=0
    )
    ft.record_dispatch(req0, w0)
    ft.evict_rollout_worker(w0, reason="pod_preempted")
    self.assertLen(engine._rollout_pool, 0)
    self.assertEqual(ft.retry_queue_size, 1)

    rejoined_w0 = FakeRolloutActor("w0")
    rejoined_w0.queued_responses.append(
        _make_response("p0", 0, worker_id="w0")
    )
    ft.register_rejoining_worker(rejoined_w0, worker_id="w0")
    self.assertLen(engine._rollout_pool, 0)

    completed = asyncio.run(engine.poll_rollouts(timeout_s=0.01))
    self.assertEqual(coord.synced_versions, [0])
    self.assertIn(rejoined_w0, engine._rollout_pool)
    self.assertLen(completed, 1)
    self.assertEqual(completed[0].traj_id, "traj_p0_g0")


if __name__ == "__main__":
  absltest.main()
