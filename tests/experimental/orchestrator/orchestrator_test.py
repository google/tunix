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

"""Unit tests for ClusterOrchestrator."""

import asyncio
import pickle
import tempfile
import threading
import time
from unittest import mock

from absl.testing import absltest
from etils import epath
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import orchestrator
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import trajectory_testing
from tunix.experimental.worker import abstract_worker
from tunix.experimental.worker import remote_execution


class ClusterOrchestratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_registry = mock.MagicMock()
    self.mock_registry.worker_ids.return_value = []
    self.mock_registry.infos.return_value = []
    self.mock_registry.group.return_value.members.return_value = []
    self.mock_lifecycle = mock.MagicMock()
    self.mock_monitor = mock.MagicMock()
    self.orch = orchestrator.ClusterOrchestrator(
        registry=self.mock_registry,
        lifecycle_driver=self.mock_lifecycle,
        monitor=self.mock_monitor,
    )

  def test_register_and_unregister_worker(self):
    mock_worker = mock.MagicMock()
    self.orch.register_worker(mock_worker)
    self.mock_registry.register.assert_called_once_with(mock_worker)

    self.orch.unregister_worker("worker_123")
    self.mock_registry.unregister.assert_called_once_with("worker_123")

  def test_bring_up_and_shutdown(self):
    self.orch.bring_up_workers("dummy_warmup_data")
    self.mock_lifecycle.bring_up.assert_called_once_with("dummy_warmup_data")

    self.orch.shutdown()
    self.mock_monitor.close.assert_called_once()
    self.mock_lifecycle.shutdown.assert_called_once()

  def test_create_engine(self):
    from tunix.experimental.worker import remote_execution

    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_critic = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_ref = mock.MagicMock(spec=remote_execution.ActorHandle)

    registry = worker_registry.WorkerRegistry()
    orch = orchestrator.ClusterOrchestrator(registry=registry)
    rollout_info = orch.register_worker_handle(
        "rollout-0", [datatypes.Role.ROLLOUT], mock_rollout
    )
    actor_info = orch.register_worker_handle(
        "actor-0", [datatypes.Role.ACTOR], mock_actor
    )
    critic_info = orch.register_worker_handle(
        "critic-0", [datatypes.Role.CRITIC], mock_critic
    )
    ref_info = orch.register_worker_handle(
        "reference-0", [datatypes.Role.REFERENCE], mock_ref
    )

    engine = orch._create_engine()
    self.assertIs(engine._rollout_workers[0], mock_rollout)
    self.assertIs(
        engine._trainer_workers[datatypes.Role.ACTOR],
        mock_actor,
    )
    self.assertIs(
        engine._trainer_workers[datatypes.Role.CRITIC],
        mock_critic,
    )
    self.assertIs(engine._inference_workers[datatypes.Role.REFERENCE], mock_ref)
    self.assertSequenceEqual(
        orch.worker_infos(), [actor_info, critic_info, ref_info, rollout_info]
    )

  def test_create_engine_with_weight_sync_shim_registrations(self):
    from tunix.experimental.worker import remote_execution

    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_actor = mock.MagicMock(spec=remote_execution.ActorHandle)

    class LocalActorWorker(abstract_worker.Worker):

      def info(self):
        return datatypes.WorkerInfo(
            worker_id="local-actor-worker",
            roles=frozenset({datatypes.Role.ACTOR.value}),
        )

      def initialize(self):
        return datatypes.Response()

      def compile(self, dummy_data=None):
        del dummy_data
        return datatypes.Response()

      def start(self):
        return datatypes.Response()

      def stop(self):
        return datatypes.Response()

      def heartbeat(self):
        return datatypes.HealthReport(state=datatypes.WorkerState.READY)

    registry = worker_registry.WorkerRegistry()
    orch = orchestrator.ClusterOrchestrator(
        registry=registry, weight_sync_mode="fallback"
    )
    orch.register_worker_handle(
        "rollout-0", [datatypes.Role.ROLLOUT], mock_rollout
    )
    orch.register_worker_handle("actor-0", [datatypes.Role.ACTOR], mock_actor)
    orch.register_worker(LocalActorWorker())

    engine = orch._create_engine()
    self.assertIsNotNone(engine._weight_sync_coordinator)

    # Assert they are properly shimmed in the registry
    self.assertIn("actor-0", orch.registry.worker_ids())
    self.assertIn("rollout-0", orch.registry.worker_ids())
    self.assertEqual(
        type(orch.registry.get("actor-0")).__name__, "RemoteWorkerShim"
    )
    self.assertEqual(
        type(orch.registry.get("rollout-0")).__name__, "RemoteWorkerShim"
    )

    local_actor_id = [
        w_id
        for w_id in orch.registry.worker_ids()
        if w_id.startswith("local-actor-") and w_id != "local-actor-worker"
    ]
    self.assertEqual(len(local_actor_id), 1)

    self.assertEqual(
        type(orch.registry.get(local_actor_id[0])).__name__, "RemoteWorkerShim"
    )

  def test_bring_up_and_shutdown_remote_worker_handles(self):
    from tunix.experimental.worker import remote_execution

    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_actor = mock.MagicMock(spec=remote_execution.ActorHandle)

    registry = worker_registry.WorkerRegistry()
    orch = orchestrator.ClusterOrchestrator(
        registry=registry,
        lifecycle_driver=self.mock_lifecycle,
        monitor=self.mock_monitor,
    )
    orch.register_worker_handle(
        "rollout-0", [datatypes.Role.ROLLOUT], mock_rollout
    )
    orch.register_worker_handle("actor-0", [datatypes.Role.ACTOR], mock_actor)

    orch.bring_up_workers(dummy_data="dummy")
    self.mock_lifecycle.bring_up.assert_called_once_with("dummy")
    mock_rollout.submit.assert_has_calls([
        mock.call("initialize"),
        mock.call("compile", "dummy"),
        mock.call("start"),
    ])
    mock_actor.submit.assert_has_calls([
        mock.call("initialize"),
        mock.call("compile", "dummy"),
        mock.call("start"),
    ])

    orch.shutdown()
    self.mock_monitor.close.assert_called_once()
    self.mock_lifecycle.shutdown.assert_called_once()
    mock_rollout.submit.assert_any_call("stop")
    mock_actor.submit.assert_any_call("stop")

  def test_shutdown_survives_a_wedged_worker(self):
    from tunix.experimental.worker import remote_execution

    wedged = mock.MagicMock(spec=remote_execution.ActorHandle)
    wedged.submit.side_effect = lambda *a, **kw: time.sleep(5)
    healthy = mock.MagicMock(spec=remote_execution.ActorHandle)

    registry = worker_registry.WorkerRegistry()
    orch = orchestrator.ClusterOrchestrator(
        registry=registry,
        lifecycle_driver=self.mock_lifecycle,
        monitor=self.mock_monitor,
    )
    orch.register_worker_handle("rollout-0", [datatypes.Role.ROLLOUT], wedged)
    orch.register_worker_handle("actor-0", [datatypes.Role.ACTOR], healthy)

    with mock.patch.object(orchestrator, "_STOP_TIMEOUT_S", 0.2):
      orch._shutdown_remote_workers()

    healthy.submit.assert_any_call("stop")

  def test_create_engine_wraps_local_workers_as_in_process_handles(self):
    from tunix.experimental.worker import remote_execution

    class LocalWorker(abstract_worker.Worker):

      def info(self):
        return datatypes.WorkerInfo(
            worker_id="rollout-0", roles=frozenset({datatypes.Role.ROLLOUT})
        )

      def initialize(self):
        return datatypes.Response()

      def compile(self, dummy_data=None):
        del dummy_data
        return datatypes.Response()

      def start(self):
        return datatypes.Response()

      def stop(self):
        return datatypes.Response()

      def heartbeat(self):
        return datatypes.HealthReport(state=datatypes.WorkerState.READY)

      def generate(self, prompts):
        del prompts
        return []

    registry = worker_registry.WorkerRegistry()
    registry.register(LocalWorker())

    orch = orchestrator.ClusterOrchestrator(registry=registry)
    engine = orch._create_engine()
    self.assertIsInstance(
        engine._rollout_workers[0], remote_execution.InProcessActorHandle
    )

  def test_worker_handles_returns_remote_and_local_workers(self):
    from tunix.experimental.worker import remote_execution

    mock_rollout_remote = mock.MagicMock(spec=remote_execution.ActorHandle)
    registry = worker_registry.WorkerRegistry()
    orch = orchestrator.ClusterOrchestrator(registry=registry)
    orch.register_worker_handle(
        "rollout-remote-0", [datatypes.Role.ROLLOUT], mock_rollout_remote
    )

    class LocalRolloutWorker(abstract_worker.Worker):

      def info(self):
        return datatypes.WorkerInfo(
            worker_id="rollout-local-0",
            roles=frozenset({"rollout"}),
        )

      def initialize(self):
        return datatypes.Response()

      def compile(self, dummy_data=None):
        del dummy_data
        return datatypes.Response()

      def start(self):
        return datatypes.Response()

      def stop(self):
        return datatypes.Response()

      def heartbeat(self):
        return datatypes.HealthReport(state=datatypes.WorkerState.READY)

    orch.register_worker(LocalRolloutWorker())

    handles_enum = orch.worker_handles(datatypes.Role.ROLLOUT)
    handles_str = orch.worker_handles("rollout")

    self.assertEqual(len(handles_enum), 2)
    self.assertEqual(len(handles_str), 2)
    self.assertIs(handles_enum[0], mock_rollout_remote)
    self.assertIsInstance(
        handles_enum[1], remote_execution.InProcessActorHandle
    )

  def test_wait_for_workers_already_available(self):
    from tunix.experimental.worker import remote_execution

    mock_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    orch = orchestrator.ClusterOrchestrator()
    orch.register_worker_handle("actor-0", [datatypes.Role.ACTOR], mock_actor)
    orch.register_worker_handle(
        "rollout-0", [datatypes.Role.ROLLOUT], mock_rollout
    )

    orch.wait_for_workers(
        {
            datatypes.Role.ACTOR: 1,
            datatypes.Role.ROLLOUT: 1,
            datatypes.Role.REFERENCE: 0,
        },
        timeout=1.0,
        poll_interval_s=0.01,
    )

  @mock.patch.object(remote_execution.ActorHandle, "from_address")
  def test_register_worker_from_hostname(self, mock_from_address):
    mock_from_address.return_value = mock.MagicMock(
        spec=remote_execution.ActorHandle
    )
    orch = orchestrator.ClusterOrchestrator()
    for port, (service_type, role) in enumerate(
        [
            ("trainer", datatypes.Role.ACTOR),
            ("rollout", datatypes.Role.ROLLOUT),
            ("inference", datatypes.Role.REFERENCE),
        ],
        start=5000,
    ):
      meta = pickle.dumps({
          "service_type": service_type,
          "service_port": port,
          "worker_id": f"{service_type}-0",
      })
      orch.register_worker_from_hostname("host", 0, meta, rpc_timeout_s=120.0)
      mock_from_address.assert_called_with(
          f"grpc://host:{port}", rpc_timeout_s=120.0
      )
      self.assertEqual(
          orch.worker_handles(role), [mock_from_address.return_value]
      )

    info_by_id = {i.worker_id: i for i in orch.worker_infos()}
    self.assertEqual(
        info_by_id["trainer-0"],
        datatypes.WorkerInfo(
            worker_id="trainer-0",
            roles=frozenset({"actor"}),
            resources={"remote": True, "address": "host:5000"},
        ),
    )

  def test_register_worker_from_hostname_unknown_service_type(self):
    orch = orchestrator.ClusterOrchestrator()
    meta = pickle.dumps({
        "service_type": "unknown",
        "service_port": 5000,
        "worker_id": "bad-0",
    })
    with self.assertRaisesRegex(RuntimeError, "unknown service type unknown"):
      orch.register_worker_from_hostname("host", 0, meta)

  def test_wait_for_workers_delayed_registration(self):
    from tunix.experimental.worker import remote_execution

    mock_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    orch = orchestrator.ClusterOrchestrator()

    def register_later():
      time.sleep(0.05)
      orch.register_worker_handle("actor-0", [datatypes.Role.ACTOR], mock_actor)

    t = threading.Thread(target=register_later)
    t.start()
    try:
      orch.wait_for_workers(
          {datatypes.Role.ACTOR: 1},
          timeout=2.0,
          poll_interval_s=0.01,
      )
    finally:
      t.join()

    self.assertEqual(len(orch.worker_handles(datatypes.Role.ACTOR)), 1)

  def test_wait_for_workers_timeout(self):
    orch = orchestrator.ClusterOrchestrator()
    with self.assertRaises(TimeoutError):
      orch.wait_for_workers(
          {datatypes.Role.ACTOR: 1},
          timeout=0.05,
          poll_interval_s=0.01,
      )

  def test_run_with_bring_up(self):
    mock_program = mock.MagicMock(spec=rl_program.RLProgram)
    mock_engine = mock.MagicMock()

    with mock.patch.object(
        self.orch, "_create_engine", return_value=mock_engine
    ):
      self.orch.run(
          program=mock_program,
          train_dataset=["batch1", "batch2"],
          max_steps=10,
          bring_up=True,
          dummy_data="dummy_init",
      )

    self.mock_lifecycle.bring_up.assert_called_once_with("dummy_init")
    self.mock_monitor.poll.assert_called_once()
    mock_program.run.assert_called_once_with(
        engine=mock_engine,
        train_dataset=["batch1", "batch2"],
        max_steps=10,
    )

  def test_run_without_bring_up(self):
    mock_program = mock.MagicMock(spec=rl_program.RLProgram)
    mock_engine = mock.MagicMock()
    self.orch.engine = mock_engine

    self.orch.run(
        program=mock_program,
        bring_up=False,
    )

    self.mock_lifecycle.bring_up.assert_not_called()
    self.mock_monitor.poll.assert_called_once()
    mock_program.run.assert_called_once_with(
        engine=mock_engine,
    )

  def test_re_register_rollout_worker_before_and_after_bring_up(self):
    h_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r0_v1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r0_v2 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r0_v3 = mock.MagicMock(spec=remote_execution.ActorHandle)

    registry = worker_registry.WorkerRegistry()
    orch = orchestrator.ClusterOrchestrator(
        registry=registry,
        lifecycle_driver=mock.MagicMock(),
        monitor=mock.MagicMock(),
        weight_sync_mode="fallback",
    )
    orch.register_worker_handle("actor-0", [datatypes.Role.ACTOR], h_actor)
    orch.register_worker_handle("rollout-0", [datatypes.Role.ROLLOUT], h_r0_v1)

    with self.assertRaisesRegex(ValueError, "duplicate worker_id"):
      orch.register_worker_handle(
          "rollout-0",
          [datatypes.Role.ROLLOUT],
          h_r0_v2,
          override=False,
      )

    # Re-register rollout-0 before bring_up_workers (default override=True)
    orch.register_worker_handle("rollout-0", [datatypes.Role.ROLLOUT], h_r0_v2)
    self.assertEqual(orch.worker_handles(datatypes.Role.ROLLOUT), [h_r0_v2])

    orch.bring_up_workers(dummy_data="warmup_batch")
    h_r0_v1.submit.assert_not_called()
    h_r0_v2.submit.assert_has_calls([
        mock.call("initialize"),
        mock.call("compile", "warmup_batch"),
        mock.call("start"),
    ])
    self.assertEqual(orch.engine._rollout_workers, [h_r0_v2])

    # Re-register rollout-0 after bring_up_workers with coordinator present ->
    # staged as PENDING_WEIGHT_SYNC until next weight sync promotes it.
    orch.register_worker_handle("rollout-0", [datatypes.Role.ROLLOUT], h_r0_v3)
    orch.wait_for_pending_bring_ups(timeout=5.0)
    h_r0_v3.submit.assert_has_calls([
        mock.call("initialize"),
        mock.call("compile", "warmup_batch"),
        mock.call("start"),
    ])
    self.assertEqual(orch.worker_handles(datatypes.Role.ROLLOUT), [])
    self.assertEqual(orch.engine._rollout_workers, [])
    self.assertEqual(
        orch.registry.state("rollout-0"),
        worker_registry.MembershipState.PENDING_WEIGHT_SYNC,
    )
    orch.registry.set_state("rollout-0", worker_registry.MembershipState.ACTIVE)
    self.assertEqual(orch.worker_handles(datatypes.Role.ROLLOUT), [h_r0_v3])
    self.assertEqual(orch.engine._rollout_workers, [h_r0_v3])
    orch.shutdown()

  def test_dynamic_worker_arrival_and_weight_sync_promotion(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    cfg = {
        "enabled": True,
        "backend": "file",
        "root_dir": str(tmp_dir),
        "run_id": "dynamic_run",
    }
    h_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r1 = mock.MagicMock(spec=remote_execution.ActorHandle)

    registry = worker_registry.WorkerRegistry()
    orch = orchestrator.ClusterOrchestrator(
        registry=registry,
        lifecycle_driver=mock.MagicMock(),
        monitor=mock.MagicMock(),
        weight_sync_mode="fallback",
        trajectory_store_config=cfg,
    )
    orch.register_worker_handle("actor-0", [datatypes.Role.ACTOR], h_actor)
    orch.register_worker_handle("rollout-0", [datatypes.Role.ROLLOUT], h_r0)
    orch.bring_up_workers(dummy_data="warmup")

    # Simulate that weight sync has already committed policy_version=1
    orch.engine._policy_version = 1

    # Dynamically register rollout-1 after bring_up_workers()
    orch.register_worker_handle("rollout-1", [datatypes.Role.ROLLOUT], h_r1)
    orch.wait_for_pending_bring_ups(timeout=5.0)

    h_r1.submit.assert_has_calls([
        mock.call("initialize"),
        mock.call("compile", "warmup"),
        mock.call("start"),
    ])
    self.assertEqual(
        orch.registry.state("rollout-1"),
        worker_registry.MembershipState.PENDING_WEIGHT_SYNC,
    )
    self.assertNotIn(h_r1, orch.engine._rollout_workers)
    self.assertTrue(
        orch.engine._weight_sync_coordinator.has_pending_destinations()
    )

    # Next sync_weights promotes rollout-1 to ACTIVE via WeightSyncCoordinator
    ws_proto = orchestrator.weight_sync_coordinator.weight_sync

    def _mock_asubmit(method_name, *args, **kwargs):
      del args, kwargs
      if method_name in ("get_weight_sync_metadata", "prepare_weight_sync"):
        return [
            ws_proto.WorkUnitMetadata(
                unit=ws_proto.WorkUnitId(job_name="w"),
                global_shape=(4,),
                item_size=4,
            )
        ]
      return {}

    for h in (h_actor, h_r0, h_r1):
      h.asubmit.side_effect = _mock_asubmit
    asyncio.run(orch.engine.sync_weights(policy_version=2))

    self.assertEqual(
        orch.registry.state("rollout-1"),
        worker_registry.MembershipState.ACTIVE,
    )
    self.assertIn(h_r1, orch.engine._rollout_workers)
    self.assertFalse(
        orch.engine._weight_sync_coordinator.has_pending_destinations()
    )
    orch.shutdown()

  def test_sync_weights_evicts_failed_destination_and_retries(self):
    ws_proto = orchestrator.weight_sync_coordinator.weight_sync

    def _mock_asubmit(method_name, *args, **kwargs):
      del args, kwargs
      if method_name in ("get_weight_sync_metadata", "prepare_weight_sync"):
        return [
            ws_proto.WorkUnitMetadata(
                unit=ws_proto.WorkUnitId(job_name="w"),
                global_shape=(4,),
                item_size=4,
            )
        ]
      return {}

    h_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_actor.asubmit.side_effect = _mock_asubmit
    h_r0.asubmit.side_effect = ConnectionError("rollout-0 died pre-quiesce")
    h_r1.asubmit.side_effect = _mock_asubmit

    registry = worker_registry.WorkerRegistry()
    orch = orchestrator.ClusterOrchestrator(
        registry=registry,
        lifecycle_driver=mock.MagicMock(),
        monitor=mock.MagicMock(),
        weight_sync_mode="fallback",
    )
    orch.register_worker_handle("actor-0", [datatypes.Role.ACTOR], h_actor)
    orch.register_worker_handle("rollout-0", [datatypes.Role.ROLLOUT], h_r0)
    orch.register_worker_handle("rollout-1", [datatypes.Role.ROLLOUT], h_r1)
    orch.bring_up_workers()

    version = asyncio.run(orch.engine.sync_weights(policy_version=3))
    self.assertEqual(version, 3)
    self.assertEqual(orch.engine._rollout_workers, [h_r1])
    self.assertEqual(orch.worker_handles(datatypes.Role.ROLLOUT), [h_r1])
    self.assertEqual(
        orch.registry.state("rollout-0"),
        worker_registry.MembershipState.EVICTED,
    )
    self.assertEqual(
        orch.registry.state("rollout-1"),
        worker_registry.MembershipState.ACTIVE,
    )
    orch.shutdown()

  def test_worker_eviction_callback_removes_dead_worker_from_orchestrator(self):
    h_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r1 = mock.MagicMock(spec=remote_execution.ActorHandle)

    registry = worker_registry.WorkerRegistry()
    orch = orchestrator.ClusterOrchestrator(
        registry=registry,
        lifecycle_driver=mock.MagicMock(),
        monitor=mock.MagicMock(),
        weight_sync_mode="fallback",
    )
    orch.register_worker_handle("actor-0", [datatypes.Role.ACTOR], h_actor)
    orch.register_worker_handle("rollout-0", [datatypes.Role.ROLLOUT], h_r0)
    orch.register_worker_handle("rollout-1", [datatypes.Role.ROLLOUT], h_r1)
    orch.bring_up_workers()

    self.assertEqual(h_r0.worker_id, "rollout-0")
    self.assertEqual(h_r1.worker_id, "rollout-1")
    self.assertEqual(orch.worker_handles(datatypes.Role.ROLLOUT), [h_r0, h_r1])
    self.assertEqual(
        orch.registry.state("rollout-0"),
        worker_registry.MembershipState.ACTIVE,
    )

    orch.engine._rollout_session.remove_actor(
        h_r0, exc=ConnectionError("pod died")
    )

    self.assertEqual(orch.worker_handles(datatypes.Role.ROLLOUT), [h_r1])
    self.assertEqual(orch.engine._rollout_workers, [h_r1])
    self.assertEqual(
        orch.registry.state("rollout-0"),
        worker_registry.MembershipState.EVICTED,
    )
    self.assertEqual(
        orch.registry.state("rollout-1"),
        worker_registry.MembershipState.ACTIVE,
    )

    # A stale eviction callback for h_r0 after rollout-0 re-registers with a
    # new handle must not evict the new incarnation.
    h_r0_v2 = mock.MagicMock(spec=remote_execution.ActorHandle)
    orch.register_worker_handle("rollout-0", [datatypes.Role.ROLLOUT], h_r0_v2)
    orch.wait_for_pending_bring_ups(timeout=5.0)
    self.assertEqual(
        orch.registry.state("rollout-0"),
        worker_registry.MembershipState.PENDING_WEIGHT_SYNC,
    )
    orch._on_engine_worker_evicted(h_r0, ConnectionError("stale callback"))
    self.assertEqual(
        orch.registry.state("rollout-0"),
        worker_registry.MembershipState.PENDING_WEIGHT_SYNC,
    )
    orch.shutdown()

  @mock.patch.object(remote_execution.ActorHandle, "from_address")
  def test_orchestrator_plumbs_max_concurrent_rollouts_per_worker(
      self, mock_from_address
  ):
    h_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_from_address.side_effect = [h_r0, h_r1]

    orch = orchestrator.ClusterOrchestrator(
        lifecycle_driver=mock.MagicMock(),
        monitor=mock.MagicMock(),
        max_concurrent_rollouts_per_worker=8,
    )
    orch.register_worker_handle("actor-0", [datatypes.Role.ACTOR], h_actor)

    # rollout-0 advertises max_concurrency=4 (< 8); rollout-1 advertises 16
    # (> 8).
    orch.register_worker_from_hostname(
        "host0",
        0,
        pickle.dumps({
            "service_type": "rollout",
            "service_port": 5001,
            "worker_id": "rollout-0",
            "max_concurrency": 4,
        }),
    )
    orch.register_worker_from_hostname(
        "host1",
        0,
        pickle.dumps({
            "service_type": "rollout",
            "service_port": 5002,
            "worker_id": "rollout-1",
            "max_concurrency": 16,
        }),
    )

    orch.bring_up_workers()
    assert orch.engine is not None
    self.assertEqual(orch.engine.max_concurrent_rollouts_per_worker, 8)
    # Effective limit is min(orchestrator_cap, worker_max_concurrency)
    self.assertEqual(orch.engine._rollout_session._get_worker_limit(h_r0), 4)
    self.assertEqual(orch.engine._rollout_session._get_worker_limit(h_r1), 8)
    orch.shutdown()

  def test_proactive_pending_weight_sync_only_syncs_pending_workers(self):
    ws_proto = orchestrator.weight_sync_coordinator.weight_sync

    def _mock_asubmit(method_name, *args, **kwargs):
      del args, kwargs
      if method_name in ("get_weight_sync_metadata", "prepare_weight_sync"):
        return [
            ws_proto.WorkUnitMetadata(
                unit=ws_proto.WorkUnitId(job_name="w"),
                global_shape=(4,),
                item_size=4,
            )
        ]
      return {}

    h_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    for h in (h_actor, h_r0, h_r1):
      h.asubmit.side_effect = _mock_asubmit

    registry = worker_registry.WorkerRegistry()
    orch = orchestrator.ClusterOrchestrator(
        registry=registry,
        lifecycle_driver=mock.MagicMock(),
        monitor=mock.MagicMock(),
        weight_sync_mode="fallback",
    )
    orch.register_worker_handle("actor-0", [datatypes.Role.ACTOR], h_actor)
    orch.register_worker_handle("rollout-0", [datatypes.Role.ROLLOUT], h_r0)
    orch.bring_up_workers()
    assert orch.engine is not None

    # Step 1 weight sync commits policy_version=1 on rollout-0
    asyncio.run(orch.engine.sync_weights(policy_version=1))
    h_r0.asubmit.reset_mock()

    # rollout-1 joins mid-step in PENDING_WEIGHT_SYNC
    orch.register_worker_handle("rollout-1", [datatypes.Role.ROLLOUT], h_r1)
    orch.wait_for_pending_bring_ups(timeout=5.0)
    self.assertEqual(
        orch.registry.state("rollout-1"),
        worker_registry.MembershipState.PENDING_WEIGHT_SYNC,
    )
    self.assertEqual(orch.worker_handles(datatypes.Role.ROLLOUT), [h_r0])

    # Proactive sync_pending_weights syncs ONLY rollout-1 without quiescing
    # rollout-0.
    synced_version = asyncio.run(orch.engine.sync_pending_weights())
    self.assertEqual(synced_version, 1)
    h_r0.asubmit.assert_not_called()
    self.assertEqual(
        orch.registry.state("rollout-1"),
        worker_registry.MembershipState.ACTIVE,
    )
    self.assertEqual(orch.worker_handles(datatypes.Role.ROLLOUT), [h_r0, h_r1])
    self.assertEqual(orch.engine._rollout_workers, [h_r0, h_r1])
    orch.shutdown()

  def test_all_workers_dead_recovery_via_proactive_pending_sync(self):
    ws_proto = orchestrator.weight_sync_coordinator.weight_sync

    def _mock_ws_asubmit(method_name, *args, **kwargs):
      del args, kwargs
      if method_name in ("get_weight_sync_metadata", "prepare_weight_sync"):
        return [
            ws_proto.WorkUnitMetadata(
                unit=ws_proto.WorkUnitId(job_name="w"),
                global_shape=(4,),
                item_size=4,
            )
        ]
      return {}

    class _RolloutActor:

      def __init__(self, worker_id: str, fail_poll: bool = False):
        self.worker_id = worker_id
        self.fail_poll = fail_poll

      def initialize(self):
        return datatypes.Response()

      def compile(self, dummy_data=None):
        del dummy_data
        return datatypes.Response()

      def start(self):
        return datatypes.Response()

      def stop(self):
        return datatypes.Response()

      def bind_weight_sync(self):
        return {}

      def get_weight_sync_metadata(self):
        return _mock_ws_asubmit("get_weight_sync_metadata")

      def pre_weight_sync(self, req):
        del req
        return {}

      def weight_sync(self, req):
        del req
        return {}

      def post_weight_sync(self, req):
        del req
        return {}

      def abort_weight_sync(self, req):
        del req
        return {}

      def get_weight_sync_status(self):
        return {"phase": "committed"}

      def generate(self, requests):
        out = []
        for req in requests:
          out.append(
              datatypes.RolloutResponse(
                  request_id=req.request_id,
                  status="COMPLETED",
                  payload=datatypes.TrajectoryItem(
                      prompt_id=req.prompt_id,
                      group_index=req.group_index,
                      traj={
                          "worker_id": self.worker_id,
                          "status": datatypes.TrajectoryStatus.SUCCEEDED,
                      },
                  ),
              )
          )
        return out

    h_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_actor.asubmit.side_effect = _mock_ws_asubmit

    r0_srv = remote_execution.InProcessRemoteExecutionServer(
        _RolloutActor("r0")
    )
    h_r0 = remote_execution.InProcessActorHandle(r0_srv)

    r1_srv = remote_execution.InProcessRemoteExecutionServer(
        _RolloutActor("r1")
    )
    h_r1 = remote_execution.InProcessActorHandle(r1_srv)

    orch = orchestrator.ClusterOrchestrator(
        lifecycle_driver=mock.MagicMock(),
        monitor=mock.MagicMock(),
        weight_sync_mode="fallback",
    )
    orch.register_worker_handle("actor-0", [datatypes.Role.ACTOR], h_actor)
    orch.register_worker_handle("rollout-0", [datatypes.Role.ROLLOUT], h_r0)
    orch.bring_up_workers()
    assert orch.engine is not None

    asyncio.run(orch.engine.sync_weights(policy_version=1))

    # Dispatch while rollout-0 is the ONLY active worker; during poll_responses,
    # rollout-1 registers in PENDING_WEIGHT_SYNC and rollout-0 crashes (N->0).
    async def _crashing_poll_responses(timeout_s=50.0):
      del timeout_s
      orch.register_worker_handle("rollout-1", [datatypes.Role.ROLLOUT], h_r1)
      orch.wait_for_pending_bring_ups(timeout=5.0)
      self.assertEqual(
          orch.registry.state("rollout-1"),
          worker_registry.MembershipState.PENDING_WEIGHT_SYNC,
      )
      raise ConnectionError("r0 crashed mid-poll")

    async def _run_dispatch_and_poll():
      with mock.patch.object(
          h_r0,
          "dispatch_task",
          side_effect=mock.AsyncMock(return_value="req_n_to_0"),
      ), mock.patch.object(
          h_r0,
          "poll_responses",
          side_effect=_crashing_poll_responses,
      ):
        await orch.engine.dispatch_rollout_requests([
            datatypes.RolloutRequest(
                request_id="req_n_to_0",
                prompt="hello",
                prompt_id="p0",
                group_index=0,
            )
        ])
        return await orch.engine.poll_rollouts(timeout_s=2.0)

    items = asyncio.run(_run_dispatch_and_poll())
    self.assertLen(items, 1)
    self.assertEqual(items[0].prompt_id, "p0")
    self.assertEqual(items[0].traj["worker_id"], "r1")
    self.assertEqual(
        orch.registry.state("rollout-0"),
        worker_registry.MembershipState.EVICTED,
    )
    self.assertEqual(
        orch.registry.state("rollout-1"),
        worker_registry.MembershipState.ACTIVE,
    )
    orch.shutdown()

  def test_orchestrator_plumbs_rollout_task_timeout_and_live_session_eviction(
      self,
  ):
    with self.assertRaisesRegex(ValueError, "must be positive"):
      orchestrator.ClusterOrchestrator(
          lifecycle_driver=mock.MagicMock(),
          monitor=mock.MagicMock(),
          rollout_task_timeout_s=-1.0,
      )

    h_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r2 = mock.MagicMock(spec=remote_execution.ActorHandle)

    orch = orchestrator.ClusterOrchestrator(
        lifecycle_driver=mock.MagicMock(),
        monitor=mock.MagicMock(),
        weight_sync_mode="fallback",
        rollout_task_timeout_s=45.0,
    )
    orch.register_worker_handle("actor-0", [datatypes.Role.ACTOR], h_actor)
    orch.register_worker_handle("rollout-0", [datatypes.Role.ROLLOUT], h_r0)
    orch.register_worker_handle("rollout-1", [datatypes.Role.ROLLOUT], h_r1)
    orch.register_worker_handle("rollout-2", [datatypes.Role.ROLLOUT], h_r2)
    orch.bring_up_workers()

    assert orch.engine is not None
    self.assertEqual(orch.engine.rollout_task_timeout_s, 45.0)
    self.assertEqual(orch.engine._rollout_session.task_timeout_s, 45.0)

    # 1. Evicting rollout-0 via WorkerRegistry removes h_r0 from live session
    orch.registry.evict("rollout-0")
    self.assertEqual(orch.engine._rollout_workers, [h_r1, h_r2])

    # 2. Unregistering rollout-1 removes h_r1 from live session and registry
    orch.unregister_worker("rollout-1")
    self.assertEqual(orch.engine._rollout_workers, [h_r2])
    self.assertNotIn("rollout-1", orch.registry)
    orch.shutdown()

  def test_late_rollout_worker_at_step_zero_is_staged_as_pending_weight_sync(
      self,
  ):
    h_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_r1 = mock.MagicMock(spec=remote_execution.ActorHandle)

    ft_cfg = datatypes.RolloutFaultToleranceConfig(
        max_task_retries=2,
        max_zero_worker_wait_s=15.0,
        recover_unknown_transfer_state=True,
    )
    orch = orchestrator.ClusterOrchestrator(
        lifecycle_driver=mock.MagicMock(),
        monitor=mock.MagicMock(),
        weight_sync_mode="fallback",
        fault_tolerance_config=ft_cfg,
    )
    orch.register_worker_handle("actor-0", [datatypes.Role.ACTOR], h_actor)
    orch.register_worker_handle("rollout-0", [datatypes.Role.ROLLOUT], h_r0)
    orch.bring_up_workers()

    # Even at policy_version == 0 (step-0 race window before initial weight
    # sync completes), a late-arriving rollout worker must be staged as
    # PENDING_WEIGHT_SYNC and excluded from active dispatch until a committed
    # weight sync round includes it.
    self.assertEqual(orch.engine.policy_version, 0)
    self.assertEqual(orch.engine.fault_tolerance_config, ft_cfg)
    self.assertEqual(orch.engine.max_zero_worker_wait_s, 15.0)
    orch.register_worker_handle("rollout-1", [datatypes.Role.ROLLOUT], h_r1)
    orch.wait_for_pending_bring_ups(timeout=5.0)

    self.assertEqual(
        orch.registry.state("rollout-1"),
        worker_registry.MembershipState.PENDING_WEIGHT_SYNC,
    )
    self.assertEqual(orch.engine._rollout_workers, [h_r0])
    orch.shutdown()


def _trajectory_store_orchestrator(
    **kwargs,
) -> orchestrator.ClusterOrchestrator:
  mock_registry = mock.MagicMock()
  mock_registry.worker_ids.return_value = []
  mock_registry.infos.return_value = []
  mock_registry.group.return_value.members.return_value = []
  return orchestrator.ClusterOrchestrator(
      registry=mock_registry,
      lifecycle_driver=mock.MagicMock(),
      monitor=mock.MagicMock(),
      **kwargs,
  )


class ClusterOrchestratorTrajectoryStoreTest(absltest.TestCase):

  def test_no_config_means_no_store(self):
    orch = _trajectory_store_orchestrator()
    self.assertIsNone(orch.trajectory_store)
    orch.shutdown()

  def test_disabled_config_means_no_store(self):
    orch = _trajectory_store_orchestrator(
        trajectory_store_config={"enabled": False, "backend": "file"}
    )
    self.assertIsNone(orch.trajectory_store)
    orch.shutdown()

  def test_enabled_file_backend_builds_store_once(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    orch = _trajectory_store_orchestrator(
        trajectory_store_config={
            "enabled": True,
            "backend": "file",
            "root_dir": str(tmp_dir),
            "run_id": "cluster_run",
        }
    )
    self.assertIsInstance(orch.trajectory_store, file_store.FileTrajectoryStore)
    orch.shutdown()

  def test_shutdown_closes_the_store(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    orch = _trajectory_store_orchestrator(
        trajectory_store_config={
            "enabled": True,
            "backend": "file",
            "root_dir": str(tmp_dir),
            "run_id": "cluster_run",
        }
    )
    store = orch.trajectory_store
    orch.shutdown()
    with self.assertRaises(RuntimeError):
      store.add_step(trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1)

  def test_shutdown_closes_the_store_even_when_a_prior_step_raises(self):
    orch = _trajectory_store_orchestrator()
    orch.trajectory_store = mock.MagicMock()
    orch.monitor.close = mock.MagicMock(
        side_effect=RuntimeError("monitor close failed")
    )
    with self.assertRaises(RuntimeError):
      orch.shutdown()
    orch.lifecycle_driver.shutdown.assert_called_once()
    orch.trajectory_store.close.assert_called_once()

  def test_shutdown_without_a_store_does_not_raise(self):
    orch = _trajectory_store_orchestrator()
    orch.shutdown()

  def test_bring_up_propagates_trajectory_store_config_to_rollout_workers(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    cfg = {
        "enabled": True,
        "backend": "file",
        "root_dir": str(tmp_dir),
        "run_id": "cluster_run",
    }
    registry = worker_registry.WorkerRegistry()
    local_rollout = mock.MagicMock()
    local_rollout.info.return_value = datatypes.WorkerInfo(
        worker_id="rollout-local-0",
        roles=frozenset({datatypes.Role.ROLLOUT}),
    )
    registry.register(local_rollout)

    mock_rollout_remote = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_actor_remote = mock.MagicMock(spec=remote_execution.ActorHandle)

    orch = orchestrator.ClusterOrchestrator(
        registry=registry,
        lifecycle_driver=mock.MagicMock(),
        monitor=mock.MagicMock(),
        trajectory_store_config=cfg,
    )
    orch.register_worker_handle(
        "rollout-remote-0", [datatypes.Role.ROLLOUT], mock_rollout_remote
    )
    orch.register_worker_handle(
        "actor-remote-0", [datatypes.Role.ACTOR], mock_actor_remote
    )

    orch.bring_up_workers()
    local_rollout.with_trajectory_store_config.assert_called_once_with(cfg)
    mock_rollout_remote.submit.assert_any_call(
        "with_trajectory_store_config", cfg
    )
    for call in mock_actor_remote.submit.call_args_list:
      self.assertNotEqual(call.args[0], "with_trajectory_store_config")
    orch.shutdown()

  def test_orchestrator_populates_run_id_when_omitted_from_trajectory_store_config(
      self,
  ):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    cfg = {
        "enabled": True,
        "backend": "file",
        "root_dir": str(tmp_dir),
    }
    registry = worker_registry.WorkerRegistry()
    mock_rollout_remote = mock.MagicMock(spec=remote_execution.ActorHandle)

    orch = orchestrator.ClusterOrchestrator(
        registry=registry,
        lifecycle_driver=mock.MagicMock(),
        monitor=mock.MagicMock(),
        trajectory_store_config=cfg,
    )
    orch.register_worker_handle(
        "rollout-remote-0", [datatypes.Role.ROLLOUT], mock_rollout_remote
    )

    self.assertTrue(orch.run_id.startswith("run_"))
    expected_cfg = {
        "enabled": True,
        "backend": "file",
        "root_dir": str(tmp_dir),
        "run_id": orch.run_id,
    }
    self.assertEqual(orch.trajectory_store_config, expected_cfg)

    orch.bring_up_workers()
    mock_rollout_remote.submit.assert_any_call(
        "with_trajectory_store_config", expected_cfg
    )
    orch.shutdown()


if __name__ == "__main__":
  absltest.main()

