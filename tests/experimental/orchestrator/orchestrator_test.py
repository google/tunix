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

import pickle
import threading
import time
from unittest import mock

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import orchestrator
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.orchestrator import worker_registry
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

    registry = worker_registry.WorkerRegistry()
    orch = orchestrator.ClusterOrchestrator(
        registry=registry, weight_sync_mode="fallback"
    )
    orch.register_worker_handle(
        "rollout-0", [datatypes.Role.ROLLOUT], mock_rollout
    )
    orch.register_worker_handle("actor-0", [datatypes.Role.ACTOR], mock_actor)

    # Local handle fallback, no worker ID in _remote_worker_handles_by_id
    local_actor = remote_execution.InProcessActorHandle(
        remote_execution.InProcessRemoteExecutionServer(mock.MagicMock())
    )
    orch._remote_worker_handles[datatypes.Role.ACTOR.value].append(local_actor)

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
        if w_id.startswith("local-actor-")
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

  def test_run_managed_program_submission(self):
    mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    mock_algo.group_size = 2
    mock_algo.mini_batch_size = 1
    mock_algo.max_turns = 1
    mock_algo.max_packed_len = 16
    mock_algo.requires_reference_kl = False

    assembler = batch_assembly.SequencePackedBatchAssembler(
        group_size=2, max_packed_len=16
    )

    with mock.patch("asyncio.run") as mock_asyncio_run:
      self.orch.run(
          algo=mock_algo,
          dataset=["prompt1"],
          reward_fns=[lambda x: 1.0],
          assembler=assembler,
          max_steps=5,
      )
      self.mock_lifecycle.bring_up.assert_called_once()
      self.mock_monitor.poll.assert_called_once()
      mock_asyncio_run.assert_called_once()

  def test_run_program_with_bring_up_and_train_dataset(self):
    mock_program = mock.MagicMock(spec=rl_program.RLProgram)
    mock_engine = mock.MagicMock()

    with mock.patch.object(
        self.orch, "_create_engine", return_value=mock_engine
    ):
      self.orch.run_program(
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

  def test_run_program_without_bring_up(self):
    mock_program = mock.MagicMock(spec=rl_program.RLProgram)
    mock_engine = mock.MagicMock()
    self.orch.engine = mock_engine

    self.orch.run_program(
        program=mock_program,
        bring_up=False,
    )

    self.mock_lifecycle.bring_up.assert_not_called()
    self.mock_monitor.poll.assert_called_once()
    mock_program.run.assert_called_once_with(
        engine=mock_engine,
    )

  def test_run_auto_instantiated_program_closes_on_success(self):
    mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    self.orch.run_program = mock.MagicMock()

    with mock.patch.object(
        rl_program.StandardRLProgram, "close", autospec=True
    ) as mock_close:
      self.orch.run(
          algo=mock_algo,
          dataset=["prompt_1"],
          program=None,
          max_steps=1,
      )
      self.orch.run_program.assert_called_once()
      created_program = self.orch.run_program.call_args.kwargs["program"]
      self.assertIsInstance(created_program, rl_program.StandardRLProgram)
      mock_close.assert_called_once_with(created_program)

  def test_run_auto_instantiated_program_closes_on_exception(self):
    mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    self.orch.run_program = mock.MagicMock(
        side_effect=RuntimeError("Engine failure")
    )

    with mock.patch.object(
        rl_program.StandardRLProgram, "close", autospec=True
    ) as mock_close:
      with self.assertRaises(RuntimeError):
        self.orch.run(
            algo=mock_algo,
            dataset=["prompt_1"],
            program=None,
            max_steps=1,
        )
      self.orch.run_program.assert_called_once()
      created_program = self.orch.run_program.call_args.kwargs["program"]
      mock_close.assert_called_once_with(created_program)

  def test_run_caller_supplied_program_preserves_external_ownership(self):
    mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    caller_program = mock.MagicMock(spec=rl_program.RLProgram)
    self.orch.run_program = mock.MagicMock()

    self.orch.run(
        algo=mock_algo,
        dataset=["prompt_1"],
        program=caller_program,
        max_steps=1,
    )
    self.orch.run_program.assert_called_once_with(
        program=caller_program,
        bring_up=False,
    )
    caller_program.close.assert_not_called()

  def test_run_auto_instantiated_program_defers_close_for_running_bg_task(self):
    mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    mock_task = mock.MagicMock()
    mock_task.done.return_value = False

    def mock_run_program(program, **kwargs):
      del kwargs
      program._bg_task = mock_task

    self.orch.run_program = mock.MagicMock(side_effect=mock_run_program)

    with mock.patch.object(
        rl_program.StandardRLProgram, "close", autospec=True
    ) as mock_close:
      self.orch.run(
          algo=mock_algo,
          dataset=["prompt_1"],
          program=None,
          max_steps=1,
      )
      self.orch.run_program.assert_called_once()
      # Ensure close() was NOT called prematurely
      mock_close.assert_not_called()
      # Ensure done callback was registered
      mock_task.add_done_callback.assert_called_once()


if __name__ == "__main__":
  absltest.main()
