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
    orch.register_worker_handle(
        "actor-0", [datatypes.Role.ACTOR], mock_actor
    )

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

  def test_run_managed_program_submission(self):
    mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    mock_algo.group_size = 2
    mock_algo.mini_batch_size = 1
    mock_algo.max_turns = 1
    mock_algo.max_packed_len = 16
    mock_algo.requires_reference_kl = False

    assembler = batch_assembly.SequencePackedBatchAssembler(max_packed_len=16)

    with mock.patch("asyncio.run") as mock_asyncio_run:
      self.orch.run(
          algo=mock_algo,
          dataset=["prompt1"],
          reward_fns=[lambda x: 1.0],
          assembler=assembler,
          num_steps=5,
      )
      self.mock_lifecycle.bring_up.assert_called_once()
      self.mock_monitor.poll.assert_called_once()
      mock_asyncio_run.assert_called_once()

  def test_run_program_with_bring_up_and_train_dataset(self):
    mock_program = mock.MagicMock(spec=rl_program.RLProgram)
    mock_engine = mock.MagicMock()

    with mock.patch.object(self.orch, "_create_engine", return_value=mock_engine):
      self.orch.run_program(
          program=mock_program,
          train_dataset=["batch1", "batch2"],
          num_steps=10,
          bring_up=True,
          dummy_data="dummy_init",
      )

    self.mock_lifecycle.bring_up.assert_called_once_with("dummy_init")
    self.mock_monitor.poll.assert_called_once()
    mock_program.run.assert_called_once_with(
        engine=mock_engine,
        train_dataset=["batch1", "batch2"],
        num_steps=10,
    )

  def test_run_program_without_bring_up(self):
    mock_program = mock.MagicMock(spec=rl_program.RLProgram)
    mock_engine = mock.MagicMock()
    self.orch.engine = mock_engine

    self.orch.run_program(
        program=mock_program,
        num_steps=2,
        bring_up=False,
    )

    self.mock_lifecycle.bring_up.assert_not_called()
    self.mock_monitor.poll.assert_called_once()
    mock_program.run.assert_called_once_with(
        engine=mock_engine,
        train_dataset=None,
        num_steps=2,
    )


if __name__ == "__main__":
  absltest.main()
