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

import tempfile
from unittest import mock

from absl.testing import absltest
from etils import epath
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import orchestrator
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory import trajectory_testing
from tunix.experimental.weight_sync import weight_sync_coordinator
from tunix.experimental.worker import abstract_worker
from tunix.experimental.worker import remote_execution


class ClusterOrchestratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_registry = mock.MagicMock()
    self.mock_registry.worker_ids.return_value = []
    self.mock_registry.infos.return_value = []
    self.mock_lifecycle = mock.MagicMock()
    self.mock_monitor = mock.MagicMock()
    self.orch = orchestrator.ClusterOrchestrator(
        registry=self.mock_registry,
        lifecycle_driver=self.mock_lifecycle,
        monitor=self.mock_monitor,
    )

  def test_empty_registry_is_preserved_in_init(self):
    empty_registry = worker_registry.WorkerRegistry()
    orch = orchestrator.ClusterOrchestrator(registry=empty_registry)
    self.assertIs(orch.registry, empty_registry)

  def test_bring_up_and_shutdown(self):
    self.orch.bring_up_workers("dummy_warmup_data")
    self.mock_lifecycle.bring_up.assert_called_once_with("dummy_warmup_data")

    self.orch.shutdown()
    self.mock_monitor.close.assert_called_once()
    self.mock_lifecycle.shutdown.assert_called_once()
    self.mock_registry.close.assert_called_once()

  def test_create_engine(self):
    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_critic = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_ref = mock.MagicMock(spec=remote_execution.ActorHandle)

    registry = worker_registry.WorkerRegistry()
    orch = orchestrator.ClusterOrchestrator(registry=registry)
    rollout_info = orch.registry.register_handle(
        "rollout-0", [datatypes.Role.ROLLOUT], mock_rollout
    )
    actor_info = orch.registry.register_handle(
        "actor-0", [datatypes.Role.ACTOR], mock_actor
    )
    critic_info = orch.registry.register_handle(
        "critic-0", [datatypes.Role.CRITIC], mock_critic
    )
    ref_info = orch.registry.register_handle(
        "reference-0", [datatypes.Role.REFERENCE], mock_ref
    )

    engine = orch._create_engine()
    self.assertEqual(
        orch.registry.handles(datatypes.Role.ROLLOUT), [mock_rollout]
    )
    self.assertIs(
        engine._role_handle(datatypes.Role.ACTOR),
        mock_actor,
    )
    self.assertIs(
        engine._role_handle(datatypes.Role.CRITIC),
        mock_critic,
    )
    self.assertIs(engine._role_handle(datatypes.Role.REFERENCE), mock_ref)
    self.assertSequenceEqual(
        orch.registry.infos(),
        [actor_info, critic_info, ref_info, rollout_info],
    )

  def test_create_engine_with_weight_sync_handle_registrations(self):
    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_actor = mock.MagicMock(spec=remote_execution.ActorHandle)

    registry = worker_registry.WorkerRegistry()
    orch = orchestrator.ClusterOrchestrator(
        registry=registry, weight_sync_mode="fallback"
    )
    orch.registry.register_handle(
        "rollout-0", [datatypes.Role.ROLLOUT], mock_rollout
    )
    orch.registry.register_handle("actor-0", [datatypes.Role.ACTOR], mock_actor)

    engine = orch._create_engine()
    self.assertIsNotNone(engine._weight_sync_coordinator)

    self.assertIn("actor-0", orch.registry.worker_ids())
    self.assertIn("rollout-0", orch.registry.worker_ids())
    self.assertEqual(orch.registry.handles(datatypes.Role.ACTOR), [mock_actor])
    self.assertEqual(
        orch.registry.handles(datatypes.Role.ROLLOUT), [mock_rollout]
    )

  def test_bring_up_and_shutdown_remote_worker_handles(self):
    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_rollout.submit.return_value = datatypes.Response()
    mock_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_actor.submit.return_value = datatypes.Response()

    registry = worker_registry.WorkerRegistry()
    orch = orchestrator.ClusterOrchestrator(
        registry=registry,
        monitor=self.mock_monitor,
    )
    orch.registry.register_handle(
        "rollout-0", [datatypes.Role.ROLLOUT], mock_rollout
    )
    orch.registry.register_handle("actor-0", [datatypes.Role.ACTOR], mock_actor)

    orch.bring_up_workers(dummy_data="dummy")
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
    mock_rollout.submit.assert_any_call("stop")
    mock_actor.submit.assert_any_call("stop")
    mock_rollout.close.assert_awaited_once()
    mock_actor.close.assert_awaited_once()

  def test_create_engine_wraps_local_workers_as_in_process_handles(self):
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

      def generate(self, requests):
        del requests
        return []

    registry = worker_registry.WorkerRegistry()
    registry.register(LocalWorker())

    orch = orchestrator.ClusterOrchestrator(registry=registry)
    engine = orch._create_engine()
    self.assertIsInstance(
        engine._rollout_handles()[0], remote_execution.InProcessActorHandle
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


def _trajectory_store_orchestrator(
    **kwargs,
) -> orchestrator.ClusterOrchestrator:
  mock_registry = mock.MagicMock()
  mock_registry.worker_ids.return_value = []
  mock_registry.infos.return_value = []
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

  def test_enabled_file_backend_builds_store_once(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    orch = _trajectory_store_orchestrator(
        trajectory_store_config={
            "backend": "file",
            "root_dir": str(tmp_dir),
            "run_id": "cluster_run",
        }
    )
    self.assertIsInstance(orch.trajectory_store, file_store.FileTrajectoryStore)
    self.assertIs(
        orch.trajectory_store.bound_metadata_cls,
        trajectory_lib.TunixTrajectoryMetadata,
    )
    orch.shutdown()

  def test_shutdown_closes_the_store(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    orch = _trajectory_store_orchestrator(
        trajectory_store_config={
            "backend": "file",
            "root_dir": str(tmp_dir),
            "run_id": "cluster_run",
        }
    )
    store = orch.trajectory_store
    orch.shutdown()
    with self.assertRaises(RuntimeError):
      store.add_step(
          trajectory_testing.TUNIX_ENV_STEP_0,
          trajectory_testing.TUNIX_METADATA_1,
      )

  def test_shutdown_closes_the_store_even_when_a_prior_step_raises(self):
    orch = _trajectory_store_orchestrator()
    orch.trajectory_store = mock.MagicMock()
    orch.monitor.close = mock.MagicMock(
        side_effect=RuntimeError("monitor close failed")
    )
    with self.assertRaises(RuntimeError):
      orch.shutdown()
    orch.lifecycle_driver.shutdown.assert_called_once()
    orch.registry.close.assert_called_once()
    orch.trajectory_store.close.assert_called_once()

  def test_shutdown_without_a_store_does_not_raise(self):
    orch = _trajectory_store_orchestrator()
    orch.shutdown()

  def test_bring_up_workers_verifies_matching_remote_rollout_store_config(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    cfg = {
        "backend": "file",
        "root_dir": str(tmp_dir),
        "run_id": "shared_run",
    }
    orch = orchestrator.ClusterOrchestrator(
        registry=worker_registry.WorkerRegistry(),
        monitor=mock.MagicMock(),
        trajectory_store_config=cfg,
    )
    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_rollout.submit.side_effect = [
        datatypes.Response(metadata={"trajectory_store_config": cfg}),
        datatypes.Response(),
        datatypes.Response(),
        datatypes.Response(),
    ]
    orch.registry.register_handle(
        "rollout-0", [datatypes.Role.ROLLOUT], mock_rollout
    )
    orch.bring_up_workers()
    orch.shutdown()

  def test_bring_up_workers_raises_on_remote_rollout_store_config_mismatch(
      self,
  ):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    orch_cfg = {
        "backend": "file",
        "root_dir": str(tmp_dir),
        "run_id": "orch_run",
    }
    worker_cfg = {
        "backend": "file",
        "root_dir": str(tmp_dir),
        "run_id": "worker_run",
    }
    orch = orchestrator.ClusterOrchestrator(
        registry=worker_registry.WorkerRegistry(),
        monitor=mock.MagicMock(),
        trajectory_store_config=orch_cfg,
    )
    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_rollout.submit.side_effect = [
        datatypes.Response(metadata={"trajectory_store_config": worker_cfg}),
        datatypes.Response(),
        datatypes.Response(),
        datatypes.Response(),
    ]
    orch.registry.register_handle(
        "rollout-0", [datatypes.Role.ROLLOUT], mock_rollout
    )
    with self.assertRaisesRegex(
        ValueError,
        "Rollout worker 'rollout-0' trajectory_store_config .* does not match",
    ):
      orch.bring_up_workers()
    orch.shutdown()

  def test_bring_up_workers_raises_when_rollout_enables_store_but_orchestrator_disables(
      self,
  ):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    worker_cfg = {
        "backend": "file",
        "root_dir": str(tmp_dir),
        "run_id": "worker_run",
    }
    orch = orchestrator.ClusterOrchestrator(
        registry=worker_registry.WorkerRegistry(),
        monitor=mock.MagicMock(),
        trajectory_store_config=None,
    )
    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_rollout.submit.side_effect = [
        datatypes.Response(metadata={"trajectory_store_config": worker_cfg}),
        datatypes.Response(),
        datatypes.Response(),
        datatypes.Response(),
    ]
    orch.registry.register_handle(
        "rollout-0", [datatypes.Role.ROLLOUT], mock_rollout
    )
    with self.assertRaisesRegex(
        ValueError,
        "Rollout worker 'rollout-0' trajectory_store_config .* does not match",
    ):
      orch.bring_up_workers()
    orch.shutdown()

  def test_bring_up_workers_raises_when_pre_timeout_shorter_than_drain(self):
    # Rollout worker drain_timeout_s=1860s with default PhaseTimeouts
    # (pre=900s) and partial_rollout=False raises ValueError.
    orch_mismatch = orchestrator.ClusterOrchestrator(
        registry=worker_registry.WorkerRegistry(),
        monitor=mock.MagicMock(),
        weight_sync_mode="fallback",
    )
    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_rollout.submit.side_effect = [
        datatypes.Response(
            metadata={"drain_timeout_s": 1860.0, "partial_rollout": False}
        ),
        datatypes.Response(),
        datatypes.Response(),
        datatypes.Response(),
    ]
    orch_mismatch.registry.register_handle(
        "rollout-slow", [datatypes.Role.ROLLOUT], mock_rollout
    )
    with self.assertRaisesRegex(
        ValueError,
        "Rollout worker 'rollout-slow' has drain_timeout_s=1860.0"
        " \\(partial_rollout=False\\), which requires PhaseTimeouts.pre >="
        " 2100.0s, but orchestrator weight_sync_timeouts.pre is 900.0s",
    ):
      orch_mismatch.bring_up_workers()
    orch_mismatch.shutdown()

  def test_bring_up_workers_accepts_aligned_weight_sync_timeouts(self):
    # Rollout worker drain_timeout_s=1860s with aligned
    # PhaseTimeouts.for_drain_timeout(1860.0) succeeds and forwards timeouts
    # to WeightSyncCoordinator.
    aligned_timeouts = weight_sync_coordinator.PhaseTimeouts.for_drain_timeout(
        1860.0
    )
    orch_aligned = orchestrator.ClusterOrchestrator(
        registry=worker_registry.WorkerRegistry(),
        monitor=mock.MagicMock(),
        weight_sync_mode="fallback",
        weight_sync_timeouts=aligned_timeouts,
    )
    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_rollout.submit.side_effect = [
        datatypes.Response(
            metadata={"drain_timeout_s": 1860.0, "partial_rollout": False}
        ),
        datatypes.Response(),
        datatypes.Response(),
        datatypes.Response(),
    ]
    orch_aligned.registry.register_handle(
        "rollout-aligned", [datatypes.Role.ROLLOUT], mock_rollout
    )
    orch_aligned.bring_up_workers()
    assert orch_aligned.engine is not None
    assert orch_aligned.engine._weight_sync_coordinator is not None  # pylint: disable=protected-access
    self.assertEqual(
        orch_aligned.engine._weight_sync_coordinator._timeouts,  # pylint: disable=protected-access
        aligned_timeouts,
    )
    orch_aligned.shutdown()

  def test_bring_up_workers_skips_drain_check_for_partial_rollout(self):
    # Rollout worker drain_timeout_s=1860s with partial_rollout=True does not
    # drain in pre_weight_sync, so default PhaseTimeouts (pre=900s) is accepted.
    orch_partial = orchestrator.ClusterOrchestrator(
        registry=worker_registry.WorkerRegistry(),
        monitor=mock.MagicMock(),
        weight_sync_mode="fallback",
    )
    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_rollout.submit.side_effect = [
        datatypes.Response(
            metadata={"drain_timeout_s": 1860.0, "partial_rollout": True}
        ),
        datatypes.Response(),
        datatypes.Response(),
        datatypes.Response(),
    ]
    orch_partial.registry.register_handle(
        "rollout-partial", [datatypes.Role.ROLLOUT], mock_rollout
    )
    orch_partial.bring_up_workers()
    orch_partial.shutdown()

  def test_bring_up_workers_raises_when_rollout_missing_timeout_metadata(self):
    orch = orchestrator.ClusterOrchestrator(
        registry=worker_registry.WorkerRegistry(),
        monitor=mock.MagicMock(),
        weight_sync_mode="fallback",
    )
    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_rollout.submit.return_value = datatypes.Response()
    orch.registry.register_handle(
        "rollout-incomplete",
        [datatypes.Role.ROLLOUT],
        mock_rollout,
        resources={"secret_key": "secret_password"},
    )
    with self.assertRaises(ValueError) as ctx:
      orch.bring_up_workers()
    self.assertIn(
        "Rollout worker 'rollout-incomplete' is missing required resource key",
        str(ctx.exception),
    )
    self.assertIn("secret_key", str(ctx.exception))
    self.assertNotIn("secret_password", str(ctx.exception))
    orch.shutdown()

  def test_bring_up_workers_skips_timeout_check_when_weight_sync_mode_none(
      self,
  ):
    for mode in (None, "none"):
      orch = orchestrator.ClusterOrchestrator(
          registry=worker_registry.WorkerRegistry(),
          monitor=mock.MagicMock(),
          weight_sync_mode=mode,
      )
      mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
      mock_rollout.submit.side_effect = [
          datatypes.Response(
              metadata={"drain_timeout_s": 1860.0, "partial_rollout": False}
          ),
          datatypes.Response(),
          datatypes.Response(),
          datatypes.Response(),
      ]
      orch.registry.register_handle(
          "rollout-nosync", [datatypes.Role.ROLLOUT], mock_rollout
      )
      orch.bring_up_workers()
      orch.shutdown()


if __name__ == "__main__":
  absltest.main()
