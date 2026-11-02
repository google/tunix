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

from unittest import mock

from absl.testing import absltest
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import fault_tolerance_manager
from tunix.dist_rl.orchestrator import lifecycle
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.testing import worker_registry_testing
from tunix.dist_rl.worker import remote_execution
from tunix.dist_rl.worker import task_session


class FaultToleranceManagerTest(absltest.TestCase):

  def test_default_and_custom_config(self):
    registry = worker_registry.WorkerRegistry()
    default_ftm = fault_tolerance_manager.FaultToleranceManager(registry)
    self.assertEqual(default_ftm.config, datatypes.FaultToleranceConfig())
    self.assertFalse(default_ftm.config.enabled)

    custom_cfg = datatypes.FaultToleranceConfig(
        enabled=True, max_task_retries=3
    )
    custom_ftm = fault_tolerance_manager.FaultToleranceManager(
        registry, config=custom_cfg
    )
    self.assertEqual(custom_ftm.config, custom_cfg)

  def test_task_session_fault_tolerance_config_derivation(self):
    registry = worker_registry.WorkerRegistry()
    disabled_ftm = fault_tolerance_manager.FaultToleranceManager(
        registry,
        config=datatypes.FaultToleranceConfig(
            enabled=False, max_task_retries=4
        ),
    )
    self.assertEqual(
        disabled_ftm.task_session_fault_tolerance_config(),
        task_session.TaskSessionFaultToleranceConfig(
            max_task_retries=0,
            on_worker_unreachable=None,
        ),
    )

    enabled_ftm = fault_tolerance_manager.FaultToleranceManager(
        registry,
        config=datatypes.FaultToleranceConfig(enabled=True, max_task_retries=4),
    )
    self.assertEqual(
        enabled_ftm.task_session_fault_tolerance_config(),
        task_session.TaskSessionFaultToleranceConfig(
            max_task_retries=4,
            on_worker_unreachable=enabled_ftm.on_worker_unreachable,
        ),
    )

  def test_on_worker_unreachable_evicts_registered_handle(self):
    registry = worker_registry.WorkerRegistry()
    h0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    registry.register_handle("rollout-0", [datatypes.Role.ROLLOUT], h0)
    ftm = fault_tolerance_manager.FaultToleranceManager(registry)
    self.assertEqual(ftm.metrics()["ft/evictions_total/task_session"], 0.0)

    ftm.on_worker_unreachable(h0, ConnectionError("peer reset connection"))

    self.assertNotIn("rollout-0", registry)
    h0.sync_close.assert_called_once()
    # Calling a second time on an already evicted handle is a safe no-op.
    ftm.on_worker_unreachable(h0, ConnectionError("duplicate"))
    h0.sync_close.assert_called_once()
    self.assertEqual(ftm.metrics()["ft/evictions_total/task_session"], 1.0)

  def test_metrics_aggregates_fleet_gauges_and_counters(self):
    registry = worker_registry.WorkerRegistry()
    for idx in range(3):
      h_active = mock.MagicMock(spec=remote_execution.ActorHandle)
      registry.register_handle(
          f"active-{idx}", [datatypes.Role.ROLLOUT], h_active
      )
      worker_registry_testing.set_status(
          registry, f"active-{idx}", worker_registry.AdmissionStatus.ACTIVE
      )
    for idx in range(2):
      h_pending = mock.MagicMock(spec=remote_execution.ActorHandle)
      registry.register_handle(
          f"pending-{idx}", [datatypes.Role.ROLLOUT], h_pending
      )
      worker_registry_testing.set_status(
          registry,
          f"pending-{idx}",
          worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
      )
    registry.register_handle(
        "registered-0",
        [datatypes.Role.ROLLOUT],
        mock.MagicMock(spec=remote_execution.ActorHandle),
    )
    h_evict = mock.MagicMock(spec=remote_execution.ActorHandle)
    registry.register_handle("evict-0", [datatypes.Role.ROLLOUT], h_evict)

    mock_lifecycle = mock.MagicMock(
        spec=lifecycle.LifecycleDriver,
        bring_up_failures_total=4,
    )
    mock_session = mock.MagicMock(
        spec=task_session.TaskSession,
        task_retries_total=5,
        task_unrecovered_total=6,
    )
    ftm = fault_tolerance_manager.FaultToleranceManager(
        registry,
        lifecycle_driver=mock_lifecycle,
    )
    ftm.on_worker_unreachable(h_evict, ConnectionError("evicted"))
    ftm.record_rejoin_promoted()
    ftm.record_rejoin_promoted()

    self.assertEqual(
        ftm.metrics(rollout_session=mock_session),
        {
            "ft/fleet/rollout_registered": 1.0,
            "ft/fleet/rollout_pending_weight_sync": 2.0,
            "ft/fleet/rollout_active": 3.0,
            "ft/evictions_total/task_session": 1.0,
            "ft/rollout/task_retries_total": 5.0,
            "ft/rollout/task_unrecovered_total": 6.0,
            "ft/rejoin/promoted_total": 2.0,
            "ft/rejoin/bring_up_failures_total": 4.0,
        },
    )

    minimal_ftm = fault_tolerance_manager.FaultToleranceManager(
        worker_registry.WorkerRegistry()
    )
    self.assertEqual(
        minimal_ftm.metrics(),
        {
            "ft/fleet/rollout_registered": 0.0,
            "ft/fleet/rollout_pending_weight_sync": 0.0,
            "ft/fleet/rollout_active": 0.0,
            "ft/evictions_total/task_session": 0.0,
            "ft/rollout/task_retries_total": 0.0,
            "ft/rollout/task_unrecovered_total": 0.0,
            "ft/rejoin/promoted_total": 0.0,
            "ft/rejoin/bring_up_failures_total": 0.0,
        },
    )


if __name__ == "__main__":
  absltest.main()
