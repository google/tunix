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
    self.assertTrue(default_ftm.config.enabled)
    self.assertEqual(default_ftm.config.min_active_rollout_ratio, 0.5)

    custom_cfg = datatypes.FaultToleranceConfig(
        enabled=False, max_task_retries=3
    )
    custom_ftm = fault_tolerance_manager.FaultToleranceManager(
        registry, config=custom_cfg
    )
    self.assertEqual(custom_ftm.config, custom_cfg)
    self.assertFalse(custom_ftm.config.enabled)

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
            "ft/fleet/rollout_active_ratio": 1.0,
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
            "ft/fleet/rollout_active_ratio": 1.0,
            "ft/evictions_total/task_session": 0.0,
            "ft/rollout/task_retries_total": 0.0,
            "ft/rollout/task_unrecovered_total": 0.0,
            "ft/rejoin/promoted_total": 0.0,
            "ft/rejoin/bring_up_failures_total": 0.0,
        },
    )

  def test_check_rollout_capacity_not_armed_before_first_active_worker(self):
    registry = worker_registry.WorkerRegistry()
    registry.register_handle(
        "reg-0",
        [datatypes.Role.ROLLOUT],
        mock.MagicMock(spec=remote_execution.ActorHandle),
    )
    registry.register_handle(
        "pending-0",
        [datatypes.Role.ROLLOUT],
        mock.MagicMock(spec=remote_execution.ActorHandle),
    )
    worker_registry_testing.set_status(
        registry,
        "pending-0",
        worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
    )
    ftm = fault_tolerance_manager.FaultToleranceManager(
        registry,
        config=datatypes.FaultToleranceConfig(
            enabled=True,
            min_active_rollout_ratio=1.0,
            min_active_rollout_grace_s=0.0,
        ),
    )
    ftm.check_rollout_capacity()
    self.assertEqual(ftm.metrics()["ft/fleet/rollout_active_ratio"], 1.0)

  def test_check_rollout_capacity_strictly_below_boundary_and_grace_window(
      self,
  ):
    registry = worker_registry.WorkerRegistry()
    for idx in range(4):
      registry.register_handle(
          f"r-{idx}",
          [datatypes.Role.ROLLOUT],
          mock.MagicMock(spec=remote_execution.ActorHandle),
      )
      worker_registry_testing.set_status(
          registry, f"r-{idx}", worker_registry.AdmissionStatus.ACTIVE
      )
    now = [0.0]
    ftm = fault_tolerance_manager.FaultToleranceManager(
        registry,
        config=datatypes.FaultToleranceConfig(
            enabled=True,
            min_active_rollout_ratio=0.5,
            min_active_rollout_grace_s=10.0,
        ),
    )
    with mock.patch.object(
        fault_tolerance_manager, "_monotonic", side_effect=lambda: now[0]
    ):
      ftm.check_rollout_capacity()

      # Exact boundary (2/4 == 0.5) is not strictly below threshold; timer stays
      # disarmed even after 100s.
      registry.unregister("r-0")
      registry.unregister("r-1")
      now[0] = 100.0
      ftm.check_rollout_capacity()
      self.assertEqual(ftm.metrics()["ft/fleet/rollout_active_ratio"], 0.5)

      # Dropping strictly below (1/4 == 0.25 < 0.5) starts the grace timer at
      # t=100.0, holds through t=109.9, and raises at t=110.0 with peak=4
      # intact.
      registry.unregister("r-2")
      ftm.check_rollout_capacity()
      self.assertEqual(ftm.metrics()["ft/fleet/rollout_active_ratio"], 0.25)
      now[0] = 109.9
      ftm.check_rollout_capacity()
      now[0] = 110.0
      with self.assertRaisesRegex(
          fault_tolerance_manager.RolloutCapacityError,
          r"for 10\.0s \(grace 10\.0s\): 1/4 active \(0\.250 < 0\.500\)",
      ):
        ftm.check_rollout_capacity()

  def test_check_rollout_capacity_recovery_resets_timer_and_late_join_raises_peak(
      self,
  ):
    registry = worker_registry.WorkerRegistry()
    for idx in range(2):
      registry.register_handle(
          f"r-{idx}",
          [datatypes.Role.ROLLOUT],
          mock.MagicMock(spec=remote_execution.ActorHandle),
      )
      worker_registry_testing.set_status(
          registry, f"r-{idx}", worker_registry.AdmissionStatus.ACTIVE
      )
    now = [0.0]
    ftm = fault_tolerance_manager.FaultToleranceManager(
        registry,
        config=datatypes.FaultToleranceConfig(
            enabled=True,
            min_active_rollout_ratio=0.75,
            min_active_rollout_grace_s=10.0,
        ),
    )
    with mock.patch.object(
        fault_tolerance_manager, "_monotonic", side_effect=lambda: now[0]
    ):
      ftm.check_rollout_capacity()

      # Drop to 1/2 (0.5 < 0.75) for 9s, then recover to 2/2 before grace
      # expires.
      registry.unregister("r-1")
      ftm.check_rollout_capacity()
      now[0] = 9.0
      ftm.check_rollout_capacity()

      registry.register_handle(
          "r-1",
          [datatypes.Role.ROLLOUT],
          mock.MagicMock(spec=remote_execution.ActorHandle),
      )
      worker_registry_testing.set_status(
          registry, "r-1", worker_registry.AdmissionStatus.ACTIVE
      )
      now[0] = 9.5
      ftm.check_rollout_capacity()

      # Promote 2 late joiners so peak rises to 4 (4/4).
      for idx in (2, 3):
        registry.register_handle(
            f"r-{idx}",
            [datatypes.Role.ROLLOUT],
            mock.MagicMock(spec=remote_execution.ActorHandle),
        )
        worker_registry_testing.set_status(
            registry, f"r-{idx}", worker_registry.AdmissionStatus.ACTIVE
        )
      now[0] = 10.0
      ftm.check_rollout_capacity()

      # Evicting 2 workers drops to 2/4 (0.5 < 0.75 against new peak=4) and must
      # start a fresh 10s grace window at t=10.0 rather than reusing t=0.0.
      registry.unregister("r-2")
      registry.unregister("r-3")
      ftm.check_rollout_capacity()
      now[0] = 19.9
      ftm.check_rollout_capacity()
      now[0] = 20.0
      with self.assertRaisesRegex(
          fault_tolerance_manager.RolloutCapacityError,
          r"for 10\.0s \(grace 10\.0s\): 2/4 active \(0\.500 < 0\.750\)",
      ):
        ftm.check_rollout_capacity()

  def test_check_rollout_capacity_ignored_when_disabled_and_metrics_is_pure(
      self,
  ):
    registry = worker_registry.WorkerRegistry()
    for idx in range(4):
      registry.register_handle(
          f"r-{idx}",
          [datatypes.Role.ROLLOUT],
          mock.MagicMock(spec=remote_execution.ActorHandle),
      )
      worker_registry_testing.set_status(
          registry, f"r-{idx}", worker_registry.AdmissionStatus.ACTIVE
      )
    ftm = fault_tolerance_manager.FaultToleranceManager(
        registry,
        config=datatypes.FaultToleranceConfig(
            enabled=False,
            min_active_rollout_ratio=1.0,
            min_active_rollout_grace_s=0.0,
        ),
    )
    with mock.patch.object(
        fault_tolerance_manager, "_monotonic", return_value=0.0
    ) as mock_monotonic:
      # metrics() is a pure read: reports 1.0 without mutating peak or calling
      # _monotonic.
      self.assertEqual(ftm.metrics()["ft/fleet/rollout_active_ratio"], 1.0)
      self.assertEqual(ftm._peak_active_rollout_workers, 0)
      mock_monotonic.assert_not_called()

      ftm.check_rollout_capacity()
      registry.unregister("r-0")
      registry.unregister("r-1")
      ftm.check_rollout_capacity()
      mock_monotonic.assert_not_called()
      self.assertEqual(ftm.metrics()["ft/fleet/rollout_active_ratio"], 0.5)

  def test_on_worker_unreachable_arms_capacity_grace_timer_without_raising(
      self,
  ):
    registry = worker_registry.WorkerRegistry()
    handles = [
        mock.create_autospec(remote_execution.ActorHandle, instance=True)
        for _ in range(2)
    ]
    for idx, handle in enumerate(handles):
      registry.register_handle(f"r-{idx}", [datatypes.Role.ROLLOUT], handle)
      worker_registry_testing.set_status(
          registry, f"r-{idx}", worker_registry.AdmissionStatus.ACTIVE
      )
    now = [0.0]
    ftm = fault_tolerance_manager.FaultToleranceManager(
        registry,
        config=datatypes.FaultToleranceConfig(
            enabled=True,
            min_active_rollout_ratio=1.0,
            min_active_rollout_grace_s=10.0,
        ),
    )
    with mock.patch.object(
        fault_tolerance_manager, "_monotonic", side_effect=lambda: now[0]
    ):
      ftm.check_rollout_capacity()

      # Eviction at t=2.0 starts the grace timer immediately; advancing to
      # t=12.0 without intermediate check_rollout_capacity() calls must raise
      # on the first check.
      now[0] = 2.0
      ftm.on_worker_unreachable(handles[1], RuntimeError("lost"))
      self.assertEqual(ftm.metrics()["ft/evictions_total/task_session"], 1.0)
      now[0] = 12.0
      with self.assertRaisesRegex(
          fault_tolerance_manager.RolloutCapacityError,
          r"for 10\.0s \(grace 10\.0s\): 1/2 active \(0\.500 < 1\.000\)",
      ):
        ftm.check_rollout_capacity()

    # Even with zero grace, on_worker_unreachable must not raise inside
    # TaskSession's callback.
    zero_grace_registry = worker_registry.WorkerRegistry()
    zero_grace_handles = [
        mock.create_autospec(remote_execution.ActorHandle, instance=True)
        for _ in range(2)
    ]
    for idx, handle in enumerate(zero_grace_handles):
      zero_grace_registry.register_handle(
          f"r-{idx}", [datatypes.Role.ROLLOUT], handle
      )
      worker_registry_testing.set_status(
          zero_grace_registry,
          f"r-{idx}",
          worker_registry.AdmissionStatus.ACTIVE,
      )
    zero_grace_ftm = fault_tolerance_manager.FaultToleranceManager(
        zero_grace_registry,
        config=datatypes.FaultToleranceConfig(
            enabled=True,
            min_active_rollout_ratio=1.0,
            min_active_rollout_grace_s=0.0,
        ),
    )
    zero_grace_ftm.check_rollout_capacity()
    zero_grace_ftm.on_worker_unreachable(
        zero_grace_handles[1], RuntimeError("lost")
    )
    self.assertEqual(
        zero_grace_ftm.metrics()["ft/evictions_total/task_session"], 1.0
    )
    with self.assertRaises(fault_tolerance_manager.RolloutCapacityError):
      zero_grace_ftm.check_rollout_capacity()


if __name__ == "__main__":
  absltest.main()
