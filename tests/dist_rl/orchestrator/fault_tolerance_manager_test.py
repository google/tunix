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
from tunix.dist_rl.orchestrator import worker_registry
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

    ftm.on_worker_unreachable(h0, ConnectionError("peer reset connection"))

    self.assertNotIn("rollout-0", registry)
    h0.sync_close.assert_called_once()
    # Calling a second time on an already evicted handle is a safe no-op.
    ftm.on_worker_unreachable(h0, ConnectionError("duplicate"))
    h0.sync_close.assert_called_once()


if __name__ == "__main__":
  absltest.main()
