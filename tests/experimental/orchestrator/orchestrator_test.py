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
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import orchestrator


class ClusterOrchestratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_registry = mock.MagicMock()
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
    mock_rollout = mock.MagicMock()
    mock_actor = mock.MagicMock()
    mock_critic = mock.MagicMock()
    mock_ref = mock.MagicMock()

    def mock_group(role):
      grp = mock.MagicMock()
      if role == datatypes.Role.ROLLOUT:
        grp.members.return_value = [mock_rollout]
      elif role == datatypes.Role.ACTOR:
        grp.members.return_value = [mock_actor]
      elif role == datatypes.Role.CRITIC:
        grp.members.return_value = [mock_critic]
      elif role == datatypes.Role.REFERENCE:
        grp.members.return_value = [mock_ref]
      else:
        grp.members.return_value = []
      return grp

    self.mock_registry.group.side_effect = mock_group

    engine = self.orch.create_engine()
    self.assertIs(
        engine._trainer_workers[datatypes.Role.ACTOR],
        mock_actor,
    )
    self.assertIs(
        engine._trainer_workers[datatypes.Role.CRITIC],
        mock_critic,
    )
    self.assertIs(engine._inference_worker, mock_ref)

  def test_run_program_basic(self):
    mock_program = mock.MagicMock()
    dataset = [["prompt1"]]
    self.orch.run_program(
        program=mock_program,
        train_dataset=dataset,
        num_steps=10,
        bring_up=True,
        dummy_data="dummy",
    )
    self.mock_lifecycle.bring_up.assert_called_once_with("dummy")
    self.mock_monitor.poll.assert_called_once()
    mock_program.run.assert_called_once_with(
        train_dataset=dataset, num_steps=10
    )


if __name__ == "__main__":
  absltest.main()
