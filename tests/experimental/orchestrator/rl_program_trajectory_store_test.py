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

"""Tests for StandardRLProgram's (non-owning) use of a Trajectory Store."""

from unittest import mock

from absl.testing import absltest
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.trajectory import in_memory_store


class StandardRLProgramTrajectoryStoreTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    self.mock_algo.group_size = 2
    self.mock_algo.mini_batch_size = 1
    self.mock_algo.max_packed_len = 16
    self.assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        group_size=2,
        mini_batch_size=1,
        max_packed_len=16,
    )

  def _create_program(self, **kwargs) -> rl_program.StandardRLProgram:
    return rl_program.StandardRLProgram(
        dataset=["prompt_0"],
        algo=self.mock_algo,
        reward_fns=[lambda x: 1.0],
        assembler=self.assembler,
        **kwargs,
    )

  def test_no_trajectory_store_by_default(self):
    program = self._create_program()
    self.assertIsNone(program._trajectory_store)  # pylint: disable=protected-access
    program.close()

  def test_holds_the_instance_it_was_given(self):
    # The program never builds its own store — it only ever holds whatever
    # instance its owner (a ClusterOrchestrator) hands it.
    store = in_memory_store.InMemoryTrajectoryStore()
    program = self._create_program(trajectory_store=store)
    self.assertIs(program._trajectory_store, store)  # pylint: disable=protected-access
    program.close()

  def test_close_does_not_close_an_injected_store(self):
    # StandardRLProgram does not own the store's lifecycle: closing it here
    # would break a ClusterOrchestrator that runs a second program against
    # the same store afterwards.
    store = mock.MagicMock(spec=in_memory_store.InMemoryTrajectoryStore)
    program = self._create_program(trajectory_store=store)
    program.close()
    store.close.assert_not_called()

  def test_close_without_a_store_does_not_raise(self):
    program = self._create_program()
    program.close()


if __name__ == "__main__":
  absltest.main()
