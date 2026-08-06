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
from tunix.experimental.orchestrator import rl_program


class RLProgramTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_driver = mock.MagicMock()
    self.mock_driver.policy_version = 0
    self.mock_driver.generate.return_value = ["mock_rollout"]
    self.mock_driver.process_results.return_value = ["mock_train_example"]
    self.mock_driver.train_step.return_value = "mock_train_result"

  def test_step_once_flow(self):
    begin_calls = []
    end_calls = []

    def on_begin(step):
      begin_calls.append(step)

    def on_end(step, result):
      end_calls.append((step, result))

    program = rl_program.RLProgram(
        driver=self.mock_driver,
        on_step_begin=on_begin,
        on_step_end=on_end,
    )

    res = program.step_once(prompts=["prompt1", "prompt2"])

    self.assertEqual(res, "mock_train_result")
    self.mock_driver.generate.assert_called_once_with(
        prompts=["prompt1", "prompt2"]
    )
    self.mock_driver.process_results.assert_called_once_with(["mock_rollout"])
    self.mock_driver.train_step.assert_called_once_with("mock_train_example")
    self.mock_driver.sync_weights.assert_called_once()
    self.assertEqual(self.mock_driver.policy_version, 1)

    self.assertEqual(begin_calls, [0])
    self.assertEqual(end_calls, [(1, "mock_train_result")])

  def test_eval_step_once_flow(self):
    program = rl_program.RLProgram(driver=self.mock_driver)
    res = program.eval_step_once(prompts=["eval_prompt"])

    self.assertEqual(res, ["mock_train_example"])
    self.mock_driver.generate.assert_called_once_with(prompts=["eval_prompt"])
    self.mock_driver.process_results.assert_called_once_with(["mock_rollout"])
    self.mock_driver.train_step.assert_not_called()
    self.mock_driver.rl_engine.sync_weights.assert_not_called()
    self.assertEqual(self.mock_driver.policy_version, 0)

  def test_run_loop(self):
    dataset = [["batch1_p1"], ["batch2_p1"], ["batch3_p1"]]
    program = rl_program.RLProgram(driver=self.mock_driver)
    program.run(train_dataset=dataset, num_steps=2)

    self.assertEqual(self.mock_driver.generate.call_count, 2)
    self.assertEqual(self.mock_driver.policy_version, 2)


if __name__ == "__main__":
  absltest.main()
