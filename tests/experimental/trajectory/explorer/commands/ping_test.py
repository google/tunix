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

"""Unit tests for PingCommand execution."""

import json
from absl.testing import absltest
from absl.testing import parameterized
from tunix.experimental.trajectory.explorer.commands import ping
from tunix.experimental.trajectory.explorer.commands import testing


class PingTest(parameterized.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.cmd = ping.PingCommand()

  @parameterized.named_parameters(
      ("empty_store", False, 0),
      ("prefilled_store", True, 2),
  )
  def test_execute_human_readable(
      self, prefill: bool, expected_count: int
  ) -> None:
    mem_store = testing.create_store(prefill=prefill)
    exit_code, out = testing.execute_command(
        self.cmd, mem_store, output_json=False
    )
    self.assertEqual(exit_code, 0)
    self.assertIn(
        "Trajectory Explorer initialized successfully."
        f" ({expected_count} trajectories accessible)",
        out,
    )

  def test_execute_json_output(self) -> None:
    mem_store = testing.create_store(prefill=True)
    exit_code, out = testing.execute_command(
        self.cmd, mem_store, output_json=True
    )
    self.assertEqual(exit_code, 0)
    data = json.loads(out)
    self.assertEqual(data, {"status": "ok", "trajectories_count": 2})


if __name__ == "__main__":
  absltest.main()
