# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for rl_loop_remote_execution_demo.py."""

import asyncio
from absl.testing import absltest
import portpicker

from tunix.experimental.worker.examples import rl_loop_remote_execution_demo as rl_demo


class RlLoopRemoteExecutionDemoTest(absltest.TestCase):

  def test_rl_training_loop(self):
    port_1 = portpicker.pick_unused_port()
    port_2 = portpicker.pick_unused_port()

    async def _run_test():
      # Start two mock rollout worker servers
      server_1 = await rl_demo.start_mock_rollout_server(
          port=port_1, worker_id="worker-1"
      )
      server_2 = await rl_demo.start_mock_rollout_server(
          port=port_2, worker_id="worker-2"
      )

      try:
        addresses = [
            f"grpc://localhost:{port_1}",
            f"grpc://localhost:{port_2}",
        ]
        metrics = await rl_demo.run_rl_training_loop(
            worker_addresses=addresses,
            num_epochs=2,
            batch_size=4,
        )

        self.assertLen(metrics, 2)
        for i, m in enumerate(metrics):
          self.assertEqual(m["epoch"], i)
          self.assertEqual(m["completed"], 4)
          self.assertEqual(m["failed"], 0)
          self.assertGreaterEqual(m["avg_reward"], 0.0)
          self.assertLessEqual(m["avg_reward"], 1.0)
      finally:
        await server_1.stop_serving()
        await server_2.stop_serving()

    asyncio.run(_run_test())


if __name__ == "__main__":
  absltest.main()
