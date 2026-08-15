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

"""Unit tests for agentic_remote_execution_demo.py in worker/examples."""

import asyncio
from absl.testing import absltest
import portpicker

from tunix.experimental.worker.examples import agentic_remote_execution_demo as demo


class AgenticRemoteExecutionDemoTest(absltest.TestCase):

  def test_distributed_rollout_workflow(self):
    port = portpicker.pick_unused_port()

    async def _run_test():
      # Start worker server
      worker_server = await demo.run_worker_node(
          port=port, worker_id="test-worker"
      )

      try:
        # Run orchestrator client
        address = f"grpc://localhost:{port}"
        response = await demo.run_orchestrator_node(address)

        # Assert response details
        self.assertEqual(response.request_id, "req_group4_pair0")
        self.assertEqual(response.metadata.get("worker_id"), "test-worker")
        self.assertIn("k8s-pod-101", response.metadata.get("observation"))
        self.assertEqual(response.metadata.get("reward"), 1.0)
      finally:
        await worker_server.stop_serving()

    asyncio.run(_run_test())


if __name__ == "__main__":
  absltest.main()
