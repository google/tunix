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

"""Runs the localhost demo end to end: real gRPC workers, CPU toy model."""

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator.examples import localhost_rl_demo


class LocalhostRlDemoTest(absltest.TestCase):

  def test_demo_trains_with_every_worker_call_over_grpc(self):
    summary = localhost_rl_demo.run_demo(prompts=["1"])

    # One gRPC endpoint per worker role, all distinct.
    self.assertCountEqual(
        summary["ports"].keys(), ["trainer", "rollout", "inference"]
    )
    self.assertLen(set(summary["ports"].values()), 3)

    # The control plane brought the remote fleet up.
    self.assertLen(summary["health"], 3)
    for state in summary["health"].values():
      self.assertEqual(state, str(datatypes.WorkerState.READY))

    # Training actually happened through the remote handles.
    self.assertEqual(summary["global_steps"], 1)
    self.assertTrue(summary["weights_changed"])


if __name__ == "__main__":
  absltest.main()
