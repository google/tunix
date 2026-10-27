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

"""Smoke tests for FakeClusterHarness and RecordingFaultHandler."""

import threading
from absl.testing import absltest
from tunix.dist_rl.testing import fake_cluster


class FakeClusterTest(absltest.TestCase):

  def test_fake_cluster_e2e_two_steps(self) -> None:
    with fake_cluster.FakeClusterHarness(
        num_rollout_workers=2,
        weight_sync_mode="fallback",
    ) as harness:
      program = harness.build_program(
          num_prompts=4,
          max_steps=2,
          num_generations=2,
          mini_batch_size=2,
          sync_weights=True,
      )
      future = harness.start_program(program)
      future.result(timeout=20.0)

      self.assertEqual(harness.fake_trainer.step_count, 2)
      self.assertEqual(harness.fake_trainer.policy_version, 2)
      for wid, sampler in harness.samplers.items():
        self.assertEqual(
            sampler.policy_version,
            2,
            msg=f"{wid} policy_version mismatch",
        )

    # After harness.close(), verify that all RunningWorkerServer threads have terminated
    remaining_worker_threads = [
        t
        for t in threading.enumerate()
        if "worker-server" in t.name and t.is_alive()
    ]
    self.assertEmpty(
        remaining_worker_threads,
        msg=f"Leaked worker threads: {remaining_worker_threads}",
    )

  def test_fake_cluster_with_recording_fault_handler(self) -> None:
    handler = fake_cluster.RecordingFaultHandler()
    with fake_cluster.FakeClusterHarness(
        num_rollout_workers=2,
        weight_sync_mode="fallback",
        weight_sync_handler=handler,
    ) as harness:
      program = harness.build_program(
          num_prompts=2,
          max_steps=1,
          num_generations=2,
          mini_batch_size=2,
          sync_weights=True,
      )
      future = harness.start_program(program)
      future.result(timeout=20.0)

      self.assertEqual(harness.fake_trainer.step_count, 1)
      self.assertNotEmpty(handler.transfers)
      self.assertNotEmpty(handler.registered_work_units)


if __name__ == "__main__":
  absltest.main()
