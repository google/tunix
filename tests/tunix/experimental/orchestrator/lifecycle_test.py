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

"""Tests for the LifecycleDriver."""

from absl.testing import absltest
from tunix.tunix.experimental.orchestrator import fake_worker
from tunix.tunix.experimental.orchestrator import lifecycle
from tunix.tunix.experimental.orchestrator import worker_registry


_DUMMY_DATA = {"max_prompt_length": 4, "max_response_tokens": 8}


class LifecycleDriverTest(absltest.TestCase):

  def test_bring_up_transitions_fakes_to_ready(self):
    registry = worker_registry.WorkerRegistry()
    rollout = fake_worker.FakeWorker(worker_id="r0", roles={"rollout"})
    trainer = fake_worker.FakeWorker(worker_id="t0", roles={"trainer"})
    registry.register(rollout)
    registry.register(trainer)

    lifecycle.LifecycleDriver(registry).bring_up(_DUMMY_DATA)

    self.assertEqual(rollout.state, "READY")
    self.assertEqual(trainer.state, "READY")

  def test_shutdown_transitions_fakes_to_stopped(self):
    registry = worker_registry.WorkerRegistry()
    trainer = fake_worker.FakeWorker(worker_id="t0", roles={"trainer"})
    registry.register(trainer)
    driver = lifecycle.LifecycleDriver(registry)
    driver.bring_up(_DUMMY_DATA)

    driver.shutdown()

    self.assertEqual(trainer.state, "STOPPED")

  def test_bring_up_runs_phase_by_phase(self):
    registry = worker_registry.WorkerRegistry()
    log: list[str] = []
    registry.register(fake_worker.FakeWorker("a", roles={"trainer"}, log=log))
    registry.register(fake_worker.FakeWorker("b", roles={"trainer"}, log=log))

    lifecycle.LifecycleDriver(registry).bring_up(_DUMMY_DATA)

    # All initializes precede all compiles, which precede all starts.
    phases = [entry.split(":")[1] for entry in log]
    self.assertEqual(
        phases,
        ["initialize", "initialize", "compile", "compile", "start", "start"],
    )

  def test_shutdown_is_best_effort_and_aggregates_failures(self):
    registry = worker_registry.WorkerRegistry()
    log: list[str] = []
    registry.register(fake_worker.FakeWorker("a", roles={"trainer"}, log=log))
    registry.register(
        fake_worker.FakeWorker("b", roles={"trainer"}, log=log, fail_stop=True)
    )
    registry.register(fake_worker.FakeWorker("c", roles={"trainer"}, log=log))
    driver = lifecycle.LifecycleDriver(registry)

    with self.assertRaises(lifecycle.LifecycleError) as ctx:
      driver.shutdown()

    # Every worker was asked to stop even though "b" raised.
    self.assertEqual(sorted(log), ["a:stop", "b:stop", "c:stop"])
    self.assertEqual([wid for wid, _ in ctx.exception.failures], ["b"])


if __name__ == "__main__":
  absltest.main()
