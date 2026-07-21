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
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import lifecycle
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.testing import fake_rollout_worker
from tunix.experimental.testing import fake_trainer_worker
from tunix.experimental.worker import abstract_worker


class _RecordingWorker(abstract_worker.Worker):
  """Records lifecycle calls into a shared log for ordering assertions."""

  def __init__(self, worker_id: str, log: list[str], *, fail_stop=False):
    self._worker_id = worker_id
    self._log = log
    self._fail_stop = fail_stop

  def initialize(self) -> None:
    self._log.append(f"{self._worker_id}:initialize")

  def compile(self, shape_config) -> None:
    del shape_config
    self._log.append(f"{self._worker_id}:compile")

  def start(self) -> None:
    self._log.append(f"{self._worker_id}:start")

  def stop(self) -> None:
    self._log.append(f"{self._worker_id}:stop")
    if self._fail_stop:
      raise RuntimeError(f"{self._worker_id} refused to stop")

  def health(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(state="READY")

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id, roles=frozenset({"trainer"})
    )


_SHAPE = datatypes.ShapeConfig(max_prompt_length=4, max_response_tokens=8)


class LifecycleDriverTest(absltest.TestCase):

  def test_bring_up_transitions_fakes_to_ready(self):
    registry = worker_registry.WorkerRegistry()
    rollout = fake_rollout_worker.FakeRolloutWorker(worker_id="r0")
    trainer = fake_trainer_worker.FakeTrainerWorker(worker_id="t0")
    registry.register(rollout)
    registry.register(trainer)

    lifecycle.LifecycleDriver(registry).bring_up(_SHAPE)

    self.assertEqual(rollout.health().state, "READY")
    self.assertEqual(trainer.health().state, "READY")

  def test_shutdown_transitions_fakes_to_stopped(self):
    registry = worker_registry.WorkerRegistry()
    trainer = fake_trainer_worker.FakeTrainerWorker(worker_id="t0")
    registry.register(trainer)
    driver = lifecycle.LifecycleDriver(registry)
    driver.bring_up(_SHAPE)

    driver.shutdown()

    self.assertEqual(trainer.health().state, "STOPPED")

  def test_bring_up_runs_phase_by_phase(self):
    registry = worker_registry.WorkerRegistry()
    log: list[str] = []
    registry.register(_RecordingWorker("a", log))
    registry.register(_RecordingWorker("b", log))

    lifecycle.LifecycleDriver(registry).bring_up(_SHAPE)

    # All initializes precede all compiles, which precede all starts.
    phases = [entry.split(":")[1] for entry in log]
    self.assertEqual(
        phases,
        ["initialize", "initialize", "compile", "compile", "start", "start"],
    )

  def test_shutdown_is_best_effort_and_aggregates_failures(self):
    registry = worker_registry.WorkerRegistry()
    log: list[str] = []
    registry.register(_RecordingWorker("a", log))
    registry.register(_RecordingWorker("b", log, fail_stop=True))
    registry.register(_RecordingWorker("c", log))
    driver = lifecycle.LifecycleDriver(registry)

    with self.assertRaises(lifecycle.LifecycleError) as ctx:
      driver.shutdown()

    # Every worker was asked to stop even though "b" raised.
    self.assertEqual(sorted(log), ["a:stop", "b:stop", "c:stop"])
    self.assertEqual([wid for wid, _ in ctx.exception.failures], ["b"])


if __name__ == "__main__":
  absltest.main()
