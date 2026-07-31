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

"""Workers that disagree about their configuration must not start a run."""

from typing import Any

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import startup_validation
from tunix.experimental.orchestrator import worker_fleet
from tunix.experimental.worker import abstract_worker


class _Worker(abstract_worker.Worker):
  """A worker that declares how it is configured."""

  def __init__(self, worker_id: str, role: str = "rollout", **resources: Any):
    self._worker_id = worker_id
    self._role = role
    self._resources = resources

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id,
        roles=frozenset({self._role}),
        resources=self._resources,
    )

  def initialize(self) -> datatypes.Response:
    return datatypes.Response()

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    del dummy_data
    return datatypes.Response()

  def start(self) -> datatypes.Response:
    return datatypes.Response()

  def stop(self) -> datatypes.Response:
    return datatypes.Response()

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(state=datatypes.WorkerState.READY)


def _configured(worker_id: str, role: str = "rollout", **overrides):
  resources = startup_validation.describe_resources(
      "sha-abc", pad_id=0, eos_id=2, vocab_size=32, temperature=0.7
  )
  resources.update(overrides)
  return _Worker(worker_id, role, **resources)


class StartupValidationTest(absltest.TestCase):

  def test_agreeing_workers_pass(self):
    problems = startup_validation.validate_workers(
        [_configured("r0"), _configured("r1"), _configured("t0", "trainer")]
    )
    self.assertEmpty(problems)

  def test_a_different_tokenizer_is_rejected(self):
    problems = startup_validation.validate_workers(
        [_configured("r0"), _configured("r1", tokenizer_hash="sha-other")]
    )

    self.assertLen(problems, 1)
    self.assertIn("tokenizer_hash", problems[0])
    self.assertIn("r1", problems[0])

  def test_a_different_sampling_temperature_is_rejected(self):
    problems = startup_validation.validate_workers(
        [_configured("r0"), _configured("r1", temperature=1.0)]
    )

    self.assertLen(problems, 1)
    self.assertIn("temperature", problems[0])

  def test_a_different_padding_token_is_rejected(self):
    problems = startup_validation.validate_workers(
        [_configured("r0"), _configured("r1", pad_id=7)]
    )

    self.assertIn("pad_id", problems[0])

  def test_every_problem_is_reported_at_once(self):
    """Finding these one restart at a time is its own failure."""
    problems = startup_validation.validate_workers([
        _configured("r0"),
        _configured("r1", tokenizer_hash="sha-other", pad_id=7, eos_id=9),
    ])

    self.assertLen(problems, 3)
    self.assertCountEqual(
        [p.split()[0] for p in problems],
        ["tokenizer_hash", "pad_id", "eos_id"],
    )

  def test_a_worker_that_declares_nothing_is_rejected(self):
    problems = startup_validation.validate_workers(
        [_configured("r0"), _Worker("silent")]
    )

    self.assertTrue(any("does not declare" in p for p in problems))

  def test_workers_agreeing_with_each_other_but_not_the_run(self):
    """Unanimity is not correctness if they all disagree with the run."""
    problems = startup_validation.validate_workers(
        [_configured("r0"), _configured("r1")],
        expected={"tokenizer_hash": "sha-expected"},
    )

    self.assertLen(problems, 1)
    self.assertIn("should be 'sha-expected'", problems[0])

  def test_require_agreement_raises_with_every_problem(self):
    with self.assertRaises(
        startup_validation.StartupValidationError
    ) as caught:
      startup_validation.require_agreement(
          [_configured("r0"), _configured("r1", pad_id=7, eos_id=9)]
      )

    self.assertLen(caught.exception.problems, 2)

  def test_no_workers_is_itself_a_problem(self):
    self.assertNotEmpty(startup_validation.validate_workers([]))


class FleetStartupValidationTest(absltest.TestCase):

  def test_bring_up_can_refuse_a_mismatched_fleet(self):
    fleet = worker_fleet.WorkerFleet(
        rollout=[_configured("r0"), _configured("r1", pad_id=7)]
    )

    with self.assertRaises(startup_validation.StartupValidationError):
      fleet.bring_up(validate=True)

  def test_bring_up_proceeds_when_the_fleet_agrees(self):
    fleet = worker_fleet.WorkerFleet(
        rollout=[_configured("r0"), _configured("r1")],
        trainer=_configured("t0", "trainer"),
    )

    fleet.bring_up(validate=True)

    self.assertLen(fleet.poll_health(), 3)

  def test_validation_is_off_unless_asked_for(self):
    """One process shares one config; there is nothing to disagree about."""
    fleet = worker_fleet.WorkerFleet(rollout=[_Worker("silent")])

    fleet.bring_up()


if __name__ == "__main__":
  absltest.main()
