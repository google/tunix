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

"""Integration tests for the RLLoopDriver against the fakes."""

import asyncio

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import lifecycle
from tunix.experimental.orchestrator import rl_loop_driver
from tunix.experimental.orchestrator import weight_sync_coordinator
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.testing import fake_rollout_worker
from tunix.experimental.testing import fake_trainer_worker
from tunix.experimental.testing import toy_trainer
from tunix.experimental.worker import trainer_worker


_TOKENIZER = datatypes.TokenizerInfo(pad_id=0, eos_id=1)
_SHAPE = datatypes.ShapeConfig(max_prompt_length=2, max_response_tokens=3)
_ROWS = [{"prompt_text": "a"}, {"prompt_text": "b"}]


def _brought_up_registry(trainer):
  registry = worker_registry.WorkerRegistry()
  registry.register(fake_rollout_worker.FakeRolloutWorker(worker_id="r0"))
  registry.register(trainer)
  lifecycle.LifecycleDriver(registry).bring_up(_SHAPE)
  return registry

def _driver(registry, **kwargs):
  return rl_loop_driver.RLLoopDriver(
      registry=registry,
      adapter=algorithm_adapter.AgenticGRPOAdapter(group_size=2),
      tokenizer_info=_TOKENIZER,
      shape_config=_SHAPE,
      group_size=2,
      **kwargs,
  )


class RLLoopDriverTest(absltest.TestCase):

  def test_run_step_trains_all_groups_and_releases_credits(self):
    registry = _brought_up_registry(
        fake_trainer_worker.FakeTrainerWorker(worker_id="t0")
    )
    driver = _driver(registry, dispatch_credit_capacity=4)

    outcome = asyncio.run(driver.run_step(_ROWS))

    self.assertEqual(outcome.num_groups_trained, 2)  # one per prompt row
    self.assertEqual(outcome.num_groups_dropped, 0)
    self.assertEqual(outcome.num_updates, 1)
    self.assertEqual(driver.credits.in_use(), 0)  # all groups terminal
    # FakeTrainerWorker advances its step on each update.
    self.assertEqual(registry.get("t0").health().policy_version, 1)

  def test_multiple_microbatch_iterations_apply_multiple_updates(self):
    registry = _brought_up_registry(
        fake_trainer_worker.FakeTrainerWorker(worker_id="t0")
    )
    driver = _driver(registry, num_microbatch_iterations=3)

    outcome = asyncio.run(driver.run_step(_ROWS))

    self.assertEqual(outcome.num_updates, 3)
    self.assertEqual(registry.get("t0").health().policy_version, 3)

  def test_step_counter_advances_across_steps(self):
    registry = _brought_up_registry(
        fake_trainer_worker.FakeTrainerWorker(worker_id="t0")
    )
    driver = _driver(registry)

    first = asyncio.run(driver.run_step(_ROWS))
    second = asyncio.run(driver.run_step(_ROWS))

    self.assertEqual(first.step, 0)
    self.assertEqual(second.step, 1)
    self.assertEqual(driver.step, 2)

  def test_run_step_syncs_weights_and_advances_policy_version(self):
    registry = _brought_up_registry(
        fake_trainer_worker.FakeTrainerWorker(worker_id="t0")
    )
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(registry)
    driver = _driver(registry, weight_sync_coordinator=coordinator)

    outcome = asyncio.run(driver.run_step(_ROWS))

    self.assertEqual(outcome.policy_version, 1)
    self.assertEqual(outcome.num_replicas_synced, 1)  # the single rollout fake
    self.assertEqual(registry.get("r0").health().policy_version, 1)

  def test_no_sync_without_a_coordinator(self):
    registry = _brought_up_registry(
        fake_trainer_worker.FakeTrainerWorker(worker_id="t0")
    )
    driver = _driver(registry)
    outcome = asyncio.run(driver.run_step(_ROWS))
    self.assertEqual(outcome.policy_version, 0)
    self.assertEqual(outcome.num_replicas_synced, 0)

  def test_drives_a_real_toy_trainer(self):
    registry = _brought_up_registry(
        trainer_worker.TrainerWorker(
            trainer_factory=toy_trainer.ToyAbstractTrainer, worker_id="t0"
        )
    )
    driver = _driver(registry)

    outcome = asyncio.run(driver.run_step(_ROWS))

    self.assertEqual(outcome.num_updates, 1)
    self.assertTrue(outcome.update_results[0].applied)
    self.assertEqual(outcome.update_results[0].step, 1)


if __name__ == "__main__":
  absltest.main()
