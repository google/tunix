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

"""Tests for DistributedRLEngine's deferred update.

`train_step(defer_update=True)` sends a step's last micro-batch with its
update left to `apply_deferred_update`, so the next step's first micro-batch
can be staged in between.
"""

import asyncio
from typing import Any

from absl.testing import absltest
import distributed_rl_engine_staging_test as st
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import trainer_worker


class _DeferringHandle(st._RecordingHandle):
  """Answers a deferred train call as a TrainerWorker that deferred it, and
  `apply_deferred_update` as one that then ran the update."""

  def __init__(self, ignores_defer: bool = False):
    super().__init__(fused=True)
    self._ignores_defer = ignores_defer

  async def asubmit(self, method_name=None, *args, **kwargs):
    if method_name == "apply_deferred_update":
      self.calls.append((method_name, kwargs))
      return datatypes.Response(
          request_id="r",
          metadata={
              "updated": True,
              "train_step": 8,
              "update_ready": True,
              "device_wait_s": 4.9,
          },
      )
    if kwargs.get("defer_update") and not self._ignores_defer:
      self.calls.append((method_name, kwargs))
      return datatypes.Response(
          request_id="r", metadata={"queued": True, "update_deferred": True}
      )
    return await super().asubmit(method_name, *args, **kwargs)


class _Trainer:
  """Records what reaches the trainer behind a real TrainerWorker."""

  def __init__(self):
    self.calls: list[str] = []
    self.policy_version = 0

  def fwd_bwd(self, payload, **kwargs):
    del payload
    self.calls.append(
        f"fwd_bwd(apply_optimizer={kwargs.get('apply_optimizer')})"
    )

  def update(self, **kwargs):
    del kwargs
    self.calls.append("update")
    return 3

  def wait_for_device(self):
    self.calls.append("wait_for_device")
    return 2.5

  def close(self):
    pass


def _real_engine(trainer):
  worker = trainer_worker.TrainerWorker(trainer_factory=lambda: trainer)
  worker.initialize()
  handle = remote_execution.InProcessActorHandle(
      remote_execution.InProcessRemoteExecutionServer(worker)
  )
  return st._engine(handle), worker


class DeferredTrainStepTest(absltest.TestCase):

  def test_defer_sends_the_flag_and_does_not_update(self):
    handle = _DeferringHandle()
    result = asyncio.run(
        st._engine(handle).train_step(
            st._payload(), apply_optimizer=True, defer_update=True
        )
    )
    self.assertEqual([m for m, _ in handle.calls], ["fwd_bwd"])
    _, kwargs = handle.calls[0]
    self.assertTrue(kwargs["defer_update"])
    self.assertTrue(kwargs["apply_optimizer"])
    self.assertTrue(result["update_deferred"])
    self.assertFalse(result["updated"])
    self.assertIsNone(result["train_step"])
    self.assertFalse(result["update_ready"])

  def test_without_defer_the_flag_is_not_sent(self):
    handle = _DeferringHandle()
    result = asyncio.run(
        st._engine(handle).train_step(st._payload(), apply_optimizer=True)
    )
    self.assertNotIn("defer_update", handle.calls[0][1])
    self.assertTrue(result["updated"])
    self.assertNotIn("update_deferred", result)

  def test_defer_needs_apply_optimizer(self):
    handle = _DeferringHandle()
    with self.assertRaisesRegex(ValueError, "needs apply_optimizer=True"):
      asyncio.run(
          st._engine(handle).train_step(
              st._payload(), apply_optimizer=False, defer_update=True
          )
      )
    self.assertEmpty(handle.calls)

  def test_a_worker_that_updated_anyway_is_taken_as_fused(self):
    # One update, never two: the reply says `updated`, not `update_deferred`.
    handle = _DeferringHandle(ignores_defer=True)
    result = asyncio.run(
        st._engine(handle).train_step(
            st._payload(), apply_optimizer=True, defer_update=True
        )
    )
    self.assertEqual([m for m, _ in handle.calls], ["fwd_bwd"])
    self.assertTrue(result["updated"])
    self.assertEqual(result["train_step"], 7)
    self.assertNotIn("update_deferred", result)

  def test_apply_deferred_update_returns_the_workers_report(self):
    handle = _DeferringHandle()
    engine = st._engine(handle)
    result = asyncio.run(engine.apply_deferred_update())
    self.assertEqual([m for m, _ in handle.calls], ["apply_deferred_update"])
    self.assertEqual(
        result,
        {
            "updated": True,
            "train_step": 8,
            "update_ready": True,
            "device_wait_s": 4.9,
        },
    )
    self.assertTrue(engine._actor_weights_dirty)

  def test_apply_deferred_update_without_a_trainer_is_an_error(self):
    engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[], trainer_workers={}
    )
    with self.assertRaisesRegex(ValueError, "No trainer worker"):
      asyncio.run(engine.apply_deferred_update())

  def test_through_a_real_worker(self):
    trainer = _Trainer()
    engine, worker = _real_engine(trainer)
    deferred = asyncio.run(
        engine.train_step(
            st._payload(), apply_optimizer=True, defer_update=True
        )
    )
    self.assertTrue(deferred["update_deferred"])
    self.assertEqual(trainer.calls, ["fwd_bwd(apply_optimizer=True)"])
    staged = asyncio.run(engine.stage_train_step(st._payload()))
    self.assertIsNone(staged.device_wait_s)
    update = asyncio.run(engine.apply_deferred_update())
    self.assertEqual(
        trainer.calls,
        ["fwd_bwd(apply_optimizer=True)", "wait_for_device", "update"],
    )
    self.assertEqual(
        update,
        {
            "updated": True,
            "train_step": 3,
            "update_ready": False,
            "device_wait_s": 2.5,
        },
    )
    asyncio.run(
        engine.train_step(
            None, apply_optimizer=False, staged_request_id=staged.request_id
        )
    )
    self.assertEqual(trainer.calls[-1], "fwd_bwd(apply_optimizer=False)")
    self.assertIsNone(worker._deferred_update)

  def test_staging_support_reports_it(self):
    engine, _ = _real_engine(_Trainer())
    support = asyncio.run(engine.trainer_staging_support())
    self.assertTrue(support["stage_before_update"])


if __name__ == "__main__":
  absltest.main()
