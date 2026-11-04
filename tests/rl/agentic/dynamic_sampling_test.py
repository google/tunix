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

"""Tests for dynamic_sampling."""

import asyncio
import types

from absl.testing import absltest
from absl.testing import parameterized
from tunix.rl.agentic import dynamic_sampling

_TIMEOUT_S = 5.0
# Time given to the sampler to (wrongly) admit more prompts before a test
# asserts that admission is blocked.
_SETTLE_S = 0.05


def _group(prompt_id, status="SUCCEEDED"):
  """Returns a one-item group with the given prompt id and episode status."""
  return [types.SimpleNamespace(prompt_id=prompt_id, traj={"status": status})]


async def _wait_until(predicate):
  """Polls `predicate` on the running loop until it holds or times out."""
  async with asyncio.timeout(_TIMEOUT_S):
    while not predicate():
      await asyncio.sleep(0.001)


class _Consumer:
  """Collects the prompts that a sampler admits, in a background task."""

  def __init__(self, sampler):
    self.admitted = []
    self.task = asyncio.create_task(self._consume(sampler))

  async def _consume(self, sampler):
    async for prompt in sampler.prompts():
      self.admitted.append(prompt)

  async def wait_admitted(self, count):
    """Waits until `count` prompts are admitted, then checks for no more."""
    await _wait_until(lambda: len(self.admitted) >= count)
    await asyncio.sleep(_SETTLE_S)
    return list(self.admitted)

  async def wait_done(self):
    """Waits until `prompts()` returns."""
    await asyncio.wait_for(self.task, _TIMEOUT_S)
    return list(self.admitted)


class DynamicSamplerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._loop = asyncio.new_event_loop()
    self.addCleanup(self._loop.close)

  def _sampler(self, num_prompts=8, step=None, loop=None, **kwargs):
    """Returns a sampler over `num_prompts` prompts in batches of 2."""
    step = step if step is not None else [0]
    options = dict(
        batches=iter(
            list(range(i, min(i + 2, num_prompts)))
            for i in range(0, num_prompts, 2)
        ),
        split=lambda batch: batch,
        loop=loop or self._loop,
        group_filters=("zero_variance",),
        max_consecutive_filtered_groups=-1,
        off_policy_steps=1,
        full_batch_size=2,
        max_in_flight_groups=None,
        global_step_fn=lambda: step[0],
    )
    options.update(kwargs)
    return dynamic_sampling.DynamicSampler(**options)

  @parameterized.parameters(1, 5)
  def test_rejects_max_in_flight_outside_batch_and_staleness(self, value):
    # batch_size=2, (off_policy_steps + 1) * batch_size = 4.
    with self.assertRaisesRegex(ValueError, "max_in_flight_groups"):
      self._sampler(max_in_flight_groups=value)

  def test_window_capacity(self):
    self.assertEqual(self._sampler().window_capacity, 4)
    self.assertEqual(self._sampler(max_in_flight_groups=3).window_capacity, 3)

  def test_check_group_raises_for_dropped_group(self):
    sampler = self._sampler()
    sampler.check_group(_group(0), [1.0, -1.0])
    with self.assertRaises(dynamic_sampling.GroupDroppedError) as cm:
      sampler.check_group(_group(1), [0.0, 0.0])
    self.assertEqual(cm.exception.reason, "zero_variance")

  def test_expected_step_counts_kept_groups_from_start_step(self):
    sampler = self._sampler(step=[3])
    self.assertEqual(sampler.expected_step, 3)
    sampler.on_group_kept()
    sampler.on_group_dropped(_group(1), "zero_variance")
    self.assertEqual(sampler.expected_step, 3)
    sampler.on_group_kept()
    self.assertEqual(sampler.expected_step, 4)

  def test_stall_guard_counts_consecutive_drops_only(self):
    sampler = self._sampler(max_consecutive_filtered_groups=2)
    sampler.on_group_dropped(_group(0), "zero_variance")
    sampler.on_group_kept()
    sampler.on_group_dropped(_group(1), "zero_variance")
    with self.assertRaisesRegex(
        dynamic_sampling.DynamicSamplingStalledError, "2 consecutive"
    ):
      sampler.on_group_dropped(_group(2), "zero_variance")

  def test_admission_follows_window_drops_and_global_steps(self):
    async def _run():
      step = [0]
      sampler = self._sampler(
          step=step, loop=asyncio.get_running_loop(), max_in_flight_groups=3
      )
      consumer = _Consumer(sampler)
      self.assertEqual(await consumer.wait_admitted(3), [0, 1, 2])

      # One dropped group admits exactly one replacement prompt.
      sampler.on_group_dropped(_group(0), "zero_variance")
      self.assertEqual(await consumer.wait_admitted(4), [0, 1, 2, 3])

      # A global step consumes a batch of 2 prompts.
      step[0] = 1
      sampler.on_global_step(done=False)
      self.assertEqual(await consumer.wait_admitted(6), list(range(6)))

      # The last step stops admission while prompts remain.
      sampler.on_global_step(done=True)
      self.assertEqual(await consumer.wait_done(), list(range(6)))

    asyncio.run(_run())

  def test_prompts_end_with_the_dataset(self):
    async def _run():
      sampler = self._sampler(num_prompts=3, loop=asyncio.get_running_loop())
      return await _Consumer(sampler).wait_done()

    self.assertEqual(asyncio.run(_run()), [0, 1, 2])

  def test_close_unblocks_a_waiting_admission(self):
    async def _run():
      sampler = self._sampler(
          loop=asyncio.get_running_loop(),
          max_in_flight_groups=2,
      )
      consumer = _Consumer(sampler)
      self.assertEqual(await consumer.wait_admitted(2), [0, 1])
      sampler.close()
      self.assertEqual(await consumer.wait_done(), [0, 1])

    asyncio.run(_run())

  def test_stop_does_not_pull_extra_batch_from_iterator(self):
    async def _run():
      pulled = 0

      def batches():
        nonlocal pulled
        for batch in ([0, 1], [2, 3], [4, 5]):
          pulled += 1
          yield batch

      sampler = self._sampler(
          batches=batches(), loop=asyncio.get_running_loop()
      )
      stream = aiter(sampler.prompts())
      self.assertEqual(await anext(stream), 0)
      self.assertEqual(await anext(stream), 1)
      self.assertEqual(pulled, 1)
      sampler.on_global_step(done=True)
      await asyncio.sleep(0)
      self.assertIsNone(await anext(stream, None))
      self.assertEqual(pulled, 1)

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
