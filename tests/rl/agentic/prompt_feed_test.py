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

"""Tests for prompt_feed."""

import asyncio
import contextlib

from absl.testing import absltest
from tunix.rl.agentic import prompt_feed


def _put_batch(q, batch):
  """Queues `batch` unchanged."""
  q.put(batch)


def _feed(num_batches, prefill_batches, **kwargs):
  """Returns a feed over batches [0, 1], [2, 3], ... of single prompts."""
  return prompt_feed.QueuePromptFeed(
      iter([2 * i, 2 * i + 1] for i in range(num_batches)),
      put_batch=_put_batch,
      split=lambda batch: batch,
      prefill_batches=prefill_batches,
      **kwargs,
  )


def _drain(feed, count):
  """Returns the first `count` prompts of `feed`."""

  async def _run():
    prompts = []
    async for prompt in feed.prompts():
      prompts.append(prompt)
      if len(prompts) == count:
        break
    return prompts

  return asyncio.run(asyncio.wait_for(_run(), timeout=5.0))


def _drain_all(feed):
  """Returns every prompt of `feed` until it is closed."""

  async def _run():
    return [prompt async for prompt in feed.prompts()]

  return asyncio.run(asyncio.wait_for(_run(), timeout=5.0))


class QueuePromptFeedTest(absltest.TestCase):

  def test_prefills_batches(self):
    self.assertEqual(_drain(_feed(5, prefill_batches=2), 4), [0, 1, 2, 3])

  def test_global_step_queues_the_next_batch_inside_the_loading_span(self):
    spans = []

    @contextlib.contextmanager
    def _span():
      spans.append("enter")
      yield

    feed = _feed(5, prefill_batches=1, loading_span=_span)
    feed.on_global_step(done=False)
    feed.close()
    self.assertEqual(_drain_all(feed), [0, 1, 2, 3])
    self.assertEqual(spans, ["enter"])

  def test_done_closes_the_feed(self):
    feed = _feed(5, prefill_batches=1)
    feed.on_global_step(done=True)
    self.assertEqual(_drain_all(feed), [0, 1])

  def test_exhausted_dataset_closes_the_feed(self):
    feed = _feed(2, prefill_batches=1)
    feed.on_global_step(done=False)
    feed.on_global_step(done=False)
    self.assertEqual(_drain_all(feed), [0, 1, 2, 3])

  def test_prefill_beyond_the_dataset_closes_the_feed(self):
    self.assertEqual(_drain_all(_feed(1, prefill_batches=3)), [0, 1])


if __name__ == "__main__":
  absltest.main()
