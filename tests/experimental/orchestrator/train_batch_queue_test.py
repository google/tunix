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

"""Tests for the bounded TrainBatchQueue."""

from absl.testing import absltest
from tunix.experimental.orchestrator import train_batch_queue


class TrainBatchQueueTest(absltest.TestCase):

  def test_fifo_order(self):
    queue = train_batch_queue.TrainBatchQueue(maxsize=3)
    for item in ("a", "b", "c"):
      self.assertTrue(queue.put(item))
    self.assertEqual([queue.get(), queue.get(), queue.get()], ["a", "b", "c"])

  def test_put_fails_when_full(self):
    queue = train_batch_queue.TrainBatchQueue(maxsize=2)
    self.assertTrue(queue.put(1))
    self.assertTrue(queue.put(2))
    self.assertTrue(queue.is_full())
    self.assertFalse(queue.put(3))  # rejected, not enqueued
    self.assertLen(queue, 2)
    self.assertEqual(queue.remaining(), 0)

  def test_get_on_empty_raises(self):
    queue = train_batch_queue.TrainBatchQueue(maxsize=1)
    with self.assertRaises(train_batch_queue.QueueEmpty):
      queue.get()
    self.assertIsNone(queue.try_get())

  def test_put_after_get_frees_a_slot(self):
    queue = train_batch_queue.TrainBatchQueue(maxsize=1)
    self.assertTrue(queue.put("x"))
    self.assertFalse(queue.put("y"))
    self.assertEqual(queue.get(), "x")
    self.assertTrue(queue.put("y"))

  def test_maxsize_must_be_positive(self):
    with self.assertRaises(ValueError):
      train_batch_queue.TrainBatchQueue(maxsize=0)


if __name__ == "__main__":
  absltest.main()
