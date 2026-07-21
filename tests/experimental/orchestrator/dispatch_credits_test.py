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

"""Tests for DispatchCredits."""

from absl.testing import absltest
from tunix.experimental.orchestrator import dispatch_credits


class DispatchCreditsTest(absltest.TestCase):

  def test_acquire_and_release_roundtrip(self):
    credits = dispatch_credits.DispatchCredits(capacity=4)
    self.assertEqual(credits.available(), 4)
    self.assertTrue(credits.try_acquire(3))
    self.assertEqual(credits.in_use(), 3)
    self.assertEqual(credits.available(), 1)
    credits.release(2)
    self.assertEqual(credits.available(), 3)

  def test_acquire_fails_when_insufficient(self):
    credits = dispatch_credits.DispatchCredits(capacity=2)
    self.assertTrue(credits.try_acquire(2))
    self.assertFalse(credits.try_acquire(1))  # none left; not acquired
    self.assertEqual(credits.in_use(), 2)

  def test_over_release_raises(self):
    credits = dispatch_credits.DispatchCredits(capacity=2)
    credits.try_acquire(1)
    with self.assertRaises(ValueError):
      credits.release(2)

  def test_negative_amounts_raise(self):
    credits = dispatch_credits.DispatchCredits(capacity=2)
    with self.assertRaises(ValueError):
      credits.try_acquire(-1)
    with self.assertRaises(ValueError):
      credits.release(-1)

  def test_zero_capacity_never_admits(self):
    credits = dispatch_credits.DispatchCredits(capacity=0)
    self.assertFalse(credits.try_acquire(1))
    self.assertTrue(credits.try_acquire(0))


if __name__ == "__main__":
  absltest.main()
