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

"""Unit tests for tunix.experimental.common.timeouts."""

from absl.testing import absltest
from absl.testing import parameterized
from tunix.experimental.common import timeouts


class TimeoutsTest(parameterized.TestCase):

  def test_default_chain_values(self):
    self.assertEqual(timeouts.DEFAULT_EPISODE_TIMEOUT_S, 600.0)
    self.assertEqual(timeouts.DEFAULT_DRAIN_TIMEOUT_S, 660.0)
    self.assertEqual(timeouts.DEFAULT_PRE_WEIGHT_SYNC_TIMEOUT_S, 900.0)
    drain_s = timeouts.drain_timeout_for(timeouts.DEFAULT_EPISODE_TIMEOUT_S)
    self.assertEqual(drain_s, timeouts.DEFAULT_DRAIN_TIMEOUT_S)
    pre_s = timeouts.pre_weight_sync_timeout_for(drain_s)
    self.assertEqual(pre_s, timeouts.DEFAULT_PRE_WEIGHT_SYNC_TIMEOUT_S)

  def test_deepswe_chain_values(self):
    drain_s = timeouts.drain_timeout_for(1800)
    self.assertEqual(drain_s, 1860.0)
    pre_s = timeouts.pre_weight_sync_timeout_for(drain_s)
    self.assertEqual(pre_s, 2100.0)

  @parameterized.named_parameters(
      ("zero", 0),
      ("negative", -1.0),
      ("nan", float("nan")),
      ("inf", float("inf")),
      ("neg_inf", float("-inf")),
      ("bool_true", True),
      ("bool_false", False),
  )
  def test_rejects_non_positive_or_non_finite_values(self, invalid):
    with self.assertRaises(ValueError):
      timeouts.drain_timeout_for(invalid)
    with self.assertRaises(ValueError):
      timeouts.pre_weight_sync_timeout_for(invalid)


if __name__ == "__main__":
  absltest.main()
