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

"""Tests for the I7 denominator-aware loss scales."""

from absl.testing import absltest
from tunix.experimental.orchestrator import loss_scale


class LossScaleTest(absltest.TestCase):

  def test_equal_denominators_give_uniform_scales(self):
    scales = loss_scale.loss_scales_from_denominators([4.0, 4.0, 4.0, 4.0])
    self.assertEqual(scales, [0.25, 0.25, 0.25, 0.25])
    self.assertAlmostEqual(sum(scales), 1.0)

  def test_scales_are_proportional_to_denominators(self):
    scales = loss_scale.loss_scales_from_denominators([3.0, 2.0])
    self.assertAlmostEqual(scales[0], 0.6)
    self.assertAlmostEqual(scales[1], 0.4)
    self.assertAlmostEqual(sum(scales), 1.0)

  def test_all_zero_denominators_give_zero_scales(self):
    self.assertEqual(
        loss_scale.loss_scales_from_denominators([0.0, 0.0]), [0.0, 0.0]
    )

  def test_empty_raises(self):
    with self.assertRaises(ValueError):
      loss_scale.loss_scales_from_denominators([])

  def test_negative_raises(self):
    with self.assertRaises(ValueError):
      loss_scale.loss_scales_from_denominators([1.0, -1.0])


if __name__ == "__main__":
  absltest.main()
