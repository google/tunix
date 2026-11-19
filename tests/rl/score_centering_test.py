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

"""Pure-NumPy tests for Score Centering coefficients and diagnostics."""

from absl.testing import absltest
import numpy as np
from tunix.rl import score_centering


class ScoreCenteringTest(absltest.TestCase):

  def test_numpy_coefficients_and_diagnostics(self):
    sampler_logps = np.array(
        [[[np.log(0.6), np.log(0.3), -np.inf], [-np.inf, -np.inf, -np.inf]]],
        dtype=np.float32,
    )
    trainer_logps = np.array(
        [[[np.log(0.2), np.log(0.5), -1.0], [-0.1, -0.2, -0.3]]],
        dtype=np.float32,
    )
    topk_ids = np.array([[[4, 0, 0], [0, 0, 0]]], dtype=np.int32)
    completion_ids = np.array([[0, 0]], dtype=np.int32)

    coeff, stats, has_valid = (
        score_centering.compute_score_centering_coefficients(
            trainer_logps,
            sampler_logps,
            sampler_is="token",
            sampler_is_threshold=2.0,
            eps=1e-6,
            completion_ids=completion_ids,
            topk_token_ids=topk_ids,
        )
    )
    np.testing.assert_array_equal(has_valid, [[True, False]])
    self.assertEqual(coeff.shape, (1, 2, 3))
    self.assertEqual(float(coeff[0, 0, 2]), 0.0)
    np.testing.assert_array_equal(coeff[0, 1], [0.0, 0.0, 0.0])

    # q = [0.6, 0.3], p = [0.2, 0.5] -> tail_q = 0.1, tail_p = 0.3, rho = 1/3.
    # TV(q_hat, p) = 0.5 * (|0.6-0.2| + |0.3-0.5|) + 0.5 * |0.7 - 0.9| = 0.4.
    self.assertAlmostEqual(float(stats["head_mass_q"][0, 0]), 0.9, places=6)
    self.assertAlmostEqual(float(stats["head_mass_p"][0, 0]), 0.7, places=6)
    self.assertAlmostEqual(
        float(stats["tail_ratio_rho"][0, 0]), 1.0 / 3.0, places=6
    )
    self.assertAlmostEqual(float(stats["tv_distance"][0, 0]), 0.4, places=6)
    self.assertEqual(float(stats["tail_eps_clamped"][0, 0]), 0.0)
    self.assertEqual(float(stats["tail_is_clipped"][0, 0]), 0.0)
    # Token 0 is valid at pos 0 (slot 1) and padded (-inf) at pos 1 -> [1, 0].
    np.testing.assert_array_equal(stats["sampled_in_topk"], [[1.0, 0.0]])

  def test_eps_clamp_and_tail_clip_flags(self):
    # q sums to 1.0 (clamped by eps); p sums to 0.2 -> rho = 1e-3 / 0.8,
    # and with q=0.2, p=0.9999 -> rho = 0.8 / 1e-3 = 800, c * rho > 1.
    sampler_logps = np.log(np.array([[[0.2, 0.0]]], dtype=np.float32))
    sampler_logps[0, 0, 1] = -np.inf
    trainer_logps = np.log(np.array([[[0.9999, 0.5]]], dtype=np.float32))
    _, stats, _ = score_centering.compute_score_centering_coefficients(
        trainer_logps,
        sampler_logps,
        sampler_is="token",
        sampler_is_threshold=2.0,
        eps=1e-3,
    )
    self.assertEqual(float(stats["tail_eps_clamped"][0, 0]), 1.0)
    self.assertEqual(float(stats["tail_is_clipped"][0, 0]), 1.0)


if __name__ == "__main__":
  absltest.main()
