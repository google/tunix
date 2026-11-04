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

"""Tests for group_filters."""

import types

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.rl.agentic import group_filters


def _group(*statuses):
  """Returns a group with one item per episode status."""
  return [types.SimpleNamespace(traj={"status": s}) for s in statuses]


class GroupFiltersTest(parameterized.TestCase):

  def test_validate_accepts_known(self):
    group_filters.validate_group_filters(())
    group_filters.validate_group_filters(("drop_failed", "zero_variance"))

  @parameterized.parameters((("nope",),), (("zero_variance", "zero_variance"),))
  def test_validate_rejects(self, filters):
    with self.assertRaises(ValueError):
      group_filters.validate_group_filters(filters)

  def test_rejects_bare_string(self):
    with self.assertRaisesRegex(ValueError, "sequence of names"):
      group_filters.validate_group_filters("zero_variance")

  @parameterized.parameters("FAILED", "ENV_TIMEOUT", "CANCELLED")
  def test_system_errors_drop(self, status):
    self.assertEqual(
        group_filters.drop_reason(
            ("drop_failed",), _group("SUCCEEDED", status), [1.0, -1.0]
        ),
        "drop_failed",
    )

  @parameterized.parameters("SUCCEEDED", "MAX_CONTEXT_LIMIT_REACHED", "TIMEOUT")
  def test_policy_outcomes_are_kept(self, status):
    self.assertIsNone(
        group_filters.drop_reason(
            ("drop_failed",), _group(status, status), [1.0, -1.0]
        )
    )

  def test_zero_variance(self):
    self.assertEqual(
        group_filters.drop_reason(
            ("zero_variance",), _group("SUCCEEDED"), [0.0, 0.0]
        ),
        "zero_variance",
    )
    self.assertIsNone(
        group_filters.drop_reason(
            ("zero_variance",), _group("SUCCEEDED"), [0.0, 0.5]
        )
    )

  def test_zero_variance_tolerates_float_noise_of_tied_rewards(self):
    # (r - mean) / (std + 1e-6) for tied non-binary rewards is ~1e-2, not 0.
    rewards = np.array([0.9] * 6)
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    self.assertEqual(
        group_filters.drop_reason(("zero_variance",), _group("SUCCEEDED"), adv),
        "zero_variance",
    )

  def test_first_matching_filter_wins_and_empty_keeps(self):
    group = _group("FAILED")
    self.assertEqual(
        group_filters.drop_reason(
            ("zero_variance", "drop_failed"), group, [0.0]
        ),
        "zero_variance",
    )
    self.assertIsNone(group_filters.drop_reason((), group, [0.0]))


if __name__ == "__main__":
  absltest.main()
