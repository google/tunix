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

"""Unit tests for the explorer's terminal formatting helpers."""

from absl.testing import absltest
from tunix.experimental.trajectory.explorer.commands import formatting
from tunix.experimental.trajectory.explorer.diagnostics import stats


class FormattingTest(absltest.TestCase):

  def test_opt_renders_none_as_missing(self) -> None:
    self.assertEqual(formatting.format_optional(None), formatting.MISSING)
    self.assertEqual(
        formatting.format_optional(None, ".3g"), formatting.MISSING
    )

  def test_opt_applies_format_spec(self) -> None:
    self.assertEqual(formatting.format_optional(3), "3")
    self.assertEqual(formatting.format_optional(2.0 / 3, ".3g"), "0.667")

  def test_status_renders_name(self) -> None:
    self.assertEqual(formatting.format_status(None), formatting.MISSING)
    self.assertEqual(
        formatting.format_status(stats.TrajectoryStatus.RUNNING), "RUNNING"
    )
    self.assertEqual(
        formatting.format_status(stats.TrajectoryStatus.SUCCEEDED), "SUCCEEDED"
    )
    self.assertEqual(
        formatting.format_status(stats.TrajectoryStatus.TIMEOUT), "TIMEOUT"
    )

  def test_steps(self) -> None:
    counts = stats.StepCounts(agent=3, user=2, system=1, other=0)
    self.assertEqual(
        formatting.format_step_counts(counts),
        "6 (agent 3, user 2, system 1, other 0)",
    )

  def test_tokens(self) -> None:
    self.assertEqual(formatting.format_token_counts(None), formatting.MISSING)
    counts = stats.TokenCounts(prompt=100, completion=50, cached=20)
    self.assertEqual(formatting.format_token_counts(counts), "150 (100+50)")


if __name__ == "__main__":
  absltest.main()
