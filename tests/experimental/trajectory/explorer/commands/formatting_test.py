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

  def test_counts_renders_empty_as_missing(self) -> None:
    self.assertEqual(formatting.format_counts({}), formatting.MISSING)

  def test_counts_ranks_most_frequent_first(self) -> None:
    self.assertEqual(
        formatting.format_counts({"search": 1, "edit": 3, "read": 2}),
        "edit x3, read x2, search x1",
    )

  def test_counts_keeps_insertion_order_on_ties(self) -> None:
    self.assertEqual(
        formatting.format_counts({"search": 2, "edit": 2}),
        "search x2, edit x2",
    )


if __name__ == "__main__":
  absltest.main()
