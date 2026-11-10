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

"""Unit tests for the Tunix schema diagnostics."""

from absl.testing import absltest
import pandas as pd
from tunix.dist_rl.trajectory import trajectory_testing
from tunix.dist_rl.trajectory.explorer.diagnostics import tunix_diagnostics


class TunixDiagnosticsTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.diagnostics = tunix_diagnostics.TunixDiagnostics()

  def test_namespace_comes_from_the_tunix_extensions_key(self) -> None:
    self.assertEqual(self.diagnostics.NAMESPACE, "tunix")

  def test_extract_reports_masked_out(self) -> None:
    self.assertEqual(
        self.diagnostics.extract(trajectory_testing.PAIRED_TUNIX_TRAJECTORY),
        {"masked_out": False},
    )
    # A rollout that was not post-processed does not report the value.
    self.assertEqual(
        self.diagnostics.extract(trajectory_testing.TUNIX_ROLLOUT_TRAJECTORY),
        {"masked_out": None},
    )

  def test_summarize_counts_masked_out_trajectories(self) -> None:
    df = pd.DataFrame(
        {"masked_out": pd.array([True, False, None, True], dtype="boolean")}
    )

    self.assertEqual(
        self.diagnostics.summarize(df),
        {"num_masked_out": 2, "num_with_masked_out": 3},
    )

  def test_summarize_without_reports_leaves_masked_out_count_unset(
      self,
  ) -> None:
    df = pd.DataFrame({"masked_out": pd.array([None, None], dtype="boolean")})

    self.assertEqual(
        self.diagnostics.summarize(df),
        {"num_masked_out": None, "num_with_masked_out": 0},
    )

  def test_format_summary_totals_reports_the_masked_out_count(self) -> None:
    lines = self.diagnostics.format_summary_totals(
        {"num_masked_out": 2, "num_with_masked_out": 3}, num_trajectories=4
    )

    self.assertEqual(list(lines), ["Masked out: 2 reported by 3/4"])

  def test_format_summary_totals_without_reports(self) -> None:
    lines = self.diagnostics.format_summary_totals(
        {"num_masked_out": None, "num_with_masked_out": 0}, num_trajectories=2
    )

    self.assertEqual(list(lines), ["Masked out: - reported by 0/2"])


if __name__ == "__main__":
  absltest.main()
