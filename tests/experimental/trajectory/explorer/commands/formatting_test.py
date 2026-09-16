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

"""Unit tests for terminal rendering of trajectory diagnostics."""

from absl.testing import absltest
from tunix.experimental.trajectory.explorer.commands import formatting
from tunix.experimental.trajectory.explorer.diagnostics import stats as stats_lib


class FormatRewardTest(absltest.TestCase):

  def test_a_zero_reward_renders_as_a_measurement(self) -> None:
    self.assertEqual(formatting.format_reward(0.0), "0.00")

  def test_an_absent_reward_renders_as_missing(self) -> None:
    self.assertEqual(formatting.format_reward(None), formatting.MISSING)


class FormatTokensTest(absltest.TestCase):

  def test_unreported_tokens_render_as_missing(self) -> None:
    self.assertEqual(
        formatting.format_tokens(stats_lib.TokenCounts()), formatting.MISSING
    )

  def test_a_measured_zero_renders_as_zeros(self) -> None:
    tokens = stats_lib.TokenCounts(reported=True)

    self.assertEqual(formatting.format_tokens(tokens), "0/0/0")

  def test_tokens_render_as_prompt_completion_total(self) -> None:
    tokens = stats_lib.TokenCounts(
        prompt=7, completion=3, cached=2, reported=True
    )

    self.assertEqual(formatting.format_tokens(tokens), "7/3/10")


class FormatStepsTest(absltest.TestCase):

  def test_steps_render_as_total_and_split(self) -> None:
    counts = stats_lib.StepCounts(total=6, agent=3, user=2, system=1)

    self.assertEqual(formatting.format_steps(counts), "6 (3/2/1)")


class FormatToolSequenceTest(absltest.TestCase):

  def test_no_tool_calls_render_as_missing(self) -> None:
    self.assertEqual(
        formatting.format_tool_sequence(stats_lib.ToolUsage()),
        formatting.MISSING,
    )

  def test_tool_sequence_is_arrow_joined(self) -> None:
    tools = stats_lib.ToolUsage(sequence=("a", "b"))

    self.assertEqual(formatting.format_tool_sequence(tools), "a -> b")

  def test_long_tool_sequences_are_truncated_with_a_remainder(self) -> None:
    tools = stats_lib.ToolUsage(sequence=("a", "b", "c", "d", "e"))

    self.assertEqual(
        formatting.format_tool_sequence(tools, max_shown=2),
        "a -> b -> ... (+3 more)",
    )

  def test_a_sequence_at_the_limit_is_not_truncated(self) -> None:
    tools = stats_lib.ToolUsage(sequence=("a", "b"))

    self.assertEqual(
        formatting.format_tool_sequence(tools, max_shown=2), "a -> b"
    )


if __name__ == "__main__":
  absltest.main()
