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

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
from tunix.dist_rl.trajectory import trajectory_testing
from tunix.dist_rl.trajectory.explorer.diagnostics import tunix_diagnostics

# Succeeded with reward 3.5, was not masked out, and trains on 2 tokens.
_VALID_TRAJECTORY = trajectory_testing.PAIRED_TUNIX_TRAJECTORY


class TunixDiagnosticsTest(parameterized.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.diagnostics = tunix_diagnostics.TunixDiagnostics()

  def test_namespace_comes_from_the_tunix_extensions_key(self) -> None:
    self.assertEqual(self.diagnostics.NAMESPACE, "tunix")

  def test_extract_reports_each_column(self) -> None:
    self.assertEqual(
        self.diagnostics.extract(_VALID_TRAJECTORY),
        {"masked_out": False, "trainable_tokens": 2, "valid": True},
    )
    # A rollout that was not post-processed reports neither `masked_out` nor a
    # status, and its steps report no masks.
    self.assertEqual(
        self.diagnostics.extract(trajectory_testing.TUNIX_ROLLOUT_TRAJECTORY),
        {"masked_out": None, "trainable_tokens": None, "valid": False},
    )

  def test_only_masked_out_is_printed(self) -> None:
    values = self.diagnostics.extract(_VALID_TRAJECTORY)

    self.assertEqual(
        list(self.diagnostics.format_show_header(values)),
        ["masked_out: False"],
    )

  @parameterized.named_parameters(
      ("status_in_lower_case", {"status": " succeeded"}, True),
      ("overlong_but_not_masked_out", {"status": "MAX_STEPS_REACHED"}, True),
      (
          "context_limit_but_not_masked_out",
          {"status": "MAX_CONTEXT_LIMIT_REACHED"},
          True,
      ),
      # Like the trainer, a status that `TrajectoryStatus` lacks is valid.
      ("unknown_status", {"status": "COMPLETED"}, True),
      ("not_post_processed", {"masked_out": None}, True),
      ("unset_status", {"status": None}, False),
      ("blank_status", {"status": " "}, False),
      ("running", {"status": "RUNNING"}, False),
      ("failed", {"status": "failed"}, False),
      ("timeout", {"status": "TIMEOUT"}, False),
      ("env_timeout", {"status": "ENV_TIMEOUT"}, False),
      ("cancelled", {"status": "CANCELLED"}, False),
      ("masked_out", {"masked_out": True}, False),
      ("unscored", {"total_reward": None}, False),
      ("non_finite_reward", {"total_reward": float("nan")}, False),
  )
  def test_validity(self, update: dict[str, Any], valid: bool) -> None:
    trajectory = _VALID_TRAJECTORY.model_copy(update=update)

    self.assertEqual(self.diagnostics.extract(trajectory)["valid"], valid)

  def test_trainable_tokens_sum_the_masks_of_agent_steps(self) -> None:
    agent_step = trajectory_testing.TUNIX_AGENT_STEP_1
    trajectory = _VALID_TRAJECTORY.model_copy(
        update={
            "steps": [
                trajectory_testing.TUNIX_ENV_STEP_0,
                agent_step.model_copy(
                    update={"assistant_masks": np.array([1, 1, 0])}
                ),
                agent_step.model_copy(update={"assistant_masks": [0, 1]}),
                agent_step.model_copy(update={"assistant_masks": None}),
            ]
        }
    )

    self.assertEqual(
        self.diagnostics.extract(trajectory)["trainable_tokens"], 3
    )

  @parameterized.named_parameters(
      ("no_trainable_tokens", np.array([0, 0]), 0, False),
      # Like the trainer, skip the check when no step reports masks.
      ("unreported_masks", None, None, True),
  )
  def test_trainable_tokens_decide_validity_only_when_reported(
      self,
      masks: np.ndarray | None,
      trainable_tokens: int | None,
      valid: bool,
  ) -> None:
    agent_step = trajectory_testing.TUNIX_AGENT_STEP_1.model_copy(
        update={"assistant_masks": masks}
    )
    trajectory = _VALID_TRAJECTORY.model_copy(update={"steps": [agent_step]})

    self.assertEqual(
        self.diagnostics.extract(trajectory),
        {
            "masked_out": False,
            "trainable_tokens": trainable_tokens,
            "valid": valid,
        },
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
