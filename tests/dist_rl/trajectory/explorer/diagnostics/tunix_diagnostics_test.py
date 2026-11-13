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

from collections.abc import Mapping
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.trajectory import trajectory_testing
from tunix.dist_rl.trajectory.explorer.diagnostics import schema_diagnostics
from tunix.dist_rl.trajectory.explorer.diagnostics import stats
from tunix.dist_rl.trajectory.explorer.diagnostics import tunix_diagnostics

# Succeeded with reward 3.5, was not masked out, and trains on 2 tokens.
_VALID_TRAJECTORY = trajectory_testing.PAIRED_TUNIX_TRAJECTORY


def _rollout(
    prompt_id: str | None, reward: float | None, **update: Any
) -> trajectory_lib.TunixTrajectory:
  """Returns a rollout of `prompt_id`, valid unless `update` says otherwise."""
  return _VALID_TRAJECTORY.model_copy(
      update={"prompt_id": prompt_id, "total_reward": reward, **update}
  )


def _run_summary(
    *trajectories: trajectory_lib.Trajectory[Any],
) -> Mapping[str, schema_diagnostics.JsonValue]:
  """Returns the Tunix run summary of a run of `trajectories`."""
  run = stats.RunStats.from_trajectories(
      trajectories, diagnostics=(tunix_diagnostics.TunixDiagnostics(),)
  )
  return run.schema_stats["tunix"]


class TunixDiagnosticsTest(parameterized.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.diagnostics = tunix_diagnostics.TunixDiagnostics()

  def test_namespace_comes_from_the_tunix_extensions_key(self) -> None:
    self.assertEqual(self.diagnostics.NAMESPACE, "tunix")

  def test_extract_reports_each_column(self) -> None:
    self.assertEqual(
        self.diagnostics.extract(_VALID_TRAJECTORY),
        {
            "masked_out": False,
            "prompt_id": "p_1",
            "trainable_tokens": 2,
            "valid": True,
        },
    )
    # A rollout that was not post-processed reports neither `masked_out` nor a
    # status, and its steps report no masks.
    self.assertEqual(
        self.diagnostics.extract(trajectory_testing.TUNIX_ROLLOUT_TRAJECTORY),
        {
            "masked_out": None,
            "prompt_id": None,
            "trainable_tokens": None,
            "valid": False,
        },
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

    values = self.diagnostics.extract(trajectory)
    self.assertEqual(values["trainable_tokens"], trainable_tokens)
    self.assertEqual(values["valid"], valid)

  def test_run_summary_counts_masked_out_trajectories(self) -> None:
    run_summary = _run_summary(
        _rollout("p", 1.0, masked_out=True),
        _rollout("p", 1.0, masked_out=False),
        _rollout("p", 1.0, masked_out=None),
        _rollout("p", 1.0, masked_out=True),
    )

    self.assertEqual(run_summary["num_masked_out"], 2)
    self.assertEqual(run_summary["num_with_masked_out"], 3)

  def test_run_summary_without_reports_leaves_masked_out_count_unset(
      self,
  ) -> None:
    run_summary = _run_summary(
        _rollout("p", 1.0, masked_out=None),
        _rollout("p", 1.0, masked_out=None),
    )

    self.assertIsNone(run_summary["num_masked_out"])
    self.assertEqual(run_summary["num_with_masked_out"], 0)

  def test_prompt_groups_are_classified_by_their_valid_rewards(self) -> None:
    run_summary = _run_summary(
        _rollout("contrastive", 1.0),
        _rollout("contrastive", 0.0),
        _rollout("dead", 0.0),
        _rollout("dead", 0.0),
        # Rewards within the tolerance count as equal.
        _rollout("saturated", 1.0),
        _rollout("saturated", 1.0 + 1e-9),
        # The masked-out rollout doesn't count, which leaves 1 valid rollout.
        _rollout("degenerate", 1.0),
        _rollout("degenerate", 0.0, masked_out=True),
        # A RUNNING rollout makes its group incomplete, whatever the others.
        _rollout("incomplete", 1.0),
        _rollout("incomplete", 0.0),
        _rollout("incomplete", None, status="RUNNING"),
        _rollout(None, 1.0),
        # Trajectories of other schemas are not counted as ungrouped.
        trajectory_testing.TRAJECTORY_1,
    )

    self.assertEqual(
        run_summary["prompt_groups"],
        {
            "num_groups": 5,
            "num_incomplete": 1,
            "num_ungrouped": 1,
            "regimes": {
                "contrastive": 1,
                "zero_std_dead": 1,
                "zero_std_saturated": 1,
                "degenerate": 1,
            },
        },
    )

  def test_rollouts_without_prompt_ids_form_no_groups(self) -> None:
    run_summary = _run_summary(_rollout(None, 1.0), _rollout(None, 0.0))

    self.assertEqual(
        run_summary["prompt_groups"],
        {
            "num_groups": 0,
            "num_incomplete": 0,
            "num_ungrouped": 2,
            "regimes": {
                "contrastive": 0,
                "zero_std_dead": 0,
                "zero_std_saturated": 0,
                "degenerate": 0,
            },
        },
    )

  def test_format_summary_totals(self) -> None:
    lines = self.diagnostics.format_summary_totals(
        {
            "num_masked_out": 127,
            "num_with_masked_out": 512,
            "prompt_groups": {
                "num_groups": 65,
                "num_incomplete": 1,
                "num_ungrouped": 0,
                "regimes": {
                    "contrastive": 18,
                    "zero_std_dead": 40,
                    "zero_std_saturated": 4,
                    "degenerate": 2,
                },
            },
        },
        num_trajectories=512,
    )

    # The shares are of the 64 complete groups.
    self.assertEqual(
        list(lines),
        [
            "Masked out: 127 reported by 512/512",
            "Prompt Groups: 65 (1 incomplete, 0 ungrouped trajectories)",
            "  contrastive (valid>=2, std>0):            18 (28.1%)",
            "  zero_std_dead (valid>=2, std=0, mean<=0): 40 (62.5%)",
            "  zero_std_saturated (std=0, mean>0):        4 ( 6.2%)",
            "  degenerate (valid<2):                      2 ( 3.1%)",
        ],
    )

  def test_format_summary_totals_without_complete_groups(self) -> None:
    lines = self.diagnostics.format_summary_totals(
        {
            "num_masked_out": None,
            "num_with_masked_out": 0,
            "prompt_groups": {
                "num_groups": 1,
                "num_incomplete": 1,
                "num_ungrouped": 2,
                "regimes": {
                    "contrastive": 0,
                    "zero_std_dead": 0,
                    "zero_std_saturated": 0,
                    "degenerate": 0,
                },
            },
        },
        num_trajectories=4,
    )

    self.assertEqual(
        list(lines),
        [
            "Masked out: - reported by 0/4",
            "Prompt Groups: 1 (1 incomplete, 2 ungrouped trajectories)",
        ],
    )


if __name__ == "__main__":
  absltest.main()
