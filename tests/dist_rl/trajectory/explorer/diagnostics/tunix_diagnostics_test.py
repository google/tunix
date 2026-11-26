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

import json
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
_NO_SAMPLE_GROUPS: dict[str, schema_diagnostics.JsonValue] = {
    "contrastive": [],
    "dead_or_degenerate": [],
}


def _rollout(
    prompt_id: str | None, reward: float | None, **update: Any
) -> trajectory_lib.TunixTrajectory:
  """Returns a rollout of `prompt_id`, valid unless `update` says otherwise."""
  return _VALID_TRAJECTORY.model_copy(
      update={"prompt_id": prompt_id, "total_reward": reward, **update}
  )


def _agent_steps(count: int) -> list[trajectory_lib.TunixAgentStep]:
  """Returns `count` agent steps, each training on 2 tokens."""
  return [
      trajectory_testing.TUNIX_AGENT_STEP_1.model_copy(update={"step_id": i})
      for i in range(count)
  ]


def _agent_step(
    masks: list[int] | None, logprobs: list[float] | None
) -> trajectory_lib.TunixAgentStep:
  """Returns an agent step with the given `assistant_masks` and `logprobs`."""
  return trajectory_testing.TUNIX_AGENT_STEP_1.model_copy(
      update={"assistant_masks": masks, "logprobs": logprobs}
  )


def _run_summary(*trajectories: trajectory_lib.Trajectory[Any]) -> Any:
  """Returns the Tunix run summary of a run of `trajectories`, as JSON."""
  run = stats.RunStats.from_trajectories(
      trajectories, diagnostics=(tunix_diagnostics.TunixDiagnostics(),)
  )
  return json.loads(json.dumps(run.to_json_dict()["schema_stats"]["tunix"]))


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
            "logp_mean": -0.35,
            "logp_min": -0.5,
            "logp_tokens": 2,
            "nonfinite_logp": 0,
            "misaligned_logp_steps": 0,
        },
    )
    # A rollout that was not post-processed reports neither `masked_out` nor a
    # status, and its steps report neither masks nor logprobs.
    self.assertEqual(
        self.diagnostics.extract(trajectory_testing.TUNIX_ROLLOUT_TRAJECTORY),
        {
            "masked_out": None,
            "prompt_id": None,
            "trainable_tokens": None,
            "valid": False,
            "logp_mean": None,
            "logp_min": None,
            "logp_tokens": None,
            "nonfinite_logp": None,
            "misaligned_logp_steps": None,
        },
    )

  def test_show_header_prints_only_the_logprobs(self) -> None:
    values = self.diagnostics.extract(_VALID_TRAJECTORY)

    self.assertEqual(
        list(self.diagnostics.format_show_header(values)),
        ["logp_mean: -0.35", "logp_min: -0.5"],
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
      # Like the trainer, skip the check when the agent steps report no masks.
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

  @parameterized.named_parameters(
      ("max_steps_reached", "MAX_STEPS_REACHED"),
      ("context_limit_reached", "MAX_CONTEXT_LIMIT_REACHED"),
  )
  def test_rollout_without_agent_steps_is_invalid(self, status: str) -> None:
    # E.g. `max_steps=0`, or a response budget spent before the first turn:
    # only the task step is recorded, and the trainer drops the rollout for its
    # empty `conversation_masks`.
    trajectory = _VALID_TRAJECTORY.model_copy(
        update={
            "status": status,
            "steps": [trajectory_testing.TUNIX_ENV_STEP_0],
        }
    )

    values = self.diagnostics.extract(trajectory)
    self.assertEqual(values["trainable_tokens"], 0)
    self.assertIs(values["valid"], False)

  @parameterized.named_parameters(
      (
          "finite_logprobs_at_mask_1",
          # The 0.0 under mask 0 is the template suffix's placeholder.
          [([1, 1, 0], [-1.0, -3.0, 0.0])],
          (-2.0, -3.0, 2, 0, 0),
      ),
      (
          "non_finite_logprobs",
          [([1, 1, 1], [-1.0, float("nan"), float("-inf")])],
          (-1.0, -1.0, 1, 2, 0),
      ),
      (
          "step_without_masks_counts_every_logprob",
          [(None, [-1.0, -2.0])],
          (-1.5, -2.0, 2, 0, 0),
      ),
      (
          "misaligned_step_is_left_out",
          [([1, 1], [-1.0]), ([1], [-2.0])],
          (-2.0, -2.0, 1, 0, 1),
      ),
      (
          "no_finite_logprob",
          [([0, 1], [0.0, float("nan")])],
          (None, None, 0, 1, 0),
      ),
      (
          "masks_without_logprobs",
          [([1, 1], None)],
          (None, None, None, None, None),
      ),
  )
  def test_logprob_columns(
      self,
      steps: list[tuple[list[int] | None, list[float] | None]],
      expected: tuple[
          float | None, float | None, int | None, int | None, int | None
      ],
  ) -> None:
    trajectory = _VALID_TRAJECTORY.model_copy(
        update={"steps": [_agent_step(masks, lps) for masks, lps in steps]}
    )

    values = self.diagnostics.extract(trajectory)
    self.assertEqual(
        (
            values["logp_mean"],
            values["logp_min"],
            values["logp_tokens"],
            values["nonfinite_logp"],
            values["misaligned_logp_steps"],
        ),
        expected,
    )

  def test_run_logprobs_weigh_rollouts_by_their_finite_logprobs(self) -> None:
    logprobs = _run_summary(
        # 3 finite logprobs of -1, then 1 of -4 next to a NaN.
        _rollout("p", 1.0, steps=[_agent_step([1, 1, 1], [-1.0, -1.0, -1.0])]),
        _rollout("p", 0.0, steps=[_agent_step([1, 1], [-4.0, float("nan")])]),
        # No trainable token, then a misaligned step, then no logprobs.
        _rollout("q", 1.0, steps=[_agent_step([0, 0], [0.0, 0.0])]),
        _rollout("q", 0.0, steps=[_agent_step([1, 1], [-1.0])]),
        _rollout("r", 1.0, steps=[_agent_step([1], None)]),
    )["logprobs"]

    self.assertEqual(
        logprobs,
        {
            "mean": -1.75,
            "min": -4.0,
            "num_non_finite": 1,
            "num_zero_mask": 1,
            "num_misaligned": 1,
        },
    )

  def test_run_without_logprobs_has_no_logprob_summary(self) -> None:
    run_summary = _run_summary(
        _rollout("p", 1.0, steps=[_agent_step([1, 1], None)])
    )

    self.assertIsNone(run_summary["logprobs"])

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

  @parameterized.named_parameters(
      ("unset", None),
      # `RolloutRequest.prompt_id` defaults to "", which the trainer treats as
      # unset too.
      ("empty", ""),
  )
  def test_rollouts_without_prompt_ids_form_no_groups(
      self, prompt_id: str | None
  ) -> None:
    rollout = _rollout(prompt_id, 1.0)
    run_summary = _run_summary(rollout, _rollout(prompt_id, 0.0))

    self.assertIsNone(self.diagnostics.extract(rollout)["prompt_id"])
    self.assertEqual(
        run_summary["prompt_groups"],
        {
            "num_groups": 0,
            "num_incomplete": 0,
            "num_ungrouped": 2,
            "regimes": None,
        },
    )

  @parameterized.named_parameters(
      # One rollout per prompt (G = 1), as in PPO or evaluation.
      (
          "one_rollout_per_prompt",
          [
              _rollout("p_1", 1.0),
              _rollout("p_2", 0.0, masked_out=True),
              _rollout("p_3", None, status="RUNNING"),
          ],
      ),
      # An incomplete group doesn't count, even with 2 rollouts.
      (
          "no_complete_group",
          [_rollout("p_1", 1.0), _rollout("p_1", None, status="RUNNING")],
      ),
  )
  def test_no_regimes_without_a_complete_group_of_2_rollouts(
      self, rollouts: list[trajectory_lib.TunixTrajectory]
  ) -> None:
    run_summary = _run_summary(*rollouts)

    self.assertIsNone(run_summary["prompt_groups"]["regimes"])
    self.assertEqual(run_summary["sample_groups"], _NO_SAMPLE_GROUPS)

  def test_regimes_report_degenerate_groups_of_2_rollouts(self) -> None:
    # A G = 2 run whose rollouts were all masked out still reports regimes.
    run_summary = _run_summary(
        _rollout("p_1", 0.0, masked_out=True),
        _rollout("p_1", 0.0, masked_out=True),
    )

    self.assertEqual(
        run_summary["prompt_groups"]["regimes"],
        {
            "contrastive": 0,
            "zero_std_dead": 0,
            "zero_std_saturated": 0,
            "degenerate": 1,
        },
    )

  def test_sample_groups_list_the_most_telling_groups(self) -> None:
    # Contrastive groups whose rewards spread more as the name grows; the
    # rewards of c0 and c1 spread equally.
    contrastive: list[trajectory_lib.TunixTrajectory] = []
    for prompt_id, reward in (
        ("c0", 1.0),
        ("c1", 1.0),
        ("c2", 2.0),
        ("c3", 3.0),
        ("c4", 4.0),
        ("c5", 5.0),
    ):
      contrastive += [_rollout(prompt_id, 0.0), _rollout(prompt_id, reward)]
    samples = _run_summary(
        *contrastive,
        # Dead groups, out of `prompt_id` order, that tie on invalid and
        # masked-out rollouts, as a rollout that was not post-processed is
        # neither.
        _rollout("dead1", 0.0, masked_out=None),
        _rollout("dead1", 0.0, steps=_agent_steps(3)),
        _rollout("dead0", 0.0),
        _rollout("dead0", 0.0),
        _rollout("degenerate", 1.0),
        _rollout(
            "degenerate", 0.0, status="MAX_STEPS_REACHED", masked_out=True
        ),
        _rollout(
            "degenerate", 0.0, status="MAX_STEPS_REACHED", masked_out=True
        ),
        # Rollouts without a status are invalid, but not running: their group
        # is complete, with no valid rollout.
        _rollout("no_status", None, status=None),
        _rollout("no_status", None, status=None),
        # Neither saturated nor incomplete groups are sampled.
        _rollout("saturated", 1.0),
        _rollout("saturated", 1.0),
        _rollout("incomplete", None, status="RUNNING"),
    )["sample_groups"]

    # By decreasing `valid_std`, then `prompt_id`, and at most 5 of them.
    self.assertEqual(
        [group["prompt_id"] for group in samples["contrastive"]],
        ["c5", "c4", "c3", "c2", "c0"],
    )
    # By decreasing number of invalid rollouts, then of masked-out ones, then
    # `prompt_id`. Rollouts without a status are not counted in `statuses`.
    self.assertEqual(
        samples["dead_or_degenerate"],
        [
            {
                "prompt_id": "degenerate",
                "valid": 1,
                "masked": 2,
                "valid_mean": 1.0,
                "valid_std": None,
                "agent_steps": {"min": 1, "mean": 1.0, "max": 1},
                "statuses": {"MAX_STEPS_REACHED": 2, "SUCCEEDED": 1},
            },
            {
                "prompt_id": "no_status",
                "valid": 0,
                "masked": 0,
                "valid_mean": None,
                "valid_std": None,
                "agent_steps": {"min": 1, "mean": 1.0, "max": 1},
                "statuses": {},
            },
            {
                "prompt_id": "dead0",
                "valid": 2,
                "masked": 0,
                "valid_mean": 0.0,
                "valid_std": 0.0,
                "agent_steps": {"min": 1, "mean": 1.0, "max": 1},
                "statuses": {"SUCCEEDED": 2},
            },
            {
                "prompt_id": "dead1",
                "valid": 2,
                "masked": 0,
                "valid_mean": 0.0,
                "valid_std": 0.0,
                "agent_steps": {"min": 1, "mean": 2.0, "max": 3},
                "statuses": {"SUCCEEDED": 2},
            },
        ],
    )
    # The most frequent status comes first.
    self.assertEqual(
        list(samples["dead_or_degenerate"][0]["statuses"]),
        ["MAX_STEPS_REACHED", "SUCCEEDED"],
    )

  def test_dead_or_degenerate_groups_come_by_invalid_rollouts(self) -> None:
    samples = _run_summary(
        # Valid rollouts only.
        _rollout("dead", 0.0),
        _rollout("dead", 0.0),
        # 2 invalid rollouts, masked out by the overlong filter.
        _rollout("overlong", 1.0),
        _rollout("overlong", 0.0, status="MAX_STEPS_REACHED", masked_out=True),
        _rollout("overlong", 0.0, status="MAX_STEPS_REACHED", masked_out=True),
        # 2 invalid rollouts, neither masked out (the overlong filter was off).
        _rollout("env_timeout", 1.0),
        _rollout("env_timeout", 0.0, status="ENV_TIMEOUT"),
        _rollout("env_timeout", 0.0, status="ENV_TIMEOUT"),
        # 3 invalid rollouts, none masked out, as an episode that FAILED is
        # not post-processed.
        _rollout("failed", None, status="FAILED", masked_out=None),
        _rollout("failed", None, status="FAILED", masked_out=None),
        _rollout("failed", None, status="FAILED", masked_out=None),
    )["sample_groups"]

    # By decreasing number of invalid rollouts, then of masked-out ones.
    self.assertEqual(
        [group["prompt_id"] for group in samples["dead_or_degenerate"]],
        ["failed", "overlong", "env_timeout", "dead"],
    )

  def test_format_summary_totals(self) -> None:
    lines = self.diagnostics.format_summary_totals(
        {
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
            "logprobs": None,
            "sample_groups": _NO_SAMPLE_GROUPS,
        },
        num_trajectories=512,
    )

    # The shares are of the 64 complete groups, and the logprobs line and the
    # sample tables without groups are left out.
    self.assertEqual(
        list(lines),
        [
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
            "prompt_groups": {
                "num_groups": 1,
                "num_incomplete": 1,
                "num_ungrouped": 2,
                "regimes": None,
            },
            "logprobs": None,
            "sample_groups": _NO_SAMPLE_GROUPS,
        },
        num_trajectories=4,
    )

    self.assertEqual(
        list(lines),
        [
            "Prompt Groups: 1 (1 incomplete, 2 ungrouped trajectories)",
        ],
    )

  def test_format_summary_totals_prints_logprobs_and_sample_tables(
      self,
  ) -> None:
    lines = self.diagnostics.format_summary_totals(
        {
            "prompt_groups": {
                "num_groups": 3,
                "num_incomplete": 0,
                "num_ungrouped": 0,
                "regimes": {
                    "contrastive": 2,
                    "zero_std_dead": 0,
                    "zero_std_saturated": 0,
                    "degenerate": 1,
                },
            },
            "logprobs": {
                "mean": -0.1423,
                "min": -18.4012,
                "num_non_finite": 0,
                "num_zero_mask": 0,
                "num_misaligned": 1,
            },
            "sample_groups": {
                "contrastive": [
                    {
                        "prompt_id": "astropy__astropy-12907",
                        "valid": 7,
                        "masked": 1,
                        "valid_mean": 0.4286,
                        "valid_std": 0.5352,
                        "agent_steps": {"min": 2, "mean": 4.625, "max": 30},
                        "statuses": {"SUCCEEDED": 7, "MAX_STEPS_REACHED": 1},
                    },
                    {
                        "prompt_id": "django__django-11099",
                        "valid": 8,
                        "masked": 0,
                        "valid_mean": 0.25,
                        "valid_std": 0.4629,
                        "agent_steps": {"min": 2, "mean": 5.125, "max": 8},
                        "statuses": {"SUCCEEDED": 8},
                    },
                ],
                "dead_or_degenerate": [
                    {
                        "prompt_id": "sympy__sympy-18057",
                        "valid": 1,
                        "masked": 7,
                        "valid_mean": 0.0,
                        "valid_std": None,
                        "agent_steps": {"min": 12, "mean": 28.375, "max": 30},
                        "statuses": {"MAX_STEPS_REACHED": 7, "SUCCEEDED": 1},
                    },
                ],
            },
        },
        num_trajectories=24,
    )

    # The logprobs line follows the prompt groups, at 3 significant digits.
    self.assertEqual(
        list(lines)[5:],
        [
            (
                "Rollout Logprobs (mask=1): mean -0.142, min -18.4"
                "  |  Anomalies: 0 non_finite, 0 zero_mask, 1 misaligned"
            ),
            "Sample Contrastive Groups:",
            (
                "  prompt_id               valid  masked  valid_mean  valid_std"
                "  agent steps (min/mean/max)  statuses"
            ),
            (
                "  astropy__astropy-12907  7      1       0.429       0.535"
                "      2 / 4.6 / 30"
                "                SUCCEEDED x7, MAX_STEPS_REACHED x1"
            ),
            (
                "  django__django-11099    8      0       0.250       0.463"
                "      2 / 5.1 / 8                 SUCCEEDED x8"
            ),
            "Sample Dead / Degenerate Groups:",
            (
                "  prompt_id           valid  masked  valid_mean  valid_std"
                "  agent steps (min/mean/max)  statuses"
            ),
            (
                "  sympy__sympy-18057  1      7       0.000       -"
                "          12 / 28.4 / 30"
                "              MAX_STEPS_REACHED x7, SUCCEEDED x1"
            ),
        ],
    )


if __name__ == "__main__":
  absltest.main()
