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

"""Tunix-specific testing fixtures and assertions for trajectory tests."""

import dataclasses
import datetime
from typing import Final

import numpy as np
import pydantic
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.trajectory import trajectory_testing
from tunix.dist_rl.trajectory import tunix_trajectory
from tunix.rl.agentic.agents import agent_types

TEST_TIMESTAMP: Final[datetime.datetime] = trajectory_testing.TEST_TIMESTAMP

TUNIX_ENV_STEP_0: Final[tunix_trajectory.TunixEnvStep] = (
    tunix_trajectory.TunixEnvStep(
        step_id=0,
        timestamp=TEST_TIMESTAMP,
        source=trajectory_lib.Source.USER,
        message="User prompt",
        observation=trajectory_lib.Observation(
            results=[
                trajectory_lib.ObservationResult(
                    source_call_id="call_env_0",
                    content="env observation content",
                    subagent_trajectory_ref=[
                        trajectory_lib.SubagentTrajectoryRef(
                            trajectory_id="sub_traj_ref_0",
                            session_id="sub_sess_ref_0",
                            extra={"ref_key": "ref_val"},
                        )
                    ],
                    extra={"obs_res_key": "obs_res_val"},
                )
            ]
        ),
        is_copied_context=False,
        llm_call_count=1,
        extra={"env_extra_key": "env_extra_val"},
        reward=1.0,
        done=False,
        env_tokens=np.array([1, 2]),
        env_masks=np.array([1, 1]),
    )
)

TUNIX_AGENT_STEP_1: Final[tunix_trajectory.TunixAgentStep] = (
    tunix_trajectory.TunixAgentStep(
        step_id=1,
        timestamp=TEST_TIMESTAMP,
        source=trajectory_lib.Source.AGENT,
        model_name="gemini-2.5-pro",
        reasoning_effort=0.8,
        message="Agent turn",
        reasoning_content="Reasoning trace",
        tool_calls=[
            trajectory_lib.ToolCall(
                tool_call_id="call_1",
                function_name="search",
                arguments={"query": "tunix"},
                extra={"tc_key": "tc_val"},
            )
        ],
        observation=trajectory_lib.Observation(
            results=[
                trajectory_lib.ObservationResult(
                    source_call_id="call_1",
                    content="search result",
                    subagent_trajectory_ref=[
                        trajectory_lib.SubagentTrajectoryRef(
                            trajectory_id="sub_traj_ref_1",
                            session_id="sub_sess_ref_1",
                            extra={"ref_key": "ref_val"},
                        )
                    ],
                    extra={"obs_res_key": "obs_res_val"},
                )
            ]
        ),
        metrics=trajectory_lib.Metrics(
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=20,
            cost_usd=0.01,
            prompt_token_ids=[1, 2],
            completion_token_ids=[3, 4],
            logprobs=[-0.1, -0.2],
            extra={"metric_key": "metric_val"},
        ),
        is_copied_context=False,
        llm_call_count=1,
        extra={"user_key": "val"},
        assistant_tokens=np.array([10, 20]),
        assistant_masks=np.array([1, 1]),
        logprobs=np.array([-0.5, -0.2]),
        policy_version=3,
        prefill_routed_experts=np.array([[[1, 2]], [[3, 4]], [[5, 6]]]),
        prefill_start=0,
        prefill_num_context=1,
    )
)

TUNIX_ENV_STEP_2: Final[tunix_trajectory.TunixEnvStep] = (
    TUNIX_ENV_STEP_0.model_copy(
        update={"step_id": 2, "reward": 1.0, "done": True}
    )
)

TUNIX_METADATA_1: Final[tunix_trajectory.TunixTrajectoryMetadata] = (
    tunix_trajectory.TunixTrajectoryMetadata(
        schema_version="ATIF-v1.7",
        session_id="sess_01",
        trajectory_id="t_atif",
        agent=trajectory_lib.Agent(
            name="agent_v1",
            version="1.0",
            model_name="gemini-2.5-pro",
            tool_definitions=[{"name": "search", "description": "search tool"}],
            extra={"agent_key": "agent_val"},
        ),
        notes="Metadata projection test",
        final_metrics=trajectory_lib.FinalMetrics(
            total_prompt_tokens=100,
            total_completion_tokens=50,
            total_cached_tokens=20,
            total_cost_usd=0.01,
            total_steps=2,
            extra={"final_key": "final_val"},
        ),
        continued_trajectory_ref="traj_prev_1000",
        extra={"user_meta": "val"},
        prompt_id="p_1",
        group_index=2,
        target_policy_versions=[2, 3],
        status="SUCCEEDED",
        total_reward=3.5,
        gamma=0.9,
        masked_out=False,
        hyperparams={"temperature": 0.7},
        env_time={"step_0": 0.05},
        reward_time={"step_1": 0.02},
    )
)

TUNIX_SUBAGENT_TRAJECTORY_1: Final[tunix_trajectory.TunixTrajectory] = (
    tunix_trajectory.TunixTrajectory(
        **TUNIX_METADATA_1.model_dump(exclude={"trajectory_id"}),
        trajectory_id="sub_traj_1",
        steps=[TUNIX_ENV_STEP_0, TUNIX_AGENT_STEP_1],
        subagent_trajectories=[],
    )
)

TUNIX_TRAJECTORY_1: Final[tunix_trajectory.TunixTrajectory] = (
    tunix_trajectory.TunixTrajectory(
        **TUNIX_METADATA_1.model_dump(),
        steps=[TUNIX_ENV_STEP_0, TUNIX_AGENT_STEP_1],
        subagent_trajectories=[TUNIX_SUBAGENT_TRAJECTORY_1],
    )
)

PAIRED_TUNIX_TRAJECTORY: Final[tunix_trajectory.TunixTrajectory] = (
    tunix_trajectory.TunixTrajectory(
        **TUNIX_METADATA_1.model_dump(exclude={"status"}),
        status="SUCCEEDED",
        steps=[TUNIX_ENV_STEP_0, TUNIX_AGENT_STEP_1, TUNIX_ENV_STEP_2],
    )
)

# The base ATIF projection of `PAIRED_TUNIX_TRAJECTORY`, as a store returns it.
PAIRED_ATIF_TRAJECTORY: Final[trajectory_lib.Trajectory] = (
    trajectory_testing.to_atif_trajectory(PAIRED_TUNIX_TRAJECTORY)
)

# A Tunix rollout with the token fields that the trajectory collector records:
# step 0 carries the prompt token IDs, each agent step's metrics report only
# completion tokens (no prompt tokens), and the terminal environment step
# records no tokens.
TUNIX_ROLLOUT_TRAJECTORY: Final[tunix_trajectory.TunixTrajectory] = (
    tunix_trajectory.TunixTrajectory(
        trajectory_id="t_rollout",
        agent=trajectory_lib.Agent(name="agent_v1", version="1.0"),
        steps=[
            tunix_trajectory.TunixEnvStep(
                step_id=0,
                source=trajectory_lib.Source.USER,
                message="Task prompt",
                env_tokens=np.array([1, 2, 3, 4]),
            ),
            tunix_trajectory.TunixAgentStep(
                step_id=1,
                source=trajectory_lib.Source.AGENT,
                message="Agent turn 0",
                metrics=trajectory_lib.Metrics(completion_tokens=3),
                assistant_tokens=np.array([10, 11, 12]),
            ),
            tunix_trajectory.TunixEnvStep(
                step_id=2,
                source=trajectory_lib.Source.SYSTEM,
                message="Observation 0",
                env_tokens=np.array([20, 21]),
            ),
            tunix_trajectory.TunixAgentStep(
                step_id=3,
                source=trajectory_lib.Source.AGENT,
                message="Agent turn 1",
                metrics=trajectory_lib.Metrics(completion_tokens=5),
                assistant_tokens=np.array([30, 31, 32, 33, 34]),
            ),
            tunix_trajectory.TunixEnvStep(
                step_id=4,
                source=trajectory_lib.Source.SYSTEM,
                message="Observation 1",
                done=True,
            ),
        ],
    )
)


class TunixTrajectoryTestCase(trajectory_testing.TrajectoryTestCase):
  """TrajectoryTestCase that also compares Tunix `agent_types.Step` dataclasses."""

  def assertStepEqual(
      self,
      actual: trajectory_lib.Step | agent_types.Step,
      expected: trajectory_lib.Step | agent_types.Step,
      msg: str | None = None,
  ) -> None:
    """Asserts that two Step instances (ATIF or Tunix `agent_types`) are equal."""
    if isinstance(actual, pydantic.BaseModel) and isinstance(
        expected, pydantic.BaseModel
    ):
      super().assertStepEqual(actual, expected, msg=msg)
      return
    self.assertEqual(type(actual), type(expected))
    if dataclasses.is_dataclass(actual):
      for field in dataclasses.fields(actual):
        v1 = getattr(actual, field.name)
        v2 = getattr(expected, field.name)
        if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
          np.testing.assert_array_equal(
              v1, v2, err_msg=msg or f"Field '{field.name}' mismatch"
          )
        else:
          self.assertEqual(v1, v2, msg=msg or f"Field '{field.name}' mismatch")
    else:
      self.assertEqual(actual, expected, msg=msg)
