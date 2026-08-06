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

"""Tests for converter module."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.experimental.trajectory import converter
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.agents import agent_types


class ConverterTest(parameterized.TestCase):

  # ---------------------------------------------------------------------------
  # tunix_to_atif_step() tests
  # ---------------------------------------------------------------------------

  def test_tunix_to_atif_step_none_returns_none(self):
    step = converter.tunix_to_atif_step(None)
    self.assertIsNone(step)

  def test_tunix_to_atif_step_only_agent_step(self):
    mock_agent_step = agent_types.Step(
        model_response="Calling bash tool",
        thought="I need to list files",
        action=agent_types.Action(
            action={"name": "bash", "arguments": {"command": "ls -la"}}
        ),
        assistant_tokens=np.array([101, 102]),
        assistant_masks=np.array([1, 1]),
        logprobs=np.array([-0.1, -0.2]),
        mc_return=1.5,
        info={"trace_id": "123"},
    )

    agent_step = converter.tunix_to_atif_step(
        mock_agent_step, step_id=1, source=trajectory_lib.Source.AGENT
    )

    self.assertIsNotNone(agent_step)
    self.assertEqual(agent_step.step_id, 1)
    self.assertEqual(agent_step.source, trajectory_lib.Source.AGENT)
    self.assertEqual(agent_step.message, "Calling bash tool")
    self.assertEqual(agent_step.reasoning_content, "I need to list files")
    self.assertIsNotNone(agent_step.tool_calls)
    self.assertEqual(agent_step.tool_calls[0].function_name, "bash")
    self.assertEqual(agent_step.tool_calls[0].arguments, {"command": "ls -la"})
    self.assertIsNotNone(agent_step.metrics)
    self.assertEqual(agent_step.metrics.completion_tokens, 2)
    self.assertEqual(agent_step.metrics.completion_token_ids, [101, 102])
    self.assertEqual(agent_step.metrics.logprobs, [-0.1, -0.2])
    self.assertEqual(agent_step.extra.get("trace_id"), "123")
    self.assertEqual(agent_step.extra.get("mc_return"), 1.5)
    self.assertIsNone(agent_step.observation)

  def test_tunix_to_atif_step_only_env_step(self):
    mock_env_step = agent_types.Step(
        observation="file1.txt\nfile2.txt",
        reward=1.0,
        done=False,
        env_tokens=np.array([201]),
        env_masks=np.array([1]),
        info={"env_meta": "test_env"},
    )

    env_step = converter.tunix_to_atif_step(
        mock_env_step, step_id=2, source=trajectory_lib.Source.SYSTEM
    )

    self.assertIsNotNone(env_step)
    self.assertEqual(env_step.step_id, 2)
    self.assertEqual(env_step.source, trajectory_lib.Source.SYSTEM)
    self.assertEqual(env_step.message, "file1.txt\nfile2.txt")
    self.assertIsNotNone(env_step.observation)
    self.assertEqual(
        env_step.observation.results[0].content, "file1.txt\nfile2.txt"
    )
    self.assertEqual(env_step.extra.get("reward"), 1.0)
    self.assertFalse(env_step.extra.get("done"))
    np.testing.assert_array_equal(
        env_step.extra.get("env_tokens"), np.array([201])
    )
    self.assertEqual(env_step.extra.get("env_meta"), "test_env")

    # Validate agent-only fields are strictly None on env_step
    self.assertIsNone(env_step.model_name)
    self.assertIsNone(env_step.reasoning_effort)
    self.assertIsNone(env_step.reasoning_content)
    self.assertIsNone(env_step.tool_calls)
    self.assertIsNone(env_step.metrics)

  def test_tunix_to_atif_step_length_mismatch_raises_value_error(self):
    mock_rl_step = agent_types.Step(
        model_response="test",
        assistant_tokens=np.array([1, 2, 3]),
        logprobs=np.array([-0.1, -0.2]),
    )
    with self.assertRaises(ValueError):
      converter.tunix_to_atif_step(mock_rl_step)

  # ---------------------------------------------------------------------------
  # atif_to_tunix_step() tests
  # ---------------------------------------------------------------------------

  def test_atif_to_tunix_step_none_returns_empty_step(self):
    dto_step = converter.atif_to_tunix_step(None, None)
    self.assertIsInstance(dto_step, agent_types.Step)
    self.assertEqual(dto_step.model_response, "")
    self.assertEqual(dto_step.thought, "")
    self.assertIsNone(dto_step.action)
    self.assertIsNone(dto_step.observation)
    self.assertEqual(dto_step.reward, 0.0)
    self.assertFalse(dto_step.done)
    self.assertIsNone(dto_step.assistant_tokens)
    self.assertIsNone(dto_step.env_tokens)

  def test_atif_to_tunix_step_only_agent_step_passed(self):
    agent_atif_step = trajectory_lib.Step(
        step_id=1,
        source=trajectory_lib.Source.AGENT,
        message="Calling search",
        reasoning_content="Search query planning",
        tool_calls=[
            trajectory_lib.ToolCall(
                tool_call_id="call_1",
                function_name="search",
                arguments={"query": "tunix"},
            )
        ],
        metrics=trajectory_lib.Metrics(
            completion_tokens=2,
            completion_token_ids=[10, 20],
            logprobs=[-0.5, -0.3],
        ),
        extra={"trace_id": "agent_only_trace", "mc_return": 2.0},
    )

    dto_step = converter.atif_to_tunix_step(
        agent_step=agent_atif_step, env_step=None
    )

    self.assertIsInstance(dto_step, agent_types.Step)
    self.assertEqual(dto_step.model_response, "Calling search")
    self.assertEqual(dto_step.thought, "Search query planning")
    self.assertIsNotNone(dto_step.action)
    self.assertEqual(
        dto_step.action.action,
        {"name": "search", "arguments": {"query": "tunix"}},
    )
    np.testing.assert_array_equal(dto_step.assistant_tokens, np.array([10, 20]))
    np.testing.assert_array_equal(dto_step.logprobs, np.array([-0.5, -0.3]))
    self.assertEqual(dto_step.mc_return, 2.0)
    self.assertEqual(dto_step.info.get("trace_id"), "agent_only_trace")
    # Env fields remain default/None
    self.assertIsNone(dto_step.observation)
    self.assertEqual(dto_step.reward, 0.0)
    self.assertFalse(dto_step.done)
    self.assertIsNone(dto_step.env_tokens)

  def test_atif_to_tunix_step_only_env_step_passed(self):
    env_atif_step = trajectory_lib.Step(
        step_id=2,
        source=trajectory_lib.Source.SYSTEM,
        message="Search completed successfully",
        observation=trajectory_lib.Observation(
            results=[
                trajectory_lib.ObservationResult(
                    content="Search completed successfully"
                )
            ]
        ),
        extra={
            "reward": 1.0,
            "done": True,
            "env_tokens": [201, 202],
            "env_masks": [1, 1],
            "env_meta": "meta_val",
        },
    )

    dto_step = converter.atif_to_tunix_step(
        agent_step=None, env_step=env_atif_step
    )

    self.assertIsInstance(dto_step, agent_types.Step)
    self.assertEqual(dto_step.observation, "Search completed successfully")
    self.assertEqual(dto_step.reward, 1.0)
    self.assertTrue(dto_step.done)
    np.testing.assert_array_equal(dto_step.env_tokens, np.array([201, 202]))
    np.testing.assert_array_equal(dto_step.env_masks, np.array([1, 1]))
    self.assertEqual(dto_step.info.get("env_meta"), "meta_val")
    # Agent fields remain default/empty
    self.assertEqual(dto_step.model_response, "")
    self.assertEqual(dto_step.thought, "")
    self.assertIsNone(dto_step.action)
    self.assertIsNone(dto_step.assistant_tokens)
    self.assertIsNone(dto_step.logprobs)

  def test_atif_to_tunix_step_both_passed(self):
    agent_atif_step = trajectory_lib.Step(
        step_id=1,
        source=trajectory_lib.Source.AGENT,
        message="Calling search",
        reasoning_content="Search query planning",
        tool_calls=[
            trajectory_lib.ToolCall(
                tool_call_id="call_1",
                function_name="search",
                arguments={"query": "tunix"},
            )
        ],
        metrics=trajectory_lib.Metrics(
            completion_tokens=2,
            completion_token_ids=[10, 20],
            logprobs=[-0.5, -0.3],
        ),
        extra={"session_id": "sess_1"},
    )
    env_atif_step = trajectory_lib.Step(
        step_id=2,
        source=trajectory_lib.Source.SYSTEM,
        message="search result",
        observation=trajectory_lib.Observation(
            results=[trajectory_lib.ObservationResult(content="search result")]
        ),
        extra={"reward": 0.8, "done": False, "env_tokens": [99]},
    )

    dto_step = converter.atif_to_tunix_step(
        agent_step=agent_atif_step, env_step=env_atif_step
    )

    self.assertIsInstance(dto_step, agent_types.Step)
    # Agent assertions
    self.assertEqual(dto_step.model_response, "Calling search")
    self.assertEqual(dto_step.thought, "Search query planning")
    self.assertIsNotNone(dto_step.action)
    self.assertEqual(
        dto_step.action.action,
        {"name": "search", "arguments": {"query": "tunix"}},
    )
    np.testing.assert_array_equal(dto_step.assistant_tokens, np.array([10, 20]))
    np.testing.assert_array_equal(dto_step.logprobs, np.array([-0.5, -0.3]))
    self.assertEqual(dto_step.info.get("session_id"), "sess_1")
    # Env assertions
    self.assertEqual(dto_step.observation, "search result")
    self.assertEqual(dto_step.reward, 0.8)
    self.assertFalse(dto_step.done)
    np.testing.assert_array_equal(dto_step.env_tokens, np.array([99]))

  def test_roundtrip_step_conversion(self):
    mock_agent_step = agent_types.Step(
        model_response="Write code",
        thought="Plan the implementation",
        action=agent_types.Action(
            action={"name": "edit", "arguments": {"path": "main.py"}}
        ),
        assistant_tokens=np.array([5, 6]),
        assistant_masks=np.array([1, 1]),
        logprobs=np.array([-0.05, -0.01]),
        mc_return=1.0,
        info={"session_id": "sess_123"},
    )
    mock_env_step = agent_types.Step(
        observation="File saved successfully",
        reward=1.0,
        done=True,
        env_tokens=np.array([42]),
        env_masks=np.array([1]),
    )

    agent_atif_step = converter.tunix_to_atif_step(
        mock_agent_step, step_id=1, source=trajectory_lib.Source.AGENT
    )
    env_atif_step = converter.tunix_to_atif_step(
        mock_env_step, step_id=2, source=trajectory_lib.Source.SYSTEM
    )

    restored_step = converter.atif_to_tunix_step(
        agent_step=agent_atif_step, env_step=env_atif_step
    )

    self.assertEqual(
        restored_step.model_response, mock_agent_step.model_response
    )
    self.assertEqual(restored_step.thought, mock_agent_step.thought)
    self.assertEqual(restored_step.action.action, mock_agent_step.action.action)
    self.assertEqual(restored_step.observation, mock_env_step.observation)
    self.assertEqual(restored_step.reward, mock_env_step.reward)
    self.assertEqual(restored_step.done, mock_env_step.done)
    self.assertEqual(restored_step.mc_return, mock_agent_step.mc_return)
    np.testing.assert_array_equal(
        restored_step.assistant_tokens, mock_agent_step.assistant_tokens
    )
    np.testing.assert_array_equal(
        restored_step.assistant_masks, mock_agent_step.assistant_masks
    )
    np.testing.assert_array_equal(
        restored_step.logprobs, mock_agent_step.logprobs
    )
    np.testing.assert_array_equal(
        restored_step.env_tokens, mock_env_step.env_tokens
    )
    np.testing.assert_array_equal(
        restored_step.env_masks, mock_env_step.env_masks
    )
    self.assertEqual(
        restored_step.info.get("session_id"),
        mock_agent_step.info.get("session_id"),
    )


if __name__ == "__main__":
  absltest.main()
