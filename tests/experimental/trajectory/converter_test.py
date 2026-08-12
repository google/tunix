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
from tunix.experimental.trajectory import store_testing
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.agents import agent_types


class CreateAgentStepTest(parameterized.TestCase):

  def test_create_agent_step_none_returns_none(self):
    step = converter.create_agent_step(None)
    self.assertIsNone(step)

  def test_create_agent_step_converts_all_fields(self):
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

    agent_step = converter.create_agent_step(
        mock_agent_step, step_id=0
    )

    expected_step = trajectory_lib.Step(
        step_id=1,
        source=trajectory_lib.Source.AGENT,
        message="Calling bash tool",
        reasoning_content="I need to list files",
        tool_calls=[
            trajectory_lib.ToolCall(
                tool_call_id="call_1",
                function_name="bash",
                arguments={"command": "ls -la"},
            )
        ],
        metrics=trajectory_lib.Metrics(
            completion_tokens=2,
            completion_token_ids=[101, 102],
            logprobs=[-0.1, -0.2],
        ),
        assistant_tokens=mock_agent_step.assistant_tokens,
        assistant_masks=mock_agent_step.assistant_masks,
        logprobs=mock_agent_step.logprobs,
        mc_return=1.5,
        extra={"trace_id": "123"},
    )
    store_testing.assert_step_equal(self, agent_step, expected_step)

  def test_create_agent_step_default_step_id(self):
    mock_agent_step = agent_types.Step(model_response="resp")
    agent_step = converter.create_agent_step(mock_agent_step)
    self.assertIsNotNone(agent_step)
    self.assertEqual(agent_step.step_id, 1)

  def test_create_agent_step_length_mismatch_raises_value_error(self):
    mock_rl_step = agent_types.Step(
        model_response="test",
        assistant_tokens=np.array([1, 2, 3]),
        logprobs=np.array([-0.1, -0.2]),
    )
    with self.assertRaises(ValueError):
      converter.create_agent_step(mock_rl_step)

  def test_create_agent_step_multiple_tool_calls(self):
    mock_agent_step = agent_types.Step(
        model_response="Calling multiple tools",
        thought="I will invoke bash and python in sequence",
        action=agent_types.Action(
            action=[
                {
                    "id": "call_101",
                    "name": "bash",
                    "arguments": {"cmd": "pwd"},
                },
                {
                    "id": "call_102",
                    "name": "python",
                    "arguments": {"code": "print(42)"},
                },
            ]
        ),
    )
    agent_step = converter.create_agent_step(mock_agent_step, step_id=1)
    self.assertIsNotNone(agent_step)
    self.assertLen(agent_step.tool_calls, 2)
    self.assertEqual(agent_step.tool_calls[0].tool_call_id, "call_101")
    self.assertEqual(agent_step.tool_calls[0].function_name, "bash")
    self.assertEqual(agent_step.tool_calls[0].arguments, {"cmd": "pwd"})
    self.assertEqual(agent_step.tool_calls[1].tool_call_id, "call_102")
    self.assertEqual(agent_step.tool_calls[1].function_name, "python")
    self.assertEqual(agent_step.tool_calls[1].arguments, {"code": "print(42)"})

  def test_create_agent_step_tool_call_fallback_keys_and_defaults(self):
    mock_agent_step = agent_types.Step(
        action=agent_types.Action(
            action=[
                {
                    "tool_call_id": "call_custom",
                    "function_name": "browse",
                    "args": {"url": "https://google.com"},
                },
                {
                    "args": {"x": 10},
                },
                {},
            ]
        )
    )
    agent_step = converter.create_agent_step(mock_agent_step, step_id=1)
    self.assertIsNotNone(agent_step)
    self.assertLen(agent_step.tool_calls, 3)
    # First item uses tool_call_id, function_name, and args
    self.assertEqual(agent_step.tool_calls[0].tool_call_id, "call_custom")
    self.assertEqual(agent_step.tool_calls[0].function_name, "browse")
    self.assertEqual(
        agent_step.tool_calls[0].arguments, {"url": "https://google.com"}
    )
    # Second item defaults to call_2 and action
    self.assertEqual(agent_step.tool_calls[1].tool_call_id, "call_2")
    self.assertEqual(agent_step.tool_calls[1].function_name, "action")
    self.assertEqual(agent_step.tool_calls[1].arguments, {"x": 10})
    # Third item defaults to call_3, action, and {}
    self.assertEqual(agent_step.tool_calls[2].tool_call_id, "call_3")
    self.assertEqual(agent_step.tool_calls[2].function_name, "action")
    self.assertEqual(agent_step.tool_calls[2].arguments, {})

  def test_create_agent_step_tool_call_payload_as_arguments_dict(self):
    mock_agent_step = agent_types.Step(
        action=agent_types.Action(action={"arg_a": 1, "arg_b": "two"})
    )
    agent_step = converter.create_agent_step(mock_agent_step, step_id=1)
    self.assertIsNotNone(agent_step)
    self.assertLen(agent_step.tool_calls, 1)
    self.assertEqual(agent_step.tool_calls[0].tool_call_id, "call_1")
    self.assertEqual(agent_step.tool_calls[0].function_name, "action")
    self.assertEqual(
        agent_step.tool_calls[0].arguments, {"arg_a": 1, "arg_b": "two"}
    )

  def test_create_agent_step_tool_call_non_dict_arguments_handled(self):
    mock_agent_step = agent_types.Step(
        action=agent_types.Action(
            action={
                "id": "c1",
                "name": "fn",
                "arguments": "invalid_string_args",
            }
        )
    )
    agent_step = converter.create_agent_step(mock_agent_step, step_id=1)
    self.assertIsNotNone(agent_step)
    self.assertLen(agent_step.tool_calls, 1)
    self.assertEqual(agent_step.tool_calls[0].arguments, {})

  def test_create_agent_step_tool_call_invalid_payloads_return_none(self):
    for invalid_action in [
        agent_types.Action(action=[]),
        agent_types.Action(action=[None, "string_item"]),
        agent_types.Action(action="string_action"),
        agent_types.Action(action=123),
        agent_types.Action(action=None),
        None,
    ]:
      step = converter.create_agent_step(
          agent_types.Step(action=invalid_action), step_id=1
      )
      self.assertIsNone(step.tool_calls)


class CreateEnvStepTest(parameterized.TestCase):

  def test_create_env_step_none_returns_none(self):
    step = converter.create_env_step(None)
    self.assertIsNone(step)

  def test_create_env_step_converts_all_fields(self):
    mock_env_step = agent_types.Step(
        observation="file1.txt\nfile2.txt",
        reward=1.0,
        done=False,
        env_tokens=np.array([201]),
        env_masks=np.array([1]),
        info={"env_meta": "test_env"},
    )

    env_step = converter.create_env_step(mock_env_step, step_id=1)

    expected_step = trajectory_lib.Step(
        step_id=2,
        source=trajectory_lib.Source.SYSTEM,
        message="file1.txt\nfile2.txt",
        observation=trajectory_lib.Observation(
            results=[
                trajectory_lib.ObservationResult(content="file1.txt\nfile2.txt")
            ]
        ),
        reward=1.0,
        done=False,
        env_tokens=mock_env_step.env_tokens,
        env_masks=mock_env_step.env_masks,
        extra={"env_meta": "test_env"},
    )
    store_testing.assert_step_equal(self, env_step, expected_step)

  def test_create_env_step_default_step_id(self):
    mock_env_step = agent_types.Step(observation="obs")
    env_step = converter.create_env_step(mock_env_step)
    self.assertIsNotNone(env_step)
    self.assertEqual(env_step.step_id, 1)

  def test_create_env_step_list_observation(self):
    mock_env_step = agent_types.Step(
        observation=["file1.txt", "file2.txt"],
        reward=1.0,
        done=False,
    )

    env_step = converter.create_env_step(mock_env_step, step_id=1)

    expected_step = trajectory_lib.Step(
        step_id=2,
        source=trajectory_lib.Source.SYSTEM,
        message="['file1.txt', 'file2.txt']",
        observation=trajectory_lib.Observation(
            results=[
                trajectory_lib.ObservationResult(content="file1.txt"),
                trajectory_lib.ObservationResult(content="file2.txt"),
            ]
        ),
        reward=1.0,
        done=False,
    )
    store_testing.assert_step_equal(self, env_step, expected_step)

  def test_create_env_step_none_observation(self):
    mock_env_step = agent_types.Step(
        observation=None,
        reward=0.5,
        done=True,
    )

    env_step = converter.create_env_step(mock_env_step, step_id=1)

    expected_step = trajectory_lib.Step(
        step_id=2,
        source=trajectory_lib.Source.SYSTEM,
        message="",
        observation=None,
        reward=0.5,
        done=True,
    )
    store_testing.assert_step_equal(self, env_step, expected_step)


class CreateTaskStepTest(parameterized.TestCase):

  def test_create_task_step_none_returns_none(self):
    step = converter.create_task_step(None)
    self.assertIsNone(step)

  def test_create_task_step_empty_string_returns_none(self):
    step = converter.create_task_step("")
    self.assertIsNone(step)

  def test_create_task_step_string_prompt(self):
    step = converter.create_task_step("Solve 2+2")
    self.assertIsNotNone(step)
    self.assertEqual(step.step_id, 0)
    self.assertEqual(step.source, trajectory_lib.Source.USER)
    self.assertEqual(step.message, "Solve 2+2")

  def test_create_task_step_dict_with_prompts_list(self):
    step = converter.create_task_step({"prompts": ["What is the capital?"]})
    self.assertIsNotNone(step)
    self.assertEqual(step.step_id, 0)
    self.assertEqual(step.source, trajectory_lib.Source.USER)
    self.assertEqual(step.message, "What is the capital?")

  def test_create_task_step_dict_with_prompts_string(self):
    step = converter.create_task_step({"prompts": "What is the capital?"})
    self.assertIsNotNone(step)
    self.assertEqual(step.step_id, 0)
    self.assertEqual(step.source, trajectory_lib.Source.USER)
    self.assertEqual(step.message, "What is the capital?")

  def test_create_task_step_dict_empty_prompts_returns_none(self):
    step = converter.create_task_step({"prompts": []})
    self.assertIsNone(step)


class ToTunixStepTest(parameterized.TestCase):

  def test_to_tunix_step_none_returns_empty_step(self):
    dto_step = converter.to_tunix_step(None, None)
    store_testing.assert_step_equal(self, dto_step, agent_types.Step())

  def test_to_tunix_step_only_agent_step_passed(self):
    agent_traj_step = trajectory_lib.Step(
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
        mc_return=2.0,
        extra={"trace_id": "agent_only_trace"},
    )

    dto_step = converter.to_tunix_step(
        agent_step=agent_traj_step, env_step=None
    )

    expected_step = agent_types.Step(
        model_response="Calling search",
        thought="Search query planning",
        action=agent_types.Action(
            action={
                "id": "call_1",
                "name": "search",
                "arguments": {"query": "tunix"},
            }
        ),
        assistant_tokens=np.array([10, 20]),
        logprobs=np.array([-0.5, -0.3]),
        mc_return=2.0,
        info={"trace_id": "agent_only_trace"},
    )
    store_testing.assert_step_equal(self, dto_step, expected_step)

  def test_to_tunix_step_only_env_step_passed(self):
    env_traj_step = trajectory_lib.Step(
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
        reward=1.0,
        done=True,
        env_tokens=[201, 202],
        env_masks=[1, 1],
        extra={"env_meta": "meta_val"},
    )

    dto_step = converter.to_tunix_step(agent_step=None, env_step=env_traj_step)

    expected_step = agent_types.Step(
        observation="Search completed successfully",
        reward=1.0,
        done=True,
        env_tokens=np.array([201, 202]),
        env_masks=np.array([1, 1]),
        info={"env_meta": "meta_val"},
    )
    store_testing.assert_step_equal(self, dto_step, expected_step)

  def test_to_tunix_step_both_passed(self):
    agent_traj_step = trajectory_lib.Step(
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
    env_traj_step = trajectory_lib.Step(
        step_id=2,
        source=trajectory_lib.Source.SYSTEM,
        message="search result",
        observation=trajectory_lib.Observation(
            results=[trajectory_lib.ObservationResult(content="search result")]
        ),
        reward=0.8,
        done=False,
        env_tokens=[99],
    )

    dto_step = converter.to_tunix_step(
        agent_step=agent_traj_step, env_step=env_traj_step
    )

    expected_step = agent_types.Step(
        model_response="Calling search",
        thought="Search query planning",
        action=agent_types.Action(
            action={
                "id": "call_1",
                "name": "search",
                "arguments": {"query": "tunix"},
            }
        ),
        observation="search result",
        reward=0.8,
        done=False,
        assistant_tokens=np.array([10, 20]),
        logprobs=np.array([-0.5, -0.3]),
        env_tokens=np.array([99]),
        info={"session_id": "sess_1"},
    )
    store_testing.assert_step_equal(self, dto_step, expected_step)

  def test_roundtrip_step_conversion(self):
    mock_agent_step = agent_types.Step(
        model_response="Write code",
        thought="Plan the implementation",
        action=agent_types.Action(
            action={
                "id": "call_1",
                "name": "edit",
                "arguments": {"path": "main.py"},
            }
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

    agent_traj_step = converter.create_agent_step(
        mock_agent_step, step_id=1
    )
    env_traj_step = converter.create_env_step(
        mock_env_step, step_id=2
    )

    restored_step = converter.to_tunix_step(
        agent_step=agent_traj_step, env_step=env_traj_step
    )

    expected_step = agent_types.Step(
        model_response="Write code",
        thought="Plan the implementation",
        action=agent_types.Action(
            action={
                "id": "call_1",
                "name": "edit",
                "arguments": {"path": "main.py"},
            }
        ),
        observation="File saved successfully",
        reward=1.0,
        done=True,
        mc_return=1.0,
        assistant_tokens=np.array([5, 6]),
        assistant_masks=np.array([1, 1]),
        logprobs=np.array([-0.05, -0.01]),
        env_tokens=np.array([42]),
        env_masks=np.array([1]),
        info={"session_id": "sess_123"},
    )
    store_testing.assert_step_equal(self, restored_step, expected_step)

  def test_to_tunix_step_multiple_tool_calls(self):
    agent_traj_step = trajectory_lib.Step(
        step_id=1,
        source=trajectory_lib.Source.AGENT,
        message="Calling multiple tools",
        tool_calls=[
            trajectory_lib.ToolCall(
                tool_call_id="call_custom_1",
                function_name="read_file",
                arguments={"path": "a.txt"},
            ),
            trajectory_lib.ToolCall(
                tool_call_id="call_custom_2",
                function_name="write_file",
                arguments={"path": "b.txt", "content": "data"},
            ),
        ],
    )

    dto_step = converter.to_tunix_step(agent_step=agent_traj_step)

    expected_step = agent_types.Step(
        model_response="Calling multiple tools",
        action=agent_types.Action(
            action=[
                {
                    "id": "call_custom_1",
                    "name": "read_file",
                    "arguments": {"path": "a.txt"},
                },
                {
                    "id": "call_custom_2",
                    "name": "write_file",
                    "arguments": {"path": "b.txt", "content": "data"},
                },
            ]
        ),
    )
    store_testing.assert_step_equal(self, dto_step, expected_step)

  def test_roundtrip_step_conversion_multiple_tool_calls(self):
    mock_agent_step = agent_types.Step(
        model_response="Execute tools",
        thought="Plan execution",
        action=agent_types.Action(
            action=[
                {
                    "id": "call_custom_1",
                    "name": "read_file",
                    "arguments": {"path": "a.txt"},
                },
                {
                    "id": "call_custom_2",
                    "name": "write_file",
                    "arguments": {"path": "b.txt", "content": "hello"},
                },
            ]
        ),
    )

    agent_traj_step = converter.create_agent_step(
        mock_agent_step, step_id=1
    )
    restored_step = converter.to_tunix_step(agent_step=agent_traj_step)

    store_testing.assert_step_equal(self, restored_step, mock_agent_step)


if __name__ == "__main__":
  absltest.main()
