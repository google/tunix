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

import dataclasses
from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.experimental.trajectory import converter
from tunix.experimental.trajectory import store_testing
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.agents import agent_types


class CreateAgentStepTest(parameterized.TestCase):

  def test_create_agent_step_none_returns_none(self):
    step = converter.create_agent_step(None, tunix_step_id=0)
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

    agent_step = converter.create_agent_step(mock_agent_step, tunix_step_id=0)

    expected_step = trajectory_lib.TunixStep(
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
        extra={
            "trace_id": "123",
            "raw_action": {"name": "bash", "arguments": {"command": "ls -la"}},
        },
    )
    store_testing.assert_step_equal(self, agent_step, expected_step)

  def test_create_agent_step_step_id_mapping(self):
    mock_agent_step = agent_types.Step(model_response="resp")
    agent_step = converter.create_agent_step(mock_agent_step, tunix_step_id=2)
    self.assertIsNotNone(agent_step)
    self.assertEqual(agent_step.step_id, 5)

  def test_create_agent_step_length_mismatch_raises_value_error(self):
    mock_rl_step = agent_types.Step(
        model_response="test",
        assistant_tokens=np.array([1, 2, 3]),
        logprobs=np.array([-0.1, -0.2]),
    )
    with self.assertRaises(ValueError):
      converter.create_agent_step(mock_rl_step, tunix_step_id=0)

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
    agent_step = converter.create_agent_step(mock_agent_step, tunix_step_id=1)
    self.assertIsNotNone(agent_step)
    self.assertLen(agent_step.tool_calls, 2)
    self.assertEqual(agent_step.step_id, 3)
    self.assertEqual(agent_step.tool_calls[0].tool_call_id, "call_101")
    self.assertEqual(agent_step.tool_calls[0].function_name, "bash")
    self.assertEqual(agent_step.tool_calls[0].arguments, {"cmd": "pwd"})
    self.assertEqual(agent_step.tool_calls[1].tool_call_id, "call_102")
    self.assertEqual(agent_step.tool_calls[1].function_name, "python")
    self.assertEqual(agent_step.tool_calls[1].arguments, {"code": "print(42)"})


class CreateEnvStepTest(parameterized.TestCase):

  def test_create_env_step_none_returns_none(self):
    step = converter.create_env_step(None, tunix_step_id=0)
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

    env_step = converter.create_env_step(mock_env_step, tunix_step_id=0)

    expected_step = trajectory_lib.TunixStep(
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

  def test_create_env_step_step_id_mapping(self):
    mock_env_step = agent_types.Step(observation="obs")
    env_step = converter.create_env_step(mock_env_step, tunix_step_id=2)
    self.assertIsNotNone(env_step)
    self.assertEqual(env_step.step_id, 6)

  def test_create_env_step_list_observation(self):
    mock_env_step = agent_types.Step(
        observation=["file1.txt", "file2.txt"],
        reward=1.0,
        done=False,
    )

    env_step = converter.create_env_step(mock_env_step, tunix_step_id=0)

    expected_step = trajectory_lib.TunixStep(
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

    env_step = converter.create_env_step(mock_env_step, tunix_step_id=0)

    expected_step = trajectory_lib.TunixStep(
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
    env_traj_step = trajectory_lib.TunixStep(
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
    env_traj_step = trajectory_lib.TunixStep(
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

    agent_traj_step = converter.create_agent_step(mock_agent_step, tunix_step_id=0)
    env_traj_step = converter.create_env_step(mock_env_step, tunix_step_id=0)

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

    agent_traj_step = converter.create_agent_step(mock_agent_step, tunix_step_id=0)
    restored_step = converter.to_tunix_step(agent_step=agent_traj_step)

    store_testing.assert_step_equal(self, restored_step, mock_agent_step)

  def test_roundtrip_openai_format(self):
    raw_action = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "search",
            "arguments": {"query": "tunix"},
        },
    }
    mock_agent_step = agent_types.Step(
        model_response="Calling search",
        action=agent_types.Action(action=raw_action),
    )
    agent_traj_step = converter.create_agent_step(mock_agent_step, tunix_step_id=0)
    self.assertIsNotNone(agent_traj_step.tool_calls)
    self.assertEqual(agent_traj_step.tool_calls[0].tool_call_id, "call_1")
    self.assertEqual(agent_traj_step.tool_calls[0].function_name, "search")
    self.assertEqual(
        agent_traj_step.tool_calls[0].arguments, {"query": "tunix"}
    )
    self.assertEqual(agent_traj_step.extra["raw_action"], raw_action)
    restored_step = converter.to_tunix_step(agent_step=agent_traj_step)
    store_testing.assert_step_equal(self, restored_step, mock_agent_step)

  def test_roundtrip_anthropic_format(self):
    raw_action = {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "weather",
        "input": {"city": "Paris"},
    }
    mock_agent_step = agent_types.Step(
        model_response="Calling weather",
        action=agent_types.Action(action=raw_action),
    )
    agent_traj_step = converter.create_agent_step(mock_agent_step, tunix_step_id=0)
    self.assertIsNotNone(agent_traj_step.tool_calls)
    self.assertEqual(agent_traj_step.tool_calls[0].tool_call_id, "toolu_1")
    self.assertEqual(agent_traj_step.tool_calls[0].function_name, "weather")
    self.assertEqual(agent_traj_step.tool_calls[0].arguments, {"city": "Paris"})
    self.assertEqual(agent_traj_step.extra["raw_action"], raw_action)
    restored_step = converter.to_tunix_step(agent_step=agent_traj_step)
    store_testing.assert_step_equal(self, restored_step, mock_agent_step)

  def test_roundtrip_xml_action(self):
    raw_action = (
        "<function=file_editor>\n"
        "<parameter=command>view</parameter>\n"
        "<parameter=path>main.py</parameter>\n"
        "</function>"
    )
    mock_agent_step = agent_types.Step(
        model_response="Calling editor",
        action=agent_types.Action(action=raw_action),
    )
    agent_traj_step = converter.create_agent_step(mock_agent_step, tunix_step_id=0)
    self.assertIsNotNone(agent_traj_step.tool_calls)
    self.assertEqual(agent_traj_step.tool_calls[0].function_name, "file_editor")
    self.assertEqual(
        agent_traj_step.tool_calls[0].arguments,
        {"command": "view", "path": "main.py"},
    )
    self.assertEqual(agent_traj_step.extra["raw_action"], raw_action)
    restored_step = converter.to_tunix_step(agent_step=agent_traj_step)
    store_testing.assert_step_equal(self, restored_step, mock_agent_step)

  def test_roundtrip_non_tool_actions(self):
    non_tool_payloads = [
        0,
        1,
        42,
        3.14,
        True,
        False,
        "I will now think about how to answer...",
        np.array([1, 2, 3]),
        {"not_a_tool": "just a dictionary payload", "count": 5},
    ]
    for raw_action in non_tool_payloads:
      mock_step = agent_types.Step(
          model_response="Non-tool step",
          action=agent_types.Action(action=raw_action),
      )
      agent_traj_step = converter.create_agent_step(mock_step, tunix_step_id=0)
      self.assertIn("raw_action", agent_traj_step.extra)
      restored_step = converter.to_tunix_step(agent_step=agent_traj_step)
      if isinstance(raw_action, np.ndarray):
        np.testing.assert_array_equal(restored_step.action.action, raw_action)
        self.assertEqual(restored_step.model_response, mock_step.model_response)
      else:
        store_testing.assert_step_equal(self, restored_step, mock_step)

  def test_roundtrip_custom_dataclass_action(self):
    @dataclasses.dataclass
    class CustomCommand:
      tool: str
      payload: dict[str, Any]
      def __eq__(self, other):
        return isinstance(other, CustomCommand) and self.tool == other.tool and self.payload == other.payload

    cmd = CustomCommand(tool="shell", payload={"cmd": "whoami"})
    mock_agent_step = agent_types.Step(
        model_response="Run command",
        action=agent_types.Action(action=cmd),
    )
    agent_traj_step = converter.create_agent_step(mock_agent_step, tunix_step_id=0)
    self.assertIs(agent_traj_step.extra["raw_action"], cmd)
    restored_step = converter.to_tunix_step(agent_step=agent_traj_step)
    store_testing.assert_step_equal(self, restored_step, mock_agent_step)
    self.assertIsInstance(restored_step.action.action, CustomCommand)

  def test_to_tunix_step_fallback_to_tool_calls_when_raw_action_absent(self):
    agent_traj_step = trajectory_lib.Step(
        step_id=1,
        source=trajectory_lib.Source.AGENT,
        message="Calling search",
        tool_calls=[
            trajectory_lib.ToolCall(
                tool_call_id="call_99",
                function_name="search",
                arguments={"q": "tunix"},
            )
        ],
        extra={"custom_tag": "external_source"},
    )
    dto_step = converter.to_tunix_step(agent_step=agent_traj_step)
    self.assertIsNotNone(dto_step.action)
    self.assertEqual(
        dto_step.action,
        agent_types.Action(
            action={
                "id": "call_99",
                "name": "search",
                "arguments": {"q": "tunix"},
            }
        ),
    )
    self.assertEqual(dto_step.info, {"custom_tag": "external_source"})

  def test_to_tunix_step_fallback_multiple_tool_calls_when_raw_action_absent(
      self,
  ):
    agent_traj_step = trajectory_lib.Step(
        step_id=1,
        source=trajectory_lib.Source.AGENT,
        message="Calling tools",
        tool_calls=[
            trajectory_lib.ToolCall(
                tool_call_id="c1",
                function_name="fn1",
                arguments={"a": 1},
            ),
            trajectory_lib.ToolCall(
                tool_call_id="c2",
                function_name="fn2",
                arguments={"b": 2},
            ),
        ],
    )
    dto_step = converter.to_tunix_step(agent_step=agent_traj_step)
    self.assertIsNotNone(dto_step.action)
    self.assertEqual(
        dto_step.action,
        agent_types.Action(
            action=[
                {"id": "c1", "name": "fn1", "arguments": {"a": 1}},
                {"id": "c2", "name": "fn2", "arguments": {"b": 2}},
            ]
        ),
    )

  def test_to_tunix_step_env_step_raw_action_excluded_from_info(self):
    agent_step = trajectory_lib.Step(
        step_id=1,
        source=trajectory_lib.Source.AGENT,
        message="agent msg",
        extra={"raw_action": "agent_action", "other_agent_info": "ok"},
    )
    env_step = trajectory_lib.Step(
        step_id=2,
        source=trajectory_lib.Source.SYSTEM,
        message="env msg",
        extra={"raw_action": "env_action", "other_env_info": "ok"},
    )
    dto_step = converter.to_tunix_step(agent_step=agent_step, env_step=env_step)
    self.assertEqual(dto_step.action, agent_types.Action(action="agent_action"))
    self.assertEqual(
        dto_step.info, {"other_agent_info": "ok", "other_env_info": "ok"}
    )
    self.assertNotIn("raw_action", dto_step.info)
    self.assertNotIn("action", dto_step.info)

  def test_roundtrip_falsy_raw_actions(self):
    falsy_actions = [0, False, "", [], {}]
    for falsy in falsy_actions:
      orig_step = agent_types.Step(
          model_response="resp",
          action=agent_types.Action(action=falsy),
      )
      agent_step = converter.create_agent_step(orig_step, tunix_step_id=0)
      self.assertIn("raw_action", agent_step.extra)
      self.assertEqual(agent_step.extra["raw_action"], falsy)
      restored = converter.to_tunix_step(agent_step=agent_step)
      store_testing.assert_step_equal(self, restored, orig_step)


class ToTunixTrajectoryTest(parameterized.TestCase):

  def test_to_tunix_trajectory_empty(self):
    traj = trajectory_lib.Trajectory(
        trajectory_id="empty_t",
        agent=trajectory_lib.Agent(name="test_agent", version="1.0"),
        steps=[],
    )
    tunix_traj = converter.to_tunix_trajectory(traj)
    self.assertIsNone(tunix_traj.task)
    self.assertEmpty(tunix_traj.steps)
    self.assertEqual(tunix_traj.reward, 0.0)
    self.assertEqual(tunix_traj.status, agent_types.TrajectoryStatus.RUNNING)

  def test_to_tunix_trajectory_with_task_and_paired_steps(self):
    traj = trajectory_lib.TunixTrajectory(
        trajectory_id="t1",
        agent=trajectory_lib.Agent(name="test_agent", version="1.0"),
        total_reward=2.5,
        status="SUCCEEDED",
        steps=[
            trajectory_lib.TunixStep(
                step_id=0,
                source=trajectory_lib.Source.USER,
                message="Calculate 3+4",
            ),
            trajectory_lib.TunixStep(
                step_id=1,
                source=trajectory_lib.Source.AGENT,
                message="7",
                reasoning_content="Simple addition",
            ),
            trajectory_lib.TunixStep(
                step_id=2,
                source=trajectory_lib.Source.SYSTEM,
                message="Correct",
                reward=2.5,
                done=True,
            ),
        ],
        env_time={"env_step": 0.05},
        reward_time={"reward_eval": 0.01},
    )
    tunix_traj = converter.to_tunix_trajectory(traj)
    self.assertEqual(tunix_traj.task, {"prompts": ["Calculate 3+4"]})
    self.assertLen(tunix_traj.steps, 1)
    self.assertEqual(tunix_traj.steps[0].model_response, "7")
    self.assertEqual(tunix_traj.steps[0].thought, "Simple addition")
    self.assertEqual(tunix_traj.steps[0].observation, "Correct")
    self.assertEqual(tunix_traj.steps[0].reward, 2.5)
    self.assertTrue(tunix_traj.steps[0].done)
    self.assertEqual(tunix_traj.reward, 2.5)
    self.assertEqual(tunix_traj.status, agent_types.TrajectoryStatus.SUCCEEDED)
    self.assertEqual(tunix_traj.env_time, {"env_step": 0.05})
    self.assertEqual(tunix_traj.reward_time, {"reward_eval": 0.01})

  def test_to_tunix_trajectory_from_dict(self):
    traj_dict = {
        "trajectory_id": "dict_traj",
        "agent": {"name": "test_agent", "version": "1.0"},
        "total_reward": 1.0,
        "status": "FAILED",
        "steps": [
            {
                "step_id": 0,
                "source": "user",
                "message": "Task from dict",
            },
            {
                "step_id": 1,
                "source": "agent",
                "message": "response",
            },
        ],
    }
    tunix_traj = converter.to_tunix_trajectory(traj_dict)
    self.assertEqual(tunix_traj.task, {"prompts": ["Task from dict"]})
    self.assertLen(tunix_traj.steps, 1)
    self.assertEqual(tunix_traj.steps[0].model_response, "response")
    self.assertEqual(tunix_traj.reward, 1.0)
    self.assertEqual(tunix_traj.status, agent_types.TrajectoryStatus.FAILED)

  def test_roundtrip_trajectory_full_with_json_dict_and_hybrid_action(self):
    orig_rl_step = agent_types.Step(
        model_response="Querying DB",
        thought="I will query the database",
        action=agent_types.Action(
            action={"query": "SELECT * FROM users", "limit": 10}
        ),
        observation="Found 10 rows",
        reward=1.0,
        done=True,
        mc_return=1.0,
        info={"session": "test_session_123"},
    )
    agent_step = converter.create_agent_step(orig_rl_step, tunix_step_id=0)
    env_step = converter.create_env_step(orig_rl_step, tunix_step_id=0)
    user_step = converter.create_task_step("Find users")

    traj = trajectory_lib.TunixTrajectory(
        trajectory_id="roundtrip_traj_1",
        agent=trajectory_lib.Agent(name="db_agent", version="1.0"),
        total_reward=1.0,
        status="SUCCEEDED",
        steps=[user_step, agent_step, env_step],
    )
    json_dict = traj.to_json_dict()
    restored_tunix_traj = converter.to_tunix_trajectory(json_dict)

    self.assertEqual(restored_tunix_traj.task, {"prompts": ["Find users"]})
    self.assertLen(restored_tunix_traj.steps, 1)
    store_testing.assert_step_equal(
        self, restored_tunix_traj.steps[0], orig_rl_step
    )

  def test_deepswe_guarded_env_steps(self):
    guard_obs = (
        "[ACTION GUARD] This exact action just failed. Repeating it will"
        " produce the same result.\nPlease try a DIFFERENT approach:\n- View"
        " the file around the relevant lines with a specific view_range"
    )
    guard_info = {
        "guard_blocked": True,
        "guard_reason": "repeated_failure",
    }
    mock_env_step = agent_types.Step(
        observation=guard_obs,
        reward=0.0,
        done=False,
        info=guard_info,
    )
    env_step = converter.create_env_step(mock_env_step, tunix_step_id=0)
    self.assertIsNotNone(env_step)
    self.assertEqual(env_step.message, guard_obs)
    self.assertEqual(env_step.observation.results[0].content, guard_obs)
    self.assertEqual(env_step.reward, 0.0)
    self.assertFalse(env_step.done)
    self.assertEqual(env_step.extra, guard_info)

    restored_env_step = converter.to_tunix_step(env_step=env_step)
    store_testing.assert_step_equal(self, restored_env_step, mock_env_step)

  def test_deepswe_token_warning_injection_step(self):
    token_warning_obs = (
        "Execution output of [execute_bash]:\nTests passed.\n"
        "\nYou are running out of tokens. Stop exploring now. Do not call"
        " file_editor, str_replace_editor, search, execute_bash, or any view"
        " command again. You must immediately submit using the final tool."
        " Output exactly this XML and nothing else:\n"
        "<function=finish>\n"
        "<parameter=command>submit</parameter>\n"
        "<parameter=result>FINAL_RESULT</parameter>\n"
        "</function>\n"
    )
    mock_env_step = agent_types.Step(
        observation=token_warning_obs,
        reward=0.0,
        done=False,
        info={"cur_tokens": 28500, "max_steps": 30},
    )
    env_traj_step = converter.create_env_step(mock_env_step, tunix_step_id=0)
    self.assertIsNotNone(env_traj_step)
    self.assertEqual(env_traj_step.message, token_warning_obs)
    self.assertEqual(env_traj_step.extra["cur_tokens"], 28500)

    restored = converter.to_tunix_step(env_step=env_traj_step)
    store_testing.assert_step_equal(self, restored, mock_env_step)

  def test_deepswe_full_multi_turn_trajectory_roundtrip_lossless(self):
    task_prompt = (
        "Consider the following github issue:\n"
        "<github_issue>\n"
        "sympy.tensor.array.ImmutableDenseNDimArray equality fails for nested"
        " arrays\n"
        "</github_issue>\n"
        "Can you help me implement the necessary changes?"
    )

    turn1_thought = "Let's search for ImmutableDenseNDimArray in the codebase."
    turn1_action_xml = (
        "<function=search>\n"
        "  <parameter=search_term>class"
        " ImmutableDenseNDimArray</parameter>\n"
        "  <parameter=path>/testbed/sympy/tensor</parameter>\n"
        "</function>"
    )
    turn1_step = agent_types.Step(
        model_response=f"{turn1_thought}\n\n{turn1_action_xml}",
        thought=turn1_thought,
        action=agent_types.Action(action=turn1_action_xml),
        observation=(
            "Matches found in:\n"
            "/testbed/sympy/tensor/array/dense_ndim_array.py: line 42"
        ),
        reward=0.0,
        done=False,
        info={"step": 1},
    )

    turn2_thought = "Let's view the implementation of ImmutableDenseNDimArray."
    turn2_action_xml = (
        "<function=file_editor>\n  <parameter=command>view</parameter>\n "
        " <parameter=path>/testbed/sympy/tensor/array/dense_ndim_array.py</parameter>\n"
        "  <parameter=view_range>[40, 60]</parameter>\n "
        " <parameter=concise>False</parameter>\n</function>"
    )
    turn2_step = agent_types.Step(
        model_response=f"{turn2_thought}\n\n{turn2_action_xml}",
        thought=turn2_thought,
        action=agent_types.Action(action=turn2_action_xml),
        observation=(
            "40: class ImmutableDenseNDimArray(DenseNDimArray):\n41:    "
            " def __eq__(self, other):\n42:         return False\n"
        ),
        reward=0.0,
        done=False,
        info={"step": 2},
    )

    turn3_thought = "Let's fix the equality method using str_replace."
    turn3_action_xml = (
        "<function=file_editor>\n  <parameter=command>str_replace</parameter>\n"
        "  <parameter=path>/testbed/sympy/tensor/array/dense_ndim_array.py</parameter>\n"
        "  <parameter=old_str>    def __eq__(self, other):\n        return"
        " False</parameter>\n  <parameter=new_str>    def __eq__(self,"
        " other):\n        return self._array =="
        " other._array</parameter>\n</function>"
    )
    turn3_step = agent_types.Step(
        model_response=f"{turn3_thought}\n\n{turn3_action_xml}",
        thought=turn3_thought,
        action=agent_types.Action(action=turn3_action_xml),
        observation=(
            "The file /testbed/sympy/tensor/array/dense_ndim_array.py has been"
            " edited."
        ),
        reward=0.0,
        done=False,
        info={"step": 3},
    )

    turn4_thought = "Let's run the test suite to verify our fix."
    turn4_action_xml = (
        "<function=execute_bash>\n"
        "  <parameter=cmd>pytest"
        " sympy/tensor/array/tests/test_ndim_array_equality.py</parameter>\n"
        "</function>"
    )
    turn4_step = agent_types.Step(
        model_response=f"{turn4_thought}\n\n{turn4_action_xml}",
        thought=turn4_thought,
        action=agent_types.Action(action=turn4_action_xml),
        observation=(
            "======================== 5 passed in 0.42s"
            " ========================"
        ),
        reward=0.0,
        done=False,
        info={"step": 4},
    )

    turn5_thought = "All tests pass. Let's finish and submit our solution."
    turn5_action_xml = (
        "<function=finish>\n  <parameter=command>submit</parameter>\n "
        " <parameter=result>Fixed ImmutableDenseNDimArray"
        " equality</parameter>\n</function>"
    )
    turn5_step = agent_types.Step(
        model_response=f"{turn5_thought}\n\n{turn5_action_xml}",
        thought=turn5_thought,
        action=agent_types.Action(action=turn5_action_xml),
        observation="<<<Finished>>>\nEvaluation score: 1.0",
        reward=1.0,
        done=True,
        mc_return=1.0,
        info={"step": 5, "resolved": True},
    )

    original_trajectory = agent_types.Trajectory(
        task={"prompts": [task_prompt]},
        steps=[turn1_step, turn2_step, turn3_step, turn4_step, turn5_step],
        reward=1.0,
        status=agent_types.TrajectoryStatus.SUCCEEDED,
        env_time={"step_time": 1.25},
        reward_time={"eval_time": 0.45},
    )

    # Convert to TunixTrajectory Steps
    converted_steps = [converter.create_task_step(original_trajectory.task)]
    for idx, s in enumerate(original_trajectory.steps):
      agent_st = converter.create_agent_step(s, tunix_step_id=idx)
      converted_steps.append(agent_st)
      env_st = converter.create_env_step(s, tunix_step_id=idx)
      converted_steps.append(env_st)

    tunix_traj_obj = trajectory_lib.TunixTrajectory(
        trajectory_id="deepswe_episode_001",
        agent=trajectory_lib.Agent(name="DeepSWEAgent", version="1.0"),
        total_reward=1.0,
        status="SUCCEEDED",
        steps=converted_steps,
        env_time={"step_time": 1.25},
        reward_time={"eval_time": 0.45},
    )

    # Verify tool calls are cleanly populated in every agent step
    self.assertEqual(converted_steps[1].tool_calls[0].function_name, "search")
    self.assertEqual(converted_steps[3].tool_calls[0].function_name, "file_editor")
    self.assertEqual(converted_steps[5].tool_calls[0].function_name, "file_editor")
    self.assertEqual(converted_steps[7].tool_calls[0].function_name, "execute_bash")
    self.assertEqual(converted_steps[9].tool_calls[0].function_name, "finish")

    # Serialize to JSON dict and deserialize
    traj_json_dict = tunix_traj_obj.to_json_dict()
    restored_tunix_traj = converter.to_tunix_trajectory(traj_json_dict)

    # Verify complete lossless roundtrip of the trajectory
    self.assertEqual(restored_tunix_traj.task, original_trajectory.task)
    self.assertEqual(restored_tunix_traj.reward, 1.0)
    self.assertEqual(
        restored_tunix_traj.status, agent_types.TrajectoryStatus.SUCCEEDED
    )
    self.assertEqual(restored_tunix_traj.env_time, {"step_time": 1.25})
    self.assertEqual(restored_tunix_traj.reward_time, {"eval_time": 0.45})
    self.assertLen(restored_tunix_traj.steps, len(original_trajectory.steps))

    for i in range(len(original_trajectory.steps)):
      store_testing.assert_step_equal(
          self, restored_tunix_traj.steps[i], original_trajectory.steps[i]
      )


class CreateTrajectoryMetadataTest(parameterized.TestCase):

  def test_create_trajectory_metadata_defaults(self):
    meta = converter.create_trajectory_metadata(traj_id="traj_100")
    self.assertIsInstance(meta, trajectory_lib.TunixTrajectoryMetadata)
    self.assertIsInstance(meta, trajectory_lib.TrajectoryMetadata)
    self.assertEqual(meta.trajectory_id, "traj_100")
    self.assertEqual(meta.agent.name, "agent")
    self.assertEqual(meta.agent.version, "1.0")
    self.assertEqual(meta.status, "RUNNING")
    self.assertIsNone(meta.prompt_id)
    self.assertIsNone(meta.group_offset_id)
    self.assertIsNone(meta.target_policy_versions)
    self.assertIsNone(meta.total_reward)
    self.assertIsNone(meta.hyperparams)
    self.assertIsNone(meta.extra)

  def test_create_trajectory_metadata_from_request_and_agent(self):
    class MockRolloutRequest:
      prompt_id = "prompt_123"
      group_offset_id = "group_offset_abc"
      generation_kwargs = {"temperature": 0.8, "top_k": 40}
      metadata = {"experiment": "exp_v1"}

    class MockAgentTrajectory:
      reward = 8.5
      env_time = {"env": 0.2}
      reward_time = {"rew": 0.05}

    class MockAgent:
      name = "custom_agent"
      version = "2.1"
      trajectory = MockAgentTrajectory()

    meta = converter.create_trajectory_metadata(
        traj_id="traj_200",
        request=MockRolloutRequest(),
        agent=MockAgent(),
        target_policy_versions=[1, 2, 3],
        status="SUCCEEDED",
        extra={"custom_tag": "run_1"},
    )
    self.assertIsInstance(meta, trajectory_lib.TunixTrajectoryMetadata)
    self.assertEqual(meta.trajectory_id, "traj_200")
    self.assertEqual(meta.agent.name, "custom_agent")
    self.assertEqual(meta.agent.version, "2.1")
    self.assertEqual(meta.prompt_id, "prompt_123")
    self.assertEqual(meta.group_offset_id, "group_offset_abc")
    self.assertEqual(meta.target_policy_versions, [1, 2, 3])
    self.assertEqual(meta.status, "SUCCEEDED")
    self.assertEqual(meta.total_reward, 8.5)
    self.assertEqual(meta.hyperparams, {"temperature": 0.8, "top_k": 40})
    self.assertEqual(meta.env_time, {"env": 0.2})
    self.assertEqual(meta.reward_time, {"rew": 0.05})
    self.assertEqual(
        meta.extra, {"experiment": "exp_v1", "custom_tag": "run_1"}
    )


if __name__ == "__main__":
  absltest.main()
