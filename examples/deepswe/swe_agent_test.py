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

"""Tests for CodeActAgent and CodeAct response parsing in DeepSWE."""

import os
from unittest import mock
from absl.testing import absltest
from examples.deepswe import openhands_utils
from examples.deepswe import sandbox_utils
from examples.deepswe import swe_agent
from examples.deepswe import swe_env
from examples.deepswe import template

SWEAction = swe_agent.SWEAction


class SweAgentTest(absltest.TestCase):

  def test_parse_codeact_markdown_fences_are_not_tool_calls(self):
    # The reference parser (vLLM qwen3_xml) reads only <tool_call>/<function=>
    # XML; a turn with just a code fence gets no action (the fake-user reply).
    for response in (
        "I need to list the files.\n```bash\ngit status\nls -la\n```\nDone.",
        "```sh\npytest tests/test_core.py\n```",
        "Let's test it.\n```python\nimport sympy\n"
        "print(sympy.__version__)\n```",
        "```ipython\n%run reproduce_issue.py\n```",
        "Executing long command:\n```bash\ngit log -n 5",
        'Via json.\n```json\n{"name": "execute_bash", "arguments": {"command":'
        ' "pwd"}}\n```',
    ):
      with self.subTest(response=response[:30]):
        thought, action = swe_agent.parse_codeact_response(response)
        self.assertEqual(action.function_name, "")
        self.assertEqual(thought, response.strip())

  def test_parse_codeact_fence_and_garbled_call_get_no_action(self):
    # A fence plus a tool call missing its <tool_call>/<function=> opener: no
    # call for qwen3_xml, so none here (it used to run execute_ipython_cell).
    response = (
        "The regex is:\n```python\nm = re.match(r\"hsl\\(\\s*("
        "\\parameter=command>\nview\n</parameter>\n"
        "<parameter=path>\n/testbed/a.py\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    _, action = swe_agent.parse_codeact_response(response)
    self.assertEqual(action.function_name, "")

  def test_parse_codeact_skips_empty_tool_call_block(self):
    # An empty <tool_call> block before a valid one: qwen3_xml returns the
    # valid call, so the empty block must not hide it.
    response = (
        "Now let's search:\n\n<tool_call>\n</function>\n</tool_call>\n\n"
        "<tool_call>\n<function=str_replace_editor>\n<parameter=command>\n"
        "view\n</parameter>\n<parameter=path>\n/testbed/Tests\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    _, action = swe_agent.parse_codeact_response(response)
    self.assertEqual(action.function_name, "str_replace_editor")
    self.assertEqual(
        action.parameters, {"command": "view", "path": "/testbed/Tests"}
    )

  def test_create_file_text_keeps_final_newline_through_env_round_trip(self):
    # The model writes "<content>\n\n</parameter>". qwen3_xml (and our model
    # parser) drop one newline; SWEEnv re-parses to_xml_string() output, which
    # must not drop the file's own final newline as well.
    response = (
        "<tool_call>\n<function=str_replace_editor>\n<parameter=command>\n"
        "create\n</parameter>\n<parameter=path>\n/testbed/t.py\n</parameter>\n"
        "<parameter=file_text>\n\n    indented = 1\nprint(indented)\n\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    _, action = swe_agent.parse_codeact_response(response)
    self.assertEqual(
        action.parameters["file_text"], "\n    indented = 1\nprint(indented)\n"
    )
    executed = openhands_utils.parse_openhands_action_str(
        action.to_xml_string()
    )
    self.assertEqual(executed.function_name, "str_replace_editor")
    self.assertEqual(executed.parameters, action.parameters)

  def test_parse_codeact_xml_function(self):
    response = (
        "I will view the file to check line numbers.\n"
        "<function=str_replace_editor>\n"
        "<parameter=command>view</parameter>\n"
        "<parameter=path>/testbed/foo.py</parameter>\n"
        "</function>"
    )
    thought, action = swe_agent.parse_codeact_response(response)
    self.assertEqual(
        thought, "I will view the file to check line numbers."
    )
    self.assertEqual(action.function_name, "str_replace_editor")
    self.assertEqual(action.parameters.get("command"), "view")
    self.assertEqual(action.parameters.get("path"), "/testbed/foo.py")

  def test_parse_codeact_json_tool_call_is_not_a_call(self):
    # qwen3_xml returns no call for a JSON body in <tool_call>.
    response = (
        "Running command via tool call.\n"
        "<tool_call>\n"
        '{"name": "execute_bash", "arguments": {"command": "git diff"}}\n'
        "</tool_call>"
    )
    thought, action = swe_agent.parse_codeact_response(response)
    self.assertEqual(thought, response.strip())
    self.assertEqual(action.function_name, "")

  def test_parse_codeact_skips_non_xml_tool_call_blocks(self):
    # A JSON or plain-text <tool_call> block before an XML one: qwen3_xml
    # returns the XML call.
    xml_call = (
        "<tool_call>\n<function=execute_bash>\n<parameter=command>\nls\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    for first_block in (
        '{"name": "think", "arguments": {"thought": "x"}}',
        "[draft]",
        "{not json",
    ):
      with self.subTest(first_block=first_block):
        response = f"<tool_call>\n{first_block}\n</tool_call>\n{xml_call}"
        _, action = swe_agent.parse_codeact_response(response)
        self.assertEqual(action.function_name, "execute_bash")
        self.assertEqual(action.parameters, {"command": "ls"})

  def test_parse_codeact_completion_trigger(self):
    response = (
        "I have resolved the issue and verified all tests pass.\n"
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    )
    thought, action = swe_agent.parse_codeact_response(response)
    self.assertIn("I have resolved the issue", thought)
    self.assertEqual(action.function_name, "submit")

  def test_codeact_agent_initialization(self):
    agent = swe_agent.CodeActAgent()
    self.assertEqual(agent.scaffold, "openhands")
    self.assertEqual(agent.system_prompt, template.OPENHANDS_SYSTEM_PROMPT)

  def test_codeact_agent_update_from_model(self):
    agent = swe_agent.CodeActAgent()
    agent.update_from_env(observation="Problem statement", reward=0.0, done=False)
    action_res = agent.update_from_model(
        "I will run pytest:\n<tool_call>\n<function=execute_bash>\n"
        "<parameter=command>\npytest\n</parameter>\n</function>\n</tool_call>"
    )
    self.assertIn("<function=execute_bash>", action_res.action)
    self.assertIn("<parameter=command>pytest</parameter>", action_res.action)

  def test_step_openhands_execute_bash(self):
    mock_workspace = mock.MagicMock()
    mock_res = mock.MagicMock(spec=["stdout", "stderr", "exit_code"])
    mock_res.exit_code = 0
    mock_res.stdout = "total 0\n-rw-r--r-- 1 root root 0 test.py"
    mock_res.stderr = ""
    mock_workspace.execute_command.return_value = mock_res

    mock_env = mock.MagicMock()
    mock_env.workspace = mock_workspace
    mock_env.max_steps = 10
    mock_env.step_timeout = 30.0
    mock_env.total_steps = 0

    action = SWEAction("execute_bash", {"command": "ls -l"})
    result = openhands_utils.step_openhands(mock_env, action)
    self.assertFalse(result.done)
    self.assertIn("test.py", result.observation)
    self.assertEqual(mock_env.total_steps, 1)
    mock_workspace.execute_command.assert_called_once()

  def test_step_openhands_execute_bash_cmd_param(self):
    mock_workspace = mock.MagicMock()
    mock_res = mock.MagicMock(spec=["stdout", "stderr", "exit_code"])
    mock_res.exit_code = 0
    mock_res.stdout = "/workspace\n"
    mock_res.stderr = ""
    mock_workspace.execute_command.return_value = mock_res

    mock_env = mock.MagicMock()
    mock_env.workspace = mock_workspace
    mock_env.max_steps = 10
    mock_env.step_timeout = 30.0
    mock_env.total_steps = 0

    action = SWEAction("execute_bash", {"cmd": "pwd"})
    result = openhands_utils.step_openhands(mock_env, action)
    self.assertFalse(result.done)
    self.assertEqual(
        result.observation,
        "/workspace\n[The command completed with exit code 0.]\n[Command"
        " finished with exit code 0]",
    )
    self.assertEqual(mock_env.total_steps, 1)

  def test_step_openhands_execute_bash_missing_command(self):
    mock_env = mock.MagicMock()
    mock_env.max_steps = 10
    action = SWEAction("execute_bash", {})
    result = openhands_utils.step_openhands(mock_env, action)
    self.assertFalse(result.done)
    self.assertEqual(
        result.observation,
        "ERROR: No previous running command to retrieve logs from.",
    )

  def test_step_openhands_execute_ipython_cell(self):
    mock_workspace = mock.MagicMock()
    mock_res = mock.MagicMock(spec=["stdout", "stderr", "exit_code"])
    mock_res.exit_code = 0
    mock_res.stdout = "42\n"
    mock_res.stderr = ""
    mock_workspace.execute_command.return_value = mock_res

    mock_env = mock.MagicMock()
    mock_env.workspace = mock_workspace
    mock_env.max_steps = 10
    mock_env.step_timeout = 30.0
    mock_env.total_steps = 0

    action = SWEAction("execute_ipython_cell", {"code": "print(42)"})
    result = openhands_utils.step_openhands(mock_env, action)
    self.assertFalse(result.done)
    self.assertEqual(result.observation, "42\n")
    self.assertEqual(mock_env.total_steps, 1)
    # Verify that base64 encoded python was executed
    cmd_arg = mock_workspace.execute_command.call_args[0][0]
    self.assertIn("python3 -c", cmd_arg)
    self.assertIn("base64", cmd_arg)

  def test_step_openhands_execute_ipython_cell_output_attr(self):
    mock_workspace = mock.MagicMock()
    mock_res = mock.MagicMock(spec=["output"])
    mock_res.output = "output_from_res"
    mock_workspace.execute_command.return_value = mock_res

    mock_env = mock.MagicMock()
    mock_env.workspace = mock_workspace
    mock_env.max_steps = 10
    mock_env.step_timeout = 30.0
    mock_env.total_steps = 0

    action = SWEAction("execute_ipython_cell", {"code": "print(1)"})
    result = openhands_utils.step_openhands(mock_env, action)
    self.assertFalse(result.done)
    self.assertEqual(result.observation, "output_from_res")
    self.assertEqual(mock_env.total_steps, 1)

  def test_step_openhands_execute_ipython_cell_missing_code(self):
    mock_env = mock.MagicMock()
    mock_env.max_steps = 10
    action = SWEAction("execute_ipython_cell", {})
    result = openhands_utils.step_openhands(mock_env, action)
    self.assertFalse(result.done)
    self.assertEqual(
        result.observation,
        "ERROR: No code specified for execute_ipython_cell.",
    )

  def test_step_openhands_submit(self):
    mock_env = mock.MagicMock()
    mock_env.max_steps = 10
    action = SWEAction("submit", {})
    result = openhands_utils.step_openhands(mock_env, action)
    self.assertTrue(result.done)
    self.assertEqual(result.observation, "Task submitted.")

  def test_step_openhands_execute_bash_local_env(self):
    mock_local_env = mock.MagicMock()
    mock_local_env.step.return_value = ("total 0", 0.0, False, {})

    mock_env = mock.MagicMock()
    mock_env.workspace = None
    mock_env.env = mock_local_env
    mock_env.max_steps = 10
    mock_env.step_timeout = 30.0
    mock_env.total_steps = 0

    action = SWEAction("execute_bash", {"command": "ls -l"})
    result = openhands_utils.step_openhands(mock_env, action)
    self.assertFalse(result.done)
    self.assertEqual(result.observation, "total 0")
    self.assertEqual(mock_env.total_steps, 1)
    mock_local_env.step.assert_called_once()

  def test_step_openhands_execute_ipython_local_env(self):
    mock_local_env = mock.MagicMock()
    mock_local_env.step.return_value = ("output_val", 0.0, False, {})

    mock_env = mock.MagicMock()
    mock_env.workspace = None
    mock_env.env = mock_local_env
    mock_env.max_steps = 10
    mock_env.step_timeout = 30.0
    mock_env.total_steps = 0

    action = SWEAction("execute_ipython_cell", {"code": "print('hi')"})
    result = openhands_utils.step_openhands(mock_env, action)
    self.assertFalse(result.done)
    self.assertEqual(result.observation, "output_val")
    self.assertEqual(mock_env.total_steps, 1)
    mock_local_env.step.assert_called_once()

  def test_codeact_agent_no_synthetic_token_or_step_warning(self):
    agent = swe_agent.CodeActAgent()
    agent.update_from_env(
        observation="Fix the bug in foo.py",
        reward=0.0,
        done=False,
        info={
            "cur_tokens": 30000,
            "max_steps": 50,
            "base_commit": "abc1234",
            "workspace_path": "/workspace",
            "repo_language": "python",
        },
    )
    self.assertNotIn("You are running out of tokens", agent.cur_step.observation)
    self.assertNotIn("Steps Remaining", agent.cur_step.observation)
    self.assertIn("<uploaded_files>\n/workspace\n</uploaded_files>", agent.cur_step.observation)
    self.assertIn(
        "compare your changes with the base commit abc1234.",
        agent.cur_step.observation,
    )
    self.assertEqual(agent.chat_completions[1]["role"], "user")

  def test_parse_codeact_qwen35_xml_tool_call_and_preserve_indentation(self):
    response = (
        "<think>\nLet's replace the indented block in foo.py.\n</think>\n\n"
        "<tool_call>\n"
        "<function=str_replace_editor>\n"
        "<parameter=command>\nstr_replace\n</parameter>\n"
        "<parameter=path>\n/workspace/foo.py\n</parameter>\n"
        "<parameter=old_str>\n    if x:\n        return 1\n</parameter>\n"
        "<parameter=new_str>\n    if x:\n        return 2\n</parameter>\n"
        "<parameter=security_risk>\nLOW\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    thought, action = swe_agent.parse_codeact_response(response)
    self.assertIn("</think>", thought)
    self.assertEqual(action.function_name, "str_replace_editor")
    self.assertEqual(action.parameters["command"], "str_replace")
    self.assertEqual(action.parameters["path"], "/workspace/foo.py")
    self.assertEqual(action.parameters["old_str"], "    if x:\n        return 1")
    self.assertEqual(action.parameters["new_str"], "    if x:\n        return 2")
    self.assertEqual(action.parameters["security_risk"], "LOW")

    reparsed = openhands_utils.parse_openhands_action_str(action.to_xml_string())
    self.assertEqual(reparsed.function_name, "str_replace_editor")
    self.assertEqual(reparsed.parameters["old_str"], "    if x:\n        return 1")
    self.assertEqual(reparsed.parameters["new_str"], "    if x:\n        return 2")

  def test_codeact_agent_tool_role_and_fake_user_response(self):
    agent = swe_agent.CodeActAgent()
    agent.update_from_env(
        observation="Fix issue",
        reward=0.0,
        done=False,
        info={"base_commit": "deadbeef"},
    )
    self.assertEqual(agent.chat_completions[1]["role"], "user")

    # Turn 1: model emits a tool call -> next observation has role="tool"
    agent.update_from_model(
        "<think>\nCheck status\n</think>\n\n"
        "<tool_call>\n<function=execute_bash>\n"
        "<parameter=command>\ngit status\n</parameter>\n"
        "<parameter=security_risk>\nLOW\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    agent.update_from_env(
        observation="On branch main",
        reward=0.0,
        done=False,
        info={},
    )
    self.assertEqual(agent.chat_completions[-1]["role"], "tool")
    self.assertEqual(agent.chat_completions[-1]["content"], "On branch main")

    # Turn 2: model emits no tool call -> next observation uses fake user response with role="user"
    agent.update_from_model("<think>\nI am thinking without calling a tool.\n</think>\nJust text.")
    agent.update_from_env(
        observation="",
        reward=0.0,
        done=False,
        info={},
    )
    self.assertEqual(agent.chat_completions[-1]["role"], "user")
    self.assertEqual(
        agent.chat_completions[-1]["content"],
        template.OPENHANDS_FAKE_USER_RESPONSE,
    )

  def test_parse_codeact_all_returns_one_call_per_tool_call_block(self):
    response = (
        "<think>\nLook first.\n</think>\n\nChecking.\n"
        "<tool_call>\n<function=execute_bash>\n"
        "<parameter=command>\nls\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n</tool_call>\n"
        '<tool_call>\n{"name": "think", "arguments": {}}\n</tool_call>\n'
        "<tool_call>\n<function=str_replace_editor>\n"
        "<parameter=command>\nview\n</parameter>\n"
        "<parameter=path>\n/testbed\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    thought, actions = swe_agent.parse_codeact_response_all(response)
    first_thought, first = swe_agent.parse_codeact_response(response)
    self.assertEqual(thought, first_thought)
    self.assertEqual(actions[0].to_xml_string(), first.to_xml_string())
    self.assertEqual(
        [(a.function_name, a.parameters) for a in actions],
        [
            ("execute_bash", {"command": "ls"}),
            ("str_replace_editor", {"command": "view", "path": "/testbed"}),
        ],
    )

  def test_parse_codeact_all_matches_qwen3_xml_call_boundaries(self):
    # Expected call lists are vLLM 0.20.0 Qwen3XMLToolParser.extract_tool_calls
    # output for the same text (after "</think>").
    def fn(value):
      return (
          f"<function=think>\n<parameter=thought>\n{value}\n</parameter>\n"
          "</function>"
      )

    def tc(body):
      return f"<tool_call>\n{body}\n</tool_call>"

    cases = {
        # A second <function=> inside one <tool_call> stays in that one call.
        # (qwen3_xml renames the call to the second function and concatenates
        # both argument JSONs, which OpenHands then rejects as invalid.)
        "tc(a + b)": (tc(fn("a") + "\n" + fn("b")), ["a"]),
        "tc a, text, tc b": (tc(fn("a")) + "\nmid\n" + tc(fn("b")), ["a", "b"]),
        "tc a, tc empty, tc b": (
            tc(fn("a")) + "\n<tool_call>\n</tool_call>\n" + tc(fn("b")),
            ["a", "b"],
        ),
        # A bare <function=> after a closed <tool_call> is a new call...
        "tc a, bare b": (tc(fn("a")) + "\n" + fn("b"), ["a", "b"]),
        "tc a, tc empty, bare b": (
            tc(fn("a")) + "\n<tool_call>\n</tool_call>\n" + fn("b"),
            ["a", "b"],
        ),
        # ...but one after another bare <function=> is dropped.
        "tc a, bare b, bare c": (
            tc(fn("a")) + "\n" + fn("b") + "\n" + fn("c"),
            ["a", "b"],
        ),
        "bare a, bare b": (fn("a") + "\n" + fn("b"), ["a"]),
        "bare a, tc b, bare c": (
            fn("a") + "\n" + tc(fn("b")) + "\n" + fn("c"),
            ["a", "b", "c"],
        ),
        "three tc": (tc(fn("a")) + tc(fn("b")) + tc(fn("c")), ["a", "b", "c"]),
        # An unclosed <tool_call> ends at the next <tool_call>.
        "tc a unclosed, tc b": (
            "<tool_call>\n" + fn("a") + "\n" + tc(fn("b")),
            ["a", "b"],
        ),
        "tc a, truncated opener": (tc(fn("a")) + "\n<function=thi", ["a"]),
    }
    for name, (text, expected) in cases.items():
      with self.subTest(name):
        _, actions = swe_agent.parse_codeact_response_all(
            "<think>\nx\n</think>\n" + text
        )
        self.assertEqual([a.parameters["thought"] for a in actions], expected)

  def test_parse_codeact_all_after_think(self):
    two_calls = (
        "<tool_call>\n<function=think>\n<parameter=thought>\na\n"
        "</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=think>\n<parameter=thought>\nb\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    for response, expected in (
        # Thinking disabled: no </think>, all text is content.
        (two_calls, ["a", "b"]),
        ("<think>\nplan\n</think>\n" + two_calls, ["a", "b"]),
        # Calls inside the reasoning are not parsed by the reference.
        ("<think>\n" + two_calls + "\n</think>\nDone.", ["a"]),
    ):
      with self.subTest(response=response):
        _, actions = swe_agent.parse_codeact_response_all(response)
        self.assertEqual([a.parameters["thought"] for a in actions], expected)

  def test_parse_codeact_all_without_tool_call(self):
    for response, name in (
        ("<think>\nhmm\n</think>\nJust text.", ""),
        ("Done. COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "submit"),
    ):
      with self.subTest(response=response):
        _, actions = swe_agent.parse_codeact_response_all(response)
        self.assertLen(actions, 1)
        self.assertEqual(actions[0].function_name, name)

  def test_codeact_agent_multi_tool_calls_flag(self):
    response = (
        "<think>\nplan\n</think>\n"
        "<tool_call>\n<function=think>\n<parameter=thought>\na\n"
        "</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=execute_bash>\n<parameter=command>\nls\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    calls = [
        str(SWEAction("think", {"thought": "a"})),
        str(SWEAction("execute_bash", {"command": "ls"})),
    ]
    for value, expected in (("", calls[0]), ("false", calls[0])):
      with self.subTest(flag=value), mock.patch.dict(
          os.environ, {"OPENHANDS_MULTI_TOOL_CALLS": value}
      ):
        agent = swe_agent.CodeActAgent()
        agent.update_from_env("Fix issue", 0.0, False, {})
        self.assertEqual(agent.update_from_model(response).action, expected)
        self.assertEqual(agent.trajectory.steps[-1].action, expected)

    with mock.patch.dict(os.environ, {"OPENHANDS_MULTI_TOOL_CALLS": "true"}):
      agent = swe_agent.CodeActAgent()
    agent.update_from_env("Fix issue", 0.0, False, {})
    self.assertEqual(agent.update_from_model(response).action, calls)
    self.assertEqual(agent.trajectory.steps[-1].action, "\n".join(calls))
    self.assertEqual(agent.chat_completions[-1]["content"], response)
    # Turns with one call keep returning a string.
    agent.update_from_env("ok", 0.0, False, {})
    single = response.split("\n<tool_call>\n<function=execute_bash>")[0]
    self.assertEqual(agent.update_from_model(single).action, calls[0])

  def test_codeact_agent_list_observation_becomes_tool_messages(self):
    with mock.patch.dict(os.environ, {"OPENHANDS_MULTI_TOOL_CALLS": "true"}):
      agent = swe_agent.CodeActAgent()
    agent.update_from_env("Fix issue", 0.0, False, {})
    agent.update_from_model(
        "<think>\nplan\n</think>\n"
        "<tool_call>\n<function=think>\n<parameter=thought>\na\n"
        "</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=think>\n<parameter=thought>\nb\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    n = len(agent.chat_completions)
    agent.update_from_env(["logged a", "logged b"], 0.0, False, {})
    self.assertEqual(
        agent.chat_completions[n:],
        [
            {"role": "tool", "content": "logged a"},
            {"role": "tool", "content": "logged b"},
        ],
    )
    self.assertEqual(agent.trajectory.steps[-1].observation, ["logged a", "logged b"])

  # r2egym is not installed in tests; SWEEnv imports its Action lazily.
  @mock.patch.object(swe_env, "Action", SWEAction)
  def test_swe_env_runs_multi_call_turn_in_order_and_counts_turns(self):
    think = "<function=think>\n<parameter=thought>{}</parameter>\n</function>"
    finish = "<function=finish>\n<parameter=message>done</parameter>\n</function>"

    env = swe_env.SWEEnv(entry={"instance_id": "t1"}, scaffold="openhands", max_steps=30)
    obs, _, done, info = env.step([think.format("a"), think.format("b")])
    self.assertEqual(obs, ["Your thought has been logged."] * 2)
    self.assertFalse(done)
    self.assertEqual(info["num_tool_calls"], 2)
    self.assertEqual(env.step_count, 2)  # Each call is one of max_steps.

    # finish ends the episode; calls after it never run.
    env = swe_env.SWEEnv(entry={"instance_id": "t1"}, scaffold="openhands", max_steps=30)
    obs, _, done, info = env.step([finish, think.format("late")])
    self.assertEqual(obs, "done")
    self.assertTrue(done)
    self.assertEqual(info["num_tool_calls"], 1)

    # At turn 29 of 30, a two-call turn runs both and ends the episode; at
    # turn 30, only the first call runs.
    for prior, expected_obs in ((28, ["Your thought has been logged."] * 2),
                                (29, "Your thought has been logged.")):
      with self.subTest(prior=prior):
        env = swe_env.SWEEnv(entry={"instance_id": "t1"}, scaffold="openhands", max_steps=30)
        env.step_count = prior
        obs, _, done, _ = env.step([think.format("a"), think.format("b")])
        self.assertEqual(obs, expected_obs)
        self.assertTrue(done)
        self.assertEqual(env.step_count, 30)

    # A single call (string) is unchanged.
    env = swe_env.SWEEnv(entry={"instance_id": "t1"}, scaffold="openhands", max_steps=30)
    obs, _, done, info = env.step(think.format("a"))
    self.assertEqual(obs, "Your thought has been logged.")
    self.assertNotIn("num_tool_calls", info)
    self.assertEqual(env.step_count, 1)

  def test_step_openhands_think_task_tracker_and_finish(self):
    mock_env = mock.MagicMock()
    mock_env.max_steps = 10
    mock_env.total_steps = 0

    think_res = openhands_utils.step_openhands(
        mock_env, SWEAction("think", {"thought": "Analyzing root cause"})
    )
    self.assertFalse(think_res.done)
    self.assertEqual(think_res.observation, "Your thought has been logged.")

    view_empty_res = openhands_utils.step_openhands(
        mock_env, SWEAction("task_tracker", {"command": "view"})
    )
    self.assertFalse(view_empty_res.done)
    self.assertIn("No task list found", view_empty_res.observation)

    plan_res = openhands_utils.step_openhands(
        mock_env,
        SWEAction(
            "task_tracker",
            {
                "command": "plan",
                "task_list": '[{"id": "1", "title": "Fix bug", "status": "in_progress"}]',
            },
        ),
    )
    self.assertFalse(plan_res.done)
    self.assertRegex(
        plan_res.observation,
        r"^Task list has been updated with 1 items\. Stored in session"
        r" directory: sessions/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{2}-"
        r"[0-9a-f]{15}/TASKS\.md$",
    )

    view_after_plan_res = openhands_utils.step_openhands(
        mock_env, SWEAction("task_tracker", {"command": "view"})
    )
    self.assertIn("Fix bug", view_after_plan_res.observation)

    finish_res = openhands_utils.step_openhands(
        mock_env,
        SWEAction("finish", {"message": "Fixed the issue and verified tests."}),
    )
    self.assertTrue(finish_res.done)
    self.assertEqual(
        finish_res.observation, "Fixed the issue and verified tests."
    )

  def test_openhands_system_prompt_and_tools_schema(self):
    tool_names = [t["function"]["name"] for t in template.OPENHANDS_TOOLS]
    self.assertEqual(
        tool_names,
        [
            "execute_bash",
            "think",
            "finish",
            "task_tracker",
            "str_replace_editor",
        ],
    )
    self.assertIn("# Tools\n\nYou have access to the following functions:\n\n<tools>", template.OPENHANDS_SYSTEM_PROMPT)
    self.assertIn("<SECURITY_RISK_ASSESSMENT>", template.OPENHANDS_SYSTEM_PROMPT)
    self.assertIn("60 seconds", template.OPENHANDS_SYSTEM_PROMPT)

  def test_sweagent_token_warning(self):
    agent = swe_agent.SWEAgent(scaffold="sweagent")
    agent.update_from_env(
        observation="Init",
        reward=0.0,
        done=False,
        info={"cur_tokens": 30000},
    )
    self.assertIn("You are running out of tokens", agent.cur_step.observation)
    self.assertIn("<function=finish>", agent.cur_step.observation)

  def test_swe_env_init_global_fleet_node_selector(self):
    mock_agent_sandbox_rl = mock.MagicMock()
    mock_fleet_inst = mock.MagicMock()
    mock_agent_sandbox_rl.SandboxFleet.return_value = mock_fleet_inst

    with mock.patch.dict(
        "sys.modules", {"agent_sandbox_rl": mock_agent_sandbox_rl}
    ), mock.patch.dict(
        os.environ,
        {
            "NODE_SELECTOR_KEY": "cloud.google.com/gke-nodepool",
            "NODE_SELECTOR_VAL": "cpu-np",
        },
    ):
      swe_env._GLOBAL_FLEET = None
      try:
        fleet = swe_env._init_global_fleet(tasks=[])
        self.assertEqual(fleet, mock_fleet_inst)
        cluster_cfg_call = mock_agent_sandbox_rl.ClusterConfig.call_args
        self.assertEqual(
            cluster_cfg_call[1]["node_selector"],
            {"cloud.google.com/gke-nodepool": "cpu-np"},
        )
      finally:
        swe_env._GLOBAL_FLEET = None

  

  def test_setup_and_restore_r2e_tests_for_reward(self):
    mock_ws = mock.MagicMock()
    mock_ws.execute_command.return_value = mock.MagicMock(exit_code=0)
    openhands_utils.setup_openhands_workspace(mock_ws, {})
    setup_cmd = mock_ws.execute_command.call_args[0][0]
    self.assertIn("/var/tmp/.r2e_grading_stash", setup_cmd)
    self.assertIn(
        "rm -rf /r2e_tests /root/r2e_tests /testbed/r2e_tests"
        " /run_tests.sh /root/run_tests.sh /testbed/run_tests.sh",
        setup_cmd,
    )

    mock_ws.reset_mock()
    openhands_utils.restore_r2e_tests_for_reward(mock_ws)
    restore_cmd = mock_ws.execute_command.call_args[0][0]
    self.assertIn("/var/tmp/.r2e_grading_stash/run_tests.sh", restore_cmd)
    self.assertIn("/root/run_tests.sh", restore_cmd)
    self.assertIn("ln -s /root/r2e_tests /testbed/r2e_tests", restore_cmd)

    # Also verify non-OpenHands (RepoEnv runtime.run) path
    mock_repo_env = mock.MagicMock(spec=["runtime"])
    mock_repo_env.runtime = mock.MagicMock()
    openhands_utils.hide_r2e_tests_for_rollout(mock_repo_env)
    self.assertIn(
        "/var/tmp/.r2e_grading_stash",
        mock_repo_env.runtime.run.call_args[0][0],
    )
    mock_repo_env.runtime.reset_mock()
    openhands_utils.restore_r2e_tests_for_reward(mock_repo_env)
    self.assertIn(
        "/root/run_tests.sh",
        mock_repo_env.runtime.run.call_args[0][0],
    )

  def test_swe_env_and_agent_backward_compatible_defaults(self):
    env = swe_env.SWEEnv(entry={"instance_id": "t1"})
    self.assertEqual(env.scaffold, "r2egym")
    self.assertEqual(env.step_timeout, 30 * 60)
    self.assertEqual(env.reward_timeout, 30 * 60)

    swe = swe_agent.SWEAgent()
    self.assertEqual(swe.scaffold, "r2egym")

    codeact = swe_agent.CodeActAgent()
    self.assertEqual(codeact.scaffold, "openhands")

  def test_parse_codeact_ignores_code_fence_inside_think_block(self):
    response = (
        "<think>\nLet's consider editing foo.py:\n"
        "```python\nprint('illustrative snippet')\n```\n"
        "</think>\n"
        "Now let's view foo.py:\n"
        "<function=str_replace_editor>\n"
        "<parameter=command>view</parameter>\n"
        "<parameter=path>/testbed/foo.py</parameter>\n"
        "</function>"
    )
    thought, action = swe_agent.parse_codeact_response(response)
    self.assertEqual(action.function_name, "str_replace_editor")
    self.assertEqual(action.parameters["command"], "view")
    self.assertEqual(action.parameters["path"], "/testbed/foo.py")
    self.assertIn("</think>", thought)

  def test_parse_codeact_truncated_xml_and_trajectory_step_action_str(self):
    response = (
        "Let's run pytest.\n"
        "<function=execute_bash>\n"
        "<parameter=command>pytest -q</parameter>"
    )
    agent = swe_agent.CodeActAgent()
    agent.update_from_env("Fix bug", 0.0, False, {"max_steps": 10})
    returned_action = agent.update_from_model(response)
    self.assertIsInstance(agent.trajectory.steps[-1].action, str)
    self.assertIn("execute_bash", agent.trajectory.steps[-1].action)
    self.assertEqual(returned_action.action, agent.trajectory.steps[-1].action)

  def test_swe_env_close_calls_both_close_and_cleanup(self):
    env = swe_env.SWEEnv(entry={"instance_id": "t1"})
    mock_ws = mock.MagicMock()
    env.workspace = mock_ws
    env.close()
    mock_ws.close.assert_called_once()
    mock_ws.cleanup.assert_called_once()
    self.assertIsNone(env.workspace)

  def test_deepswe_dist_agent_registration_and_openhands_dispatch(self):
    from tunix.experimental.examples.deepswe_dist import deepswe  # pylint: disable=g-import-not-at-top
    from tunix.experimental.rl.agentic import registry  # pylint: disable=g-import-not-at-top

    self.assertEqual(deepswe.get_agent_name("openhands"), "codeact_agent")
    self.assertEqual(deepswe.get_agent_name("r2egym"), "deepswe_agent")
    self.assertEqual(deepswe.get_agent_name("sweagent"), "deepswe_agent")

    codeact_cls = registry.AGENT_REGISTRY.get(deepswe.CODEACT_AGENT_NAME)
    codeact_agent = codeact_cls(scaffold="openhands")
    self.assertIsInstance(codeact_agent, swe_agent.CodeActAgent)
    self.assertEqual(codeact_agent.name, "codeact_agent")
    self.assertEqual(codeact_agent.scaffold, "openhands")

    # Even if a caller requests deepswe_agent with scaffold="openhands",
    # DeepSWEAgent.__new__ dispatches to DeepSWECodeActAgent.
    deepswe_cls = registry.AGENT_REGISTRY.get(deepswe.DEEPSWE_AGENT_NAME)
    dispatched_agent = deepswe_cls(scaffold="openhands")
    self.assertIsInstance(dispatched_agent, swe_agent.CodeActAgent)
    self.assertEqual(dispatched_agent.name, "codeact_agent")

    r2e_agent = deepswe_cls(scaffold="r2egym")
    self.assertIsInstance(r2e_agent, swe_agent.SWEAgent)
    self.assertNotIsInstance(r2e_agent, swe_agent.CodeActAgent)
    self.assertEqual(r2e_agent.name, "deepswe_agent")

    # scaffold passed positionally (4th arg) dispatches the same way.
    positional_agent = deepswe_cls(None, False, False, "openhands")
    self.assertIsInstance(positional_agent, swe_agent.CodeActAgent)
    self.assertEqual(positional_agent.scaffold, "openhands")

    prompt_item = deepswe.build_prompt_item(
        entry={"instance_id": "inst_1", "problem_statement": "fix it"},
        prompt_idx=0,
        max_turns=10,
        max_response_length=1024,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        step_timeout_secs=60,
        reward_timeout_secs=120,
        env_backend="kubernetes",
        use_agent_sandbox=True,
        scaffold="openhands",
        env_verbose=False,
    )
    self.assertEqual(prompt_item["metadata"]["agent_name"], "codeact_agent")
    self.assertEqual(
        prompt_item["metadata"]["agent_config"], {"scaffold": "openhands"}
    )

  def test_oh_editor_create_view_str_replace_insert_undo_and_clip(self):
    import tempfile  # pylint: disable=g-import-not-at-top

    with tempfile.TemporaryDirectory() as tmpdir:
      hist_file = os.path.join(tmpdir, ".oh_editor_history.json")
      file_path = os.path.join(tmpdir, "sample.py")

      # 1. create
      create_out = openhands_utils.run_oh_editor_locally(
          {
              "command": "create",
              "path": file_path,
              "file_text": "def foo():\n    return 1\n",
          },
          history_file=hist_file,
      )
      self.assertIn("File created successfully", create_out)

      # 2. view
      view_out = openhands_utils.run_oh_editor_locally(
          {"command": "view", "path": file_path},
          history_file=hist_file,
      )
      self.assertIn("Here's the result of running `cat -n`", view_out)
      self.assertIn("1\tdef foo():", view_out)
      self.assertIn("2\t    return 1", view_out)

      # 3. str_replace
      replace_out = openhands_utils.run_oh_editor_locally(
          {
              "command": "str_replace",
              "path": file_path,
              "old_str": "    return 1",
              "new_str": "    return 2",
          },
          history_file=hist_file,
      )
      self.assertIn("has been edited", replace_out)
      self.assertIn("return 2", replace_out)

      # 4. insert
      insert_out = openhands_utils.run_oh_editor_locally(
          {
              "command": "insert",
              "path": file_path,
              "insert_line": 1,
              "new_str": "    # comment",
          },
          history_file=hist_file,
      )
      self.assertIn("has been edited", insert_out)
      self.assertIn("# comment", insert_out)

      # 5. undo_edit (reverts insert)
      undo_out = openhands_utils.run_oh_editor_locally(
          {"command": "undo_edit", "path": file_path},
          history_file=hist_file,
      )
      self.assertIn("Last edit", undo_out)
      self.assertNotIn("# comment", undo_out)
      self.assertIn("return 2", undo_out)

      # 6. no line cap; long files are clipped by characters, before the
      # line numbers are added, with the reference notice.
      long_path = os.path.join(tmpdir, "long.py")
      with open(long_path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"line_{i}" for i in range(1, 701)))
      long_view = openhands_utils.run_oh_editor_locally(
          {"command": "view", "path": long_path},
          history_file=hist_file,
      )
      self.assertIn("   700\tline_700", long_view)
      self.assertNotIn("<response clipped>", long_view)
      huge_path = os.path.join(tmpdir, "huge.py")
      with open(huge_path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"value_{i} = {'x' * 40}" for i in range(1, 2001)))
      huge_view = openhands_utils.run_oh_editor_locally(
          {"command": "view", "path": huge_path},
          history_file=hist_file,
      )
      self.assertTrue(
          huge_view.endswith(openhands_utils.FILE_CLIPPED_NOTICE + "\n")
      )
      self.assertIn("`grep -n`", huge_view)
      binary_path = os.path.join(tmpdir, "blob.bin")
      with open(binary_path, "wb") as f:
        f.write(b"abc\x00def\n")
      binary_view = openhands_utils.run_oh_editor_locally(
          {"command": "view", "path": binary_path},
          history_file=hist_file,
      )
      self.assertEqual(
          binary_view,
          "ERROR_BINARY_FILE\n[Error occurred in processing last action]",
      )
      binary_edit = openhands_utils.run_oh_editor_locally(
          {
              "command": "str_replace",
              "path": binary_path,
              "old_str": "abc",
              "new_str": "x",
          },
          history_file=hist_file,
      )
      self.assertEqual(
          binary_edit,
          f"ERROR:\nFile validation failed for {binary_path}: File appears"
          " to be binary and this file type cannot be read or edited by this"
          " tool.",
      )
      pyc_path = os.path.join(tmpdir, "mod.pyc")
      with open(pyc_path, "w", encoding="utf-8") as f:
        f.write("text\n")
      self.assertEqual(
          openhands_utils.run_oh_editor_locally(
              {"command": "view", "path": pyc_path}, history_file=hist_file
          ),
          "ERROR_BINARY_FILE\n[Error occurred in processing last action]",
      )

      # 7. remote command serialization round-trip
      import subprocess  # pylint: disable=g-import-not-at-top

      remote_cmd = openhands_utils._build_oh_editor_remote_cmd(
          {"command": "view", "path": file_path, "view_range": [1, 2]}
      )
      proc = subprocess.run(
          remote_cmd, shell=True, capture_output=True, text=True, check=True
      )
      self.assertIn("return 2", proc.stdout)

  def test_step_openhands_bash_persistent_cwd_and_timeout(self):
    mock_workspace = mock.MagicMock()
    mock_res = mock.MagicMock(
        spec=["stdout", "stderr", "exit_code", "timeout_occurred"]
    )
    mock_res.exit_code = -1
    mock_res.stdout = "collected 10 items\n"
    mock_res.stderr = "Command timed out after 60.0 seconds"
    mock_res.timeout_occurred = True
    mock_workspace.execute_command.return_value = mock_res

    mock_env = mock.MagicMock()
    mock_env.workspace = mock_workspace
    mock_env.max_steps = 10
    mock_env.step_timeout = 60.0
    mock_env.total_steps = 0

    action = SWEAction(
        "execute_bash",
        {"command": "pytest", "timeout": "45", "security_risk": "LOW"},
    )
    result = openhands_utils.step_openhands(mock_env, action)
    self.assertFalse(result.done)
    self.assertTrue(result.info.get("command_timed_out"))
    self.assertEqual(
        result.observation,
        "collected 10 items\n[The command timed out after 45.0 seconds. You"
        " may wait longer to see additional output by sending empty command"
        " '', send other commands to interact with the current process, send"
        ' keys ("C-c", "C-z", "C-d") to interrupt/kill the previous command'
        " before sending your new command, or use the timeout parameter in"
        " execute_bash for future commands.]",
    )
    called_cmd = mock_workspace.execute_command.call_args[0][0]
    self.assertIn("/var/tmp/.oh_cwd", called_cmd)
    self.assertEqual(
        mock_workspace.execute_command.call_args.kwargs["timeout"], 45.0
    )

  def test_step_openhands_bash_wrapper_keeps_heredoc_and_comment_valid(self):
    import subprocess  # pylint: disable=g-import-not-at-top
    import tempfile  # pylint: disable=g-import-not-at-top

    mock_workspace = mock.MagicMock()
    mock_res = mock.MagicMock(spec=["stdout", "stderr", "exit_code"])
    mock_res.exit_code = 0
    mock_res.stdout = ""
    mock_res.stderr = ""
    mock_workspace.execute_command.return_value = mock_res

    mock_env = mock.MagicMock()
    mock_env.workspace = mock_workspace
    mock_env.max_steps = 10
    mock_env.step_timeout = 60.0
    mock_env.total_steps = 0

    with tempfile.TemporaryDirectory() as tmpdir:
      tmpdir = os.path.realpath(tmpdir)
      cwd_file = os.path.join(tmpdir, ".oh_cwd")
      for command, expected in (
          ("cat <<'EOF'\nhello from heredoc\nEOF", "hello from heredoc"),
          ("cat << 'END'\nline1\nline2\nEND", "line1\nline2"),
          ("echo ok  # trailing comment", "ok"),
      ):
        with open(cwd_file, "w", encoding="utf-8") as f:
          f.write(tmpdir)
        openhands_utils.step_openhands(
            mock_env, SWEAction("execute_bash", {"command": command})
        )
        wrapped = mock_workspace.execute_command.call_args[0][0]
        # Run the wrapped command for real, with its cwd state file moved
        # into the temp dir so nothing is written outside it.
        proc = subprocess.run(
            ["sh", "-c", wrapped.replace("/var/tmp/.oh_cwd", cwd_file)],
            capture_output=True,
            text=True,
        )
        res = mock.MagicMock(spec=["stdout", "stderr", "exit_code"])
        res.stdout, res.stderr, res.exit_code = (
            proc.stdout,
            proc.stderr,
            proc.returncode,
        )
        obs, timed_out = openhands_utils._format_command_result(
            res, 60.0, 0.1, bash_observation=True
        )
        self.assertFalse(timed_out)
        self.assertTrue(
            obs.startswith(
                f"{expected}\n[The command completed with exit code 0.]\n"
                f"[Current working directory: {tmpdir}]\n"
            ),
            msg=f"{command!r}: {obs!r}",
        )
        self.assertTrue(obs.endswith("\n[Command finished with exit code 0]"))

  def test_bash_observation_matches_reference_rendering(self):
    meta = openhands_utils._BASH_META_SENTINEL
    res = mock.MagicMock(spec=["stdout", "stderr", "exit_code"])
    res.stdout = f"  out line\n\n{meta}/testbed\t/testbed/.venv/bin/python\n"
    res.stderr = "warn line\n"
    res.exit_code = 2
    obs, _ = openhands_utils._format_command_result(
        res, 60.0, 0.1, bash_observation=True
    )
    self.assertEqual(
        obs,
        "out line\nwarn line\n[The command completed with exit code 2.]\n"
        "[Current working directory: /testbed]\n"
        "[Python interpreter: /testbed/.venv/bin/python]\n"
        "[Command finished with exit code 2]",
    )
    # Long output keeps the head and tail halves, with the suffix in the tail.
    res.stdout = "a" * 20000 + "b" * 20000 + f"\n{meta}/testbed\t\n"
    res.stderr = ""
    res.exit_code = 0
    obs, _ = openhands_utils._format_command_result(
        res, 60.0, 0.1, bash_observation=True
    )
    self.assertEqual(len(obs), 30047)
    self.assertTrue(
        obs.startswith("a" * 15000 + "\n[... Observation truncated")
    )
    self.assertTrue(obs.endswith("[Command finished with exit code 0]"))
    # Every line is rstripped, like the reference tmux capture.
    res.stdout = f"diff\n \n+x  \n{meta}/testbed\t\n"
    obs, _ = openhands_utils._format_command_result(
        res, 60.0, 0.1, bash_observation=True
    )
    self.assertTrue(obs.startswith("diff\n\n+x\n[The command completed"))

  def test_step_openhands_bash_echoes_commands_the_pane_rewrites(self):
    meta = openhands_utils._BASH_META_SENTINEL
    mock_env = mock.MagicMock()
    mock_env.max_steps = 10
    mock_env.step_timeout = 60.0
    res = mock.MagicMock(spec=["stdout", "stderr", "exit_code"])
    res.stdout = f"1\n2\n{meta}/testbed\t/testbed/.venv/bin/python\n"
    res.stderr = ""
    res.exit_code = 0
    mock_env.workspace.execute_command.return_value = res
    suffix = (
        "[The command completed with exit code 0.]\n"
        "[Current working directory: /testbed]\n"
        "[Python interpreter: /testbed/.venv/bin/python]\n"
        "[Command finished with exit code 0]"
    )
    # An empty line, or a whitespace-only one, keeps the echo; the pane shows
    # the command without empty lines and with each line rstripped.
    result = openhands_utils.step_openhands(
        mock_env,
        SWEAction(
            "execute_bash",
            {"command": 'python3 -c "\nprint(1)\n\nprint(2)\n  \n"'},
        ),
    )
    self.assertEqual(
        result.observation,
        'python3 -c "\nprint(1)\nprint(2)\n\n"\n1\n2\n' + suffix,
    )
    # A command the pane shows verbatim is not echoed.
    result = openhands_utils.step_openhands(
        mock_env,
        SWEAction("execute_bash", {"command": 'python3 -c "\nprint(1)\n"'}),
    )
    self.assertEqual(result.observation, "1\n2\n" + suffix)

  def test_step_openhands_editor_view_is_not_truncated(self):
    mock_env = mock.MagicMock()
    mock_env.max_steps = 10
    mock_env.step_timeout = 60.0
    res = mock.MagicMock(spec=["stdout", "stderr", "exit_code"])
    res.stdout = "v" * 40000
    res.stderr = ""
    res.exit_code = 0
    mock_env.workspace.execute_command.return_value = res
    view = openhands_utils.step_openhands(
        mock_env,
        SWEAction("str_replace_editor", {"command": "view", "path": "/a"}),
    )
    self.assertEqual(view.observation, "v" * 40000)
    edit = openhands_utils.step_openhands(
        mock_env,
        SWEAction(
            "str_replace_editor",
            {"command": "str_replace", "path": "/a", "old_str": "x",
             "new_str": "y"},
        ),
    )
    self.assertEqual(len(edit.observation), 30047)

  def test_step_openhands_bash_send_keys_without_running_command(self):
    mock_env = mock.MagicMock()
    mock_env.max_steps = 10
    result = openhands_utils.step_openhands(
        mock_env,
        SWEAction("execute_bash", {"command": "C-c", "is_input": "true"}),
    )
    self.assertEqual(
        result.observation,
        "ERROR: No previous running command to interact with.",
    )
    mock_env.workspace.execute_command.assert_not_called()

  def test_wrapper_strips_pyinstaller_library_path(self):
    import subprocess  # pylint: disable=g-import-not-at-top

    probe = (
        openhands_utils._STRIP_PYINSTALLER_LD_PATH
        + 'echo "${LD_LIBRARY_PATH-unset}"'
    )
    for value, expected in (
        ("/tmp/_MEIabc:/opt/a:/opt/b", "/opt/a:/opt/b"),
        ("/tmp/_MEIabc", "unset"),
        ("/tmp/_MEIabc:", "unset"),
        ("/oh/glibc236", "unset"),
        ("/oh/glibc236:", "unset"),
        ("/tmp/_MEIabc:/oh/glibc236", "unset"),
        ("/tmp/_MEIabc:/oh/glibc236:", "unset"),
        ("/tmp/_MEIabc:/oh/glibc236:/opt/a:", "/opt/a"),
        ("/tmp/_MEIx::/usr/lib", "/usr/lib"),
        (":/usr/lib", "/usr/lib"),
        ("", "unset"),
        ("/usr/lib", "/usr/lib"),
    ):
      out = subprocess.run(
          ["sh", "-c", probe],
          capture_output=True,
          text=True,
          env={"PATH": os.environ.get("PATH", ""), "LD_LIBRARY_PATH": value},
      ).stdout.strip()
      self.assertEqual(out, expected, msg=value)

    out_unset = subprocess.run(
        ["sh", "-c", probe],
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", "")},
    ).stdout.strip()
    self.assertEqual(out_unset, "unset")

  def test_oh_editor_remote_driver_postpones_annotations(self):
    import __future__  # pylint: disable=g-import-not-at-top
    import ast  # pylint: disable=g-import-not-at-top
    import base64  # pylint: disable=g-import-not-at-top
    import re  # pylint: disable=g-import-not-at-top
    import shlex  # pylint: disable=g-import-not-at-top
    import subprocess  # pylint: disable=g-import-not-at-top
    import sys  # pylint: disable=g-import-not-at-top
    import tempfile  # pylint: disable=g-import-not-at-top

    with tempfile.TemporaryDirectory() as tmpdir:
      file_path = os.path.join(tmpdir, "sample.py")
      with open(file_path, "w", encoding="utf-8") as f:
        f.write("def foo():\n    return 1\n")
      cmd = openhands_utils._build_oh_editor_remote_cmd(
          {"command": "view", "path": file_path}
      )
      self.assertTrue(
          cmd.startswith(openhands_utils._STRIP_PYINSTALLER_LD_PATH)
      )
      argv = shlex.split(cmd[len(openhands_utils._STRIP_PYINSTALLER_LD_PATH) :])
      self.assertEqual(argv[0], "python3")
      driver = base64.b64decode(
          re.search(r"b64decode\('([^']+)'\)", cmd).group(1)
      ).decode("utf-8")
      # Sandboxes run python 3.7/3.8, where `dict[str, Any]` annotations fail
      # unless their evaluation is postponed.
      self.assertTrue(
          driver.startswith("from __future__ import annotations\n")
      )
      code = compile(driver, "<driver>", "exec", dont_inherit=True)
      self.assertTrue(code.co_flags & __future__.annotations.compiler_flag)
      ast.parse(driver, feature_version=(3, 7))

      # Run the shipped driver with its history file moved into the temp dir.
      driver = driver.replace(
          "/var/tmp/.oh_editor_history.json",
          os.path.join(tmpdir, ".oh_editor_history.json"),
      )
      out = subprocess.run(
          [sys.executable, "-c", driver, argv[-1]],
          capture_output=True,
          text=True,
      )
      self.assertEqual(out.returncode, 0, msg=out.stderr)
      self.assertIn("cat -n", out.stdout)
      self.assertIn("def foo():", out.stdout)

  def test_openhands_editor_path_description_matches_reference(self):
    tools = {
        t["function"]["name"]: t["function"] for t in template.OPENHANDS_TOOLS
    }
    path_param = tools["str_replace_editor"]["parameters"]["properties"]["path"]
    self.assertEqual(
        path_param["description"],
        "Absolute path to file or directory, e.g."
        " `/openhands_setup/OpenHands/file.py` or"
        " `/openhands_setup/OpenHands`.",
    )

  def test_execute_ipython_cell_uses_testbed_cwd_and_venv_python(self):
    import subprocess  # pylint: disable=g-import-not-at-top
    import tempfile  # pylint: disable=g-import-not-at-top

    mock_workspace = mock.MagicMock()
    mock_res = mock.MagicMock(spec=["stdout", "stderr", "exit_code"])
    mock_res.exit_code = 0
    mock_res.stdout = "ok\n"
    mock_res.stderr = ""
    mock_workspace.execute_command.return_value = mock_res

    mock_env = mock.MagicMock()
    mock_env.workspace = mock_workspace
    mock_env.max_steps = 10
    mock_env.step_timeout = 60.0
    mock_env.total_steps = 0

    with tempfile.TemporaryDirectory() as tmpdir:
      fake_testbed = os.path.join(tmpdir, "testbed")
      os.makedirs(fake_testbed)
      openhands_utils.step_openhands(
          mock_env,
          SWEAction(
              "execute_ipython_cell",
              {"code": "import os; print(os.getcwd())"},
          ),
      )
      wrapped = mock_workspace.execute_command.call_args[0][0]
      self.assertTrue(
          wrapped.startswith(openhands_utils._STRIP_PYINSTALLER_LD_PATH)
      )
      # Execute the wrapped command pointing /testbed to fake_testbed and verify
      # the cd persists into the Python process.
      proc = subprocess.run(
          ["sh", "-c", wrapped.replace("/testbed", fake_testbed)],
          capture_output=True,
          text=True,
          check=True,
      )
      self.assertEqual(proc.stdout.strip(), os.path.realpath(fake_testbed))

  def test_remove_binary_diffs_and_extract_agent_patch_real_git_repo(self):
    import subprocess  # pylint: disable=g-import-not-at-top
    import tempfile  # pylint: disable=g-import-not-at-top

    raw_patch = (
        "diff --git a/foo.py b/foo.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
        "diff --git a/bin.dat b/bin.dat\n"
        "new file mode 100644\n"
        "Binary files /dev/null and b/bin.dat differ\n"
        "diff --git a/bar.py b/bar.py\n"
        "index 3333333..4444444 100644\n"
        "--- a/bar.py\n"
        "+++ b/bar.py\n"
        "@@ -1 +1 @@\n"
        "-y = 1\n"
        "+y = 2\n"
    )
    cleaned = openhands_utils.remove_binary_diffs(raw_patch)
    self.assertIn("diff --git a/foo.py b/foo.py", cleaned)
    self.assertIn("diff --git a/bar.py b/bar.py", cleaned)
    self.assertNotIn("bin.dat", cleaned)

    with tempfile.TemporaryDirectory() as tmpdir:
      repo = os.path.join(tmpdir, "testbed")
      os.makedirs(repo)
      subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
      subprocess.run(
          ["git", "config", "user.email", "test@example.com"],
          cwd=repo,
          check=True,
      )
      subprocess.run(
          ["git", "config", "user.name", "Test User"],
          cwd=repo,
          check=True,
      )
      src_file = os.path.join(repo, "app.py")
      with open(src_file, "w", encoding="utf-8") as f:
        f.write("def solve():\n    return 0\n")
      subprocess.run(["git", "add", "app.py"], cwd=repo, check=True)
      subprocess.run(
          ["git", "commit", "-m", "initial"],
          cwd=repo,
          check=True,
          capture_output=True,
      )
      base_commit = subprocess.run(
          ["git", "rev-parse", "HEAD"],
          cwd=repo,
          check=True,
          capture_output=True,
          text=True,
      ).stdout.strip()

      # Modify tracked source file, add nested .git dir (including spaces), and add binary file (with spaces).
      with open(src_file, "w", encoding="utf-8") as f:
        f.write("def solve():\n    return 42\n")
      nested_git = os.path.join(repo, "sub repo", ".git")
      os.makedirs(nested_git)
      with open(os.path.join(nested_git, "HEAD"), "w", encoding="utf-8") as f:
        f.write("ref: refs/heads/main\n")
      with open(
          os.path.join(repo, "sub repo", "helper.py"), "w", encoding="utf-8"
      ) as f:
        f.write("HELPER = True\n")
      bin_file = os.path.join(repo, "compiled binary.out")
      with open(bin_file, "wb") as f:
        f.write(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64)
      os.chmod(bin_file, 0o755)

      class _LocalWorkspace:

        def execute_command(self, cmd, timeout=60.0):
          del timeout
          p = subprocess.run(
              ["bash", "-c", cmd], capture_output=True, text=True
          )
          res = mock.MagicMock()
          res.stdout = p.stdout
          res.stderr = p.stderr
          res.exit_code = p.returncode
          return res

      patch = openhands_utils.extract_agent_patch(
          _LocalWorkspace(),
          base_commit=base_commit,
          workspace_path=repo,
      )
      self.assertFalse(os.path.exists(nested_git))
      self.assertIn("diff --git a/app.py b/app.py", patch)
      self.assertIn("+    return 42", patch)
      self.assertIn("sub repo/helper.py", patch)
      self.assertNotIn("compiled binary.out", patch)
      self.assertTrue(patch.endswith("\n"))

  def test_cleanup_rollout_container_processes(self):
    mock_ws = mock.MagicMock()
    openhands_utils.cleanup_rollout_container_processes(mock_ws)
    mock_ws.execute_command.assert_called_once()
    cmd = mock_ws.execute_command.call_args[0][0]
    self.assertIn("kill -TERM", cmd)
    self.assertIn("kill -KILL", cmd)
    self.assertIn("openhands-agent-server", cmd)
    self.assertIn("/dev/shm", cmd)

  def test_evaluate_patch_in_fresh_container_empty_and_failed_apply_and_success(
      self,
  ):
    mock_eval_env = mock.MagicMock()
    mock_runtime = mock.MagicMock()
    mock_runtime.repo_path = "/testbed"
    mock_eval_env.runtime = mock_runtime
    orig_reward = mock.MagicMock(return_value=1.0)

    # 1. Empty patch -> 0.0 without running git apply or orig_reward
    self.assertEqual(
        openhands_utils.evaluate_patch_in_fresh_container(
            mock_eval_env, "   \n", orig_reward
        ),
        0.0,
    )
    mock_runtime.run.assert_not_called()
    orig_reward.assert_not_called()

    # 2. Failed git apply -> 0.0 without running setup_env or orig_reward
    mock_runtime.run.side_effect = [
        ("run_tests.sh\nexpected_test_output.json\n", "0"),
        ("error: patch failed", "Error: Exit code 1"),
    ]
    self.assertEqual(
        openhands_utils.evaluate_patch_in_fresh_container(
            mock_eval_env,
            "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n",
            orig_reward,
        ),
        0.0,
    )
    mock_runtime.setup_env.assert_not_called()
    orig_reward.assert_not_called()

    # 3. Successful apply -> runs setup_env() and orig_reward() in eval container
    mock_runtime.reset_mock()
    mock_runtime.run.side_effect = [
        ("run_tests.sh\nexpected_test_output.json\nuntracked file.txt\n", "0"),
        ("", "0"),
    ]
    reward = openhands_utils.evaluate_patch_in_fresh_container(
        mock_eval_env,
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n",
        orig_reward,
    )
    self.assertEqual(reward, 1.0)
    self.assertEqual(
        mock_runtime._target_container,
        sandbox_utils.EVAL_CONTAINER_NAME,
    )
    self.assertEqual(mock_runtime.run.call_count, 2)
    ls_cmd = mock_runtime.run.call_args_list[0][0][0]
    apply_cmd = mock_runtime.run.call_args_list[1][0][0]
    self.assertEqual(ls_cmd, "git ls-files --others --exclude-standard")
    self.assertIn("git apply --whitespace=fix", apply_cmd)
    self.assertIn("--exclude=run_tests.sh", apply_cmd)
    self.assertIn("--exclude=expected_test_output.json", apply_cmd)
    self.assertIn("--exclude=untracked file.txt", apply_cmd)
    mock_runtime.setup_env.assert_called_once()
    orig_reward.assert_called_once()

    # 4. Real git repo with DockerRuntime._run_kubernetes `cd <workdir> && timeout <sec> <code>` semantics
    import subprocess  # pylint: disable=g-import-not-at-top
    import tempfile  # pylint: disable=g-import-not-at-top

    with tempfile.TemporaryDirectory() as tmpdir:
      repo = os.path.join(tmpdir, "testbed")
      os.makedirs(repo)
      subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
      subprocess.run(
          ["git", "config", "user.email", "test@example.com"],
          cwd=repo,
          check=True,
      )
      subprocess.run(
          ["git", "config", "user.name", "Test User"],
          cwd=repo,
          check=True,
      )
      target_file = os.path.join(repo, "calc.py")
      with open(target_file, "w", encoding="utf-8") as f:
        f.write("val = 1\n")
      subprocess.run(["git", "add", "calc.py"], cwd=repo, check=True)
      subprocess.run(
          ["git", "commit", "-m", "init"],
          cwd=repo,
          check=True,
          capture_output=True,
      )
      with open(os.path.join(repo, "run_tests.sh"), "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\nexit 0\n")

      def _k8s_like_run(code, timeout=30):
        full_cmd = f"cd {repo} && timeout {timeout} {code}"
        p = subprocess.run(
            ["/bin/sh", "-c", full_cmd], capture_output=True, text=True
        )
        out = (p.stdout or "") + (p.stderr or "")
        return (out, "0" if p.returncode == 0 else f"Error: Exit code {p.returncode}")

      real_eval_env = mock.MagicMock()
      real_runtime = mock.MagicMock()
      real_runtime.repo_path = repo
      real_runtime.run.side_effect = _k8s_like_run
      real_eval_env.runtime = real_runtime
      patch_str = (
          "diff --git a/calc.py b/calc.py\n"
          "--- a/calc.py\n"
          "+++ b/calc.py\n"
          "@@ -1 +1 @@\n"
          "-val = 1\n"
          "+val = 99\n"
      )
      res_reward = openhands_utils.evaluate_patch_in_fresh_container(
          real_eval_env,
          patch_str,
          lambda: 1.0,
      )
      self.assertEqual(res_reward, 1.0)
      with open(target_file, "r", encoding="utf-8") as f:
        self.assertEqual(f.read(), "val = 99\n")

      # 5. Large patch (>64KB base64) chunked write path + extract_agent_patch
      # ignoring bash_events/, conversations/, and install.sh
      subprocess.run(["git", "reset", "--hard"], cwd=repo, check=True)
      os.makedirs(os.path.join(repo, "bash_events"), exist_ok=True)
      with open(
          os.path.join(repo, "bash_events", "evt1"), "w", encoding="utf-8"
      ) as f:
        f.write("big event log " * 5000)
      os.makedirs(os.path.join(repo, "conversations"), exist_ok=True)
      with open(
          os.path.join(repo, "conversations", "conv1"), "w", encoding="utf-8"
      ) as f:
        f.write("conversation state")
      with open(os.path.join(repo, "install.sh"), "w", encoding="utf-8") as f:
        f.write("uv pip install -e .\n")

      class _LocalWs:

        def execute_command(self, cmd, timeout=60.0):
          del timeout
          p = subprocess.run(
              ["/bin/sh", "-c", cmd], capture_output=True, text=True
          )
          return mock.MagicMock(
              exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr
          )

      empty_patch = openhands_utils.extract_agent_patch(
          _LocalWs(), base_commit="HEAD", workspace_path=repo
      )
      self.assertEqual(empty_patch, "")

      large_content = "\n".join(f"line_{i} = {i}" for i in range(6000)) + "\n"
      with open(target_file, "w", encoding="utf-8") as f:
        f.write(large_content)
      extracted_large = openhands_utils.extract_agent_patch(
          _LocalWs(), base_commit="HEAD", workspace_path=repo
      )
      self.assertIn("diff --git a/calc.py b/calc.py", extracted_large)
      self.assertNotIn("bash_events", extracted_large)
      self.assertNotIn("conversations", extracted_large)
      self.assertNotIn("install.sh", extracted_large)
      self.assertGreater(len(extracted_large), 50000)

      subprocess.run(["git", "reset", "--hard"], cwd=repo, check=True)
      res_large_reward = openhands_utils.evaluate_patch_in_fresh_container(
          real_eval_env,
          extracted_large,
          lambda: 1.0,
      )
      self.assertEqual(res_large_reward, 1.0)
      with open(target_file, "r", encoding="utf-8") as f:
        self.assertEqual(f.read(), large_content)

      # 6. If repository tracked install.sh or run_tests.sh at base_commit,
      # extract_agent_patch must reset their index state to base_commit rather
      # than emitting a deletion diff.
      subprocess.run(["git", "reset", "--hard"], cwd=repo, check=True)
      tracked_install = os.path.join(repo, "install.sh")
      with open(tracked_install, "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\necho repo tracked install\n")
      subprocess.run(["git", "add", "install.sh"], cwd=repo, check=True)
      subprocess.run(
          ["git", "commit", "-m", "track install.sh"],
          cwd=repo,
          check=True,
          capture_output=True,
      )
      tracked_base = subprocess.check_output(
          ["git", "rev-parse", "HEAD"], cwd=repo, text=True
      ).strip()
      os.remove(tracked_install)
      with open(target_file, "w", encoding="utf-8") as f:
        f.write("val = 123\n")
      patch_with_tracked_install = openhands_utils.extract_agent_patch(
          _LocalWs(), base_commit=tracked_base, workspace_path=repo
      )
      self.assertIn("diff --git a/calc.py b/calc.py", patch_with_tracked_install)
      self.assertNotIn("install.sh", patch_with_tracked_install)

  def test_swe_env_agent_sandbox_grades_in_fresh_eval_container(self):
    mock_asrl = mock.MagicMock()
    mock_r2egym_adapter = mock.MagicMock()
    mock_oh_adapter = mock.MagicMock()

    mock_handle = mock.MagicMock()
    mock_fleet = mock.MagicMock()
    mock_fleet.acquire.return_value = mock_handle

    mock_ws = mock.MagicMock()
    setup_res = mock.MagicMock(exit_code=0, stdout="deadbeef1234567\n")
    diff_res = mock.MagicMock(
        exit_code=0,
        stdout=(
            "diff --git a/fix.py b/fix.py\n"
            "--- a/fix.py\n"
            "+++ b/fix.py\n"
            "@@ -1 +1 @@\n"
            "-a = 1\n"
            "+a = 2\n"
        ),
    )
    cleanup_res = mock.MagicMock(exit_code=0, stdout="")
    mock_ws.execute_command.side_effect = [setup_res, diff_res, cleanup_res]
    mock_oh_adapter.make_handle_workspace.return_value = mock_ws

    mock_repo_env = mock.MagicMock()
    mock_runtime = mock.MagicMock()
    mock_runtime.repo_path = "/testbed"
    mock_runtime.run.side_effect = [
        ("run_tests.sh\n", "0"),
        ("", "0"),
    ]
    mock_repo_env.runtime = mock_runtime
    mock_repo_env.compute_reward = mock.MagicMock(return_value=1.0)

    captured_handle_state = {}

    def _fake_make_fleet_repo_env(handle, **kwargs):
      del kwargs
      captured_handle_state["target_container"] = getattr(
          handle, "_target_container", None
      )
      captured_handle_state["defer_setup_env"] = getattr(
          handle, "_defer_setup_env", None
      )
      return mock_repo_env

    mock_r2egym_adapter.make_fleet_repo_env.side_effect = (
        _fake_make_fleet_repo_env
    )

    with mock.patch.dict(
        "sys.modules",
        {
            "agent_sandbox_rl": mock_asrl,
            "agent_sandbox_rl.adapters": mock.MagicMock(),
            "agent_sandbox_rl.adapters.r2egym": mock_r2egym_adapter,
            "agent_sandbox_rl.adapters.openhands": mock_oh_adapter,
        },
    ):
      env = swe_env.SWEEnv(
          entry={
              "instance_id": "inst_1",
              "docker_image": "img:v1",
              "problem_statement": "Fix bug",
          },
          scaffold="openhands",
          use_agent_sandbox=True,
          fleet=mock_fleet,
      )
      obs, info = env.reset()
      self.assertEqual(obs, "Fix bug")
      self.assertEqual(info.get("base_commit"), "deadbeef1234567")
      self.assertEqual(
          captured_handle_state["target_container"],
          sandbox_utils.EVAL_CONTAINER_NAME,
      )
      self.assertTrue(captured_handle_state["defer_setup_env"])

      reward = env.final_reward_fn()
      self.assertEqual(reward, 1.0)
      self.assertEqual(mock_ws.execute_command.call_count, 3)
      extract_cmd = mock_ws.execute_command.call_args_list[1][0][0]
      self.assertIn("git diff --no-color --cached deadbeef1234567", extract_cmd)
      cleanup_cmd = mock_ws.execute_command.call_args_list[2][0][0]
      self.assertIn("kill -TERM", cleanup_cmd)
      mock_runtime.setup_env.assert_called_once()


if __name__ == "__main__":
  absltest.main()



