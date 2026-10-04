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
from examples.deepswe import swe_agent
from examples.deepswe import swe_env
from examples.deepswe import template

SWEAction = swe_agent.SWEAction


class SweAgentTest(absltest.TestCase):

  def test_parse_codeact_bash_markdown(self):
    response = (
        "I need to inspect the git diff and list the files.\n"
        "```bash\n"
        "git status\n"
        "ls -la\n"
        "```\n"
        "This will help us understand the current workspace."
    )
    thought, action = swe_agent.parse_codeact_response(response)
    self.assertEqual(
        thought, "I need to inspect the git diff and list the files."
    )
    self.assertEqual(action.function_name, "execute_bash")
    self.assertEqual(action.parameters.get("command"), "git status\nls -la")

  def test_parse_codeact_sh_markdown(self):
    response = "```sh\npytest tests/test_core.py\n```"
    thought, action = swe_agent.parse_codeact_response(response)
    self.assertEqual(thought, "")
    self.assertEqual(action.function_name, "execute_bash")
    self.assertEqual(action.parameters.get("command"), "pytest tests/test_core.py")

  def test_parse_codeact_python_markdown(self):
    response = (
        "Let's write a small script to test reproduction.\n"
        "```python\n"
        "import sympy\n"
        "print(sympy.__version__)\n"
        "```"
    )
    thought, action = swe_agent.parse_codeact_response(response)
    self.assertEqual(
        thought, "Let's write a small script to test reproduction."
    )
    self.assertEqual(action.function_name, "execute_ipython_cell")
    self.assertEqual(
        action.parameters.get("code"),
        "import sympy\nprint(sympy.__version__)",
    )

  def test_parse_codeact_ipython_markdown(self):
    response = "```ipython\n%run reproduce_issue.py\n```"
    thought, action = swe_agent.parse_codeact_response(response)
    self.assertEqual(thought, "")
    self.assertEqual(action.function_name, "execute_ipython_cell")
    self.assertEqual(action.parameters.get("code"), "%run reproduce_issue.py")

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

  def test_parse_codeact_json_tool_call(self):
    response = (
        "Running command via tool call.\n"
        "<tool_call>\n"
        '{"name": "execute_bash", "arguments": {"command": "git diff"}}\n'
        "</tool_call>"
    )
    thought, action = swe_agent.parse_codeact_response(response)
    self.assertEqual(thought, "Running command via tool call.")
    self.assertEqual(action.function_name, "execute_bash")
    self.assertEqual(action.parameters.get("command"), "git diff")

  def test_parse_codeact_completion_trigger(self):
    response = (
        "I have resolved the issue and verified all tests pass.\n"
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    )
    thought, action = swe_agent.parse_codeact_response(response)
    self.assertIn("I have resolved the issue", thought)
    self.assertEqual(action.function_name, "submit")

  def test_parse_codeact_unclosed_block(self):
    # Simulates response truncated by max tokens
    response = (
        "Executing long command:\n"
        "```bash\n"
        "git log -n 5"
    )
    thought, action = swe_agent.parse_codeact_response(response)
    self.assertEqual(thought, "Executing long command:")
    self.assertEqual(action.function_name, "execute_bash")
    self.assertEqual(action.parameters.get("command"), "git log -n 5")

  def test_codeact_agent_initialization(self):
    agent = swe_agent.CodeActAgent()
    self.assertEqual(agent.scaffold, "openhands")
    self.assertEqual(agent.system_prompt, template.OPENHANDS_SYSTEM_PROMPT)

  def test_codeact_agent_update_from_model(self):
    agent = swe_agent.CodeActAgent()
    agent.update_from_env(observation="Problem statement", reward=0.0, done=False)
    action_res = agent.update_from_model(
        "I will run pytest:\n```bash\npytest\n```"
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
    self.assertEqual(result.observation, "/workspace\n")
    self.assertEqual(mock_env.total_steps, 1)

  def test_step_openhands_execute_bash_missing_command(self):
    mock_env = mock.MagicMock()
    mock_env.max_steps = 10
    action = SWEAction("execute_bash", {})
    result = openhands_utils.step_openhands(mock_env, action)
    self.assertFalse(result.done)
    self.assertEqual(
        result.observation, "ERROR: No command specified for execute_bash."
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

  def test_parse_codeact_markdown_json_tool_call(self):
    response = (
        "Running command via markdown json.\n"
        "```json\n"
        '{"name": "execute_bash", "arguments": {"command": "pwd"}}\n'
        "```"
    )
    thought, action = swe_agent.parse_codeact_response(response)
    self.assertEqual(thought, "Running command via markdown json.")
    self.assertEqual(action.function_name, "execute_bash")
    self.assertEqual(action.parameters.get("command"), "pwd")

  def test_codeact_agent_step_and_token_warning(self):
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
    self.assertIn("You are running out of tokens", agent.cur_step.observation)
    self.assertIn("<function=finish>", agent.cur_step.observation)
    self.assertIn("Steps Remaining: 49", agent.cur_step.observation)
    self.assertIn(
        "<uploaded_files>\n/workspace\n</uploaded_files>",
        agent.cur_step.observation,
    )
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
    self.assertIn("Task list has been updated with 1 items.", plan_res.observation)

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

  def test_oh_editor_create_view_str_replace_insert_undo_and_600_line_limit(self):
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

      # 6. 600-line view limit clipping
      long_path = os.path.join(tmpdir, "long.py")
      with open(long_path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"line_{i}" for i in range(1, 701)))
      long_view = openhands_utils.run_oh_editor_locally(
          {"command": "view", "path": long_path},
          history_file=hist_file,
      )
      self.assertIn("line_600", long_view)
      self.assertNotIn("line_601", long_view)
      self.assertIn(openhands_utils.FILE_CLIPPED_NOTICE, long_view)

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
    self.assertIn("Command timed out after 45 seconds", result.observation)
    called_cmd = mock_workspace.execute_command.call_args[0][0]
    self.assertIn("/var/tmp/.oh_cwd", called_cmd)
    self.assertTrue(called_cmd.endswith("\n)"))
    self.assertIn("/opt/miniconda3/envs/testbed/bin", called_cmd)
    self.assertEqual(
        mock_workspace.execute_command.call_args.kwargs["timeout"], 45.0
    )

  def test_oh_editor_preserves_tabs_clamps_view_range_and_strip_fallback(self):
    import tempfile  # pylint: disable=g-import-not-at-top

    with tempfile.TemporaryDirectory() as tmpdir:
      hist_file = os.path.join(tmpdir, "history.json")
      tab_path = os.path.join(tmpdir, "tabbed.py")
      with open(tab_path, "w", encoding="utf-8") as f:
        f.write("def f():\n\tx = 1\n\ty = 2\n\treturn x + y\n")

      # 1. view preserves tabs and clamps view_range upper bound > n_lines
      view_out = openhands_utils.run_oh_editor_locally(
          {"command": "view", "path": tab_path, "view_range": [1, 999]},
          history_file=hist_file,
      )
      self.assertIn("\t\tx = 1", view_out)
      self.assertNotIn("Error: Invalid `view_range`", view_out)

      # 2. str_replace preserves tabs in file and supports stripped fallback
      replace_out = openhands_utils.run_oh_editor_locally(
          {
              "command": "str_replace",
              "path": tab_path,
              "old_str": "\n\tx = 1\n",
              "new_str": "\n\tx = 10\n",
          },
          history_file=hist_file,
      )
      self.assertIn("has been edited", replace_out)
      with open(tab_path, "r", encoding="utf-8") as f:
        updated_content = f.read()
      self.assertIn("\tx = 10\n\ty = 2\n", updated_content)

      # 3. FILE_CLIPPED_NOTICE is not stripped by _format_command_result
      huge_path = os.path.join(tmpdir, "huge.py")
      with open(huge_path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"line_{i}_" + ("x" * 40) for i in range(1, 500)))
      clipped_view = openhands_utils.run_oh_editor_locally(
          {"command": "view", "path": huge_path},
          history_file=hist_file,
      )
      self.assertTrue(
          clipped_view.endswith(openhands_utils.FILE_CLIPPED_NOTICE)
      )
      mock_res = mock.MagicMock(
          spec=["stdout", "stderr", "exit_code", "timeout_occurred"]
      )
      mock_res.stdout = clipped_view
      mock_res.stderr = ""
      mock_res.exit_code = 0
      mock_res.timeout_occurred = False
      formatted_obs, _ = openhands_utils._format_command_result(
          mock_res, 60.0, 0.1
      )
      self.assertTrue(
          formatted_obs.endswith(openhands_utils.FILE_CLIPPED_NOTICE)
      )

  def test_format_command_result_preserves_stderr_on_zero_exit(self):
    mock_res = mock.MagicMock(
        spec=["stdout", "stderr", "exit_code", "timeout_occurred"]
    )
    mock_res.exit_code = 0
    mock_res.stdout = ""
    mock_res.stderr = "Ran 3 tests in 0.01s\n\nOK\n"
    mock_res.timeout_occurred = False
    obs, timed_out = openhands_utils._format_command_result(
        mock_res, 60.0, 0.05
    )
    self.assertFalse(timed_out)
    self.assertIn("Ran 3 tests in 0.01s", obs)

  def test_parse_codeact_multiple_tool_calls_prefers_actionable(self):
    response = (
        "<think>Update task list and run pytest.</think>\n"
        "<tool_call>\n"
        "<function=task_tracker>\n"
        "<parameter=command>plan</parameter>\n"
        "<parameter=task_list>[]</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=execute_bash>\n"
        "<parameter=command>pytest -q</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    thought, action = swe_agent.parse_codeact_response(response)
    self.assertEqual(action.function_name, "execute_bash")
    self.assertEqual(action.parameters["command"], "pytest -q")
    self.assertIn("</think>", thought)

  def test_resolve_base_commit_and_strip_issue_tags(self):
    self.assertEqual(
        openhands_utils.resolve_base_commit({"commit_hash": "deadbeef123"}),
        "deadbeef123",
    )
    env = swe_env.SWEEnv(
        entry={"instance_id": "t1", "problem_statement": "[ISSUE]\nFix bug\n[/ISSUE]"},
        scaffold="openhands",
    )
    env.workspace = mock.MagicMock()
    env.env = mock.MagicMock()
    obs = env._initial_observation()
    self.assertEqual(obs, "Fix bug")


if __name__ == "__main__":
  absltest.main()
