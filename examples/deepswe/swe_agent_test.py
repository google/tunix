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
from r2egym.agenthub.action.action import Action as SWEAction
from examples.deepswe import openhands_utils
from examples.deepswe import swe_agent
from examples.deepswe import swe_env
from examples.deepswe import template


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

  def test_codeact_agent_token_warning(self):
    agent = swe_agent.CodeActAgent()
    agent.update_from_env(
        observation="Init",
        reward=0.0,
        done=False,
        info={"cur_tokens": 30000},
    )
    self.assertIn("You are running out of tokens", agent.cur_step.observation)
    self.assertIn("<function=submit>", agent.cur_step.observation)

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


if __name__ == "__main__":
  absltest.main()

