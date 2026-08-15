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

"""Tests for action_converter module."""

import dataclasses
import re
import shlex
from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.experimental.trajectory import action_converter
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.tools import base_tool


class MockR2EAction:
  """Replicates r2egym.agenthub.action.Action interface and behavior."""

  def __init__(
      self,
      function_name: str,
      parameters: dict[str, Any],
      function_id: str | None = None,
  ):
    self.function_name = function_name
    self.parameters = parameters
    self.function_id = function_id

  @classmethod
  def from_string(cls, action_str: str) -> "MockR2EAction":
    fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_str)
    function_name = fn_match.group(1).strip() if fn_match else ""
    pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
    param_matches = re.findall(pattern, action_str, flags=re.DOTALL)
    params = {k.strip(): v.strip() for k, v in param_matches}
    return cls(function_name, params)

  def to_xml_string(self) -> str:
    xml_str = f"<function={self.function_name}>\n"
    for param_key, param_value in self.parameters.items():
      xml_str += f"  <parameter={param_key}>{param_value}</parameter>\n"
    xml_str += "</function>"
    return xml_str

  def to_dict(self) -> dict[str, Any]:
    return {"function": self.function_name, "parameters": self.parameters}

  def to_bashcmd(self) -> str:
    if not self.function_name:
      return ""
    if self.function_name in ("finish", "submit"):
      return "echo '<<<Finished>>>'"
    cmd_parts = [shlex.quote(self.function_name)]
    base_command = self.parameters.get("command")
    if base_command is not None:
      cmd_parts.append(shlex.quote(str(base_command)))
    for param_key, param_value in self.parameters.items():
      if param_key == "command":
        continue
      cmd_parts.append(f"--{param_key}")
      cmd_parts.append(shlex.quote(str(param_value)))
    return " ".join(cmd_parts)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, MockR2EAction):
      return False
    return (
        self.function_name == other.function_name
        and self.parameters == other.parameters
    )

  def __repr__(self) -> str:
    return (
        f"MockR2EAction(function_name={self.function_name!r},"
        f" parameters={self.parameters!r})"
    )


class ActionConverterTest(parameterized.TestCase):

  def test_extract_tool_calls_none(self):
    self.assertIsNone(action_converter.extract_tool_calls(None))

  def test_extract_tool_calls_openai_format_stringified_args(self):
    action = agent_types.Action(
        action={
            "id": "call_openai_1",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": '{"query": "tunix", "limit": 10}',
            },
        }
    )
    calls = action_converter.extract_tool_calls(action)
    self.assertIsNotNone(calls)
    self.assertLen(calls, 1)
    self.assertEqual(calls[0].tool_call_id, "call_openai_1")
    self.assertEqual(calls[0].function_name, "search")
    self.assertEqual(calls[0].arguments, {"query": "tunix", "limit": 10})

  def test_extract_tool_calls_openai_format_dict_args(self):
    action = agent_types.Action(
        action={
            "id": "call_openai_2",
            "type": "function",
            "function": {
                "name": "calc",
                "arguments": {"x": 5, "y": 10},
            },
        }
    )
    calls = action_converter.extract_tool_calls(action)
    self.assertIsNotNone(calls)
    self.assertLen(calls, 1)
    self.assertEqual(calls[0].tool_call_id, "call_openai_2")
    self.assertEqual(calls[0].function_name, "calc")
    self.assertEqual(calls[0].arguments, {"x": 5, "y": 10})

  def test_extract_tool_calls_openai_format_list(self):
    action = agent_types.Action(
        action=[
            {
                "id": "call_10",
                "type": "function",
                "function": {"name": "f1", "arguments": '{"a": 1}'},
            },
            {
                "id": "call_20",
                "type": "function",
                "function": {"name": "f2", "arguments": {"b": 2}},
            },
        ]
    )
    calls = action_converter.extract_tool_calls(action)
    self.assertIsNotNone(calls)
    self.assertLen(calls, 2)
    self.assertEqual(calls[0].tool_call_id, "call_10")
    self.assertEqual(calls[0].function_name, "f1")
    self.assertEqual(calls[0].arguments, {"a": 1})
    self.assertEqual(calls[1].tool_call_id, "call_20")
    self.assertEqual(calls[1].function_name, "f2")
    self.assertEqual(calls[1].arguments, {"b": 2})

  def test_extract_tool_calls_anthropic_format_dict_input(self):
    action = agent_types.Action(
        action={
            "type": "tool_use",
            "id": "toolu_01A",
            "name": "get_weather",
            "input": {"location": "San Francisco, CA"},
        }
    )
    calls = action_converter.extract_tool_calls(action)
    self.assertIsNotNone(calls)
    self.assertLen(calls, 1)
    self.assertEqual(calls[0].tool_call_id, "toolu_01A")
    self.assertEqual(calls[0].function_name, "get_weather")
    self.assertEqual(calls[0].arguments, {"location": "San Francisco, CA"})

  def test_extract_tool_calls_anthropic_format_stringified_input(self):
    action = agent_types.Action(
        action={
            "type": "tool_use",
            "id": "toolu_02B",
            "name": "calculator",
            "input": '{"expr": "2+2"}',
        }
    )
    calls = action_converter.extract_tool_calls(action)
    self.assertIsNotNone(calls)
    self.assertLen(calls, 1)
    self.assertEqual(calls[0].tool_call_id, "toolu_02B")
    self.assertEqual(calls[0].function_name, "calculator")
    self.assertEqual(calls[0].arguments, {"expr": "2+2"})

  def test_extract_tool_calls_anthropic_format_list(self):
    action = agent_types.Action(
        action=[
            {
                "type": "tool_use",
                "id": "t1",
                "name": "weather",
                "input": {"city": "Paris"},
            },
            {
                "type": "tool_use",
                "id": "t2",
                "name": "time",
                "input": {"timezone": "CET"},
            },
        ]
    )
    calls = action_converter.extract_tool_calls(action)
    self.assertIsNotNone(calls)
    self.assertLen(calls, 2)
    self.assertEqual(calls[0].tool_call_id, "t1")
    self.assertEqual(calls[0].function_name, "weather")
    self.assertEqual(calls[0].arguments, {"city": "Paris"})
    self.assertEqual(calls[1].tool_call_id, "t2")
    self.assertEqual(calls[1].function_name, "time")
    self.assertEqual(calls[1].arguments, {"timezone": "CET"})

  def test_extract_tool_calls_flat_dict_alias_combinations(self):
    cases = [
        (
            {"call_id": "c1", "func": "f1", "params": {"x": 1}},
            "c1",
            "f1",
            {"x": 1},
        ),
        (
            {"tool_call_id": "c2", "tool_name": "f2", "args": {"y": 2}},
            "c2",
            "f2",
            {"y": 2},
        ),
        (
            {"function_id": "c3", "function": "f3", "parameters": {"z": 3}},
            "c3",
            "f3",
            {"z": 3},
        ),
        (
            {"id": "c4", "function_name": "f4", "input": {"w": 4}},
            "c4",
            "f4",
            {"w": 4},
        ),
    ]
    for action_dict, exp_id, exp_name, exp_args in cases:
      calls = action_converter.extract_tool_calls(
          agent_types.Action(action=action_dict)
      )
      self.assertIsNotNone(calls)
      self.assertLen(calls, 1)
      self.assertEqual(calls[0].tool_call_id, exp_id)
      self.assertEqual(calls[0].function_name, exp_name)
      self.assertEqual(calls[0].arguments, exp_args)

  def test_extract_tool_calls_flat_dict_inlined_parameters(self):
    action = agent_types.Action(
        action={
            "id": "c_inline",
            "name": "execute_bash",
            "command": "echo hello",
            "timeout": 30,
        }
    )
    calls = action_converter.extract_tool_calls(action)
    self.assertIsNotNone(calls)
    self.assertLen(calls, 1)
    self.assertEqual(calls[0].tool_call_id, "c_inline")
    self.assertEqual(calls[0].function_name, "execute_bash")
    self.assertEqual(
        calls[0].arguments, {"command": "echo hello", "timeout": 30}
    )

  def test_extract_tool_calls_dataclass_and_custom_objects(self):
    # base_tool.ToolCall
    bt_call = base_tool.ToolCall(name="bash", arguments={"command": "ls -la"})
    calls1 = action_converter.extract_tool_calls(
        agent_types.Action(action=bt_call)
    )
    self.assertIsNotNone(calls1)
    self.assertLen(calls1, 1)
    self.assertEqual(calls1[0].tool_call_id, "call_1")
    self.assertEqual(calls1[0].function_name, "bash")
    self.assertEqual(calls1[0].arguments, {"command": "ls -la"})

    # Custom SWEAction object
    class MockSWEAction:

      def __init__(self, function_name, parameters):
        self.function_name = function_name
        self.parameters = parameters

      def to_dict(self):
        return {"function": self.function_name, "parameters": self.parameters}

    swe_action = MockSWEAction(
        function_name="file_editor",
        parameters={"command": "view", "path": "test.py"},
    )
    calls2 = action_converter.extract_tool_calls(
        agent_types.Action(action=swe_action)
    )
    self.assertIsNotNone(calls2)
    self.assertLen(calls2, 1)
    self.assertEqual(calls2[0].tool_call_id, "call_1")
    self.assertEqual(calls2[0].function_name, "file_editor")
    self.assertEqual(
        calls2[0].arguments, {"command": "view", "path": "test.py"}
    )

    # Custom dataclass
    @dataclasses.dataclass
    class CustomDataclassCall:
      tool_name: str
      args: dict[str, Any]
      call_id: str

    custom_dc = CustomDataclassCall(
        tool_name="python_exec", args={"code": "1+1"}, call_id="custom_99"
    )
    calls3 = action_converter.extract_tool_calls(
        agent_types.Action(action=custom_dc)
    )
    self.assertIsNotNone(calls3)
    self.assertLen(calls3, 1)
    self.assertEqual(calls3[0].tool_call_id, "custom_99")
    self.assertEqual(calls3[0].function_name, "python_exec")
    self.assertEqual(calls3[0].arguments, {"code": "1+1"})

    # Custom object with to_dict returning OpenAI format
    class CustomObjWithToDict:

      def to_dict(self):
        return {
            "id": "c_obj_1",
            "type": "function",
            "function": {"name": "obj_fn", "arguments": {"k": "v"}},
        }

    calls4 = action_converter.extract_tool_calls(
        agent_types.Action(action=CustomObjWithToDict())
    )
    self.assertIsNotNone(calls4)
    self.assertLen(calls4, 1)
    self.assertEqual(calls4[0].tool_call_id, "c_obj_1")
    self.assertEqual(calls4[0].function_name, "obj_fn")
    self.assertEqual(calls4[0].arguments, {"k": "v"})

    # Dataclass with inlined arguments
    @dataclasses.dataclass
    class InlinedDataclassCall:
      name: str = "bash"
      command: str = "ls -la"
      timeout: int = 30

    calls5 = action_converter.extract_tool_calls(InlinedDataclassCall())
    self.assertIsNotNone(calls5)
    self.assertLen(calls5, 1)
    self.assertEqual(calls5[0].function_name, "bash")
    self.assertEqual(calls5[0].arguments, {"command": "ls -la", "timeout": 30})

    # Custom object with __dict__ and inlined arguments
    class CustomClassWithDict:

      def __init__(self):
        self.tool_name = "python"
        self.code = "print(42)"

    calls6 = action_converter.extract_tool_calls(CustomClassWithDict())
    self.assertIsNotNone(calls6)
    self.assertLen(calls6, 1)
    self.assertEqual(calls6[0].function_name, "python")
    self.assertEqual(calls6[0].arguments, {"code": "print(42)"})

    # Custom object with to_dict returning Anthropic format
    class CustomAnthropicObj:

      def to_dict(self):
        return {
            "type": "tool_use",
            "id": "toolu_abc",
            "name": "weather",
            "input": {"city": "Berlin"},
        }

    calls7 = action_converter.extract_tool_calls(CustomAnthropicObj())
    self.assertIsNotNone(calls7)
    self.assertLen(calls7, 1)
    self.assertEqual(calls7[0].tool_call_id, "toolu_abc")
    self.assertEqual(calls7[0].function_name, "weather")
    self.assertEqual(calls7[0].arguments, {"city": "Berlin"})

  def test_extract_tool_calls_xml_string_formats(self):
    # SWE-agent standard format
    xml_swe = (
        "<function=file_editor>\n"
        "<parameter=command>view</parameter>\n"
        "<parameter=path>main.py</parameter>\n"
        "</function>"
    )
    calls1 = action_converter.extract_tool_calls(xml_swe)
    self.assertIsNotNone(calls1)
    self.assertLen(calls1, 1)
    self.assertEqual(calls1[0].tool_call_id, "call_1")
    self.assertEqual(calls1[0].function_name, "file_editor")
    self.assertEqual(
        calls1[0].arguments, {"command": "view", "path": "main.py"}
    )

    # SWE-agent with quotes around function and parameter names
    xml_swe_quotes = (
        '<function="file_editor">\n'
        '<parameter="command">view</parameter>\n'
        '<parameter="path">main.py</parameter>\n'
        "</function>"
    )
    calls_quotes = action_converter.extract_tool_calls(xml_swe_quotes)
    self.assertIsNotNone(calls_quotes)
    self.assertLen(calls_quotes, 1)
    self.assertEqual(calls_quotes[0].function_name, "file_editor")
    self.assertEqual(
        calls_quotes[0].arguments, {"command": "view", "path": "main.py"}
    )

    # SWE-agent / XML with name= attribute syntax
    xml_swe_name_attr = (
        '<function name="bash">\n'
        '<parameter name="command">ls -la</parameter>\n'
        "</function>"
    )
    calls_name_attr = action_converter.extract_tool_calls(xml_swe_name_attr)
    self.assertIsNotNone(calls_name_attr)
    self.assertLen(calls_name_attr, 1)
    self.assertEqual(calls_name_attr[0].function_name, "bash")
    self.assertEqual(calls_name_attr[0].arguments, {"command": "ls -la"})

    # Anthropic invoke XML format
    xml_anthropic_invoke = (
        '<invoke name="calculator">\n'
        "<parameter name='expr'>1+1</parameter>\n"
        "</invoke>"
    )
    calls_invoke = action_converter.extract_tool_calls(xml_anthropic_invoke)
    self.assertIsNotNone(calls_invoke)
    self.assertLen(calls_invoke, 1)
    self.assertEqual(calls_invoke[0].function_name, "calculator")
    self.assertEqual(calls_invoke[0].arguments, {"expr": "1+1"})

    # Anthropic function_calls container wrapping invoke
    xml_fc_invoke = (
        "<function_calls>\n"
        '<invoke name="calc">\n'
        "<parameter name='x'>1</parameter>\n"
        "</invoke>\n"
        "</function_calls>"
    )
    calls_fc_invoke = action_converter.extract_tool_calls(xml_fc_invoke)
    self.assertIsNotNone(calls_fc_invoke)
    self.assertLen(calls_fc_invoke, 1)
    self.assertEqual(calls_fc_invoke[0].function_name, "calc")
    self.assertEqual(calls_fc_invoke[0].arguments, {"x": "1"})

    # SWE-agent multiline parameter format
    xml_multiline = (
        "<function=execute_bash>\n"
        "<parameter=command>\n"
        "cat << 'EOF' > test.py\n"
        "print('hello')\n"
        "EOF\n"
        "</parameter>\n"
        "</function>"
    )
    calls2 = action_converter.extract_tool_calls(xml_multiline)
    self.assertIsNotNone(calls2)
    self.assertLen(calls2, 1)
    self.assertEqual(calls2[0].function_name, "execute_bash")
    self.assertEqual(
        calls2[0].arguments,
        {"command": "cat << 'EOF' > test.py\nprint('hello')\nEOF"},
    )

    # SWE-agent multiple function calls
    xml_multi = (
        "<function=list_files>\n</function>\n<function=submit>\n</function>"
    )
    calls3 = action_converter.extract_tool_calls(xml_multi)
    self.assertIsNotNone(calls3)
    self.assertLen(calls3, 2)
    self.assertEqual(calls3[0].tool_call_id, "call_1")
    self.assertEqual(calls3[0].function_name, "list_files")
    self.assertEqual(calls3[0].arguments, {})
    self.assertEqual(calls3[1].tool_call_id, "call_2")
    self.assertEqual(calls3[1].function_name, "submit")
    self.assertEqual(calls3[1].arguments, {})

    # SWE-agent empty body
    xml_finish = "<function=finish>\n</function>"
    calls4 = action_converter.extract_tool_calls(xml_finish)
    self.assertIsNotNone(calls4)
    self.assertLen(calls4, 1)
    self.assertEqual(calls4[0].function_name, "finish")
    self.assertEqual(calls4[0].arguments, {})

    # Qwen/Hermes <tool_call> JSON format
    xml_qwen_json = (
        '<tool_call>\n{"name": "calculator", "arguments": {"expr":'
        ' "100*2"}}\n</tool_call>'
    )
    calls5 = action_converter.extract_tool_calls(xml_qwen_json)
    self.assertIsNotNone(calls5)
    self.assertLen(calls5, 1)
    self.assertEqual(calls5[0].function_name, "calculator")
    self.assertEqual(calls5[0].arguments, {"expr": "100*2"})

    # Qwen/Hermes <tool_call> with markdown code fence
    xml_qwen_fenced = (
        '<tool_call>\n```json\n{"name": "calculator", "arguments": {"expr":'
        ' "100*2"}}\n```\n</tool_call>'
    )
    calls5_fenced = action_converter.extract_tool_calls(xml_qwen_fenced)
    self.assertIsNotNone(calls5_fenced)
    self.assertLen(calls5_fenced, 1)
    self.assertEqual(calls5_fenced[0].function_name, "calculator")
    self.assertEqual(calls5_fenced[0].arguments, {"expr": "100*2"})

    # Qwen/Hermes multiple <tool_call> tags
    xml_qwen_multi = (
        '<tool_call>{"name": "fn1", "arguments": {"a": 1}}</tool_call>\n'
        '<tool_call>{"name": "fn2", "arguments": {"b": 2}}</tool_call>'
    )
    calls6 = action_converter.extract_tool_calls(xml_qwen_multi)
    self.assertIsNotNone(calls6)
    self.assertLen(calls6, 2)
    self.assertEqual(calls6[0].function_name, "fn1")
    self.assertEqual(calls6[0].arguments, {"a": 1})
    self.assertEqual(calls6[1].function_name, "fn2")
    self.assertEqual(calls6[1].arguments, {"b": 2})

    # Qwen <tool_call> wrapping <function=...> XML
    xml_qwen_nested = (
        "<tool_call>\n"
        "<function=search>\n"
        "<parameter=query>tunix</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    calls7 = action_converter.extract_tool_calls(xml_qwen_nested)
    self.assertIsNotNone(calls7)
    self.assertLen(calls7, 1)
    self.assertEqual(calls7[0].function_name, "search")
    self.assertEqual(calls7[0].arguments, {"query": "tunix"})

    # XML <function_call> tag with <name> and <arguments>
    xml_function_call_tag = (
        "<function_call>\n"
        "<name>calc</name>\n"
        '<arguments>{"x": 1}</arguments>\n'
        "</function_call>"
    )
    calls_fc_tag = action_converter.extract_tool_calls(xml_function_call_tag)
    self.assertIsNotNone(calls_fc_tag)
    self.assertLen(calls_fc_tag, 1)
    self.assertEqual(calls_fc_tag[0].function_name, "calc")
    self.assertEqual(calls_fc_tag[0].arguments, {"x": 1})

    # Llama 3 / Functionary JSON inside <function=...>
    xml_llama3 = '<function=search>{"query": "machine learning"}</function>'
    calls8 = action_converter.extract_tool_calls(xml_llama3)
    self.assertIsNotNone(calls8)
    self.assertLen(calls8, 1)
    self.assertEqual(calls8[0].function_name, "search")
    self.assertEqual(calls8[0].arguments, {"query": "machine learning"})

  def test_extract_tool_calls_json_stringified_formats(self):
    # Stringified dict
    json_dict_str = '{"name": "bash", "arguments": {"cmd": "pwd"}}'
    calls1 = action_converter.extract_tool_calls(json_dict_str)
    self.assertIsNotNone(calls1)
    self.assertLen(calls1, 1)
    self.assertEqual(calls1[0].function_name, "bash")
    self.assertEqual(calls1[0].arguments, {"cmd": "pwd"})

    # Stringified dict inside markdown code fences
    json_fenced_str = (
        '```json\n{"name": "bash", "arguments": {"cmd": "pwd"}}\n```'
    )
    calls1_fenced = action_converter.extract_tool_calls(json_fenced_str)
    self.assertIsNotNone(calls1_fenced)
    self.assertLen(calls1_fenced, 1)
    self.assertEqual(calls1_fenced[0].function_name, "bash")
    self.assertEqual(calls1_fenced[0].arguments, {"cmd": "pwd"})

    # Stringified list of dicts
    json_list_str = (
        '[{"name": "fn1", "arguments": {"a": 1}}, {"name": "fn2",'
        ' "arguments": {"b": 2}}]'
    )
    calls2 = action_converter.extract_tool_calls(json_list_str)
    self.assertIsNotNone(calls2)
    self.assertLen(calls2, 2)
    self.assertEqual(calls2[0].function_name, "fn1")
    self.assertEqual(calls2[0].arguments, {"a": 1})
    self.assertEqual(calls2[1].function_name, "fn2")
    self.assertEqual(calls2[1].arguments, {"b": 2})

    # Stringified OpenAI format
    json_openai_str = (
        '{"id": "call_json_1", "type": "function", "function": {"name":'
        ' "browse", "arguments": "{\\"url\\": \\"https://google.com\\"}"}}'
    )
    calls3 = action_converter.extract_tool_calls(json_openai_str)
    self.assertIsNotNone(calls3)
    self.assertLen(calls3, 1)
    self.assertEqual(calls3[0].tool_call_id, "call_json_1")
    self.assertEqual(calls3[0].function_name, "browse")
    self.assertEqual(calls3[0].arguments, {"url": "https://google.com"})

  def test_extract_tool_calls_unwrapped_actions(self):
    # Unwrapped dict
    calls1 = action_converter.extract_tool_calls(
        {"name": "bash", "arguments": {"cmd": "ls"}}
    )
    self.assertIsNotNone(calls1)
    self.assertLen(calls1, 1)
    self.assertEqual(calls1[0].function_name, "bash")
    self.assertEqual(calls1[0].arguments, {"cmd": "ls"})

    # Unwrapped base_tool.ToolCall
    calls2 = action_converter.extract_tool_calls(
        base_tool.ToolCall(name="bash", arguments={"cmd": "ls"})
    )
    self.assertIsNotNone(calls2)
    self.assertLen(calls2, 1)
    self.assertEqual(calls2[0].function_name, "bash")
    self.assertEqual(calls2[0].arguments, {"cmd": "ls"})

    # Unwrapped XML string
    calls3 = action_converter.extract_tool_calls("<function=finish></function>")
    self.assertIsNotNone(calls3)
    self.assertLen(calls3, 1)
    self.assertEqual(calls3[0].function_name, "finish")
    self.assertEqual(calls3[0].arguments, {})

    # Unwrapped list of dicts
    calls4 = action_converter.extract_tool_calls([
        {"name": "tool1", "arguments": {}},
        {"name": "tool2", "arguments": {}},
    ])
    self.assertIsNotNone(calls4)
    self.assertLen(calls4, 2)
    self.assertEqual(calls4[0].function_name, "tool1")
    self.assertEqual(calls4[1].function_name, "tool2")

  def test_extract_tool_calls_non_tool_actions_return_none(self):
    non_tool_actions = [
        0,
        1,
        42,
        "0",
        "1",
        "I will now think about how to answer...",
        np.array([0, 1, 0]),
        0.5,
        True,
        False,
        [0, 1, 2],
    ]
    for non_tool in non_tool_actions:
      self.assertIsNone(
          action_converter.extract_tool_calls(
              agent_types.Action(action=non_tool)
          )
      )
      self.assertIsNone(action_converter.extract_tool_calls(non_tool))

  def test_extract_tool_calls_malformed_and_empty_inputs_return_none(self):
    invalid_actions = [
        None,
        agent_types.Action(action=None),
        {},
        agent_types.Action(action={}),
        {"id": "call_1"},
        "",
        "   \n\t  ",
        [],
        [None, "random invalid text", 123],
        "<unknown_tag>test</unknown_tag>",
        "{broken json string",
        "<function=",
    ]
    for invalid_action in invalid_actions:
      self.assertIsNone(action_converter.extract_tool_calls(invalid_action))

  def test_extract_tool_calls_already_tool_call(self):
    tc = trajectory_lib.ToolCall(
        tool_call_id="call_x",
        function_name="fn_x",
        arguments={"k": "v"},
    )
    calls = action_converter.extract_tool_calls(tc)
    self.assertIsNotNone(calls)
    self.assertEqual(calls[0], tc)

  def test_to_dict_helper(self):
    # dict
    self.assertEqual(action_converter._to_dict({"a": 1}), {"a": 1})

    # dataclass
    @dataclasses.dataclass
    class SampleDC:
      val: int

    self.assertEqual(action_converter._to_dict(SampleDC(val=5)), {"val": 5})

    # to_dict method
    class SampleWithToDict:

      def to_dict(self):
        return {"b": 2}

    self.assertEqual(action_converter._to_dict(SampleWithToDict()), {"b": 2})

    # non-dict / primitive
    self.assertIsNone(action_converter._to_dict(123))
    self.assertIsNone(action_converter._to_dict("string"))

  def test_clean_markdown_code_blocks(self):
    self.assertEqual(
        action_converter._clean_markdown_code_blocks("```json\n{}\n```"), "{}"
    )
    self.assertEqual(
        action_converter._clean_markdown_code_blocks("```\nhello\n```"), "hello"
    )
    self.assertEqual(
        action_converter._clean_markdown_code_blocks("plain text"), "plain text"
    )


class DeepSWEActionConverterTest(parameterized.TestCase):
  """Comprehensive tests for DeepSWE action formats."""

  @parameterized.parameters(
      (
          (
              "<function=file_editor>\n"
              "  <parameter=command>view</parameter>\n"
              "  <parameter=path>/testbed/sympy/core/basic.py</parameter>\n"
              "  <parameter=view_range>[10, 20]</parameter>\n"
              "  <parameter=concise>True</parameter>\n"
              "</function>"
          ),
          "file_editor",
          {
              "command": "view",
              "path": "/testbed/sympy/core/basic.py",
              "view_range": "[10, 20]",
              "concise": "True",
          },
      ),
      (
          (
              "<function=file_editor>\n"
              "  <parameter=command>create</parameter>\n"
              "  <parameter=path>/testbed/reproduce_issue.py</parameter>\n"
              "  <parameter=file_text>\n"
              "import sympy\n"
              "from sympy import Symbol\n"
              "x = Symbol('x')\n"
              "print(x + x)\n"
              "  </parameter>\n"
              "</function>"
          ),
          "file_editor",
          {
              "command": "create",
              "path": "/testbed/reproduce_issue.py",
              "file_text": (
                  "import sympy\nfrom sympy import Symbol\nx ="
                  " Symbol('x')\nprint(x + x)"
              ),
          },
      ),
      (
          (
              "<function=file_editor>\n"
              "  <parameter=command>str_replace</parameter>\n"
              "  <parameter=path>/testbed/sympy/core/basic.py</parameter>\n"
              "  <parameter=old_str>\n"
              "    def __eq__(self, other):\n"
              "        return False\n"
              "  </parameter>\n"
              "  <parameter=new_str>\n"
              "    def __eq__(self, other):\n"
              "        if self is other:\n"
              "            return True\n"
              "        return super().__eq__(other)\n"
              "  </parameter>\n"
              "</function>"
          ),
          "file_editor",
          {
              "command": "str_replace",
              "path": "/testbed/sympy/core/basic.py",
              "old_str": "def __eq__(self, other):\n        return False",
              "new_str": (
                  "def __eq__(self, other):\n        if self is other:\n       "
                  "     return True\n        return super().__eq__(other)"
              ),
          },
      ),
      (
          (
              "<function=file_editor>\n "
              " <parameter=command>insert</parameter>\n "
              " <parameter=path>/testbed/sympy/core/basic.py</parameter>\n "
              " <parameter=insert_line>42</parameter>\n  <parameter=new_str>   "
              " # Helper comment\n    pass</parameter>\n</function>"
          ),
          "file_editor",
          {
              "command": "insert",
              "path": "/testbed/sympy/core/basic.py",
              "insert_line": "42",
              "new_str": "# Helper comment\n    pass",
          },
      ),
      (
          (
              "<function=file_editor>\n"
              "  <parameter=command>undo_edit</parameter>\n"
              "  <parameter=path>/testbed/sympy/core/basic.py</parameter>\n"
              "</function>"
          ),
          "file_editor",
          {
              "command": "undo_edit",
              "path": "/testbed/sympy/core/basic.py",
          },
      ),
  )
  def test_deepswe_xml_file_editor_commands(
      self, xml_action, exp_function, exp_args
  ):
    calls = action_converter.extract_tool_calls(
        agent_types.Action(action=xml_action)
    )
    self.assertIsNotNone(calls)
    self.assertLen(calls, 1)
    tc = calls[0]
    self.assertEqual(tc.tool_call_id, "call_1")
    self.assertEqual(tc.function_name, exp_function)
    self.assertEqual(tc.arguments, exp_args)

  def test_deepswe_xml_execute_bash_r2egym_and_sweagent(self):
    # r2egym format uses parameter "cmd"
    r2egym_bash = (
        "<function=execute_bash>\n"
        "  <parameter=cmd>pytest sympy/core/tests/test_basic.py -k"
        " 'test_equality'</parameter>\n"
        "</function>"
    )
    calls1 = action_converter.extract_tool_calls(r2egym_bash)
    self.assertIsNotNone(calls1)
    self.assertEqual(calls1[0].function_name, "execute_bash")
    self.assertEqual(
        calls1[0].arguments,
        {"cmd": "pytest sympy/core/tests/test_basic.py -k 'test_equality'"},
    )

    # sweagent format uses parameter "command"
    sweagent_bash = (
        "<function=execute_bash>\n"
        "  <parameter=command>python -m pytest"
        " tests/test_domain_py.py::test_pymethod_options</parameter>\n"
        "</function>"
    )
    calls2 = action_converter.extract_tool_calls(sweagent_bash)
    self.assertIsNotNone(calls2)
    self.assertEqual(calls2[0].function_name, "execute_bash")
    self.assertEqual(
        calls2[0].arguments,
        {
            "command": (
                "python -m pytest"
                " tests/test_domain_py.py::test_pymethod_options"
            )
        },
    )

    # Complex bash command with heredoc and multiline script
    complex_bash = (
        "<function=execute_bash>\n"
        "  <parameter=cmd>\n"
        "cat << 'EOF' > reproduce.py\n"
        "import sys\n"
        "print('Reproducing...')\n"
        "EOF\n"
        "python reproduce.py\n"
        "  </parameter>\n"
        "</function>"
    )
    calls3 = action_converter.extract_tool_calls(complex_bash)
    self.assertIsNotNone(calls3)
    self.assertEqual(calls3[0].function_name, "execute_bash")
    self.assertEqual(
        calls3[0].arguments,
        {
            "cmd": (
                "cat << 'EOF' > reproduce.py\nimport"
                " sys\nprint('Reproducing...')\nEOF\npython reproduce.py"
            )
        },
    )

  def test_deepswe_xml_search_tool(self):
    search_xml = (
        "<function=search>\n"
        "  <parameter=search_term>class ImmutableDenseNDimArray</parameter>\n"
        "  <parameter=path>/testbed/sympy/tensor</parameter>\n"
        "</function>"
    )
    calls = action_converter.extract_tool_calls(search_xml)
    self.assertIsNotNone(calls)
    self.assertEqual(calls[0].function_name, "search")
    self.assertEqual(
        calls[0].arguments,
        {
            "search_term": "class ImmutableDenseNDimArray",
            "path": "/testbed/sympy/tensor",
        },
    )

  def test_deepswe_xml_finish_and_submit_tools(self):
    finish_xml = (
        "<function=finish>\n"
        "  <parameter=command>submit</parameter>\n"
        "  <parameter=result>Fixed bug in dense_ndim_array.py</parameter>\n"
        "</function>"
    )
    calls1 = action_converter.extract_tool_calls(finish_xml)
    self.assertIsNotNone(calls1)
    self.assertEqual(calls1[0].function_name, "finish")
    self.assertEqual(
        calls1[0].arguments,
        {
            "command": "submit",
            "result": "Fixed bug in dense_ndim_array.py",
        },
    )

    submit_xml = "<function=submit>\n</function>"
    calls2 = action_converter.extract_tool_calls(submit_xml)
    self.assertIsNotNone(calls2)
    self.assertEqual(calls2[0].function_name, "submit")
    self.assertEqual(calls2[0].arguments, {})

  def test_deepswe_xml_str_replace_editor_sweagent(self):
    editor_xml = (
        "<function=str_replace_editor>\n"
        "  <parameter=command>view</parameter>\n"
        "  <parameter=path>/repo/sympy/core/basic.py</parameter>\n"
        "  <parameter=view_range>[50, 100]</parameter>\n"
        "</function>"
    )
    calls = action_converter.extract_tool_calls(editor_xml)
    self.assertIsNotNone(calls)
    self.assertEqual(calls[0].function_name, "str_replace_editor")
    self.assertEqual(
        calls[0].arguments,
        {
            "command": "view",
            "path": "/repo/sympy/core/basic.py",
            "view_range": "[50, 100]",
        },
    )

  def test_deepswe_r2egym_action_objects(self):
    action_obj = MockR2EAction(
        function_name="file_editor",
        parameters={
            "command": "view",
            "path": "./sympy/tensor/array/dense_ndim_array.py",
            "concise": "True",
        },
    )
    # Wrapped in Action
    calls1 = action_converter.extract_tool_calls(
        agent_types.Action(action=action_obj)
    )
    self.assertIsNotNone(calls1)
    self.assertEqual(calls1[0].function_name, "file_editor")
    self.assertEqual(
        calls1[0].arguments,
        {
            "command": "view",
            "path": "./sympy/tensor/array/dense_ndim_array.py",
            "concise": "True",
        },
    )

    # Unwrapped
    calls2 = action_converter.extract_tool_calls(action_obj)
    self.assertIsNotNone(calls2)
    self.assertEqual(calls2[0].function_name, "file_editor")

  def test_deepswe_openai_function_calling_format(self):
    openai_raw_action = {
        "id": "call_swe_oai_1",
        "type": "function",
        "function": {
            "name": "file_editor",
            "arguments": (
                '{"command": "view", "path": "/testbed/sympy/core/basic.py"}'
            ),
        },
    }
    calls = action_converter.extract_tool_calls(openai_raw_action)
    self.assertIsNotNone(calls)
    self.assertEqual(calls[0].tool_call_id, "call_swe_oai_1")
    self.assertEqual(calls[0].function_name, "file_editor")
    self.assertEqual(
        calls[0].arguments,
        {"command": "view", "path": "/testbed/sympy/core/basic.py"},
    )


class LogWarningsTest(parameterized.TestCase):

  def test_normalize_arguments_invalid_json_logs_warning(self):
    with self.assertLogs(level="WARNING") as log_output:
      result = action_converter._normalize_arguments("{invalid json")
      self.assertEqual(result, {})
      self.assertTrue(
          any(
              "Failed to parse arguments JSON string" in msg
              for msg in log_output.output
          )
      )

  def test_parse_xml_function_blocks_invalid_json_logs_warning(self):
    with self.assertLogs(level="WARNING") as log_output:
      calls = action_converter._parse_xml_function_blocks(
          "<function=test>{broken json</function>"
      )
      self.assertIsNotNone(calls)
      self.assertEqual(calls[0].arguments, {})
      self.assertTrue(
          any(
              "Failed to parse XML function body JSON" in msg
              for msg in log_output.output
          )
      )

  def test_parse_string_tool_calls_invalid_json_logs_warning(self):
    with self.assertLogs(level="WARNING") as log_output:
      calls = action_converter._parse_string_tool_calls("{invalid json string")
      self.assertIsNone(calls)
      self.assertTrue(
          any(
              "Failed to parse JSON tool calls string" in msg
              for msg in log_output.output
          )
      )

  def test_parse_string_tool_calls_invalid_tool_call_block_json_logs_warning(
      self,
  ):
    with self.assertLogs(level="WARNING") as log_output:
      calls = action_converter._parse_string_tool_calls(
          "<tool_call>{invalid json</tool_call>"
      )
      self.assertIsNone(calls)
      self.assertTrue(
          any(
              "Failed to parse tool call block JSON" in msg
              for msg in log_output.output
          )
      )


if __name__ == "__main__":
  absltest.main()
