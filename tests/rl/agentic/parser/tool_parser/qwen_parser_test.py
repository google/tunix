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

from absl.testing import absltest
from absl.testing import parameterized
from tunix.rl.agentic.parser.tool_parser import qwen_parser


def _call(body: str) -> str:
  return f"<tool_call>{body}</tool_call>"


CALC = _call('{"name": "calc", "arguments": {"a": 1}}')
ECHO = _call('{"name": "echo", "arguments": {"text": "hi"}}')


class QwenToolParserTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.parser = qwen_parser.QwenToolParser()

  def test_single_valid_call(self):
    self.assertEqual(
        self.parser.parse_qwen_tool_calls(CALC),
        [{"name": "calc", "arguments": {"a": 1}}],
    )

  def test_multiple_valid_calls_keep_order(self):
    self.assertEqual(
        self.parser.parse_qwen_tool_calls(f"first {CALC}\n{ECHO} done"),
        [
            {"name": "calc", "arguments": {"a": 1}},
            {"name": "echo", "arguments": {"text": "hi"}},
        ],
    )

  def test_invalid_json_is_skipped_and_next_call_kept(self):
    self.assertEqual(
        self.parser.parse_qwen_tool_calls(_call("{not json") + CALC),
        [{"name": "calc", "arguments": {"a": 1}}],
    )

  @parameterized.named_parameters(
      ("missing_arguments", '{"name": "x"}'),
      ("missing_name", '{"arguments": {"a": 1}}'),
      ("list", "[1, 2]"),
      ("string", '"hi"'),
      ("number", "42"),
      ("null", "null"),
      ("name_not_a_string", '{"name": 1, "arguments": {}}'),
      ("arguments_not_a_dict", '{"name": "x", "arguments": "{}"}'),
  )
  def test_malformed_call_is_skipped_without_affecting_valid_calls(self, body):
    text = CALC + _call(body) + ECHO

    self.assertEqual(
        self.parser.parse_qwen_tool_calls(text),
        [
            {"name": "calc", "arguments": {"a": 1}},
            {"name": "echo", "arguments": {"text": "hi"}},
        ],
    )
    self.assertEqual(self.parser.parse_qwen_tool_calls(_call(body)), [])

  def test_text_without_tool_call_returns_empty_list(self):
    self.assertEqual(self.parser.parse_qwen_tool_calls("just some text"), [])

  def test_unclosed_tool_call_returns_only_what_came_before(self):
    text = CALC + '<tool_call>{"name": "echo", "arguments": {}}'
    self.assertEqual(
        self.parser.parse_qwen_tool_calls(text),
        [{"name": "calc", "arguments": {"a": 1}}],
    )

  def test_parse_returns_tool_calls_for_valid_calls_only(self):
    tool_calls = self.parser.parse(CALC + _call('{"name": "x"}') + ECHO)

    self.assertEqual([c.name for c in tool_calls], ["calc", "echo"])
    self.assertEqual(tool_calls[0].arguments, {"a": 1})


if __name__ == "__main__":
  absltest.main()
