# Copyright 2025 Google LLC
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

"""Tests for chat_template_parser."""

from unittest import mock

from absl.testing import absltest

from tunix.rl.agentic.parser.chat_template_parser import parser


class DefaultChatTemplateParserTest(absltest.TestCase):

  def test_parse_uses_tokenizer_template(self):
    mock_tokenizer = mock.Mock()
    messages = [{'role': 'user', 'content': 'Hello'}]

    p = parser.DefaultChatTemplateParser(mock_tokenizer)
    p.parse(messages, add_generation_prompt=True)

    mock_tokenizer.apply_chat_template.assert_called_once_with(
        messages, tokenize=False, add_generation_prompt=True
    )


class QwenChatTemplateParserTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_tokenizer = mock.Mock()
    self.mock_tokenizer.bos_token = '<bos>'
    self.mock_tokenizer.eos_token = '<eos>'

  def test_parse_with_system_message(self):
    p = parser.QwenChatTemplateParser(self.mock_tokenizer)
    messages = [
        {'role': 'system', 'content': 'You are Qwen.'},
        {'role': 'user', 'content': 'Hello'},
    ]
    result = p.parse(messages)
    expected = ('<|im_start|>system\nYou are Qwen.<|im_end|>\n'
                '<|im_start|>user\nHello<|im_end|>\n')
    self.assertEqual(result, expected)

  def test_parse_without_system_message_and_is_first_msg(self):
    p = parser.QwenChatTemplateParser(self.mock_tokenizer)
    messages = [{'role': 'user', 'content': 'Hello'}]
    result = p.parse(messages, is_first_msg=True)
    expected = ('<|im_start|>system\n'
                'You are Qwen, created by Alibaba Cloud. You are a helpful'
                ' assistant.<|im_end|>\n'
                '<|im_start|>user\nHello<|im_end|>\n')
    self.assertEqual(result, expected)

  def test_parse_with_add_generation_prompt(self):
    p = parser.QwenChatTemplateParser(self.mock_tokenizer)
    messages = [{'role': 'user', 'content': 'Hello'}]
    result = p.parse(messages, add_generation_prompt=True)
    expected = ('<|im_start|>user\nHello<|im_end|>\n'
                '<|im_start|>assistant\n')
    self.assertEqual(result, expected)

  def test_parse_with_tool_message(self):
    p = parser.QwenChatTemplateParser(self.mock_tokenizer)
    messages = [{'role': 'tool', 'content': 'Tool output'}]
    result = p.parse(messages)
    expected = ('<|im_start|>user\n'
                '<tool_response>\nTool output\n</tool_response>'
                '<|im_end|>\n')
    self.assertEqual(result, expected)

  def test_parse_with_disable_thinking(self):
    p = parser.QwenChatTemplateParser(
        self.mock_tokenizer, disable_thinking=True
    )
    messages = [{'role': 'assistant', 'content': 'Thinking...'}]
    result = p.parse(messages, add_generation_prompt=True)
    expected = (
        '<|im_start|>assistant\n<think>\\n\\n</think>\\n\\nThinking...<|im_end|>\n'
        '<|im_start|>assistant\n<think>\\n\\n</think>\\n\\n'
    )
    self.assertEqual(result, expected)


class LlamaChatTemplateParserTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_tokenizer = mock.Mock()

  def test_parse_with_system_message_and_is_first_msg(self):
    p = parser.LlamaChatTemplateParser(self.mock_tokenizer)
    messages = [
        {'role': 'system', 'content': 'You are Llama.'},
        {'role': 'user', 'content': 'Hello'},
    ]
    result = p.parse(messages, is_first_msg=True)
    expected = (
        '<|begin_of_text|>'
        '<|start_header_id|>system<|end_header_id|>\n\nYou are Llama.<|eot_id|>'
        '<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>'
    )
    self.assertEqual(result, expected)

  def test_parse_without_is_first_msg(self):
    p = parser.LlamaChatTemplateParser(self.mock_tokenizer)
    messages = [{'role': 'user', 'content': 'Hello'}]
    result = p.parse(messages, is_first_msg=False)
    expected = '<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>'
    self.assertEqual(result, expected)

  def test_parse_with_add_generation_prompt(self):
    p = parser.LlamaChatTemplateParser(self.mock_tokenizer)
    messages = [{'role': 'user', 'content': 'Hello'}]
    result = p.parse(messages, add_generation_prompt=True)
    expected = ('<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>'
                '<|start_header_id|>assistant<|end_header_id|>\n\n')
    self.assertEqual(result, expected)

  def test_parse_with_tool_message(self):
    p = parser.LlamaChatTemplateParser(self.mock_tokenizer)
    messages = [{'role': 'tool', 'content': 'Tool output'}]
    result = p.parse(messages)
    expected = ('<|start_header_id|>user<|end_header_id|>\n\n'
                '<|start_header_id|>tool_response<|end_header_id|>\n\n'
                'Tool output'
                '<|eot_id|>'
                '<|eot_id|>')
    self.assertEqual(result, expected)

  def test_parse_with_unsupported_role_raises_error(self):
    p = parser.LlamaChatTemplateParser(self.mock_tokenizer)
    messages = [{'role': 'unsupported', 'content': 'Hello'}]
    with self.assertRaises(NotImplementedError):
      p.parse(messages)


class GemmaChatTemplateParserTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_tokenizer = mock.Mock()
    self.parser = parser.GemmaChatTemplateParser(self.mock_tokenizer)

  def test_preprocess_with_system_and_user(self):
    messages = [
        {'role': 'system', 'content': 'System prompt.'},
        {'role': 'user', 'content': 'User prompt.'},
    ]
    processed = self.parser.preprocess_messages(messages)
    self.assertEqual(len(processed), 1)
    self.assertEqual(processed[0]['role'], 'user')
    self.assertEqual(
        processed[0]['content'], 'System prompt.\nUser prompt.'
    )

  def test_preprocess_with_system_only(self):
    messages = [{'role': 'system', 'content': 'System prompt.'}]
    processed = self.parser.preprocess_messages(messages)
    self.assertEqual(len(processed), 1)
    self.assertEqual(processed[0]['role'], 'user')
    self.assertEqual(processed[0]['content'], 'System prompt.')

  def test_preprocess_with_no_system_message(self):
    messages = [
        {'role': 'user', 'content': 'User prompt.'},
        {'role': 'assistant', 'content': 'Assistant response.'},
    ]
    processed = self.parser.preprocess_messages(messages)
    self.assertEqual(processed, messages)

  def test_parse_raises_error_for_direct_system_message(self):
    message = {'role': 'system', 'content': 'System prompt.'}
    with self.assertRaises(ValueError):
      # Directly test _parse_message to ensure it raises for system messages
      # that might bypass the preprocessing in parse().
      self.parser._parse_message(message)


if __name__ == '__main__':
  absltest.main()
