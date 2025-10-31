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

"""Tests for agentic utility functions."""

import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from tunix.rl.agentic import utils


class RecentMessagesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'empty_list',
          [],
          (None, []),
      ),
      (
          'no_assistant_message',
          [
              {'role': 'user', 'content': 'Hello'},
              {'role': 'tool', 'content': 'Tool output'},
          ],
          (None, [
              {'role': 'user', 'content': 'Hello'},
              {'role': 'tool', 'content': 'Tool output'},
          ]),
      ),
      (
          'only_assistant_message',
          [{'role': 'assistant', 'content': 'Hi there'}],
          ({'role': 'assistant', 'content': 'Hi there'}, []),
      ),
      (
          'multiple_turns',
          [
              {'role': 'user', 'content': 'Hi'},
              {'role': 'assistant', 'content': 'Hello!'},
              {'role': 'user', 'content': 'How are you?'},
              {'role': 'assistant', 'content': 'I am fine.'},
              {'role': 'user', 'content': 'Good.'},
          ],
          (
              {'role': 'assistant', 'content': 'I am fine.'},
              [{'role': 'user', 'content': 'Good.'}],
          ),
      ),
      (
          'multiple_env_messages_after_last_assistant',
          [
              {'role': 'assistant', 'content': 'Thinking...'},
              {'role': 'user', 'content': 'Query'},
              {'role': 'tool', 'content': 'Result'},
              {'role': 'user', 'content': 'Another query'},
          ],
          (
              {'role': 'assistant', 'content': 'Thinking...'},
              [
                  {'role': 'user', 'content': 'Query'},
                  {'role': 'tool', 'content': 'Result'},
                  {'role': 'user', 'content': 'Another query'},
              ],
          ),
      ),
      (
          'no_env_messages_after_last_assistant',
          [
              {'role': 'user', 'content': 'Hi'},
              {'role': 'assistant', 'content': 'Hello!'},
          ],
          ({'role': 'assistant', 'content': 'Hello!'}, []),
      ),
  )
  def test_get_recent_assistant_user_messages(
      self, messages, expected_output
  ):
    self.assertEqual(
        utils.get_recent_assistant_user_messages(messages), expected_output
    )


class MessagesToTokensTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_tokenizer = mock.Mock()
    self.mock_parser = mock.Mock(spec=['parse', 'assistant_token'])
    self.mock_parser.assistant_token = ''

    # Simple tokenizer mock: returns list of char codes
    self.mock_tokenizer.encode.side_effect = lambda s, add_special_tokens: [
        ord(c) for c in s
    ]

    # Simple parser mock: returns a formatted string
    def parse_side_effect(messages, add_generation_prompt, is_first_msg):
      msg = messages[0]
      text = f'role:{msg["role"]} content:{msg["content"]}'
      if is_first_msg:
        text = f'first:{text}'
      if add_generation_prompt:
        text = f'{text}:gen'
      return text

    self.mock_parser.parse.side_effect = parse_side_effect

  def test_empty_messages(self):
    tokens, masks = utils.tokenize_and_generate_masks(
        messages=[],
        tokenizer=self.mock_tokenizer,
        parser=self.mock_parser,
    )
    self.assertEqual(tokens, [])
    self.assertEqual(masks, [])

  def test_single_user_message(self):
    messages = [{'role': 'user', 'content': 'hi'}]
    expected_text = 'role:user content:hi'
    expected_tokens = [ord(c) for c in expected_text]

    tokens, masks = utils.tokenize_and_generate_masks(
        messages=messages,
        tokenizer=self.mock_tokenizer,
        parser=self.mock_parser,
    )

    self.assertEqual(tokens, expected_tokens)
    self.assertEqual(masks, [0] * len(expected_tokens))
    self.mock_parser.parse.assert_called_once_with(
        messages=[messages[0]], add_generation_prompt=False, is_first_msg=False
    )

  def test_single_assistant_message(self):
    messages = [{'role': 'assistant', 'content': 'hello'}]
    expected_text = 'role:assistant content:hello'
    expected_tokens = [ord(c) for c in expected_text]

    tokens, masks = utils.tokenize_and_generate_masks(
        messages=messages,
        tokenizer=self.mock_tokenizer,
        parser=self.mock_parser,
    )

    self.assertEqual(tokens, expected_tokens)
    self.assertEqual(masks, [1] * len(expected_tokens))
    self.mock_parser.parse.assert_called_once_with(
        messages=[messages[0]], add_generation_prompt=False, is_first_msg=False
    )

  def test_multiple_messages(self):
    messages = [
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'hello'},
    ]
    text1 = 'role:user content:hi'
    tokens1 = [ord(c) for c in text1]
    masks1 = [0] * len(tokens1)

    text2 = 'role:assistant content:hello'
    tokens2 = [ord(c) for c in text2]
    masks2 = [1] * len(tokens2)

    tokens, masks = utils.tokenize_and_generate_masks(
        messages=messages,
        tokenizer=self.mock_tokenizer,
        parser=self.mock_parser,
    )

    self.assertEqual(tokens, tokens1 + tokens2)
    self.assertEqual(masks, masks1 + masks2)
    self.assertEqual(self.mock_parser.parse.call_count, 2)

  def test_with_contains_first_msg(self):
    messages = [
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'hello'},
    ]
    utils.tokenize_and_generate_masks(
        messages=messages,
        tokenizer=self.mock_tokenizer,
        parser=self.mock_parser,
        contains_first_msg=True,
    )
    calls = self.mock_parser.parse.call_args_list
    self.assertEqual(len(calls), 2)
    self.assertEqual(
        calls[0],
        mock.call(
            messages=[messages[0]],
            add_generation_prompt=False,
            is_first_msg=True,
        ),
    )
    self.assertEqual(
        calls[1],
        mock.call(
            messages=[messages[1]],
            add_generation_prompt=False,
            is_first_msg=False,
        ),
    )

  def test_with_contains_generation_msg(self):
    messages = [
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'hello'},
    ]
    utils.tokenize_and_generate_masks(
        messages=messages,
        tokenizer=self.mock_tokenizer,
        parser=self.mock_parser,
        contains_generation_msg=True,
    )
    calls = self.mock_parser.parse.call_args_list
    self.assertEqual(len(calls), 2)
    self.assertEqual(
        calls[0],
        mock.call(
            messages=[messages[0]],
            add_generation_prompt=False,
            is_first_msg=False,
        ),
    )
    self.assertEqual(
        calls[1],
        mock.call(
            messages=[messages[1]],
            add_generation_prompt=True,
            is_first_msg=False,
        ),
    )

  def test_with_assistant_token_stripping(self):
    self.mock_parser.assistant_token = '<|asst|>'

    def parse_side_effect_with_token(messages, **kwargs):
      del kwargs
      msg = messages[0]
      text = f'content:{msg["content"]}'
      if msg['role'] == 'assistant':
        text = f'<|asst|>{text}'
      return text

    self.mock_parser.parse.side_effect = parse_side_effect_with_token

    messages = [{'role': 'assistant', 'content': 'hi'}]
    expected_text = 'content:hi'  # The token should be stripped
    expected_tokens = [ord(c) for c in expected_text]

    tokens, masks = utils.tokenize_and_generate_masks(
        messages=messages,
        tokenizer=self.mock_tokenizer,
        parser=self.mock_parser,
    )

    self.assertEqual(tokens, expected_tokens)
    self.assertEqual(masks, [1] * len(expected_tokens))
    self.mock_tokenizer.encode.assert_called_once_with(
        expected_text, add_special_tokens=False
    )

  def test_with_preprocess_messages(self):
    mock_parser_with_preprocess = mock.Mock(
        spec=['parse', 'assistant_token', 'preprocess_messages']
    )
    mock_parser_with_preprocess.assistant_token = ''
    mock_parser_with_preprocess.parse.side_effect = (
        self.mock_parser.parse.side_effect
    )
    messages = [
        {'role': 'system', 'content': 'system'},
        {'role': 'user', 'content': 'user'},
    ]
    processed_messages = [{'role': 'user', 'content': 'processed'}]
    mock_parser_with_preprocess.preprocess_messages.return_value = (
        processed_messages
    )

    utils.tokenize_and_generate_masks(
        messages=messages,
        tokenizer=self.mock_tokenizer,
        parser=mock_parser_with_preprocess,
    )

    mock_parser_with_preprocess.preprocess_messages.assert_called_once_with(
        messages
    )
    # Check that parse was called on the preprocessed message.
    calls = mock_parser_with_preprocess.parse.call_args_list
    self.assertEqual(len(calls), 1)
    self.assertEqual(
        calls[0],
        mock.call(
            messages=[processed_messages[0]],
            add_generation_prompt=False,
            is_first_msg=False,
        ),
    )

  def test_with_tokenizer_without_add_special_tokens_arg(self):
    mock_tokenizer_simple = mock.Mock(spec=['encode'])
    # Simple tokenizer mock: returns list of char codes
    mock_tokenizer_simple.encode.side_effect = lambda s: [ord(c) for c in s]

    messages = [{'role': 'user', 'content': 'hi'}]
    expected_text = 'role:user content:hi'
    expected_tokens = [ord(c) for c in expected_text]

    tokens, masks = utils.tokenize_and_generate_masks(
        messages=messages,
        tokenizer=mock_tokenizer_simple,
        parser=self.mock_parser,
    )

    self.assertEqual(tokens, expected_tokens)
    self.assertEqual(masks, [0] * len(expected_tokens))
    mock_tokenizer_simple.encode.assert_called_with(expected_text)


if __name__ == '__main__':
  absltest.main()
