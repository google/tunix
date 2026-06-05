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
from tunix.utils import token_sanitization


class TokenSanitizationTest(parameterized.TestCase):

  @parameterized.parameters(
      ('', ''),
      (None, None),
      ('hello world', 'hello world'),
      ('<start_of_turn>user<end_of_turn>', 'user'),
      ('<bos>Hello<eos>', 'Hello'),
      ('<|begin_of_text|>System message<|end_of_text|>', 'System message'),
      ('<|start_header_id|>assistant<|end_header_id|><|eot_id|>', 'assistant'),
      ('<|im_start|>user\nHello!<|im_end|>', 'user\nHello!'),
      ('Nested <start_of_turn><bos>tokens<eos><end_of_turn>', 'Nested tokens'),
      ('<start_<start_of_turn>of_turn>', ''),
      ('Recursive <bos<bos>>', 'Recursive '),
  )
  def test_sanitize_control_tokens(self, content, expected):
    self.assertEqual(
        token_sanitization.sanitize_control_tokens(content), expected
    )

  def test_sanitize_control_tokens_with_extra(self):
    content = '[CUSTOM]user\nHello![/CUSTOM]'
    extra_tokens = ['[CUSTOM]', '[/CUSTOM]']
    expected = 'user\nHello!'
    self.assertEqual(
        token_sanitization.sanitize_control_tokens(
            content, extra_tokens=extra_tokens
        ),
        expected,
    )

  def test_sanitize_control_tokens_recursive_with_extra(self):
    content = '[CUSTOM[CUSTOM]]nested[/CUSTOM]'
    extra_tokens = ['[CUSTOM]', '[/CUSTOM]']
    expected = 'nested'
    self.assertEqual(
        token_sanitization.sanitize_control_tokens(
            content, extra_tokens=extra_tokens
        ),
        expected,
    )

  def test_sanitize_control_tokens_with_empty_extra(self):
    content = 'hello world'
    extra_tokens = ['', None]
    expected = 'hello world'
    self.assertEqual(
        token_sanitization.sanitize_control_tokens(
            content, extra_tokens=extra_tokens
        ),
        expected,
    )

  def test_sanitize_control_tokens_no_default(self):
    content = '<bos>[CUSTOM]hello[/CUSTOM]<eos>'
    extra_tokens = ['[CUSTOM]', '[/CUSTOM]']
    # <bos> and <eos> should remain
    expected = '<bos>hello<eos>'
    self.assertEqual(
        token_sanitization.sanitize_control_tokens(
            content, extra_tokens=extra_tokens, include_default=False
        ),
        expected,
    )

  def test_sanitize_control_tokens_no_default_no_extra(self):
    content = '<bos>hello<eos>'
    expected = '<bos>hello<eos>'
    self.assertEqual(
        token_sanitization.sanitize_control_tokens(
            content, include_default=False
        ),
        expected,
    )


if __name__ == '__main__':
  absltest.main()
