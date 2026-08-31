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

"""Tests for the trajectory collector."""

from absl.testing import absltest
from tunix.experimental.rollout import collector


class _RecordingParser:

  def __init__(self):
    self.calls = []

  def parse(self, msgs, add_generation_prompt=False, is_first_msg=False):
    self.calls.append((msgs, add_generation_prompt, is_first_msg))
    return "PARSED"


class BuildPromptTest(absltest.TestCase):

  def test_chat_messages_are_parsed(self):
    parser = _RecordingParser()
    msgs = [{"role": "user", "content": "hi"}]
    self.assertEqual(collector._build_prompt(parser, msgs), "PARSED")
    self.assertEqual(parser.calls, [(msgs, True, True)])

  def test_string_prompt_passes_through(self):
    parser = _RecordingParser()
    self.assertEqual(collector._build_prompt(parser, "raw"), "raw")
    self.assertEmpty(parser.calls)

  def test_no_parser_passes_through(self):
    msgs = [{"role": "user", "content": "hi"}]
    self.assertIs(collector._build_prompt(None, msgs), msgs)


if __name__ == "__main__":
  absltest.main()