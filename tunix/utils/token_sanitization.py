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

"""Safety utilities for RL agents."""

import re
from typing import Iterable

_CONTROL_TOKENS = [
    '<start_of_turn>',
    '<end_of_turn>',
    '<bos>',
    '<eos>',
    '<|begin_of_text|>',
    '<|end_of_text|>',
    '<|start_header_id|>',
    '<|end_header_id|>',
    '<|eot_id|>',
    '<|im_start|>',
    '<|im_end|>',
]

_CONTROL_TOKENS_RE = re.compile('|'.join(map(re.escape, _CONTROL_TOKENS)))


def sanitize_control_tokens(
    content: str,
    extra_tokens: Iterable[str] | None = None,
    include_default: bool = True,
) -> str:
  """Sanitize control tokens from the content.

  Args:
    content: The content to sanitize.
    extra_tokens: Additional tokens to sanitize.
    include_default: Whether to include the default control tokens.

  Returns:
    The sanitized content.
  """
  if not content:
    return content

  if include_default and not extra_tokens:
    regex = _CONTROL_TOKENS_RE
  else:
    all_tokens = set()
    if include_default:
      all_tokens.update(_CONTROL_TOKENS)
    if extra_tokens:
      all_tokens.update(t for t in extra_tokens if t)

    if not all_tokens:
      return content

    regex = re.compile('|'.join(map(re.escape, sorted(all_tokens))))

  # Strip known model control tokens to prevent prompt injection.
  # We use a loop to handle cases where tokens are nested to bypass sanitization.
  sanitized = content
  while True:
    new_content = regex.sub('', sanitized)
    if new_content == sanitized:
      break
    sanitized = new_content
  return sanitized
