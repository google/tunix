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

"""Utility functions for agentic models."""

from typing import Any, Optional

from tunix.rl.agentic.parser.chat_template_parser import parser as chat_template_parser


def get_recent_assistant_user_messages(
    chat_completions_messages: list[dict[str, Any]]
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
  """Extracts the most recent assistant message and environment messages (user/tool) from a chat completions list.

  Args:
      chat_completions_messages (list[dict]): List of message dictionaries from
        chat completions.

  Returns:
      tuple[dict, list[dict]]: A tuple containing:
          - The most recent assistant message (or None if not found)
          - A list of environment messages (user/tool) that occurred after the
          last assistant message, in chronological order.
  """
  # Loop backwards to get the last assistant message and environment messages
  env_messages = []
  assistant_message = None
  for message in reversed(chat_completions_messages):
    role = message.get("role", None)
    if role == "assistant":
      assistant_message = message
      break
    elif role in ["user", "tool"] and assistant_message is None:
      env_messages.append(message)
  # Reverse the env_messages to maintain chronological order
  env_messages = list(reversed(env_messages))

  return assistant_message, env_messages


def convert_single_message(
    msg: dict[str, str],
    tokenizer: Any,
    parser: chat_template_parser.BaseChatTemplateParser,
    is_first: bool = False,
    is_generation: bool = False,
) -> tuple[list[int], list[int]]:
  """Converts a single message to tokens and a mask.

  Args:
    msg: The message to convert.
    tokenizer: The tokenizer to use.
    parser: The chat template parser.
    is_first: Whether this is the first message in a sequence.
    is_generation: Whether this message is for generation.

  Returns:
    A tuple containing (tokens, mask).
  """
  # Parse message to text
  msg_text = parser.parse(
      messages=[msg],
      add_generation_prompt=is_generation,
      is_first_msg=is_first,
  )

  # Remove assistant token if present (it's in the prior generation prompt).
  if msg["role"] == "assistant" and hasattr(parser, "assistant_token"):
    assistant_token = parser.assistant_token
    if msg_text.startswith(assistant_token):
      msg_text = msg_text[len(assistant_token) :]

  # Tokenize
  tokens = tokenizer.encode(msg_text, add_special_tokens=False)

  # Create mask (1 for assistant, 0 for others)
  mask_value = 1 if msg["role"] == "assistant" else 0
  masks = [mask_value] * len(tokens)

  return tokens, masks


def tokenize_and_generate_masks(
    messages: list[dict[str, str]],
    tokenizer: Any,
    parser: chat_template_parser.BaseChatTemplateParser,
    contains_first_msg: bool = False,
    contains_generation_msg: bool = False,
) -> tuple[list[int], list[int]]:
  """Converts multiple messages to tokens and masks.

  Args:
    messages: The messages to convert.
    tokenizer: The tokenizer to use.
    parser: The chat template parser.
    contains_first_msg: Whether the first message in `messages` is the absolute
      first message of the conversation. This is used by the parser to add
      special beginning-of-sequence tokens.
    contains_generation_msg: Whether to add a generation prompt after the last
      message in `messages`. This signals the model to start generating a
      response (e.g., by adding an assistant role token).

  Returns:
    A tuple containing (all_tokens, all_masks).
  """
  all_tokens = []
  all_masks = []

  # Process each message
  for i, msg in enumerate(messages):
    is_first = contains_first_msg and i == 0
    is_generation = contains_generation_msg and i == len(messages) - 1
    tokens, masks = convert_single_message(
        msg, tokenizer, parser, is_first, is_generation
    )
    all_tokens.extend(tokens)
    all_masks.extend(masks)

  return all_tokens, all_masks
