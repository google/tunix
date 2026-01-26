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

import threading
from typing import Any, Optional

import numpy as np
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_template_parser


def pad_prompt_and_completion(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    max_prompt_length: int,
    max_generation_steps: int,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Pads prompt tokens to the left and completion tokens to the right."""

  def left_pad(x, length, pad):
    x = np.asarray(x, dtype=np.int32)
    if x.size >= length:
      return x[-length:]
    pad_part = np.full(length - x.size, pad, np.int32)
    return np.concatenate([pad_part, x], axis=0)

  def right_pad(x, length, pad):
    x = np.asarray(x, dtype=np.int32)
    if x.size >= length:
      return x[:length]
    pad_part = np.full(length - x.size, pad, np.int32)
    return np.concatenate([x, pad_part], axis=0)

  left_padded_prompt_tokens = left_pad(prompt_tokens, max_prompt_length, pad_id)
  right_padded_completion_tokens = right_pad(
      completion_tokens, max_generation_steps, pad_id
  )
  tokens = np.concatenate(
      [left_padded_prompt_tokens, right_padded_completion_tokens], axis=0
  )

  return left_padded_prompt_tokens, right_padded_completion_tokens, tokens


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
  # Some tokenizers (like raw sentencepiece) have an `encode` method that
  # doesn't accept `add_special_tokens`.
  try:
    tokens = tokenizer.encode(msg_text, add_special_tokens=False)
  except TypeError:
    tokens = tokenizer.encode(msg_text)

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
  # For parsers that require preprocessing (e.g., merging system messages),
  # apply it before iterating message by message.
  if hasattr(parser, "preprocess_messages"):
    messages = parser.preprocess_messages(messages)

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


class RolloutSyncLock:
  """A lock object that allows many simultaneous rollouts or one exclusive weight sync.

  Sync preference: Once a weight sync requests the lock, new rollouts will be
  blocked until the sync has acquired and released the lock.
  """

  def __init__(self):
    self._mutex = threading.Lock()
    self._can_rollout = threading.Condition(self._mutex)
    self._can_weight_sync = threading.Condition(self._mutex)
    self._rollouts = 0
    # Number of active weight syncs. This should only be 0 or 1.
    self._weight_syncs = 0
    self._weight_syncs_waiting = 0

  def acquire_rollout(self):
    """Acquire a rollout lock."""
    with self._mutex:
      while self._weight_syncs > 0 or self._weight_syncs_waiting > 0:
        self._can_rollout.wait()
      self._rollouts += 1

  def release_rollout(self):
    """Release a rollout lock."""
    with self._mutex:
      self._rollouts -= 1
      if self._rollouts == 0 and self._weight_syncs_waiting > 0:
        self._can_weight_sync.notify()

  def acquire_weight_sync(self):
    """Acquire a weight sync lock."""
    with self._mutex:
      while self._rollouts > 0 or self._weight_syncs > 0:
        self._weight_syncs_waiting += 1
        self._can_weight_sync.wait()
        self._weight_syncs_waiting -= 1
      self._weight_syncs += 1

  def release_weight_sync(self):
    """Release a weight sync lock."""
    with self._mutex:
      self._weight_syncs -= 1
      # Notify waiting sync first, then all rollouts
      self._can_weight_sync.notify()
      self._can_rollout.notify_all()
