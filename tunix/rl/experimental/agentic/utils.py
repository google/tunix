"""Utility functions for agentic models."""

from typing import Any, Dict, List, Tuple


def get_recent_assistant_user_messages(chat_completions_messages):
  """Extracts the most recent assistant message and environment messages (user/tool) from a chat completions list.

  Args:
      chat_completions_messages (List[Dict]): List of message dictionaries from
        chat completions.

  Returns:
      Tuple[Dict, List[Dict]]: A tuple containing:
          - The most recent assistant message (or None if not found)
          - A list of environment messages (user/tool) that occurred after the
          last assistant message,
            in chronological order.
  """
  # Loop backwards to get the last assistant message and environment messages
  env_messages = []
  assistant_message = None
  seen_assistant_message = False
  for message in reversed(chat_completions_messages):
    role = message.get("role", None)
    if role == "assistant":
      if assistant_message:
        break
      seen_assistant_message = True
      assistant_message = message
    elif role in ["user", "tool"] and not seen_assistant_message:
      env_messages.append(message)
  # Reverse the env_messages to maintain chronological order
  env_messages = list(reversed(env_messages))

  return assistant_message, env_messages


def convert_messages_to_tokens_and_masks(
    messages: List[Dict[str, str]],
    tokenizer: Any,
    parser,
    contains_first_msg: bool = False,
    contains_generation_msg: bool = False,
) -> Tuple[List[int], List[int]]:
  """Converts multiple messages to tokens and masks.

  Args:
      messages: The messages to convert
      tokenizer: The tokenizer to use
      parser: The chat template parser
      contains_first_msg: Whether the first message is special
      contains_generation_msg: Whether the last message needs generation prompt

  Returns:
      Tuple containing (all_tokens, all_masks)
  """
  all_tokens = []
  all_masks = []

  def convert_single_message(
      msg: Dict[str, str], is_first: bool = False, is_generation: bool = False
  ) -> Tuple[List[int], List[int]]:
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

  # Process each message
  for i, msg in enumerate(messages):
    is_first = contains_first_msg and i == 0
    is_generation = contains_generation_msg and i == len(messages) - 1
    tokens, masks = convert_single_message(msg, is_first, is_generation)
    all_tokens.extend(tokens)
    all_masks.extend(masks)

  return all_tokens, all_masks
