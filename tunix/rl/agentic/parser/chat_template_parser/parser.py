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

"""Base classes and factory for parsing chat messages into model-specific formats."""

import abc
import dataclasses
from typing import Dict, List


dataclass = dataclasses.dataclass
abstractmethod = abc.abstractmethod
ABC = abc.ABC


@dataclass
class TokenConfig:
  """Token configuration for different chat templates."""

  bos_token: str = ""
  eos_token: str = ""
  eot_token: str = ""
  system_token: str = ""
  user_token: str = ""
  assistant_token: str = ""
  tool_start_token: str = ""
  tool_end_token: str = ""
  tool_response_start_token: str = ""
  tool_response_end_token: str = ""


class BaseChatTemplateParser(ABC):
  """Abstract base class for chat template parsers."""

  def __init__(self, tokenizer, disable_thinking: bool = False):
    self.tokenizer = tokenizer
    self.disable_thinking = disable_thinking
    self.tokens = self._init_tokens()
    self.generation_prompt = self._init_generation_prompt()

  @abstractmethod
  def _init_tokens(self) -> TokenConfig:
    """Initialize token configuration."""
    pass

  @abstractmethod
  def _init_generation_prompt(self) -> str:
    """Initialize generation prompt."""
    pass

  def parse(
      self,
      messages: List[Dict[str, str]],
      add_generation_prompt: bool = False,
      is_first_msg: bool = False,
  ) -> str:
    """Parse messages into chat template format."""
    result = ""

    if is_first_msg:
      result += self._handle_first_message(messages)

    for message in messages:
      result += self._parse_message(message)

    if add_generation_prompt:
      result += self.generation_prompt

    return result

  def _handle_first_message(self, messages: List[Dict[str, str]]) -> str:
    """Handle special logic for first message."""
    del messages  # Unused in the base implementation.
    return self.tokens.bos_token

  def _parse_message(self, message: Dict[str, str]) -> str:
    """Parse a single message based on its role."""
    role = message["role"]

    parser_map = {
        "system": self._parse_system,
        "user": self._parse_user,
        "assistant": self._parse_assistant,
        "tool": self._parse_tool,
    }

    if role not in parser_map:
      raise NotImplementedError(f"Unsupported message role: {role}")

    return parser_map[role](message)

  def _parse_system(self, message: Dict[str, str]) -> str:
    return self.tokens.system_token + message["content"] + self.tokens.eot_token

  def _parse_user(self, message: Dict[str, str]) -> str:
    return self.tokens.user_token + message["content"] + self.tokens.eot_token

  def _parse_assistant(self, message: Dict[str, str]) -> str:
    return (
        self.tokens.assistant_token + message["content"] + self.tokens.eot_token
    )

  def _parse_tool(self, message: Dict[str, str]) -> str:
    return (
        self.tokens.user_token
        + self.tokens.tool_response_start_token
        + message["content"]
        + self.tokens.tool_response_end_token
        + self.tokens.eot_token
    )


class DefaultChatTemplateParser(BaseChatTemplateParser):
  """Default parser using tokenizer's built-in chat template."""

  def _init_tokens(self) -> TokenConfig:
    return TokenConfig()

  def _init_generation_prompt(self) -> str:
    return ""

  def parse(
      self,
      messages: List[Dict[str, str]],
      add_generation_prompt: bool = False,
      is_first_msg: bool = False,
  ) -> str:
    return self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


class QwenChatTemplateParser(BaseChatTemplateParser):
  """Parser for Qwen models."""

  def _init_tokens(self) -> TokenConfig:
    return TokenConfig(
        bos_token=self.tokenizer.bos_token,
        eos_token=self.tokenizer.eos_token,
        eot_token="<|im_end|>\n",
        system_token="<|im_start|>system\n",
        user_token="<|im_start|>user\n",
        assistant_token=self._get_assistant_token(),
        tool_start_token="\n<tool_call>\n",
        tool_end_token="\n</tool_call>",
        tool_response_start_token="<tool_response>\n",
        tool_response_end_token="\n</tool_response>",
    )

  def _get_assistant_token(self) -> str:
    token = "<|im_start|>assistant\n"
    if self.disable_thinking:
      token += "<think>\\n\\n</think>\\n\\n"
    return token

  def _init_generation_prompt(self) -> str:
    return self.tokens.assistant_token

  def _handle_first_message(self, messages: List[Dict[str, str]]) -> str:
    """Add default system message if first message is not system."""
    if messages[0]["role"] != "system":
      return self._parse_system({
          "role": "system",
          "content": (
              "You are Qwen, created by Alibaba Cloud. You are a helpful"
              " assistant."
          ),
      })
    return ""


class LlamaChatTemplateParser(BaseChatTemplateParser):
  """Parser for Llama models."""

  def _init_tokens(self) -> TokenConfig:
    return TokenConfig(
        bos_token="<|begin_of_text|>",
        eot_token="<|eot_id|>",
        system_token="<|start_header_id|>system<|end_header_id|>\n\n",
        user_token="<|start_header_id|>user<|end_header_id|>\n\n",
        assistant_token="<|start_header_id|>assistant<|end_header_id|>\n\n",
        tool_start_token="<|start_header_id|>tool<|end_header_id|>\n\n",
        tool_end_token="<|eot_id|>",
        tool_response_start_token=(
            "<|start_header_id|>tool_response<|end_header_id|>\n\n"
        ),
        tool_response_end_token="<|eot_id|>",
    )

  def _init_generation_prompt(self) -> str:
    return self.tokens.assistant_token


class GemmaChatTemplateParser(BaseChatTemplateParser):
  """Parser for Gemma models."""

  def _init_tokens(self) -> TokenConfig:
    return TokenConfig(
        bos_token="<bos>",
        eot_token="<end_of_turn>\n",
        user_token="<start_of_turn>user\n",
        assistant_token="<start_of_turn>model\n",
    )

  def _parse_assistant(self, message: Dict[str, str]) -> str:
    return self.tokens.assistant_token + message["content"]

  def _parse_system(self, message: Dict[str, str]) -> str:
    # This should not be called if parse() is used, as it handles the system
    # prompt by merging it. Raise error for unexpected system messages.
    del message  # Unused.
    raise ValueError(
        "Gemma models do not support system messages directly. The system"
        " prompt should be the first message and is handled by merging with"
        " the first user message."
    )

  def preprocess_messages(
      self, messages: List[Dict[str, str]]
  ) -> List[Dict[str, str]]:
    """Preprocesses messages, merging system prompt into the first user prompt."""
    processed_messages = list(messages)
    if processed_messages and processed_messages[0]["role"] == "system":
      system_message = processed_messages.pop(0)
      if processed_messages and processed_messages[0]["role"] == "user":
        processed_messages[0]["content"] = (
            system_message["content"] + "\n" + processed_messages[0]["content"]
        )
      else:
        processed_messages.insert(
            0, {"role": "user", "content": system_message["content"]}
        )
    return processed_messages

  def parse(
      self,
      messages: List[Dict[str, str]],
      add_generation_prompt: bool = False,
      is_first_msg: bool = False,
  ) -> str:
    """Parses messages for Gemma. System prompt is prepended to user prompt."""
    processed_messages = self.preprocess_messages(messages)

    result = ""

    for message in processed_messages:
      result += self._parse_message(message)

    if add_generation_prompt:
      result += self.generation_prompt

    return result

  def _init_generation_prompt(self) -> str:
    return self.tokens.assistant_token
