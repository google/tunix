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
from typing import Dict, List, Tuple

import numpy as np
from tunix.utils import token_sanitization

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
  # Separator inserted between consecutive messages by `parse()`. Match what
  # the model's `apply_chat_template` writes between successive messages so
  # incremental per-message rendering can be concatenated without a fixup.
  message_separator: str = ""


class BaseChatTemplateParser(ABC):
  """Abstract base class for chat template parsers."""

  def __init__(self, tokenizer, enable_thinking: bool = True):
    self.tokenizer = tokenizer
    self.enable_thinking = enable_thinking
    self.tokens = self._init_tokens()
    self.generation_prompt = self._init_generation_prompt()
    # message_separator is a per-turn formatting hint (e.g. "\n"), not a
    # control token; including it would strip every newline from message
    # content.
    self._tokens_to_sanitize = {
        v
        for k, v in dataclasses.asdict(self.tokens).items()
        if k != "message_separator" and isinstance(v, str) and v
    }

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
    """Parse messages into chat template format.

    When `is_first_msg=False` the call renders a continuation of an existing
    conversation, so the result is prefixed with `message_separator` to
    re-introduce the inter-message boundary that the previous turn's eot did
    not emit. This lets per-message incremental rendering be concatenated
    directly to prior tokens without external fixups.
    """
    parts = []

    if is_first_msg:
      first_chunk = self._handle_first_message(messages)
      if first_chunk:
        parts.append(first_chunk)

    for message in messages:
      parts.append(self._parse_message(message))

    if add_generation_prompt:
      parts.append(self.generation_prompt)

    sep = self.tokens.message_separator
    result = sep.join(parts)
    if not is_first_msg and result:
      result = sep + result
    return result

  def _handle_first_message(self, messages: List[Dict[str, str]]) -> str:
    """Handle special logic for first message."""
    del messages  # Unused in the base implementation.
    return self.tokens.bos_token

  def _parse_message(self, message: Dict[str, str]) -> str:
    """Parse a single message based on its role."""
    role = message["role"]
    content = token_sanitization.sanitize_control_tokens(
        message["content"],
        extra_tokens=self._tokens_to_sanitize,
        include_default=not self._tokens_to_sanitize,
    )

    parser_map = {
        "system": self._parse_system,
        "user": self._parse_user,
        "assistant": self._parse_assistant,
        "tool": self._parse_tool,
    }

    if role not in parser_map:
      raise NotImplementedError(f"Unsupported message role: {role}")

    return parser_map[role](content)

  def _parse_system(self, content: str) -> str:
    return self.tokens.system_token + content + self.tokens.eot_token

  def _parse_user(self, content: str,) -> str:
    return self.tokens.user_token + content + self.tokens.eot_token

  def _parse_assistant(self, content: str) -> str:
    return self.tokens.assistant_token + content + self.tokens.eot_token

  def _parse_tool(self, content: str) -> str:
    return (
        self.tokens.user_token
        + self.tokens.tool_response_start_token
        + content
        + self.tokens.tool_response_end_token
        + self.tokens.eot_token
    )

  @property
  def newline_tokens(self) -> list[int]:
    """Returns the tokenized newline "\n"."""
    return []


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
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=self.enable_thinking,
    )


class QwenChatTemplateParser(BaseChatTemplateParser):
  """Parser for Qwen models."""

  def _init_tokens(self) -> TokenConfig:
    return TokenConfig(
        bos_token=self.tokenizer.bos_token,
        eos_token=self.tokenizer.eos_token,
        eot_token="<|im_end|>",
        system_token="<|im_start|>system\n",
        user_token="<|im_start|>user\n",
        assistant_token=self._get_assistant_token(),
        tool_start_token="\n<tool_call>\n",
        tool_end_token="\n</tool_call>",
        tool_response_start_token="<tool_response>\n",
        tool_response_end_token="\n</tool_response>",
        message_separator="\n",
    )

  def _get_assistant_token(self) -> str:
    token = "<|im_start|>assistant\n"
    if not self.enable_thinking:
      token += "<think>\n\n</think>\n\n"
    return token

  def _init_generation_prompt(self) -> str:
    return self.tokens.assistant_token

  def _handle_first_message(self, messages: List[Dict[str, str]]) -> str:
    """Add default system message if first message is not system."""
    if messages[0]["role"] != "system":
      return self._parse_system(
          "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
      )
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

  def _parse_assistant(self, content: str) -> str:
    return self.tokens.assistant_token + content

  def _parse_system(self, content: str) -> str:
    # This should not be called if parse() is used, as it handles the system
    # prompt by merging it. Raise error for unexpected system messages.
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


class Gemma4ChatTemplateParser(BaseChatTemplateParser):
  """Parser for Gemma 4 models."""

  def __init__(
      self,
      tokenizer,
      enable_thinking: bool = True,
  ):
    super().__init__(tokenizer, enable_thinking=enable_thinking)
    # no strip with disable thinking lowers the  difference between initial rollout and trainer logps from 0.4  to 0.3.
    self._strip_past_thinking = strip_past_thinking and enable_thinking
    # Also sanitize the base <turn|> token (without trailing newline) to guard
    # against model-generated control tokens trailing in message contents.
    self._tokens_to_sanitize.add("<turn|>")

  def _init_tokens(self) -> TokenConfig:
    return TokenConfig(
        bos_token="<bos>",
        eot_token="<turn|>\n",
        system_token=self._get_system_token(),
        user_token="<|turn>user\n",
        assistant_token=self._get_assistant_token(),
        tool_start_token="<|tool_call>",
        tool_end_token="<tool_call|>",
        tool_response_start_token="<|tool_response>",
        tool_response_end_token="<tool_response|>",
        message_separator="",
    )

  def _get_system_token(self) -> str:
    token = "<|turn>system\n"
    if self.enable_thinking:
      token += "<|think|>"
    return token

  def _get_assistant_token(self) -> str:
    token = "<|turn>model\n"
    if not self.enable_thinking:
      token += "<|channel>thought\n<channel|>"
    return token

  def _init_generation_prompt(self) -> str:
    return self.tokens.assistant_token

  def _handle_first_message(self, messages: List[Dict[str, str]]) -> str:
    """Prepend bos and system think token if needed."""
    prefix = self.tokens.bos_token
    if messages and messages[0]["role"] not in ("system", "developer"):
      if self.enable_thinking:
        prefix += "<|turn>system\n<|think|><turn|>\n"
    return prefix

  def _parse_tool(self, content: str) -> str:
    return (
        self.tokens.tool_response_start_token
        + content
        + self.tokens.tool_response_end_token
    )

  def _parse_system(self, content: str) -> str:
    return self.tokens.system_token + content.strip() + self.tokens.eot_token

  def _parse_user(self, content: str) -> str:
    return self.tokens.user_token + content.strip() + self.tokens.eot_token

  def _parse_assistant(self, content: str) -> str:
    """Parses an assistant message, optionally stripping thinking blocks."""
    # TODO(linchai): Support stripping thinking tokens in multi-turn.
    # Historical thinking tokens are typically stripped per official Gemma 4
    # guidelines to match inference-time prompt formatting.
    cleaned_content = content.strip()
    prefix = (
        "<|turn>model\n"
        if self.enable_thinking
        else self.tokens.assistant_token
    )
    if cleaned_content.endswith("<turn|>"):
      return prefix + cleaned_content + "\n"
    return prefix + cleaned_content + self.tokens.eot_token

  def update_assistant_end_tokens(
      self,
      tokens: np.ndarray,
  ) -> Tuple[np.ndarray, int]:
    """Ensures the assistant response ends with the correct tokens."""
    # We append "\n" when pasing assistant message above so here we need to match the tokens with newline,
    # otherwise there will be logprobs mismatch between rollout and trainer.
    tokens = np.concatenate(
        [tokens, np.array(self.newline_tokens, dtype=np.int32)], axis=0
    )
    return tokens, len(self.newline_tokens)

  @property
  def newline_tokens(self) -> list[int]:
    """Returns the tokenized newline "\n"."""
    suffix_str = "\n"
    try:
      suffix_tokens = self.tokenizer.encode(suffix_str, add_special_tokens=False)
    except TypeError:
      suffix_tokens = self.tokenizer.encode(suffix_str)
    bos_id = getattr(self.tokenizer, "bos_id", None)
    if bos_id is not None:
      bos_id = bos_id() if callable(bos_id) else bos_id
    eos_id = getattr(self.tokenizer, "eos_id", None)
    if eos_id is not None:
      eos_id = eos_id() if callable(eos_id) else eos_id
    return [t for t in suffix_tokens if t != bos_id and t != eos_id]


