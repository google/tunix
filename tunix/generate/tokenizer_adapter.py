# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapt tokenizers to a common interface."""

import enum
import inspect
from typing import Any

from etils import epath
import numpy as np

import sentencepiece as spm


class TokenizerType(enum.Enum):
  SP: str = 'sp'  # sentencepiece tokenizer
  HF: str = 'hf'  # huggingface tokenizer
  NONE: str = 'none'  # Represents no tokenizer


class TokenizerAdapter:
  """Wrapper for different tokenizers used in sampler."""

  def __init__(self, tokenizer: Any):
    self._tokenizer = tokenizer

    missing_methods = self._missing_methods()

    if isinstance(self._tokenizer, spm.SentencePieceProcessor):
      self._tokenizer_type = TokenizerType.SP
    elif self._is_hf_tokenizer():
      self._tokenizer_type = TokenizerType.HF
    elif not missing_methods:
      self._tokenizer_type = TokenizerType.NONE
    else:
      raise ValueError(
          'Your tokenizer should either be a `spm.SentencePieceProcessor` '
          'tokenizer, a HuggingFace tokenizer, or it should have '
          'the following methods: '
          '`["encode", "decode", "bos_id", "eos_id", "pad_id"]`. Received: '
          f'`type(tokenizer)` = {type(tokenizer)}, with missing methods: '
          f'{missing_methods}.'
      )

  def encode(self, text: str, **kwargs) -> list[int]:
    if self._tokenizer_type == TokenizerType.SP:
      return self._tokenizer.EncodeAsIds(text, **kwargs)
    elif self._tokenizer_type == TokenizerType.HF:
      return self._tokenizer.encode(text, **kwargs)
    else:
      return self._tokenizer.encode(text, **kwargs)

  def decode(self, ids: list[int], **kwargs) -> str:
    if self._tokenizer_type == TokenizerType.SP:
      return self._tokenizer.DecodeIds(ids, **kwargs)
    elif self._tokenizer_type == TokenizerType.HF:
      return self._tokenizer.decode(ids, **kwargs)
    else:
      return self._tokenizer.decode(ids, **kwargs)

  def bos_id(self) -> int:
    if self._tokenizer_type == TokenizerType.SP:
      return self._tokenizer.bos_id()
    elif self._tokenizer_type == TokenizerType.HF:
      return self._tokenizer.bos_token_id
    else:
      return self._tokenizer.bos_id()

  def eos_id(self) -> int:
    if self._tokenizer_type == TokenizerType.SP:
      return self._tokenizer.eos_id()
    elif self._tokenizer_type == TokenizerType.HF:
      return self._tokenizer.eos_token_id
    else:
      return self._tokenizer.eos_id()

  def pad_id(self) -> int:
    """Returns the pad token id."""
    if self._tokenizer_type == TokenizerType.SP:
      ret_id = self._tokenizer.pad_id()
      if ret_id == -1:
        raise ValueError('SentencePiece tokenizer has a undefined pad_id.')
      return ret_id
    elif self._tokenizer_type == TokenizerType.HF:
      # e.g. llama3 HF tokenizers do not have pad_id
      if self._tokenizer.pad_token_id is None:
        self._tokenizer.pad_token = self._tokenizer.eos_token
      return self._tokenizer.pad_token_id
    else:
      return self._tokenizer.pad_id()

  def dedup_bos_ids(self, ids: list[int]) -> list[int]:
    """Deduplicates the bos_id at the beginning of the list."""
    i = 0
    while i < len(ids) - 1 and ids[i] == ids[i + 1] == self.bos_id():
      i += 1
    return ids[i:]

  def _missing_methods(self) -> list[str]:
    """Checks if the tokenizer has any missing methods."""
    required_methods = ['encode', 'decode', 'bos_id', 'eos_id', 'pad_id']
    missing_methods = []
    for method in required_methods:
      if not hasattr(self._tokenizer, method):
        missing_methods.append(method)
    return missing_methods

  def _is_hf_tokenizer(self) -> bool:
    """Checks if the tokenizer is a huggingface tokenizer."""
    baseclasses = inspect.getmro(type(self._tokenizer))
    baseclass_names = [
        baseclass.__module__ + '.' + baseclass.__name__
        for baseclass in baseclasses
    ]
    if (
        'transformers.tokenization_utils_base.PreTrainedTokenizerBase'
        in baseclass_names
    ):
      return True
    return False

  @property
  def tokenizer(self) -> Any:
    return self._tokenizer

  def apply_chat_template(
      self,
      messages: list[dict[str, str]],
      add_generation_prompt: bool = True,
      tokenize: bool = False,
      **kwargs,
  ) -> str | list[int]:
    """Applies a chat template to format a list of messages.

    Primarily for HuggingFace tokenizers, this formats conversation history
    into a single string or token sequence.

    Args:
      messages: Conversation turns, each with 'role' and 'content'.
      add_generation_prompt: Whether to append a generation prompt.
      tokenize: If True, returns token IDs; otherwise, returns a string.
      **kwargs: Additional args for the underlying `apply_chat_template`.

    Returns:
      The formatted chat as a string or list of token IDs.

    Raises:
      NotImplementedError: If chat templating is not supported by the tokenizer.
    """
    if self._tokenizer_type == TokenizerType.HF:
      return self._tokenizer.apply_chat_template(
          messages,
          add_generation_prompt=add_generation_prompt,
          tokenize=tokenize,
          **kwargs,
      )
    elif self._tokenizer_type == TokenizerType.SP:
      # TODO(haoyugao): The tokenizer type is SP but the template is for
      # Gemma. This should be aligned, for example by adding a 'gemma'
      # tokenizer type or a flag. For now, all sentencepiece tokenizers default
      # to using the Gemma chat template.
      return self._apply_gemma_chat_template(
          messages, add_generation_prompt, tokenize
      )
    else:
      if hasattr(self._tokenizer, 'apply_chat_template'):
        return self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )
      # Implements the Gemma chat template format as a fallback.
      return self._apply_gemma_chat_template(
          messages, add_generation_prompt, tokenize
      )

  def _apply_gemma_chat_template(
      self,
      messages: list[dict[str, str]],
      add_generation_prompt: bool,
      tokenize: bool,
  ) -> str | list[int]:
    """Applies the Gemma chat template format."""
    chat_str = ''
    for message in messages:
      role = message.get('role')
      content = message.get('content')
      if role in ('user', 'model'):
        chat_str += f'<start_of_turn>{role}\n{content}<end_of_turn>\n'

    if add_generation_prompt:
      chat_str += '<start_of_turn>model\n'

    if tokenize:
      return self.encode(chat_str)
    return chat_str


class Tokenizer(TokenizerAdapter):
  """Tokenizing and encoding/decoding text using TokenizerAdapter."""

  def __init__(
      self,
      tokenizer_type: str = 'sentencepiece',
      tokenizer_path: str = 'gs://gemma-data/tokenizers/tokenizer_gemma2.model',
      add_bos: bool | None = True,
      add_eos: bool | None = True,
      hf_access_token: str | None = None,
  ):

    self.tokenizer_type = tokenizer_type
    if tokenizer_type == 'huggingface':
      import transformers  # pylint: disable=g-import-not-at-top

      tokenizer = transformers.AutoTokenizer.from_pretrained(
          pretrained_model_name_or_path=tokenizer_path,
          add_bos_token=add_bos,
          add_eos_token=add_eos,
          token=hf_access_token,
      )
    elif tokenizer_type == 'sentencepiece':
      model_proto = epath.Path(tokenizer_path).read_bytes()
      tokenizer = spm.SentencePieceProcessor()
      tokenizer.LoadFromSerializedProto(model_proto)
      options = []
      if add_bos:
        options.append('bos')
      if add_eos:
        options.append('eos')

      extra_options_str = ':'.join(options)
      if extra_options_str:
        tokenizer.SetEncodeExtraOptions(extra_options_str)
    else:
      raise ValueError(f'Unsupported tokenizer_type: {tokenizer_type}')
    super().__init__(tokenizer)

  def tokenize(
      self,
      example: str,
      prefix: str = '',
      suffix: str = '',
      add_eos: bool = True,
  ) -> np.ndarray:
    """The tokenization function.

    Args:
      example: Input string to tokenize.
      prefix:  Prefix to add to the input string.
      suffix:  Suffix to add to the input string.
      add_eos: If True, add an "end of sentence" token at the end of the output
        sequence.

    Returns:
      Tokens corresponding to the input string.
    """
    int_list = []
    if self.bos_id():
      int_list.append(self.bos_id())
    if self.tokenizer_type == 'huggingface':
      int_list.extend(
          self.encode(prefix + example + suffix, add_special_tokens=False)
      )
    else:
      # sentencepiece
      int_list.extend(self.tokenizer.EncodeAsIds(prefix + example + suffix))
    if add_eos:
      int_list.append(self.eos_id())
    return np.array(int_list, dtype=np.int32)
