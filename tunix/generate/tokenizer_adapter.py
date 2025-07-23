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
    if not missing_methods:
      self._tokenizer_type = TokenizerType.NONE
    elif isinstance(self._tokenizer, spm.SentencePieceProcessor):
      self._tokenizer_type = TokenizerType.SP
    elif self._is_hf_tokenizer():
      self._tokenizer_type = TokenizerType.HF
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
