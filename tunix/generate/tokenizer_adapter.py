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


# Make SentencePiece optional so HF-only setups don't crash on import.
try:
  import sentencepiece as spm
except Exception:
  spm = None

# Best-effort import; if transformers isn't installed, leave as sentinel.
try:
  from transformers.processing_utils import ProcessorMixin
except Exception:
  ProcessorMixin = None


class TokenizerType(enum.Enum):
  SP: str = 'sp'   # sentencepiece tokenizer
  HF: str = 'hf'   # huggingface tokenizer
  NONE: str = 'none'  # Represents no tokenizer


# ---- Tiny shim so a Processor behaves like a text tokenizer, if needed. ----
class _ProcessorShim:
  """Minimal text-tokenizer shim for HF Processors.

  Implements encode/decode/bos_id/eos_id/pad_id by delegating to the underlying
  text tokenizer when present; for encode, can fall back to calling the Processor.
  """
  def __init__(self, processor: Any):
    self._p = processor
    self._tok = getattr(processor, "tokenizer", None) or getattr(processor, "text_tokenizer", None)

  def encode(self, text: str, **kwargs) -> list[int]:
    if self._tok is not None and hasattr(self._tok, "encode"):
      return self._tok.encode(text, **kwargs)

    # Fallback: call processor to get input_ids (ensure Python lists, not tensors)
    proc_kwargs = dict(kwargs)
    proc_kwargs.setdefault("return_tensors", None)
    proc_kwargs.setdefault("add_special_tokens", True)
    out = self._p(text, **proc_kwargs)
    ids = out.get("input_ids", None)
    if ids is None:
      for k, v in out.items():
        if "input_ids" in k:
          ids = v
          break
    if ids is None:
      raise ValueError("Processor did not yield 'input_ids' for text encode().")
    if hasattr(ids, "tolist"):
      ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
      ids = ids[0]
    return [int(x) for x in ids]

  def decode(self, ids: list[int], **kwargs) -> str:
    if self._tok is None or not hasattr(self._tok, "decode"):
      raise AttributeError(
        "This Processor lacks a text tokenizer for decode(). "
        "Provide a processor with `.tokenizer`/`.text_tokenizer`."
      )
    return self._tok.decode(ids, **kwargs)

  def bos_id(self) -> int:
    if self._tok is not None and getattr(self._tok, "bos_token_id", None) is not None:
      return self._tok.bos_token_id
    # fall back to eos if BOS is undefined
    eos = self.eos_id()
    if eos is None:
      raise ValueError("No BOS/EOS token id available for this Processor.")
    return eos

  def eos_id(self) -> int:
    if self._tok is not None and getattr(self._tok, "eos_token_id", None) is not None:
      return self._tok.eos_token_id
    raise ValueError("Processor's tokenizer has no eos_token_id defined.")

  def pad_id(self) -> int:
    if self._tok is None:
      raise ValueError("Processor lacks a text tokenizer to query pad_id().")
    if getattr(self._tok, "pad_token_id", None) is None:
      # common fallback for LLaMA-like tokenizers
      try:
        self._tok.pad_token = self._tok.eos_token
      except Exception:
        pass
    if getattr(self._tok, "pad_token_id", None) is None:
      raise ValueError("Processor's tokenizer has no pad_token_id and could not set one.")
    return self._tok.pad_token_id
# ---------------------------------------------------------------------------


class TokenizerAdapter:
  """Wrapper for different tokenizers used in sampler."""

  def __init__(self, tokenizer: Any):
    # If a HF Processor is passed, unwrap to its text tokenizer if possible;
    # otherwise wrap it with a shim that provides the required methods.
    proc_tok = self._unwrap_hf_processor(tokenizer)
    self._tokenizer = proc_tok if proc_tok is not None else tokenizer

    missing_methods = self._missing_methods()
    if not missing_methods:
      self._tokenizer_type = TokenizerType.NONE
    elif spm is not None and isinstance(self._tokenizer, spm.SentencePieceProcessor):
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
      # e.g., LLaMA-style HF tokenizers may not have pad_id
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
    return (
        'transformers.tokenization_utils_base.PreTrainedTokenizerBase'
        in baseclass_names
    )

  def _unwrap_hf_processor(self, obj: Any):
    """If obj is a HF Processor, return a text tokenizer or a shim."""
    if ProcessorMixin is None:
      return None
    try:
      if isinstance(obj, ProcessorMixin):
        tok = getattr(obj, "tokenizer", None) or getattr(obj, "text_tokenizer", None)
        return tok if tok is not None else _ProcessorShim(obj)
    except Exception:
      pass
    return None

  @property
  def tokenizer(self) -> Any:
    return self._tokenizer