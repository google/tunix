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

"""Asynchronous teacher scoring on the student's original token sequence."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING
import urllib.parse

import httpx
import numpy as np

if TYPE_CHECKING:
  from tunix.dist_rl.common import datatypes


@runtime_checkable
class TeacherScorer(Protocol):
  """Scores unbatched, unpacked completion tokens under a frozen teacher."""

  async def score(
      self, payloads: Sequence[datatypes.RLTrainerPayload]
  ) -> Sequence[np.ndarray]:
    """Returns one fp32 logprob vector per payload, aligned to completion_ids."""
    ...


def _token_semantics(value: Any) -> Any:
  """Retains AddedToken flags rather than comparing only its printed content."""
  if isinstance(value, Mapping):
    return {str(k): _token_semantics(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_token_semantics(v) for v in value]
  if hasattr(value, "content"):
    return {
        key: getattr(value, key, None)
        for key in (
            "content",
            "single_word",
            "lstrip",
            "rstrip",
            "normalized",
            "special",
        )
    }
  return value


def _tokenizer_semantics(tokenizer: Any) -> dict[str, Any]:
  get_vocab = getattr(tokenizer, "get_vocab", None)
  if not callable(get_vocab):
    raise ValueError("OPD tokenizers must expose get_vocab().")
  vocab = get_vocab()
  if not isinstance(vocab, Mapping) or not vocab:
    raise ValueError("OPD tokenizer vocabulary must be a nonempty mapping.")
  if any(
      not isinstance(token, str)
      or isinstance(token_id, (bool, np.bool_))
      or not isinstance(token_id, (int, np.integer))
      or token_id < 0
      for token, token_id in vocab.items()
  ):
    raise ValueError(
        "OPD vocabulary must map strings to nonnegative token IDs."
    )
  get_added = getattr(tokenizer, "get_added_vocab", None)
  backend = getattr(tokenizer, "backend_tokenizer", None)
  serialize = getattr(backend, "to_str", None)
  if not callable(serialize):
    raise ValueError(
        "OPD requires a verifiable fast tokenizer backend; load the tokenizer"
        " with use_fast=True. Slow tokenizers are not supported."
    )
  try:
    backend_config = json.loads(serialize())
  except (TypeError, ValueError) as exc:
    raise ValueError(
        "OPD tokenizer backend must serialize to valid JSON."
    ) from exc
  if not isinstance(backend_config, dict) or not backend_config.get("model"):
    raise ValueError("OPD tokenizer backend must describe its token model.")
  special_ids = {
      name: _token_semantics(getattr(tokenizer, name, None))
      for name in (
          "bos_token_id",
          "eos_token_id",
          "pad_token_id",
          "unk_token_id",
          "sep_token_id",
          "cls_token_id",
          "mask_token_id",
          "additional_special_tokens_ids",
          "all_special_ids",
      )
  }
  return {
      "vocab": {token: int(token_id) for token, token_id in vocab.items()},
      "added_vocab": dict(get_added()) if callable(get_added) else {},
      "added_tokens": _token_semantics(
          getattr(tokenizer, "added_tokens_decoder", {})
      ),
      "special_ids": special_ids,
      "special_tokens": _token_semantics(
          getattr(tokenizer, "special_tokens_map_extended", None)
          or getattr(tokenizer, "special_tokens_map", {})
      ),
      # Padding/truncation are mutable per-call settings, not token semantics.
      "backend": {
          key: backend_config.get(key)
          for key in (
              "model",
              "decoder",
              "normalizer",
              "pre_tokenizer",
              "post_processor",
          )
      },
  }


def _token_vector(value: Any, name: str) -> np.ndarray:
  vector = np.asarray(value)
  if vector.ndim != 1 or vector.dtype.kind not in "iu":
    raise ValueError(f"{name} must be an unbatched integer token vector.")
  return vector


class SGLangTeacherScorer:
  """Scores temperature-one OPD tokens using SGLang's /generate endpoint.

  The tokenizers must describe the models actually served and trained. This
  checks token-ID semantics; the caller owns checking the server's checkpoint
  identity. No text decoding or retokenization occurs. Both a server base URL
  and a complete /generate URL are accepted. Tokenizers must have verifiable
  fast backends; runtime padding and truncation settings are not compared.
  """

  def __init__(
      self,
      teacher_url: str,
      *,
      student_tokenizer: Any,
      teacher_tokenizer: Any,
      timeout_seconds: float = 60.0,
      max_concurrency: int = 8,
  ):
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
      raise ValueError("timeout_seconds must be finite and positive.")
    if (
        isinstance(max_concurrency, bool)
        or not isinstance(max_concurrency, int)
        or max_concurrency <= 0
    ):
      raise ValueError("max_concurrency must be a positive integer.")
    parsed = urllib.parse.urlsplit(teacher_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
      raise ValueError("teacher_url must be an absolute HTTP or HTTPS URL.")
    endpoint = parsed.path.rstrip("/")
    if not endpoint.endswith("/generate"):
      endpoint += "/generate"
    self.teacher_url = urllib.parse.urlunsplit(parsed._replace(path=endpoint))

    student = _tokenizer_semantics(student_tokenizer)
    teacher = _tokenizer_semantics(teacher_tokenizer)
    for key in student:
      if student[key] != teacher[key]:
        raise ValueError(
            f"Student and teacher tokenizer {key} differ; OPD requires the"
            " same token-ID semantics."
        )
    self.tokenizer_fingerprint = hashlib.sha256(
        json.dumps(student, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    self._token_ids = frozenset(student["vocab"].values())
    self._timeout_seconds = float(timeout_seconds)
    self._max_concurrency = max_concurrency
    self._semaphore = asyncio.Semaphore(max_concurrency)

  async def _request(self, client: httpx.AsyncClient, tokens: list[int]) -> Any:
    body = {
        "input_ids": tokens,
        "sampling_params": {
            "temperature": 1.0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    try:
      # post() reads the complete response body before returning. The outer
      # asyncio deadline also covers slow-drip bodies, unlike an idle timeout.
      response = await client.post(self.teacher_url, json=body)
      response.raise_for_status()
      return response.json()
    except httpx.HTTPStatusError as exc:
      raise RuntimeError(
          f"Teacher scoring failed with HTTP status {exc.response.status_code}."
      ) from exc
    except httpx.RequestError as exc:
      raise RuntimeError("Teacher scoring request failed.") from exc

  async def _score_tokens(
      self,
      client: httpx.AsyncClient,
      tokens: list[int],
      completion_mask: np.ndarray,
  ) -> np.ndarray:
    try:
      # Start before acquiring a slot: queued requests share the same bounded
      # wait as network I/O. Cancellation closes HTTPX's in-flight socket.
      async with asyncio.timeout(self._timeout_seconds):
        async with self._semaphore:
          response = await self._request(client, tokens)
    except TimeoutError as exc:
      raise TimeoutError("Teacher scoring request timed out.") from exc
    return self._completion_logps(response, tokens, completion_mask)

  @staticmethod
  def _completion_logps(
      response: Any, tokens: list[int], completion_mask: np.ndarray
  ) -> np.ndarray:
    completion_length = len(completion_mask)
    prompt_length = len(tokens) - completion_length
    if prompt_length < 0:
      raise ValueError("Completion mask exceeds the scored input length.")
    try:
      rows = response["meta_info"]["input_token_logprobs"]
    except (KeyError, TypeError) as exc:
      raise ValueError(
          "Teacher response is missing meta_info.input_token_logprobs."
      ) from exc
    if not isinstance(rows, list) or len(rows) != len(tokens):
      raise ValueError(
          "Teacher logprob row count must match the complete input token count."
      )
    result = np.zeros(completion_length, dtype=np.float32)
    for position, (row, expected_id) in enumerate(zip(rows, tokens)):
      if not isinstance(row, (list, tuple)) or len(row) < 2:
        raise ValueError(
            f"Malformed teacher logprob row at position {position}."
        )
      logprob, token_id = row[:2]
      placeholder = position == 0 and (
          logprob is None
          or isinstance(logprob, (int, float))
          and math.isnan(logprob)
      )
      if not (placeholder and token_id is None) and (
          isinstance(token_id, bool)
          or not isinstance(token_id, int)
          or token_id != expected_id
      ):
        raise ValueError(f"Teacher token ID mismatch at position {position}.")
      completion_position = position - prompt_length
      if not (
          0 <= completion_position < completion_length
          and completion_mask[completion_position] > 0
      ):
        continue
      if (
          isinstance(logprob, bool)
          or not isinstance(logprob, (int, float))
          or not math.isfinite(logprob)
      ):
        raise ValueError(f"Invalid teacher logprob at position {position}.")
      with np.errstate(over="ignore"):
        result[completion_position] = logprob
    if not np.all(np.isfinite(result)):
      raise ValueError("Teacher completion logprobs must be finite in fp32.")
    return result

  async def score(
      self, payloads: Sequence[datatypes.RLTrainerPayload]
  ) -> Sequence[np.ndarray]:
    """Scores raw unbatched rows; entirely masked completions return zeros."""
    rows = []
    for payload in payloads:
      if getattr(payload, "segment_ids", None) is not None:
        raise ValueError("Teacher scoring requires unpacked payloads.")
      completion = _token_vector(payload.completion_ids, "completion_ids")
      mask = np.asarray(payload.completion_mask)
      if mask.shape != completion.shape or not np.all(
          (mask == 0) | (mask == 1)
      ):
        raise ValueError("completion_mask must be a token-aligned binary mask.")
      if not np.any(mask):
        # Failed rollouts can carry placeholder/out-of-vocabulary token IDs.
        # They contribute no loss and must not fail scoring healthy peers.
        rows.append(([], mask.copy(), False))
        continue
      prompt = _token_vector(payload.prompt_ids, "prompt_ids")
      prompt_mask = np.asarray(payload.prompt_mask)
      if prompt_mask.shape != prompt.shape or not np.all(prompt_mask == 1):
        raise ValueError("Teacher scoring requires unpadded prompt tokens.")
      tokens = prompt.tolist() + completion.tolist()
      if any(token_id not in self._token_ids for token_id in tokens):
        raise ValueError(
            "Payload contains token IDs outside the tokenizer vocabulary."
        )
      if not prompt.size:
        raise ValueError(
            "Active completions require at least one prompt token."
        )
      rows.append((tokens, mask.copy(), True))

    async def score_row(
        client: httpx.AsyncClient,
        tokens: list[int],
        mask: np.ndarray,
        active: bool,
    ) -> np.ndarray:
      if not active:
        return np.zeros(len(mask), dtype=np.float32)
      return await self._score_tokens(client, tokens, mask)

    if not any(active for _, _, active in rows):
      return [np.zeros(len(mask), dtype=np.float32) for _, mask, _ in rows]
    async with httpx.AsyncClient(
        timeout=None,
        limits=httpx.Limits(max_connections=self._max_concurrency),
    ) as client:
      tasks = [asyncio.create_task(score_row(client, *row)) for row in rows]
      try:
        return await asyncio.gather(*tasks)
      finally:
        for task in tasks:
          if not task.done():
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
