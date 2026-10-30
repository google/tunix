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

"""Held-out GSM8K evaluation of the actual synchronized rollout policy.

Call only at a training barrier, before training starts or after the training
program finishes. Evaluation uses synchronous engine.generate RPCs rather than
the training dispatch/poll queues, and never performs an optimizer update or
teacher-scoring request. Gold answers remain on the control plane.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import dataclasses
from decimal import Decimal
from decimal import InvalidOperation
from fractions import Fraction
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.utils import gsm8k_vtc

_NUMBER = (
    r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)" r"(?:[eE][+-]?\d+)?"
)
_DECIMAL_RE = re.compile(rf"^{_NUMBER}$")
_FRACTION_RE = re.compile(rf"^({_NUMBER})\s*/\s*({_NUMBER})$")
_LATEX_FRACTION_RE = re.compile(
    rf"^\\(?:d|t)?frac\s*\{{\s*({_NUMBER})\s*\}}\s*\{{\s*({_NUMBER})\s*\}}$"
)
_HELD_OUT_SPLIT = re.compile(r"^test(?:\[(?:\d+%?)?:(?:\d+%?)?\])?$")
_VALID_STATUSES = frozenset(
    {"SUCCEEDED", "MAX_STEPS_REACHED", "MAX_CONTEXT_LIMIT_REACHED"}
)


def _positive_integer(
    value: Any, name: str, *, allow_zero: bool = False
) -> None:
  if (
      isinstance(value, bool)
      or not isinstance(value, numbers.Integral)
      or value < (0 if allow_zero else 1)
  ):
    raise ValueError(
        f"{name} must be {'nonnegative' if allow_zero else 'positive'} integer."
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class EvaluationConfig:
  """Fixed greedy pass@1 settings for an official held-out GSM8K split."""

  dataset_name: str = "gsm8k:1.0.0"
  split: str = "test"
  batch_size: int = 8
  max_response_length: int = 2048
  seed: int = 42
  timeout_seconds: float = 600.0
  max_examples: int = 0

  def __post_init__(self) -> None:
    if self.dataset_name != "gsm8k:1.0.0":
      raise ValueError("Evaluation requires the pinned dataset gsm8k:1.0.0.")
    if not isinstance(self.split, str) or not _HELD_OUT_SPLIT.fullmatch(
        self.split
    ):
      raise ValueError(
          "Evaluation split must be test or a test[start:stop] slice."
      )
    for name in ("batch_size", "max_response_length"):
      _positive_integer(getattr(self, name), name)
    for name in ("seed", "max_examples"):
      _positive_integer(getattr(self, name), name, allow_zero=True)
    if (
        isinstance(self.timeout_seconds, bool)
        or not isinstance(self.timeout_seconds, numbers.Real)
        or not math.isfinite(self.timeout_seconds)
        or self.timeout_seconds <= 0
    ):
      raise ValueError("timeout_seconds must be finite and positive.")

  @classmethod
  def from_dict(cls, value: Mapping[str, Any]) -> EvaluationConfig:
    if not isinstance(value, Mapping):
      raise ValueError("Evaluation configuration must be a JSON object.")
    unknown = set(value) - {field.name for field in dataclasses.fields(cls)}
    if unknown:
      raise ValueError(f"Unknown evaluation settings: {sorted(unknown)}.")
    return cls(**dict(value))


@dataclasses.dataclass(frozen=True, kw_only=True)
class EvaluationExample:
  """Gold is kept out of engine requests, environment metadata, and prompts."""

  example_id: str
  question: str
  gold_answer: str

  def __post_init__(self) -> None:
    for name in ("example_id", "question", "gold_answer"):
      value = getattr(self, name)
      if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Evaluation {name} must be a nonempty string.")
    if canonical_number(self.gold_answer) is None:
      raise ValueError(f"Non-numeric GSM8K gold answer for {self.example_id}.")


def canonical_number(value: str | None) -> str | None:
  """Canonicalizes only explicit numeric answers; never guesses a last number.

  Decimals and simple fractions are compared exactly. NaN, infinity, units,
  arbitrary expressions, very large exponents, and malformed answers fail.
  """
  if not isinstance(value, str):
    return None
  text = value.strip()
  if not text or len(text) > 128:
    return None

  def decimal_fraction(number: str) -> Fraction:
    parsed = Decimal(number.replace(",", ""))
    if not parsed.is_finite() or abs(parsed.as_tuple().exponent) > 64:
      raise ValueError("Unsupported numeric range.")
    return Fraction(parsed)

  try:
    if _DECIMAL_RE.fullmatch(text):
      result = decimal_fraction(text)
    else:
      match = _FRACTION_RE.fullmatch(text) or _LATEX_FRACTION_RE.fullmatch(text)
      if match is None:
        return None
      result = decimal_fraction(match[1]) / decimal_fraction(match[2])
  except (InvalidOperation, ValueError, ZeroDivisionError, OverflowError):
    return None
  return str(result)


def load_examples(
    data_dir: str, config: EvaluationConfig
) -> list[EvaluationExample]:
  """Loads held-out questions in their original order, without shuffling."""
  from tunix.dist_rl.examples.math_gsm8k_dist import gsm8k  # pylint: disable=g-import-not-at-top

  dataset = gsm8k.load_gsm8k_dataset(
      split=config.split,
      data_dir=data_dir,
      shuffle=False,
      seed=config.seed,
      dataset_name=config.dataset_name,
  )
  count = len(dataset)
  if config.split == "test" and not config.max_examples and count != 1319:
    raise ValueError(
        "The complete pinned GSM8K test split must have 1319 examples."
    )
  if config.max_examples:
    count = min(count, config.max_examples)
  return [
      EvaluationExample(
          example_id=f"gsm8k:{config.split}:{index:06d}",
          question=gsm8k_vtc.as_text(dataset[index]["question"]),
          gold_answer=gsm8k_vtc.as_text(dataset[index]["answer"]),
      )
      for index in range(count)
  ]


def _digest(value: Any) -> str:
  return hashlib.sha256(
      json.dumps(
          value, sort_keys=True, ensure_ascii=True, allow_nan=False
      ).encode()
  ).hexdigest()


def _response_record(
    item: datatypes.TrajectoryItem,
    example: EvaluationExample,
    *,
    policy_version: int,
    batch_seconds: float,
) -> dict[str, Any]:
  traj = getattr(item, "traj", None)
  if not isinstance(traj, Mapping):
    raise ValueError("Evaluation response must carry a token-mode trajectory.")
  metadata = getattr(item, "metadata", {})
  if isinstance(metadata, Mapping) and metadata.get("error"):
    raise ValueError(f"Evaluation rollout failed for {example.example_id}.")
  status = traj.get("status")
  status = status.name if hasattr(status, "name") else str(status).upper()
  if status not in _VALID_STATUSES:
    raise ValueError(f"Evaluation rollout has invalid status {status}.")
  if status == "SUCCEEDED" and not getattr(item, "is_valid", True):
    raise ValueError("Evaluation received an invalid successful trajectory.")
  version = getattr(item, "policy_version", None)
  if (
      isinstance(version, bool)
      or not isinstance(version, numbers.Integral)
      or version != policy_version
  ):
    raise ValueError(
        "Evaluation response policy version does not match the synchronized"
        " policy."
    )
  versions = getattr(item, "policy_versions", None)
  if versions is not None and any(v != policy_version for v in versions):
    raise ValueError("Evaluation trajectory mixed policy versions.")
  if "conversation_text" not in traj:
    raise ValueError("Evaluation response is missing conversation_text.")
  conversation = traj["conversation_text"]
  if isinstance(conversation, list):
    assistant_messages = [
        message
        for message in conversation
        if isinstance(message, Mapping) and message.get("role") == "assistant"
    ]
    if len(assistant_messages) != 1 or not isinstance(
        assistant_messages[0].get("content"), str
    ):
      raise ValueError(
          "Single-turn evaluation requires one textual assistant response."
      )
  elif not isinstance(conversation, str):
    raise ValueError(
        "Evaluation conversation_text must be text or chat messages."
    )
  text = datatypes.assistant_text(conversation)
  tokens = np.asarray(traj.get("conversation_tokens"))
  if tokens.ndim != 1 or tokens.dtype.kind not in "iu" or tokens.size == 0:
    raise ValueError(
        "Evaluation response must contain a nonempty integer completion token"
        " vector."
    )
  masks = traj.get("conversation_masks")
  if masks is not None:
    masks = np.asarray(masks)
    if masks.shape != tokens.shape or not np.all((masks == 0) | (masks == 1)):
      raise ValueError(
          "Evaluation action mask does not match completion tokens."
      )
  extracted = gsm8k_vtc.extract_boxed_answer(text)
  predicted = canonical_number(extracted)
  gold = canonical_number(example.gold_answer)
  reward, format_ok, strict_correct, _ = gsm8k_vtc.vtc_completion_outcome(
      text, example.gold_answer
  )
  return {
      "example_id": example.example_id,
      "prompt_id": str(item.prompt_id),
      "question": example.question,
      "gold_answer": example.gold_answer,
      "completion": text,
      "extracted_answer": extracted,
      "canonical_answer": predicted,
      "answer_correct": predicted is not None and predicted == gold,
      "strict_boxed_correct": bool(strict_correct),
      "format_correct": bool(format_ok),
      "vtc_reward": float(reward),
      "completion_token_count": int(tokens.size),
      "truncated": status != "SUCCEEDED",
      "status": status,
      "policy_version": int(version),
      "batch_latency_seconds": batch_seconds,
  }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = None
  try:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
      temporary = Path(stream.name)
      json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
      stream.write("\n")
      stream.flush()
      os.fsync(stream.fileno())
    if path.exists():
      raise FileExistsError(f"Evaluation report already exists: {path}")
    os.replace(temporary, path)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


async def evaluate_engine(
    engine: Any,
    examples: Sequence[EvaluationExample],
    *,
    config: EvaluationConfig,
    output_path: str | Path,
    label: str,
    model_identity: Mapping[str, Any],
    policy_version: int,
) -> dict[str, Any]:
  """Evaluates the live synchronized student without training or teacher I/O.

  Invalid or missing responses fail the evaluation; they are never omitted
  from an accuracy denominator or replaced with fabricated completions. A
  report is published atomically only after every example completes.
  """
  if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
    raise ValueError(
        "Evaluation label must contain only letters, numbers, dot, underscore,"
        " or hyphen."
    )
  _positive_integer(policy_version, "policy_version", allow_zero=True)
  output = Path(output_path)
  if output.exists():
    raise FileExistsError(f"Evaluation report already exists: {output}")
  rows = list(examples)
  if config.max_examples:
    rows = rows[: config.max_examples]
  if not rows or any(not isinstance(row, EvaluationExample) for row in rows):
    raise ValueError("Evaluation requires at least one validated example.")
  ids = [row.example_id for row in rows]
  if len(set(ids)) != len(ids):
    raise ValueError("Evaluation example IDs must be unique.")
  if not isinstance(model_identity, Mapping) or not model_identity:
    raise ValueError("Evaluation requires explicit model identity/provenance.")
  identity = json.loads(json.dumps(dict(model_identity), allow_nan=False))
  dataset_identity = [dataclasses.asdict(row) for row in rows]
  records = []
  started = time.monotonic()
  for start in range(0, len(rows), config.batch_size):
    batch = rows[start : start + config.batch_size]
    prompt_ids = [
        f"eval_{label}_v{policy_version}_{start + index:06d}"
        for index in range(len(batch))
    ]
    prompts = []
    for prompt_id, example in zip(prompt_ids, batch, strict=True):
      prompt = gsm8k_vtc.build_prompt(example.question)
      prompts.append({
          "prompt_id": prompt_id,
          "prompt": prompt,
          "max_turns": 1,
          "max_response_length": config.max_response_length,
          "metadata": {
              "evaluation": True,
              "env_config": {"prompt": prompt, "question": example.question},
          },
      })
    generation = datatypes.GenerationArgs(
        max_generation_steps=config.max_response_length,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=config.seed,
        return_logprobs=False,
    )
    batch_started = time.monotonic()
    async with asyncio.timeout(config.timeout_seconds):
      results = await engine.generate(prompts, generation_args=generation)
    batch_seconds = time.monotonic() - batch_started
    if not isinstance(results, Sequence) or len(results) != len(batch):
      raise ValueError(
          "Evaluation result count must match dispatched questions."
      )
    by_id = {}
    for item in results:
      key = str(getattr(item, "prompt_id", ""))
      if key not in prompt_ids or key in by_id:
        raise ValueError("Evaluation returned duplicate or unknown prompt IDs.")
      if getattr(item, "group_index", None) != 0:
        raise ValueError(
            "Greedy pass@1 evaluation expects one response per question."
        )
      by_id[key] = item
    for prompt_id, example in zip(prompt_ids, batch, strict=True):
      records.append(
          _response_record(
              by_id[prompt_id],
              example,
              policy_version=policy_version,
              batch_seconds=batch_seconds,
          )
      )
  count = len(records)
  correct = sum(row["answer_correct"] for row in records)
  report = {
      "schema_version": 1,
      "task": "gsm8k",
      "label": label,
      "model": identity,
      "policy_version": int(policy_version),
      "evaluation": dataclasses.asdict(config),
      "sampling": {
          "temperature": 0.0,
          "top_p": 1.0,
          "top_k": 0,
          "base_seed": config.seed,
          "samples_per_question": 1,
          "max_generation_steps": config.max_response_length,
      },
      "scoring": "boxed_numeric_exact_match_v1",
      "dataset": {
          "name": config.dataset_name,
          "split": config.split,
          "size": count,
          "sha256": _digest(dataset_identity),
      },
      "prompt_template_sha256": _digest(gsm8k_vtc.VTC_PROMPT_TEMPLATE),
      "metrics": {
          "num_examples": count,
          "num_correct": correct,
          "answer_accuracy": correct / count,
          "strict_boxed_accuracy": (
              sum(row["strict_boxed_correct"] for row in records) / count
          ),
          "format_accuracy": (
              sum(row["format_correct"] for row in records) / count
          ),
          "mean_vtc_reward": sum(row["vtc_reward"] for row in records) / count,
          "mean_completion_tokens": (
              sum(row["completion_token_count"] for row in records) / count
          ),
          "max_completion_tokens": max(
              row["completion_token_count"] for row in records
          ),
          "truncated_count": sum(row["truncated"] for row in records),
          "elapsed_seconds": time.monotonic() - started,
      },
      "results": records,
  }
  _write_report(output, report)
  return report
