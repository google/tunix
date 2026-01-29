#!/usr/bin/env python3
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

"""Reward function for the NL2SQL GRPO example."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable, List

from tunix.cli import config as config_lib

_EXEC_WEIGHT = 0.2
_RESULT_WEIGHT = 0.8
_DEBUG_ONCE = False


class _Nl2SqlHelper:
  """Helper utilities kept off the reward function registry."""

  _FORBIDDEN_SQL = re.compile(
      r"\b(update|delete|insert|drop|alter|create|attach|detach|pragma|vacuum|"
      r"replace|truncate)\b",
      re.IGNORECASE,
  )
  _CODE_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

  @staticmethod
  def completion_to_text(completion) -> str:
    if completion is None:
      return ""
    if isinstance(completion, str):
      return completion
    if isinstance(completion, dict):
      content = completion.get("content")
      if isinstance(content, str):
        return content
    return str(completion)

  @staticmethod
  def truncate_text(text: str, limit: int = 200) -> str:
    if len(text) <= limit:
      return text
    return f"{text[:limit]}..."

  @staticmethod
  def extract_sql(completion: str) -> str:
    text = _Nl2SqlHelper.completion_to_text(completion).strip()
    fence_match = _Nl2SqlHelper._CODE_FENCE.search(text)
    if fence_match:
      text = fence_match.group(1).strip()

    select_match = re.search(r"\bselect\b", text, flags=re.IGNORECASE)
    if not select_match:
      return ""

    text = text[select_match.start():].strip()
    semicolon_index = text.find(";")
    if semicolon_index != -1:
      text = text[:semicolon_index].strip()
    return text

  @staticmethod
  def normalize_value(value):
    if isinstance(value, bytes):
      return value.decode("utf-8")
    if isinstance(value, float):
      return round(value, 2)
    return value

  @staticmethod
  def normalize_rows(rows: Iterable[Iterable[object]]) -> List[List[object]]:
    return [[_Nl2SqlHelper.normalize_value(v) for v in row] for row in rows]

  @staticmethod
  def execute_sql(db_path: Path, sql: str) -> List[List[object]] | None:
    try:
      with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
      return _Nl2SqlHelper.normalize_rows(rows)
    except sqlite3.Error:
      return None


def _ensure_list(values) -> list:
  if values is None:
    return []
  if isinstance(values, list):
    return values
  try:
    return list(values)
  except TypeError:
    return [values]


def _coerce_id(value) -> str:
  if value is None:
    return ""
  if isinstance(value, (list, tuple)):
    if not value:
      return ""
    if len(value) == 1:
      return _coerce_id(value[0])
    return _coerce_id(value[0])
  try:
    import numpy as np  # pylint: disable=import-error

    if isinstance(value, np.ndarray):
      if value.size == 0:
        return ""
      if value.size == 1:
        return str(value.item())
      return str(value.flat[0])
  except Exception:
    pass
  return str(value)


def _expand_gold_results(
    gold_results: List[List[List[object]]] | None, target_len: int
) -> List[List[List[object]] | None]:
  if target_len <= 0:
    return []
  gold_results = _ensure_list(gold_results)
  if not gold_results:
    return [None] * target_len
  if len(gold_results) == target_len:
    return gold_results
  if target_len % len(gold_results) == 0:
    group_size = target_len // len(gold_results)
    return [gold_results[i // group_size] for i in range(target_len)]
  return (gold_results * target_len)[:target_len]


def _align_to_length(values, target_len: int, fill_value) -> list:
  values = _ensure_list(values)
  if target_len <= 0:
    return []
  if not values:
    return [fill_value] * target_len
  if len(values) == target_len:
    return values
  if target_len % len(values) == 0:
    group_size = target_len // len(values)
    return [values[i // group_size] for i in range(target_len)]
  if len(values) > target_len:
    return values[:target_len]
  repeats = (target_len // len(values)) + 1
  return (values * repeats)[:target_len]


def nl2sql_reward(prompts, completions, gold_result, **kwargs):
  """Computes NL2SQL rewards using execution success and result match."""
  global _DEBUG_ONCE
  project_root = config_lib.get_project_root()
  db_path = project_root / "examples/rl/grpo/nl2sql/example.sqlite"

  target_len = len(prompts)
  aligned_completions = _align_to_length(completions, target_len, "")
  aligned_ids = _align_to_length(kwargs.get("id"), target_len, None)
  if isinstance(gold_result, dict):
    expanded_gold = [
        gold_result.get(_coerce_id(example_id)) for example_id in aligned_ids
    ]
  else:
    expanded_gold = _expand_gold_results(gold_result, target_len)

  rewards = []
  debug_enabled = not _DEBUG_ONCE
  if debug_enabled:
    print("DEBUG nl2sql_reward kwargs keys:", sorted(kwargs.keys()))
    print("DEBUG nl2sql_reward prompt len:", len(prompts))
    print("DEBUG nl2sql_reward completion len:", len(completions))
    if prompts:
      print("DEBUG prompt[0]:", prompts[0])
    if aligned_completions:
      print("DEBUG completion[0]:", aligned_completions[0])
    if aligned_ids:
      print("DEBUG id[0]:", _coerce_id(aligned_ids[0]))

  for idx, (completion, expected) in enumerate(
      zip(aligned_completions, expanded_gold)
  ):
    if expected is None:
      rewards.append(0.0)
      continue
    sql = _Nl2SqlHelper.extract_sql(completion)
    if not sql or _Nl2SqlHelper._FORBIDDEN_SQL.search(sql):
      if debug_enabled and idx < 3:
        completion_text = _Nl2SqlHelper.completion_to_text(completion)
        print(
            "DEBUG reward sample:",
            {
                "id": _coerce_id(aligned_ids[idx]) if aligned_ids else None,
                "reason": "no_sql" if not sql else "forbidden_sql",
                "completion": _Nl2SqlHelper.truncate_text(completion_text),
            },
        )
      rewards.append(0.0)
      continue

    rows = _Nl2SqlHelper.execute_sql(db_path, sql)
    if rows is None:
      if debug_enabled and idx < 3:
        print(
            "DEBUG reward sample:",
            {
                "id": _coerce_id(aligned_ids[idx]) if aligned_ids else None,
                "reason": "sql_error",
                "pred_sql": sql,
            },
        )
      rewards.append(0.0)
      continue

    normalized_expected = _Nl2SqlHelper.normalize_rows(expected)
    result_score = 1.0 if rows == normalized_expected else 0.0
    reward = _EXEC_WEIGHT * 1.0 + _RESULT_WEIGHT * result_score
    rewards.append(reward)
    if debug_enabled and idx < 3:
      example_id = aligned_ids[idx] if aligned_ids else None
      print(
          "DEBUG reward sample:",
          {"id": _coerce_id(example_id), "pred_sql": sql, "reward": reward},
      )
  if debug_enabled:
    _DEBUG_ONCE = True
  return rewards

