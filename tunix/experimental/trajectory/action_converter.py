# Copyright 2026 Google LLC
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

"""Action and ToolCall converter for extracting ATIF ToolCalls from action payloads."""

import dataclasses
import json
import re
from typing import Any
from absl import logging
import numpy as np
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.agents import agent_types

_ID_KEYS = ("id", "tool_call_id", "call_id", "function_id")
_NAME_KEYS = ("name", "function_name", "tool_name", "func", "function")
_ARG_KEYS = ("arguments", "args", "parameters", "params", "input", "inputs")
_META_KEYS = frozenset(_ID_KEYS + _NAME_KEYS + ("type",))
_NON_TOOL_TYPES = (int, float, bool, np.number, np.ndarray)


def _get_first(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
  """Returns the first non-None value in d matching any of the given keys."""
  for k in keys:
    val = d.get(k)
    if val is not None:
      return val
  return None


def _get_attr_first(
    obj: Any, keys: tuple[str, ...], default: Any = None
) -> Any:
  """Returns the first non-None attribute in obj matching any given keys."""
  for k in keys:
    val = getattr(obj, k, None)
    if val is not None:
      return val
  return default


def _clean_markdown_code_blocks(s: str) -> str:
  """Strips markdown code fences (e.g. ```json ... ```) from a string."""
  stripped = s.strip()
  if stripped.startswith("```"):
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
      lines = lines[1:]
    if lines and lines[-1].strip() == "```":
      lines = lines[:-1]
    stripped = "\n".join(lines).strip()
  return stripped


def _to_dict(obj: Any) -> dict[str, Any] | None:
  """Converts a dataclass, to_dict(), or __dict__ into a dictionary."""
  if isinstance(obj, dict):
    return obj
  if callable(getattr(obj, "to_dict", None)):
    d = obj.to_dict()
    if isinstance(d, dict):
      return d
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    return dataclasses.asdict(obj)
  if hasattr(obj, "__dict__") and not isinstance(obj, type):
    return dict(obj.__dict__)
  return None


def _normalize_arguments(raw_args: Any) -> dict[str, Any]:
  """Normalizes arguments into a dictionary."""
  if raw_args is None:
    return {}
  d = _to_dict(raw_args)
  if d is not None:
    return d
  if isinstance(raw_args, str):
    raw_str = _clean_markdown_code_blocks(raw_args)
    if raw_str:
      try:
        parsed = json.loads(raw_str)
        if isinstance(parsed, dict):
          return parsed
      except ValueError as e:
        logging.warning("Failed to parse arguments JSON string: %s", e)
  return {}


def _parse_dict_tool_call(
    item: dict[str, Any], default_id: str = "call_1"
) -> trajectory_lib.ToolCall | None:
  """Parses a dictionary into a ToolCall object."""
  if not isinstance(item, dict):
    return None

  call_id = _get_first(item, _ID_KEYS) or default_id

  # 1. OpenAI nested tool call format:
  # {"id": "...", "type": "function", "function": {"name": "...",
  # "arguments": ...}}
  if isinstance(item.get("function"), dict):
    fn_dict = item["function"]
    fn_name = _get_first(fn_dict, _NAME_KEYS) or "action"
    raw_args = _get_first(fn_dict, _ARG_KEYS)
    if raw_args is None and len(fn_dict) > 1:
      raw_args = {k: v for k, v in fn_dict.items() if k not in _META_KEYS}

  # 2. Anthropic tool use format:
  # {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
  elif item.get("type") == "tool_use":
    fn_name = _get_first(item, _NAME_KEYS) or "action"
    raw_args = _get_first(item, _ARG_KEYS)
    if raw_args is None and len(item) > 2:
      raw_args = {k: v for k, v in item.items() if k not in _META_KEYS}

  # 3. Flat dict format with aliases
  else:
    fn_name = _get_first(item, _NAME_KEYS)
    raw_args = _get_first(item, _ARG_KEYS)
    if raw_args is None:
      if fn_name is not None:
        raw_args = {k: v for k, v in item.items() if k not in _META_KEYS}
      else:
        other_keys = {
            k: v for k, v in item.items() if k not in _ID_KEYS and k != "type"
        }
        if not other_keys and item:
          return None
        fn_name = "action"
        raw_args = item

  return trajectory_lib.ToolCall(
      tool_call_id=str(call_id),
      function_name=str(fn_name or "action"),
      arguments=_normalize_arguments(raw_args),
  )


def _parse_single_tool_call(
    item: Any, default_id: str = "call_1"
) -> trajectory_lib.ToolCall | None:
  """Parses a single tool call item (dict, object, ToolCall) into a ToolCall."""
  if item is None or isinstance(item, _NON_TOOL_TYPES):
    return None

  if isinstance(item, trajectory_lib.ToolCall):
    return item

  d = _to_dict(item)
  if d is not None:
    return _parse_dict_tool_call(d, default_id=default_id)

  # Fallback: check properties via getattr
  fn_name = _get_attr_first(item, _NAME_KEYS)
  if fn_name is not None:
    call_id = _get_attr_first(item, _ID_KEYS, default_id)
    raw_args = _get_attr_first(item, _ARG_KEYS)
    return trajectory_lib.ToolCall(
        tool_call_id=str(call_id),
        function_name=str(fn_name),
        arguments=_normalize_arguments(raw_args),
    )

  return None


def _parse_xml_function_blocks(
    text: str, start_idx: int = 1
) -> list[trajectory_lib.ToolCall] | None:
  """Extracts ToolCalls from <function=NAME> or <invoke name=NAME> XML blocks."""
  function_matches = list(
      re.finditer(
          r"<(?:function|invoke)(?:\s*=\s*|\s+name=\s*)([^>]+)>(.*?)(?:</(?:function|invoke)>|$)",
          text,
          re.DOTALL,
      )
  )
  if not function_matches:
    return None

  calls = []
  for idx_offset, match in enumerate(function_matches):
    idx = start_idx + idx_offset
    fn_name = match.group(1).strip().strip("\"'")
    body = match.group(2).strip()
    if not fn_name:
      continue

    param_matches = re.findall(
        r"<parameter(?:\s*=\s*|\s+name=\s*)([^>]+)>(.*?)(?:</parameter>|$)",
        body,
        re.DOTALL,
    )
    cleaned_body = _clean_markdown_code_blocks(body)
    if param_matches:
      args = {k.strip().strip("\"'"): v.strip() for k, v in param_matches}
    elif cleaned_body.startswith("{"):
      try:
        parsed_body = json.loads(cleaned_body)
        args = parsed_body if isinstance(parsed_body, dict) else {}
      except ValueError as e:
        logging.warning("Failed to parse XML function body JSON: %s", e)
        args = {}
    elif not body:
      args = {}
    else:
      args = {"input": body}

    calls.append(
        trajectory_lib.ToolCall(
            tool_call_id=f"call_{idx}",
            function_name=fn_name,
            arguments=args,
        )
    )

  return calls or None


def _parse_string_tool_calls(
    text: str,
) -> list[trajectory_lib.ToolCall] | None:
  """Parses tool calls from a string (JSON or XML formats)."""
  text_str = text.strip()
  if not text_str:
    return None

  cleaned_text = _clean_markdown_code_blocks(text_str)

  # 1. Try parsing JSON string (dict or list of dicts)
  if cleaned_text.startswith(("{", "[")):
    try:
      parsed_json = json.loads(cleaned_text)
      if isinstance(parsed_json, (dict, list)):
        calls = _extract_tool_calls_from_payload(parsed_json)
        if calls:
          return calls
    except ValueError as e:
      logging.warning("Failed to parse JSON tool calls string: %s", e)

  # 2. Qwen/Hermes <tool_call> tags or <function_call> tags
  if "<tool_call>" in text_str or "<function_call>" in text_str:
    tool_call_blocks = re.findall(
        r"<(?:tool_call|function_call)>(.*?)(?:</(?:tool_call|function_call)>|$)",
        text_str,
        re.DOTALL,
    )
    calls = []
    for block in tool_call_blocks:
      block_str = _clean_markdown_code_blocks(block.strip())
      if not block_str:
        continue
      if block_str.startswith(("{", "[")):
        try:
          parsed = json.loads(block_str)
          if isinstance(parsed, dict):
            tc = _parse_dict_tool_call(
                parsed, default_id=f"call_{len(calls) + 1}"
            )
            if tc:
              calls.append(tc)
              continue
          elif isinstance(parsed, list):
            sub_calls = _extract_tool_calls_from_payload(parsed)
            if sub_calls:
              calls.extend(sub_calls)
              continue
        except ValueError as e:
          logging.warning("Failed to parse tool call block JSON: %s", e)
      if "<name>" in block_str or "<function_name>" in block_str:
        fn_match = re.search(
            r"<(?:name|function_name)>(.*?)</(?:name|function_name)>",
            block_str,
            re.DOTALL,
        )
        args_match = re.search(
            r"<(?:arguments|parameters|args|params|input)>(.*?)</(?:arguments|parameters|args|params|input)>",
            block_str,
            re.DOTALL,
        )
        if fn_match:
          fn_n = fn_match.group(1).strip()
          raw_a = args_match.group(1).strip() if args_match else "{}"
          norm_a = _normalize_arguments(raw_a)
          calls.append(
              trajectory_lib.ToolCall(
                  tool_call_id=f"call_{len(calls) + 1}",
                  function_name=fn_n,
                  arguments=norm_a,
              )
          )
          continue
      if "<function" in block_str or "<invoke" in block_str:
        sub_calls = _parse_xml_function_blocks(
            block_str, start_idx=len(calls) + 1
        )
        if sub_calls:
          calls.extend(sub_calls)
          continue
    if calls:
      return calls

  # 3. SWE-agent / XML format: <function=NAME>...<parameter=KEY>VAL...
  if "<function" in text_str or "<invoke" in text_str:
    calls = _parse_xml_function_blocks(text_str, start_idx=1)
    if calls:
      return calls

  return None


def _extract_tool_calls_from_payload(
    payload: Any,
) -> list[trajectory_lib.ToolCall] | None:
  """Extracts ToolCalls from an unwrapped action payload."""
  if payload is None or isinstance(payload, _NON_TOOL_TYPES):
    return None

  if not payload and isinstance(payload, (dict, list, tuple, str)):
    return None

  if isinstance(payload, str):
    return _parse_string_tool_calls(payload)

  if isinstance(payload, (list, tuple)):
    calls = []
    for item in payload:
      if isinstance(item, str):
        parsed_str_calls = _parse_string_tool_calls(item)
        if parsed_str_calls:
          calls.extend(parsed_str_calls)
      else:
        tc = _parse_single_tool_call(item, default_id=f"call_{len(calls) + 1}")
        if tc:
          calls.append(tc)
    return calls or None

  single_tc = _parse_single_tool_call(payload, default_id="call_1")
  return [single_tc] if single_tc else None


def extract_tool_calls(
    action: Any,
) -> list[trajectory_lib.ToolCall] | None:
  """Extracts ATIF ToolCalls from an RL Action or raw action payload.

  Args:
    action: An agent_types.Action instance, tool call object, dict, string, or
      list of tool calls.

  Returns:
    A list of trajectory_lib.ToolCall objects, or None if no valid tool calls
    are found.
  """
  if action is None:
    return None

  payload = action
  while isinstance(payload, agent_types.Action) or (
      hasattr(payload, "action")
      and not isinstance(payload, (trajectory_lib.ToolCall, dict))
      and not any(hasattr(payload, k) for k in _NAME_KEYS)
  ):
    payload = payload.action
    if payload is None:
      return None

  return _extract_tool_calls_from_payload(payload)
