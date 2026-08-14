"""Exact, opt-in compatibility normalization for Q4 R2E-Gym actions.

The normal DeepSWE/Qwen3-32B path uses R2E-Gym's strict XML-like syntax and
does not call this module.  Qwen3-4B evaluation may explicitly enable this
compatibility mode because that model sometimes emits a small, observed
dialect: inline-valued parameter tags and top-level file-editor command names.

Only deterministic rewrites are admitted.  The raw model response remains in
the trajectory and the canonical action is stored in ``Step.action``.
"""

from __future__ import annotations

import re


STRICT_XML_MODE = "strict_xml"
Q4_R2EGYM_COMPAT_MODE = "q4_r2egym_xml_v2"
ACTION_COMPAT_MODES = frozenset({STRICT_XML_MODE, Q4_R2EGYM_COMPAT_MODE})


class ActionCompatibilityInternalError(RuntimeError):
  """Raised only when the compatibility layer itself corrupts an action."""


_FUNCTION = re.compile(r"<function\s*=\s*([^>]+)>")
# The closed form must be repaired before the open-only form.  In particular,
# ``<parameter=cmd=ls</parameter>`` must not treat ``ls</parameter`` as the
# value and create ``</parameter</parameter>``.
_INLINE_CLOSED_PARAMETER = re.compile(
    r"<parameter\s*=\s*([A-Za-z_][A-Za-z0-9_-]*)=([^<>\r\n]+)"
    r"</parameter\s*>"
)
_NESTED_PARAMETER_KEY = re.compile(
    r"<parameter\s*=\s*parameter\s*=\s*"
    r"([A-Za-z_][A-Za-z0-9_-]*)\s*>(.*?)</parameter\s*>",
    re.S,
)
_INLINE_OPEN_PARAMETER = re.compile(
    r"<parameter\s*=\s*([A-Za-z_][A-Za-z0-9_-]*)=([^<>\r\n]+)>"
)
_FILE_EDITOR_COMMAND = re.compile(
    r"<parameter\s*=\s*(view|create|str_replace|insert|undo_edit)\s*>"
)
_EXPLICIT_COMMAND_VALUE = re.compile(
    r"<parameter\s*=\s*command\s*>([^<>\r\n]+)</parameter\s*>"
)
_SUPPORTED_FUNCTIONS = frozenset({
    "execute_bash",
    "file_editor",
    "finish",
    "search",
})
_TOP_LEVEL_FILE_EDITOR = frozenset({
    "view",
    "create",
    "str_replace",
    "insert",
    "undo_edit",
})


def canonicalize_r2egym_action(action_text: str) -> tuple[str, int]:
  """Canonicalizes only the signed Q4 action dialect.

  Unknown tools and contradictory top-level editor shorthands are left
  untouched so they remain observable model errors instead of being guessed
  into a different program.
  """
  if not isinstance(action_text, str) or not action_text:
    return action_text, 0
  function_match = _FUNCTION.search(action_text)
  if not function_match:
    return action_text, 0
  raw_function = function_match.group(1).strip()
  if raw_function not in _SUPPORTED_FUNCTIONS | _TOP_LEVEL_FILE_EDITOR:
    return action_text, 0

  repair_count = 0

  def replace_parameter(match: re.Match[str]) -> str:
    nonlocal repair_count
    repair_count += 1
    return (
        f"<parameter={match.group(1)}>"
        f"{match.group(2).strip()}</parameter>"
    )

  canonical = _NESTED_PARAMETER_KEY.sub(replace_parameter, action_text)
  canonical = _INLINE_CLOSED_PARAMETER.sub(replace_parameter, canonical)
  canonical = _INLINE_OPEN_PARAMETER.sub(replace_parameter, canonical)

  if raw_function in _TOP_LEVEL_FILE_EDITOR:
    commands = [
        match.group(1).strip()
        for match in _EXPLICIT_COMMAND_VALUE.finditer(canonical)
    ]
    if commands and any(command != raw_function for command in commands):
      return action_text, 0
    canonical = (
        canonical[: function_match.start()]
        + "<function=file_editor>"
        + canonical[function_match.end() :]
    )
    repair_count += 1
    if not commands:
      opening_end = canonical.find(">", canonical.find("<function=")) + 1
      canonical = (
          canonical[:opening_end]
          + f"\n<parameter=command>{raw_function}</parameter>"
          + canonical[opening_end:]
      )
      repair_count += 1

  if raw_function == "file_editor" and not _EXPLICIT_COMMAND_VALUE.search(
      canonical
  ):

    def replace_command(match: re.Match[str]) -> str:
      nonlocal repair_count
      repair_count += 1
      return f"<parameter=command>{match.group(1)}</parameter>"

    canonical = _FILE_EDITOR_COMMAND.sub(replace_command, canonical)

  # This exact string was produced by the previous greedy implementation.  A
  # new repair must never synthesize malformed XML of the same class.
  if (
      repair_count
      and "</parameter</parameter>" in canonical
      and "</parameter</parameter>" not in action_text
  ):
    raise ActionCompatibilityInternalError(
        "Q4 action compatibility emitted a double closing tag"
    )
  return canonical, repair_count
