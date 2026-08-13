"""Compatibility normalization for R2E-Gym XML-like tool actions.

R2E-Gym expects parameter values between an opening and closing tag, for
example ``<parameter=command>view</parameter>``.  Small instruction-tuned
models sometimes put the value in the opening tag instead, for example
``<parameter=command=view>``.  The upstream parser accepts that text but
interprets ``command=view`` as the parameter name.  Its bash adapter then
emits ``--command=view`` even though ``file_editor`` requires ``view`` as a
positional argument.

Only the observed R2E tool functions and file-editor command shorthands are
repaired here.  The original model response remains in the trajectory, while
the canonical action executed by the environment is visible in the step's
``action`` field.
"""

from __future__ import annotations

import re


_FUNCTION = re.compile(r"<function\s*=\s*([^>]+)>")
_INLINE_PARAMETER = re.compile(
    r"<parameter\s*=\s*([A-Za-z_][A-Za-z0-9_-]*)=([^>\r\n]+)>"
)
_FILE_EDITOR_COMMAND = re.compile(
    r"<parameter\s*=\s*(view|create|str_replace|insert|undo_edit)\s*>"
)
_EXPLICIT_COMMAND = re.compile(r"<parameter\s*=\s*command\s*>")
_SUPPORTED_FUNCTIONS = frozenset({
    "execute_bash",
    "file_editor",
    "finish",
    "search",
})


def canonicalize_r2egym_action(action_text: str) -> tuple[str, int]:
  """Repairs the observed inline-valued R2E parameter tags.

  Extra closing ``</parameter>`` tags left by a nested malformed response are
  harmless to the pinned R2E-Gym parser.  Keeping them also avoids rewriting
  any model-provided multiline payload beyond the exact malformed open tags.

  Args:
    action_text: The first XML-like function block from the model response.

  Returns:
    A tuple of ``(canonical_text, repair_count)``.
  """
  if not isinstance(action_text, str) or not action_text:
    return action_text, 0
  function_match = _FUNCTION.search(action_text)
  if not function_match:
    return action_text, 0
  function_name = function_match.group(1).strip()
  if function_name not in _SUPPORTED_FUNCTIONS:
    return action_text, 0

  repair_count = 0

  def replace(match: re.Match[str]) -> str:
    nonlocal repair_count
    repair_count += 1
    key = match.group(1)
    value = match.group(2).strip()
    return f"<parameter={key}>{value}</parameter>"

  canonical = _INLINE_PARAMETER.sub(replace, action_text)
  if function_name == "file_editor" and not _EXPLICIT_COMMAND.search(canonical):

    def replace_command(match: re.Match[str]) -> str:
      nonlocal repair_count
      repair_count += 1
      return f"<parameter=command>{match.group(1)}</parameter>"

    canonical = _FILE_EDITOR_COMMAND.sub(replace_command, canonical)
  return canonical, repair_count
