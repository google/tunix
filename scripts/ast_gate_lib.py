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
"""Core AST traversal and scope utilities for the Tunix CL complexity check."""

from __future__ import annotations

import ast
import re

FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
BLOCK_NODES = (ast.stmt, ast.ExceptHandler, ast.match_case)


def try_parse_ast(source: str) -> tuple[ast.Module | None, str]:
  """Parses `source` into an AST or returns `(None, error_detail)`."""
  try:
    return ast.parse(source), ""
  except (SyntaxError, ValueError) as exc:
    return None, f"{type(exc).__name__}: {exc}"


def has_docstring(node: ast.AST) -> bool:
  """Returns True if `node` is a module, class, or function with a docstring."""
  return (
      isinstance(node, (ast.Module, ast.ClassDef, *FUNC_TYPES))
      and ast.get_docstring(node, clean=False) is not None
  )


def iter_scope_defs(
    tree: ast.Module,
) -> list[tuple[str, ast.stmt | ast.expr, list[ast.stmt]]]:
  """Collects `(qualname, node, body)` across modules, classes, and blocks."""
  stack: list[tuple[str, ast.AST]] = [("", s) for s in reversed(tree.body)]
  out: list[tuple[str, ast.stmt | ast.expr, list[ast.stmt]]] = []
  while stack:
    pref, node = stack.pop()
    if isinstance(node, FUNC_TYPES):
      doc = has_docstring(node)
      body = [s for i, s in enumerate(node.body) if i > 0 or not doc]
      out.append((f"{pref}{node.name}", node, body))
    elif isinstance(node, ast.ClassDef):
      stack.extend((f"{pref}{node.name}.", c) for c in reversed(node.body))
    elif (
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, (ast.Lambda, ast.Call))
        and re.match(
            r"^[^=]+= *(lambda\b|(?:functools\.)?partial\s*\()",
            unparsed := ast.unparse(node),
        )
    ):
      lhs = unparsed.split("=", 1)[0].split(":", 1)[0].strip()
      if isinstance(node.value, ast.Lambda):
        out.append(
            (f"{pref}{lhs}", node.value, [ast.Return(value=node.value.body)])
        )
      else:
        out.append((f"{pref}{lhs}", node, [node]))
    else:
      kids = [
          c for c in ast.iter_child_nodes(node) if isinstance(c, BLOCK_NODES)
      ]
      stack.extend((pref, c) for c in reversed(kids))
  return out
