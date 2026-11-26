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
"""Unit tests for ast_gate_lib."""

from __future__ import annotations

import ast
import pathlib
import sys
import unittest

if str(pathlib.Path(__file__).resolve().parent) not in sys.path:
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
  from scripts import ast_gate_lib  # pylint: disable=g-import-not-at-top
except ImportError:
  import ast_gate_lib  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]


class AstGateLibTest(unittest.TestCase):

  def test_try_parse_ast_and_has_docstring(self):
    tree, err = ast_gate_lib.try_parse_ast('"""Module doc."""\nx = 1\n')
    self.assertIsNotNone(tree)
    self.assertEqual(err, "")
    assert tree is not None
    self.assertTrue(ast_gate_lib.has_docstring(tree))
    self.assertFalse(ast_gate_lib.has_docstring(tree.body[1]))

    bad_tree, bad_err = ast_gate_lib.try_parse_ast("def broken(\n")
    self.assertIsNone(bad_tree)
    self.assertIn("SyntaxError", bad_err)

  def test_iter_scope_defs_extracts_methods_lambdas_and_partials(self):
    scope_src = (
        "class C:\n"
        '  def m(self):\n    """doc"""\n    if True:\n      return 1\n'
        "lam = lambda x: x + 1\n"
        "part: object = functools.partial(int, base=2)\n"
        "top = 42\n"
    )
    scope_tree = ast.parse(scope_src)
    defs = ast_gate_lib.iter_scope_defs(scope_tree)
    self.assertEqual([q for q, _, _ in defs], ["C.m", "lam", "part"])


if __name__ == "__main__":
  unittest.main()
