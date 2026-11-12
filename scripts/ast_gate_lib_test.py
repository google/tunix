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
import json
import os
import pathlib
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

if str(pathlib.Path(__file__).resolve().parent) not in sys.path:
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
  from scripts import ast_gate_lib  # pylint: disable=g-import-not-at-top
except ImportError:
  import ast_gate_lib  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]


class AstGateLibTest(unittest.TestCase):

  def test_try_parse_ast_and_iter_scope_defs(self):
    tree, err = ast_gate_lib.try_parse_ast("x = 1\n")
    self.assertIsNotNone(tree)
    self.assertEqual(err, "")
    bad_tree, bad_err = ast_gate_lib.try_parse_ast("def broken(\n")
    self.assertIsNone(bad_tree)
    self.assertIn("SyntaxError", bad_err)

    scope_src = (
        "class C:\n"
        '  def m(self):\n    """doc"""\n    return 1\n'
        "lam = lambda x: x + 1\n"
        "part = functools.partial(int, base=2)\n"
    )
    scope_tree = ast.parse(scope_src)
    defs = ast_gate_lib.iter_scope_defs(scope_tree)
    self.assertEqual([q for q, _, _ in defs], ["C.m", "lam", "part"])

  def test_format_public_signature(self):
    code = (
        "GLOBAL_CONST: int = 100\n"
        "import sys\n"
        "def add(a: int, b: int=0) -> int:\n"
        "  return a + b\n"
        "async def fetch(url: str):\n"
        "  pass\n"
        "class Model:\n"
        "  name: str\n"
        "  def __init__(self, name: str):\n"
        "    self.name = name\n"
        "class Empty:\n"
        "  pass\n"
        "def _private():\n"
        "  pass\n"
    )
    sigs = ast_gate_lib.extract_public_signatures(code)
    self.assertIn("GLOBAL_CONST", sigs)
    self.assertIn("GLOBAL_CONST: int", sigs["GLOBAL_CONST"])
    self.assertIn("add", sigs)
    self.assertIn("def add(a: int, b: int=0) -> int", sigs["add"])
    self.assertIn("fetch", sigs)
    self.assertIn("async def fetch(url: str)", sigs["fetch"])
    self.assertIn("Model", sigs)
    self.assertIn("class Model: __init__(self, name: str)", sigs["Model"])
    self.assertIn("Empty", sigs)
    self.assertNotIn("_private", sigs)
    self.assertNotIn("sys", sigs)
    self.assertEqual(
        ast_gate_lib.extract_public_signatures("def broken(\n"), {}
    )

  def test_extract_dict_and_typed_dict_keys(self):
    code = (
        "def parse(d):\n"
        "  a = d['target']\n"
        "  b = d.get('optional_key', 42)\n"
        "  c = d.pop('remove_me', None)\n"
        "  d.setdefault('default_k', 'val')\n"
        "  other = d.items()\n"
        "  indexed = d[0]\n"
        "  call_fn = fn(1)\n"
    )
    keys = ast_gate_lib.extract_dict_key_accesses(code)
    self.assertEqual(keys, {"target", "optional_key", "remove_me", "default_k"})
    self.assertEqual(
        ast_gate_lib.extract_dict_key_accesses("def broken(\n"), set()
    )

    td_code = (
        "class Config:\n"
        "  host: str\n"
        "  port: int\n"
        "class MethodsOnly:\n"
        "  def run(self): pass\n"
    )
    defs = ast_gate_lib.extract_typed_dict_keys(td_code)
    self.assertEqual(defs, {"Config": {"host", "port"}})
    self.assertEqual(ast_gate_lib.extract_typed_dict_keys("def broken(\n"), {})

  def test_extract_schema_properties_and_test_assertions(self):
    schema = {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "deps": {"type": "array"},
                    },
                },
            },
        },
    }
    self.assertEqual(
        ast_gate_lib.extract_schema_properties(schema),
        {"task", "nodes", "id", "deps"},
    )
    self.assertEqual(
        ast_gate_lib.extract_schema_properties(json.dumps(schema)),
        {"task", "nodes", "id", "deps"},
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json") as fp:
      json.dump(schema, fp)
      fp.flush()
      self.assertEqual(
          ast_gate_lib.extract_schema_properties(fp.name),
          {"task", "nodes", "id", "deps"},
      )
    self.assertEqual(
        ast_gate_lib.extract_schema_properties("invalid json {"), set()
    )
    self.assertEqual(
        ast_gate_lib.extract_schema_properties(123),  # pyrefly: ignore[bad-argument-type]
        set(),
    )

    test_code = (
        "def test_foo(self):\n"
        "  assert 1 == 1\n"
        "  self.assertEqual(2, 2)\n"
        "  self.assertTrue(True)\n"
        "  x = 3\n"
    )
    self.assertEqual(ast_gate_lib.count_test_assertions(test_code), 3)
    self.assertEqual(ast_gate_lib.count_test_assertions("def broken(\n"), 0)

  def test_hygiene_scanners_and_evaluate_hygiene(self):
    prod_bad = textwrap.dedent("""\
        from unittest import mock
        def run():
          try:
            return mock.MagicMock()
          except Exception:
            pass
          except BaseException:
            continue
          except ValueError:
            pass
          except Exception:  # noqa: BLE001
            pass
    """)
    prod_tree = ast.parse(prod_bad)
    self.assertEqual(len(ast_gate_lib.scan_mock_ast(prod_tree)), 2)
    self.assertEqual(
        ast_gate_lib.scan_silent_except_ast(prod_tree, prod_bad.splitlines()),
        [
            (5, "[G1:silent-except] Silent broad except"),
            (7, "[G1:silent-except] Silent broad except"),
        ],
    )
    contract_src = textwrap.dedent("""\
        import dataclasses
        from typing import Protocol, TypedDict

        @dataclasses.dataclass
        class OneKnob:
          lr: float = 1e-3

        class OneDict(TypedDict):
          lr: float

        class OneProto(Protocol):
          def step(self, x: int) -> int:
            ...

        @dataclasses.dataclass
        class TwoKnobs:
          lr: float
          wd: float
    """)
    self.assertEqual(
        ast_gate_lib.scan_trivial_contract_ast(ast.parse(contract_src)),
        [
            (5, "[G1:trivial-contract] 1-field OneKnob; prefer plain typed arg"),
            (8, "[G1:trivial-contract] 1-field OneDict; prefer plain typed arg"),
            (
                11,
                "[G1:trivial-contract] 1-method Protocol OneProto;"
                " prefer Callable",
            ),
        ],
    )
    self.assertEqual(
        len(ast_gate_lib.scan_source_hygiene("pkg/prod.py", prod_bad)), 4
    )
    self.assertEqual(
        ast_gate_lib.scan_source_hygiene("pkg/prod.py", prod_bad, prod_bad), []
    )
    self.assertEqual(
        ast_gate_lib.scan_source_hygiene("pkg/testdata/a.py", prod_bad), []
    )
    self.assertEqual(
        ast_gate_lib.scan_source_hygiene("pkg/prod.py", "def broken(\n"), []
    )
    duck_src = (
        "def f(x, d):\n"
        "  a = getattr(x, 'k', 0)\n"
        "  b = getattr(x, 'k')\n"
        "  c = hasattr(x, 'k')\n"
        "  e = d.get('k', 1)\n"
        "  g = d.get('k')\n"
        "  return a, b, c, e, g\n"
    )
    self.assertEqual(ast_gate_lib.count_duck_typing_calls(duck_src), 3)
    self.assertEqual(ast_gate_lib.count_duck_typing_calls("def broken(\n"), 0)
    self.assertEqual(
        ast_gate_lib.sum_duck_typing_delta(
            {"a.py": (duck_src, "def f(x):\n  return hasattr(x, 'k')\n")},
            ["a.py"],
        ),
        2,
    )

    test_bad = textwrap.dedent("""\
        import unittest
        @unittest.skip("legacy")
        def test_skipped():
          assert 1 == 1
        def test_call_skip(self):
          self.skipTest("unconditional")
        def test_empty():
          \"\"\"Only docstring.\"\"\"
          pass
    """)
    test_tree = ast.parse(test_bad)
    with mock.patch.dict(os.environ, {"ALLOW_TEST_SKIP_WITH_BUG": "true"}):
      self.assertEqual(len(ast_gate_lib.scan_test_skip_ast(test_tree)), 2)
    with mock.patch.dict(os.environ, {"ALLOW_TEST_SKIP_WITH_BUG": "#42"}):
      self.assertEqual(ast_gate_lib.scan_test_skip_ast(test_tree), [])
    self.assertEqual(
        ast_gate_lib.scan_vacuous_test_ast(test_tree),
        [(7, "[G2:vacuous-test] Empty test_empty")],
    )

    with tempfile.TemporaryDirectory() as tmp:
      ws = pathlib.Path(tmp)
      (ws / "prod.py").write_text(prod_bad, encoding="utf-8")
      (ws / "prod_test.py").write_text(test_bad, encoding="utf-8")
      self.assertEqual(
          ast_gate_lib.read_candidate_source(
              ws, "missing.py", {"in_mem.py": "x = 1\n"}
          ),
          ("missing.py", ""),
      )
      self.assertEqual(
          ast_gate_lib.read_candidate_source(
              ws, str(ws / "in_mem.py"), {"in_mem.py": "x = 1\n"}
          ),
          ("in_mem.py", "x = 1\n"),
      )
      viols = ast_gate_lib.evaluate_hygiene(
          ws,
          ["prod.py", "prod_test.py"],
          bases={"prod.py": prod_bad},
          ratchet_base=True,
      )
      self.assertTrue(any("prod_test.py" in v for v in viols))
      self.assertFalse(any("Hygiene prod.py:" in v for v in viols))

  def test_coverage_targets_and_evaluate_test_coverage(self):
    self.assertEqual(
        ast_gate_lib.extract_coverage_targets("def broken(\n"), ([], set())
    )
    self.assertEqual(
        ast_gate_lib.extract_coverage_targets("def s():\n  ...\n"), ([], set())
    )
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      self.assertEqual(ast_gate_lib.evaluate_test_coverage(root, []), [])
      mod_p = root / "calc.py"
      mod_p.write_text(
          "def add(a: int, b: int) -> int:\n  return a + b\n"
          "def sub(a: int, b: int) -> int:\n  return a - b\n",
          encoding="utf-8",
      )
      miss = ast_gate_lib.evaluate_test_coverage(root, ["calc.py"])
      self.assertTrue(any("G2_MISSING_UNIT_TEST" in v for v in miss))

      test_p = root / "calc_test.py"
      test_p.write_text(
          "import unittest\nimport calc\n"
          "class T(unittest.TestCase):\n"
          "  def test_add(self):\n"
          "    self.assertEqual(calc.add(1, 2), 3)\n",
          encoding="utf-8",
      )
      uncov = ast_gate_lib.evaluate_test_coverage(root, ["calc.py"])
      self.assertTrue(any("G2_UNCOVERED_FUNCTIONS" in v for v in uncov))

      low_p = root / "branchy.py"
      low_p.write_text(
          "def run(flag: bool) -> int:\n"
          "  if flag:\n    return 1\n"
          "  a = 1\n  b = 2\n  c = 3\n  d = 4\n  e = 5\n"
          "  f = 6\n  g = 7\n  h = 8\n  i = 9\n"
          "  return a + b + c + d + e + f + g + h + i\n",
          encoding="utf-8",
      )
      (root / "branchy_test.py").write_text(
          "import unittest\nimport branchy\n"
          "class T(unittest.TestCase):\n"
          "  def test_flag(self):\n"
          "    self.assertEqual(branchy.run(True), 1)\n",
          encoding="utf-8",
      )
      low_v = ast_gate_lib.evaluate_test_coverage(root, ["branchy.py"])
      self.assertTrue(any("G2_LOW_STMT_COVERAGE" in v for v in low_v))

      test_p.write_text(
          "import unittest\nimport calc\n"
          "class T(unittest.TestCase):\n"
          "  def test_all(self):\n"
          "    self.assertEqual(calc.add(1, 2), 3)\n"
          "    self.assertEqual(calc.sub(3, 1), 2)\n",
          encoding="utf-8",
      )
      self.assertEqual(
          ast_gate_lib.evaluate_test_coverage(root, ["calc.py"]), []
      )
      with (
          mock.patch.object(ast_gate_lib.sys, "executable", None),
          mock.patch.object(ast_gate_lib.shutil, "which", return_value=None),
      ):
        self.assertEqual(
            ast_gate_lib.evaluate_test_coverage(root, ["calc.py"]), []
        )

      test_p.write_text("raise RuntimeError('broken')\n", encoding="utf-8")
      fail_v = ast_gate_lib.evaluate_test_coverage(root, ["calc.py"])
      self.assertTrue(any("G2_UNIT_TEST_FAILURE" in v for v in fail_v))
      with (
          mock.patch.object(ast_gate_lib.sys, "executable", None),
          mock.patch.object(ast_gate_lib.shutil, "which", return_value=None),
      ):
        fail_inproc = ast_gate_lib.evaluate_test_coverage(root, ["calc.py"])
        self.assertTrue(any("G2_UNIT_TEST_FAILURE" in v for v in fail_inproc))


if __name__ == "__main__":
  unittest.main()
