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
"""Unit tests for the Tunix 3D CL Understandability & Complexity Gate."""

from __future__ import annotations

import collections
import contextlib
import io
import os
import pathlib
import textwrap
from unittest import mock

from absl.testing import absltest
from scripts import check_cl_complexity

_CONFIG = pathlib.Path(__file__).parent.parent / "pyproject.toml"


class CheckClComplexityTest(absltest.TestCase):

  def test_load_config_reads_canonical_toml_thresholds(self):
    cfg = check_cl_complexity.load_config(_CONFIG)
    self.assertEqual(cfg.max_prod_loc_fail, 400)
    self.assertEqual(cfg.max_cognitive_fail, 15)
    self.assertEqual(cfg.max_call_depth_fail, 3)
    self.assertEqual(cfg.max_trivial_helpers_fail, 3)

  def test_count_code_loc_excludes_comments_and_docstrings(self):
    src = textwrap.dedent('''\
        """Module
        multiline docstring.
        """
        # Leading comment.
        def compute(x: int) -> int:
          """Function docstring."""
          return x + 1
    ''')
    self.assertEqual(check_cl_complexity.count_code_loc(src), {5, 7})

  def test_analyze_functions_computes_cognitive_and_closures(self):
    src = textwrap.dedent("""\
        def branchy(items: list[int]) -> int:
          total = 0
          def _inner(val: int) -> int:
            return val + 1
          for x in items:
            if x > 0 and x < 10:
              if x % 2 == 0:
                total += _inner(x)
          return total

        class Walker:
          def step(self, n: int) -> int:
            return self.step(n - 1) if n > 0 else 0

        _alias = functools.partial(branchy, fn=lambda v: v + 1)
    """)
    funcs = check_cl_complexity.analyze_functions(src, collections.Counter())
    self.assertLen(funcs, 3)
    fn, method_fn, alias_fn = funcs
    self.assertEqual(fn.qualname, "branchy")
    self.assertEqual(fn.cognitive, 8)
    self.assertEqual(fn.delta_cognitive, 8)
    self.assertEqual(fn.cyclomatic, 6)
    self.assertEqual(fn.max_nesting, 3)
    self.assertEqual(fn.nested_closures, ("branchy._inner",))
    self.assertFalse(fn.is_pass_through)
    self.assertEqual(method_fn.qualname, "Walker.step")
    self.assertEqual(method_fn.cognitive, 2)
    self.assertIn("Walker.step", method_fn.calls)
    self.assertEqual(alias_fn.qualname, "_alias")
    self.assertEqual(alias_fn.cognitive, 1)
    self.assertEqual(alias_fn.cyclomatic, 2)
    self.assertEqual(alias_fn.nested_closures, ("_alias.<lambda>",))
    self.assertFalse(alias_fn.is_pass_through)
    self.assertIn("branchy", alias_fn.calls)

  def test_compute_call_graph_metrics_detects_deep_chain_and_trivial(self):
    src = textwrap.dedent("""\
        def entry(x: int) -> int:
          return _step1(x)

        def _step1(x: int) -> int:
          return _step2(x)

        def _step2(x: int) -> int:
          return _step3(x)

        def _step3(x: int) -> int:
          return _step4(x)

        def _step4(x: int) -> int:
          return x + 1
    """)
    cfg = check_cl_complexity.load_config(_CONFIG)
    funcs = check_cl_complexity.analyze_functions(src, collections.Counter())
    self.assertTrue(funcs[0].is_pass_through)
    self.assertFalse(funcs[-1].is_pass_through)
    hops, chain, trivial = check_cl_complexity.compute_call_graph_metrics(
        funcs, cfg
    )
    self.assertEqual(hops, 4)
    self.assertEqual(
        chain, ["entry", "_step1", "_step2", "_step3", "_step4"]
    )
    self.assertEqual(trivial, ["_step1", "_step2", "_step3", "_step4"])

  def test_evaluate_gate_passes_clean_module_and_fails_ravioli_module(self):
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    clean_py = tmp_dir / "clean.py"
    clean_py.write_text(
        textwrap.dedent("""\
            def transform(items: list[int]) -> list[int]:
              return [x * 2 for x in items if x > 0]
        """),
        encoding="utf-8",
    )
    cfg = check_cl_complexity.load_config(_CONFIG)
    pass_report = check_cl_complexity.evaluate_gate(
        tmp_dir, ["clean.py"], collections.defaultdict(str), cfg
    )
    self.assertTrue(pass_report.passed)
    self.assertEmpty(pass_report.violations)

    ravioli_py = tmp_dir / "ravioli.py"
    ravioli_py.write_text(
        textwrap.dedent("""\
            def run(x: int) -> int:
              return _h1(x)
            def _h1(x: int) -> int:
              return _h2(x)
            def _h2(x: int) -> int:
              return _h3(x)
            def _h3(x: int) -> int:
              return _h4(x)
            def _h4(x: int) -> int:
              return x + 1
        """),
        encoding="utf-8",
    )
    fail_code = check_cl_complexity.main([
        "--cwd",
        str(tmp_dir),
        "--config",
        str(_CONFIG),
        "--files",
        "ravioli.py",
        "--strict",
    ])
    self.assertEqual(fail_code, 1)
    with mock.patch.dict(os.environ, {"SKIP_CL_COMPLEXITY": "migration"}):
      skip_code = check_cl_complexity.main([
          "--cwd",
          str(tmp_dir),
          "--config",
          str(_CONFIG),
          "--files",
          "ravioli.py",
          "--strict",
      ])
    self.assertEqual(skip_code, 0)


if __name__ == "__main__":
  absltest.main()
