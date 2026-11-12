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
"""Unit tests for the Tunix CL Complexity Gate."""

from __future__ import annotations

import ast
import collections
import contextlib
import dataclasses
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock

if str(pathlib.Path(__file__).resolve().parent) not in sys.path:
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
  from absl.testing import absltest  # pylint: disable=g-import-not-at-top
except ImportError:

  class _FallbackTestCase(unittest.TestCase):

    def create_tempdir(self) -> types.SimpleNamespace:
      d = self.enterContext(tempfile.TemporaryDirectory())
      return types.SimpleNamespace(full_path=d)

    def assertLen(self, container, expected_len: int, msg=None) -> None:
      self.assertEqual(len(container), expected_len, msg)

    def assertEmpty(self, container, msg=None) -> None:
      self.assertEqual(len(container), 0, msg)

    def assertNotEmpty(self, container, msg=None) -> None:
      self.assertGreater(len(container), 0, msg)

  absltest = types.SimpleNamespace(  # pylint: disable=invalid-name
      TestCase=_FallbackTestCase, main=unittest.main
  )

try:
  from scripts import ast_gate_lib  # pylint: disable=g-import-not-at-top
  from scripts import check_cl_complexity  # pylint: disable=g-import-not-at-top
except ImportError:
  import ast_gate_lib  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]
  import check_cl_complexity  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

_CAND_CFG = pathlib.Path(__file__).parent.parent / "pyproject.toml"
_CONFIG = (
    _CAND_CFG
    if _CAND_CFG.is_file()
    else pathlib.Path(__file__).with_name("cl_complexity_config.toml")
)


class CheckClComplexityTest(absltest.TestCase):

  def test_load_config_reads_canonical_toml_thresholds(self):
    cfg = check_cl_complexity.load_config(_CONFIG)
    self.assertEqual(cfg.enabled_dimensions, (1, 2, 3, 4))
    self.assertEqual(cfg.max_cognitive_fail, 15)
    self.assertEqual(cfg.max_call_depth_fail, 2)
    self.assertEqual(cfg.max_trivial_helpers_fail, 3)

    tmp_cfg = pathlib.Path(self.create_tempdir().full_path) / "minimal.toml"
    tmp_cfg.write_text(
        textwrap.dedent("""\
            [weights]
            prod_loc = 0.15
            cognitive_excess = 2.0
            call_depth_excess = 15.0
            trivial_helper = 10.0
            pass_through_func = 10.0
            nested_closure = 5.0
            duplicate_clone = 20.0
            [thresholds]
            max_cognitive_warn = 12
            max_cognitive_fail = 15
            max_cyclomatic_fail = 15
            max_nesting_fail = 4
            max_legacy_cog_delta_fail = 5
            max_legacy_cyc_delta_fail = 5
            max_call_depth_warn = 2
            max_call_depth_fail = 2
            max_trivial_helpers_fail = 3
            max_pass_through_ratio_fail = 0.25
            max_nested_closures_warn = 2
            max_nested_closures_fail = 4
            trivial_helper_max_stmts = 4
            trivial_helper_max_cognitive = 1
            max_sprawl_helpers_fail = 5
            max_file_cog_delta_sprawl = 40
            min_clone_stmts = 7
            min_clone_cognitive = 4
            min_clone_jaccard = 0.60
        """),
        encoding="utf-8",
    )
    fallback_cfg = check_cl_complexity.load_config(tmp_cfg)
    self.assertEqual(fallback_cfg.enabled_dimensions, (1, 2, 3, 4))
    self.assertEqual(fallback_cfg.intentional_forks, ())
    empty_cwd = pathlib.Path(self.create_tempdir().full_path)
    prev_cwd = pathlib.Path.cwd()
    os.chdir(empty_cwd)
    try:
      pkg_cfg = check_cl_complexity.load_config()
    finally:
      os.chdir(prev_cwd)
    self.assertEqual(pkg_cfg.max_cognitive_fail, 15)
    with self.assertRaises(FileNotFoundError):
      check_cl_complexity.load_config(empty_cwd / "nonexistent.toml")

  def test_count_code_loc_and_decorator_info(self):
    src = textwrap.dedent('''\
        """Module
        multiline docstring.
        """
        # Leading comment.
        def compute(x: int) -> int:
          """Function docstring."""
          return x + 1

        def nodoc(y: int) -> int:
          return y * 2
    ''')
    self.assertEqual(check_cl_complexity.count_code_loc(src), {5, 7, 9, 10})
    self.assertEqual(check_cl_complexity.count_code_loc("   \n"), set())

    dec_tree = ast.parse(textwrap.dedent("""\
            class Box:
              @jax.jit(donate_argnums=(0,))
              def f(self, x):
                return x
              @val.setter
              def val(self, v):
                self._v = v
              @val.deleter
              def val(self):
                del self._v
        """))
    cls_node = dec_tree.body[0]
    self.assertIsInstance(cls_node, ast.ClassDef)
    self.assertEqual(
        check_cl_complexity.decorator_info(cls_node.body[0]), ({"jit"}, "")
    )
    self.assertEqual(
        check_cl_complexity.decorator_info(cls_node.body[1]),
        ({"setter"}, ":setter"),
    )
    self.assertEqual(
        check_cl_complexity.decorator_info(cls_node.body[2]),
        ({"deleter"}, ":deleter"),
    )

  def test_analyze_functions_computes_cognitive_and_closures(self):
    src = textwrap.dedent("""\
        def branchy(items: list[int]) -> int:
          total = 0
          def _inner(val: int) -> int:
            return _inner(val - 1) if val > 1 else val
          for x in items:
            if x > 0 and x < 10:
              if x % 2 == 0:
                total += _inner(x)
          return total

        class Walker:
          def step(self, n: int) -> int:
            _ = [
                y if y > 0 else 0
                for y in range(n)
                if (y > 0 and y < 5) or y == 9
            ]
            return self.step(n - 1) if n > 0 else 0

        _alias = functools.partial(branchy, fn=lambda v: v + 1)
        _passthrough_alias = functools.partial(branchy, [1])
    """)
    funcs = check_cl_complexity.analyze_functions(src)
    self.assertLen(funcs, 4)
    fn, method_fn, alias_fn, passthrough_alias_fn = funcs
    self.assertEqual(fn.qualname, "branchy")
    self.assertEqual(fn.cognitive, 10)
    self.assertEqual(fn.delta_cognitive, 10)
    self.assertEqual(fn.cyclomatic, 7)
    self.assertEqual(fn.max_nesting, 3)
    self.assertEqual(fn.nested_closures, ("branchy._inner",))
    self.assertFalse(fn.is_pass_through)
    self.assertEqual(method_fn.qualname, "Walker.step")
    self.assertEqual(method_fn.cognitive, 7)
    self.assertEqual(method_fn.cyclomatic, 6)
    self.assertEqual(method_fn.max_nesting, 2)
    self.assertIn("Walker.step", method_fn.calls)
    self.assertEqual(alias_fn.qualname, "_alias")
    self.assertEqual(alias_fn.cognitive, 0)
    self.assertTrue(alias_fn.is_pass_through)
    self.assertEqual(passthrough_alias_fn.qualname, "_passthrough_alias")
    self.assertTrue(passthrough_alias_fn.is_pass_through)
    self.assertEqual(alias_fn.cyclomatic, 1)
    self.assertEqual(alias_fn.nested_closures, ())
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
    funcs = check_cl_complexity.analyze_functions(src)
    self.assertTrue(funcs[0].is_pass_through)
    self.assertFalse(funcs[-1].is_pass_through)
    hops, chain, trivial = check_cl_complexity.compute_call_graph_metrics(
        funcs, cfg
    )
    self.assertEqual(hops, 4)
    self.assertEqual(
        chain, ["entry", "_step1", "_step2", "_step3", "_step4"]
    )
    self.assertEqual(trivial, ["_step1", "_step2", "_step3"])

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
    with mock.patch.dict(
        os.environ, {"PR_BODY": "SKIP_CL_COMPLEXITY=true"}, clear=False
    ):
      self.assertEqual(check_cl_complexity.resolve_skip_reason(tmp_dir), "")
    with mock.patch.dict(
        os.environ,
        {"PR_BODY": "Summary\nSKIP_CL_COMPLEXITY=mechanical_rename\n"},
        clear=False,
    ):
      self.assertEqual(
          check_cl_complexity.resolve_skip_reason(tmp_dir), "mechanical_rename"
      )

  def test_dimensions_filter_isolates_dim3_only(self):
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    (tmp_dir / "ravioli.py").write_text(
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
    self.assertEqual(
        check_cl_complexity.main([
            "--cwd",
            str(tmp_dir),
            "--config",
            str(_CONFIG),
            "--dimensions",
            "3",
            "--files",
            "ravioli.py",
            "--strict",
        ]),
        1,
    )
    self.assertEqual(
        check_cl_complexity.main([
            "--cwd",
            str(tmp_dir),
            "--config",
            str(_CONFIG),
            "--dimensions",
            "2",
            "--files",
            "ravioli.py",
            "--strict",
        ]),
        0,
    )

  def test_cross_file_move_guard_relief_cohesive_leaf_and_clone_hygiene(self):
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    cfg = check_cl_complexity.load_config(_CONFIG)
    legacy_fn = textwrap.dedent("""\
        def complex_legacy_step(items: list[int], mode: int) -> int:
          total = 0
          for x in items:
            if x > 0:
              if mode == 1:
                if x % 2 == 0:
                  total += x
                elif x % 3 == 0:
                  total -= x
              elif mode == 2:
                for y in range(x):
                  if y > 1 and y < 5:
                    total += y
          return total
    """)
    guarded_moved = legacy_fn.replace(
        "  total = 0\n",
        "  if not items:\n    return 0\n  total = 0\n",
    )
    (tmp_dir / "moved.py").write_text(guarded_moved, encoding="utf-8")
    bases = collections.defaultdict(str, {"old.py": legacy_fn})
    move_report = check_cl_complexity.evaluate_gate(
        tmp_dir, ["old.py", "moved.py"], bases, cfg
    )
    self.assertTrue(move_report.passed, move_report.violations)
    self.assertEqual(move_report.net_cog_delta, 1)

    base_file = tmp_dir / "base_old.py"
    base_file.write_text(legacy_fn, encoding="utf-8")
    self.assertEqual(
        check_cl_complexity.main([
            "--cwd",
            str(tmp_dir),
            "--config",
            str(_CONFIG),
            "--files",
            "moved.py",
            "--base",
            str(base_file),
            "--strict",
        ]),
        0,
    )

    leaf_py = tmp_dir / "cohesive_leaf.py"
    leaf_py.write_text(
        textwrap.dedent("""\
            def public_api(xs: list[int]) -> list[int]:
              return _normalize_values(xs)

            def _normalize_values(xs: list[int]) -> list[int]:
              scale = sum(xs) or 1
              return [x // scale for x in xs]
        """),
        encoding="utf-8",
    )
    leaf_report = check_cl_complexity.evaluate_gate(
        tmp_dir, ["cohesive_leaf.py"], collections.defaultdict(str), cfg
    )
    self.assertTrue(leaf_report.passed, leaf_report.violations)
    self.assertEmpty(leaf_report.trivial_single_call_helpers)

    clone_a = tmp_dir / "tunix/sft/peft_trainer.py"
    clone_b = tmp_dir / "tunix/dist_rl/train/peft_trainer_v2.py"
    for p in (clone_a, clone_b):
      p.parent.mkdir(parents=True, exist_ok=True)
      p.write_text(legacy_fn, encoding="utf-8")
    fork_report = check_cl_complexity.evaluate_gate(
        tmp_dir,
        [
            "tunix/sft/peft_trainer.py",
            "tunix/dist_rl/train/peft_trainer_v2.py",
        ],
        collections.defaultdict(str, {"tunix/sft/peft_trainer.py": legacy_fn}),
        cfg,
    )
    self.assertEmpty(fork_report.duplicate_clones)

    clone_c = tmp_dir / "tunix/rl/unexempt_copy.py"
    clone_c.parent.mkdir(parents=True, exist_ok=True)
    clone_c.write_text(legacy_fn, encoding="utf-8")
    dup_report = check_cl_complexity.evaluate_gate(
        tmp_dir,
        ["tunix/rl/unexempt_copy.py"],
        collections.defaultdict(str),
        cfg,
    )
    self.assertFalse(dup_report.passed)
    self.assertNotEmpty(dup_report.duplicate_clones)

    gen_py = tmp_dir / "stream_mismatch.py"
    gen_py.write_text(
        textwrap.dedent("""\
            def _drain_data_queue(q):
              while True:
                item = q.get(block=True)
                if item is None:
                  break
                yield item

            def _load_all_batches(q):
              yield q.get()

            def _collect_valid_records(chunks: list[bytes]) -> list[bytes]:
              def _iter_non_empty():
                for c in chunks:
                  if c:
                    yield c
              return list(_iter_non_empty())
        """),
        encoding="utf-8",
    )
    gen_report = check_cl_complexity.evaluate_gate(
        tmp_dir, ["stream_mismatch.py"], collections.defaultdict(str), cfg
    )
    self.assertFalse(gen_report.passed)
    self.assertTrue(
        any(
            "_drain_data_queue" in v
            and "_load_all_batches" in v
            and "_collect_valid_records" not in v
            for v in gen_report.violations
        ),
        gen_report.violations,
    )

    renamed_gen_report = check_cl_complexity.evaluate_gate(
        tmp_dir,
        ["stream_mismatch.py"],
        collections.defaultdict(
            str,
            {
                "stream_mismatch.py": textwrap.dedent("""\
                    def _iter_data_queue(q):
                      while True:
                        item = q.get(block=True)
                        if item is None:
                          break
                        yield item
                """)
            },
        ),
        cfg,
    )
    self.assertFalse(renamed_gen_report.passed)
    self.assertTrue(
        any("_drain_data_queue" in v for v in renamed_gen_report.violations),
        renamed_gen_report.violations,
    )

  def test_pass_through_move_shim_arrowhead_and_distinct_shingles(self):
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    cfg = check_cl_complexity.load_config(_CONFIG)

    # 1. Dataclass default_factory and string constants do not false-positive.
    dataclass_src = textwrap.dedent("""\
        import dataclasses, functools
        DOC_HINT = "Avoid using lambda functions or partial(f) in hot loops."
        @dataclasses.dataclass
        class TrainerConfig:
          batch_size: int = 8
          sorted_keys: list[str] = dataclasses.field(
              default_factory=functools.partial(
                  sorted, ["b", "a"], key=lambda s: s
              )
          )
        def run_trainer(cfg: TrainerConfig) -> list[str]:
          return cfg.sorted_keys
    """)
    (tmp_dir / "cfg_mod.py").write_text(dataclass_src, encoding="utf-8")
    cfg_report = check_cl_complexity.evaluate_gate(
        tmp_dir, ["cfg_mod.py"], collections.defaultdict(str), cfg
    )
    self.assertTrue(cfg_report.passed, cfg_report.violations)
    self.assertEmpty(cfg_report.pass_through_funcs)

    # 2. Moving a 3-hop chain across files or leaving a deprecation shim passes.
    chain_src = textwrap.dedent("""\
        def entry(x: int) -> int:
          y = x + 1
          return _h1(y)
        def _h1(x: int) -> int:
          y = x * 2
          return _h2(y)
        def _h2(x: int) -> int:
          y = x - 1
          return _h3(y)
        def _h3(x: int) -> int:
          return x * 3
    """)
    (tmp_dir / "new_chain.py").write_text(chain_src, encoding="utf-8")
    chain_move = check_cl_complexity.evaluate_gate(
        tmp_dir,
        ["old_chain.py", "new_chain.py"],
        collections.defaultdict(str, {"old_chain.py": chain_src}),
        cfg,
    )
    self.assertTrue(chain_move.passed, chain_move.violations)

    # 3. 5-deep arrowhead `if` fails max_nesting_fail = 4.
    arrowhead_src = textwrap.dedent("""\
        def deep_nested(a: int, b: int, c: int, d: int, e: int) -> int:
          if a:
            if b:
              if c:
                if d:
                  if e:
                    return 1
          return 0
    """)
    (tmp_dir / "arrowhead.py").write_text(arrowhead_src, encoding="utf-8")
    arrow_report = check_cl_complexity.evaluate_gate(
        tmp_dir, ["arrowhead.py"], collections.defaultdict(str), cfg
    )
    self.assertFalse(arrow_report.passed)
    self.assertEqual(arrow_report.max_nesting, 5)

    # 4. Distinct numerical loops with different operators do not collide.
    distinct_loops = textwrap.dedent("""\
        def compute_attention_dropout_mask(
            scores, threshold: float, scale: float
        ):
          total = 0.0
          out = []
          for row in scores:
            if row is not None:
              for val in row:
                if val > threshold:
                  total += val * scale
                  out.append(val)
                elif val == 0.0:
                  total -= 1.0
          return out, total

        def accumulate_gradient_norm_penalty(
            grads, limit: float, weight: float
        ):
          acc = 1.0
          clipped = []
          for tensor in grads:
            if tensor is not None:
              for item in tensor:
                if item < limit:
                  acc *= item + weight
                  clipped.append(item)
                elif item >= 100.0:
                  acc /= 2.0
          return clipped, acc
    """)
    (tmp_dir / "distinct.py").write_text(distinct_loops, encoding="utf-8")
    distinct_report = check_cl_complexity.evaluate_gate(
        tmp_dir, ["distinct.py"], collections.defaultdict(str), cfg
    )
    self.assertTrue(distinct_report.passed, distinct_report.violations)
    self.assertEmpty(distinct_report.duplicate_clones)

  def test_ast_stmts_soft_scope_and_clean_workdir_fallback(self):
    compact = "def f(a, b, c):\n  return a + b + c\n"
    wrapped = (
        "def f(\n    a,\n    b,\n    c,\n):\n"
        '  """Docstring."""\n  return (\n      a\n      + b\n      + c\n  )\n'
    )
    self.assertEqual(
        check_cl_complexity.count_ast_stmts(compact),
        check_cl_complexity.count_ast_stmts(wrapped),
    )
    fn_node = ast.parse("@p.setter\n@p.deleter\ndef p(self): pass").body[0]
    self.assertIsInstance(fn_node, ast.FunctionDef)
    self.assertEqual(check_cl_complexity.decorator_info(fn_node)[1], ":deleter")
    if_multi = ast.parse("if c:\n  x = 1\n  return x").body[0]
    if_else = ast.parse("if c:\n  return 1\nelse:\n  return 2").body[0]
    self.assertIsInstance(if_multi, ast.If)
    self.assertIsInstance(if_else, ast.If)
    self.assertFalse(check_cl_complexity.is_early_exit_guard(if_multi))
    self.assertFalse(check_cl_complexity.is_early_exit_guard(if_else))
    self.assertEqual(
        check_cl_complexity.canonical_ast_digest(
            ast.parse("return (lambda a: a + 1)(x)").body
        ),
        check_cl_complexity.canonical_ast_digest(
            ast.parse("return (lambda b: b + 1)(y)").body
        ),
    )
    self.assertNotEqual(
        check_cl_complexity.canonical_ast_digest(
            ast.parse("return (lambda a: a + 1)(x)").body
        ),
        check_cl_complexity.canonical_ast_digest(
            ast.parse("return (lambda b: b + 2)(y)").body
        ),
    )
    bad_pt = ast.parse("def f(x):\n  r = g(x)\n  return r + 1").body[0]
    good_pt = ast.parse("def f(x):\n  r = g(x)\n  return r").body[0]
    assign_only = ast.parse("def f(x):\n  r = g(x)").body[0]
    self.assertIsInstance(bad_pt, ast.FunctionDef)
    self.assertIsInstance(good_pt, ast.FunctionDef)
    self.assertIsInstance(assign_only, ast.FunctionDef)
    self.assertFalse(check_cl_complexity.is_pass_through(bad_pt, bad_pt.body))
    self.assertTrue(check_cl_complexity.is_pass_through(good_pt, good_pt.body))
    self.assertFalse(
        check_cl_complexity.is_pass_through(assign_only, assign_only.body)
    )
    elif_fm = check_cl_complexity.analyze_functions(
        "def f(a, b, c):\n  if a:\n    return 1\n"
        "  elif b and c:\n    if a:\n      return 2\n  else:\n    return 3\n",
        "elif.py",
    )[0]
    self.assertEqual((elif_fm.cognitive, elif_fm.cyclomatic), (6, 5))
    branch_fm = check_cl_complexity.analyze_functions(
        "def g(x, y):\n  match x and y:\n    case 1 if x and y:\n      pass\n"
        "  for i in (x and y):\n    pass\n  else:\n    pass\n"
        "  while x and y:\n    foo(x and y)\n"
        "  try:\n    pass\n  except ValueError:\n    pass\n  else:\n    pass\n"
        "  return [u for v in x for u in v if u] if y else []\n",
        "branch.py",
    )[0]
    self.assertEqual((branch_fm.cognitive, branch_fm.cyclomatic), (14, 13))

    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    cfg = check_cl_complexity.load_config(_CONFIG)
    large_clean = "\n\n".join(
        f"def helper_{i}(x: int) -> int:\n"
        + "".join(f"  v_{j} = x + {j}\n" for j in range(10))
        + "  return v_9\n"
        for i in range(42)
    )
    (tmp_dir / "large_clean.py").write_text(large_clean, encoding="utf-8")
    report = check_cl_complexity.evaluate_gate(
        tmp_dir, ["large_clean.py"], collections.defaultdict(str), cfg
    )
    self.assertGreater(report.net_loc_delta, 400)
    self.assertTrue(report.passed, report.violations)
    self.assertEmpty(report.violations)
    sub_dir = tmp_dir / "sub"
    sub_dir.mkdir()
    (sub_dir / "clean.py").write_text("def f(x: int) -> int:\n  return x + 1\n")
    self.assertEqual(
        check_cl_complexity.main([
            "--cwd",
            str(sub_dir),
            "--config",
            str(_CONFIG),
            "--files",
            "clean.py",
            "--strict",
        ]),
        0,
    )

  def test_visitor_and_gate_mutation_invariants(self):
    cfg = check_cl_complexity.load_config(_CONFIG)
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    dec_node = ast.parse("@foo\n@p.setter\ndef p(self): pass").body[0]
    self.assertEqual(
        check_cl_complexity.decorator_info(dec_node),
        ({"foo", "setter"}, ":setter"),
    )
    self.assertEqual(
        check_cl_complexity.canonical_ast_digest(
            ast.parse("return (lambda a: a + 1)(x)").body
        ),
        check_cl_complexity.canonical_ast_digest(
            ast.parse("return (lambda long_arg: long_arg + 1)(long_var)").body
        ),
    )
    f_assign = check_cl_complexity.analyze_functions(
        "def _c():\n  return 1\ndef f():\n  a, b = 1, 2\n  h = _c\n"
        "  s = [_c]\n  d = {'k': _c}\n  return h(), s, d, a, b\n"
    )[1]
    self.assertEqual(f_assign.calls, ("_c", "_c", "_c", "_c"))
    f_lam = check_cl_complexity.analyze_functions(
        "def f(x, y):\n  @jax.jit\n  def jitted(z):\n    if z:\n"
        "      return 1\n    return 0\n  fn = lambda z: z\n"
        "  return foo(lambda z: x and y), fn, jitted\n"
    )[0]
    self.assertEqual(f_lam.nested_closures, ("f.<lambda>",))
    self.assertEqual(f_lam.cognitive, 3)

    cg_funcs = check_cl_complexity.analyze_functions(
        "def a(x):\n  return _shared(x) + _sem(x) + _triv_body(x)\n"
        "def b(x):\n  return _shared(x) + _pt_ext(x)\n"
        "@jax.jit\ndef _sem(x):\n  return _leaf(x)\n"
        "def _shared(x):\n  return _leaf(x)\n"
        "def _triv_body(x):\n  y = _leaf(x)\n  return y + 1\n"
        "def _pt_ext(x):\n  return ext(x)\n"
        "def _leaf(x):\n  return x + 1\n"
    )
    hops, _, triv = check_cl_complexity.compute_call_graph_metrics(
        cg_funcs, cfg
    )
    self.assertEqual(hops, 1)
    self.assertEqual(triv, ["_pt_ext", "_triv_body"])

    for field, delta_field, val in [
        ("ast_digest", None, "diff"),
        ("cognitive", "delta_cognitive", 5),
        ("cyclomatic", "delta_cyclomatic", 5),
        ("max_nesting", "delta_nesting", 3),
    ]:
      nf = check_cl_complexity.analyze_functions("def f(x):\n  return x\n")[0]
      bf = check_cl_complexity.analyze_functions("def f(x):\n  return x\n")[0]
      setattr(nf, field, val)
      check_cl_complexity.assign_baseline(nf, bf)
      self.assertTrue(nf.is_new_or_modified, field)
      if delta_field:
        self.assertEqual(
            getattr(nf, delta_field), val - getattr(bf, field), delta_field
        )
    self.assertLen(
        check_cl_complexity.stmt_shingles(ast.parse("a = 1\nb = 2\n").body), 1
    )
    inner_cg = check_cl_complexity.analyze_functions(
        "class A:\n  def _dup(self, x):\n    return x\n"
        "  def _uniq_m(self, x):\n    return x\n"
        "class B:\n  def _dup(self, x):\n    return x\n"
        "def outer(x):\n  return _uniq_m(x)\n"
    )
    self.assertEqual(
        check_cl_complexity.compute_call_graph_metrics(inner_cg, cfg)[0], 1
    )
    dup_cg = check_cl_complexity.analyze_functions(
        "class A:\n  def _dup(self, x):\n    return x\n"
        "class B:\n  def _dup(self, x):\n    return x\n"
        "def outer(x):\n  return _dup(x)\n"
    )
    self.assertEqual(
        check_cl_complexity.compute_call_graph_metrics(dup_cg, cfg)[0], 0
    )
    self.assertIsNone(
        check_cl_complexity.extract_callee_name(
            ast.Attribute(
                value=ast.Name(id="other", ctx=ast.Load()),
                attr="foo",
                ctx=ast.Load(),
            ),
            "Cls.m",
            {},
        )
    )
    self.assertIsNone(
        check_cl_complexity.extract_callee_name(
            ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id="fn", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                ),
                attr="foo",
                ctx=ast.Load(),
            ),
            "Cls.m",
            {},
        )
    )
    self.assertIsNone(
        check_cl_complexity.extract_callee_name(
            ast.Starred(
                value=ast.Name(id="self", ctx=ast.Load()), ctx=ast.Load()
            ),
            "Cls.m",
            {},
        )
    )
    self.assertEqual(
        check_cl_complexity.extract_callee_name(
            ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr="foo",
                ctx=ast.Load(),
            ),
            "top_fn",
            {},
        ),
        "foo",
    )
    if_guard = ast.parse("if x:\n  return 1\n").body[0]
    if_non_guard = ast.parse("if x:\n  a = 1\n  return a\n").body[0]
    assert isinstance(if_guard, ast.If) and isinstance(if_non_guard, ast.If)
    self.assertTrue(check_cl_complexity.is_early_exit_guard(if_guard))
    self.assertFalse(check_cl_complexity.is_early_exit_guard(if_non_guard))
    self.assertEqual(
        check_cl_complexity.stmt_shingles(
            ast.parse("if x:\n  a = 1\n  b = 2\n").body
        ),
        collections.Counter({
            "d0:If:[]:[]=>d1:Assign:[]:[]": 1,
            "d1:Assign:[]:[]=>d1:Assign:[]:[]": 1,
        }),
    )
    non_pt = check_cl_complexity.analyze_functions(
        "def f(x):\n  return g(x)\n  a = 1\n"
    )[0]
    self.assertFalse(non_pt.is_pass_through)
    alias_edge = check_cl_complexity.analyze_functions(
        "def target(x):\n  return x\n"
        "def caller(self, x):\n"
        "  a = b = target\n"
        "  self.c = target\n"
        "  return a(x) + target(caller)\n"
    )
    self.assertEqual(alias_edge[1].calls, ("a", "x", "target", "caller"))
    self.assertEqual(alias_edge[1].cognitive, 0)
    guard_blocks = check_cl_complexity.analyze_functions(
        "def f(xs):\n"
        "  for x in xs:\n"
        "    if x < 0:\n      continue\n"
        "    if x == 0:\n      return 0\n"
        "    if x == 1:\n      a = 1\n      continue\n"
        "    for y in x:\n"
        "      if y < 0:\n        continue\n"
        "  else:\n"
        "    if xs:\n      return 1\n"
        "  if xs:\n    pass\n  elif not xs:\n    pass\n  else:\n"
        "    if xs:\n      return 2\n"
        "  match xs:\n    case [1]:\n      if xs:\n        return 3\n"
        "  try:\n    pass\n  except Exception:\n    if xs:\n      return 4\n"
        "  else:\n    if xs:\n      return 5\n"
    )[0]
    self.assertEqual(guard_blocks.cognitive, 26)
    dec_node = ast.fix_missing_locations(
        ast.FunctionDef(
            name="p",
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[ast.Pass()],
            decorator_list=[],
        )
    )
    nested_bool = ast.BoolOp(
        op=ast.And(),
        values=[
            ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Name(id="a", ctx=ast.Load()),
                    ast.Name(id="b", ctx=ast.Load()),
                ],
            ),
            ast.Name(id="c", ctx=ast.Load()),
        ],
    )
    v_bool = check_cl_complexity._ComplexityVisitor(dec_node, "p", [], set())
    v_bool.visit_BoolOp(nested_bool)
    self.assertEqual(v_bool.cognitive, 1)
    v_mut = check_cl_complexity._ComplexityVisitor(dec_node, "p", [], set())
    v_mut.arg_lambdas.add(id(dec_node))
    v_mut.visit_FunctionDef(dec_node)
    self.assertEqual(v_mut.closures, ["p.p"])
    ws_doc_fn = check_cl_complexity.analyze_functions(
        'def f():\n  "   "\n  return 1\n'
    )[0]
    self.assertEqual(ws_doc_fn.stmts, 1)
    fake_assign = ast.Assign(
        targets=[ast.Name(id="f", ctx=ast.Store())],
        value=ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=ast.Constant(value=1),
        ),
    )
    setattr(fake_assign, "targets", [*fake_assign.targets, dec_node])
    ast.fix_missing_locations(fake_assign)
    self.assertLen(
        check_cl_complexity.iter_scope_defs(
            ast.Module(body=[fake_assign], type_ignores=[])
        ),
        1,
    )
    ov_fns = check_cl_complexity.analyze_functions(
        "@typing.overload\ndef f(x: int) -> int:\n  ...\n"
        '@typing.overload\ndef f(x: str) -> str:\n  """doc"""\n'
        "def f(x):\n  return x\n"
    )
    self.assertEqual([fn.qualname for fn in ov_fns], ["f"])
    ov_non_stub = check_cl_complexity.analyze_functions(
        "@typing.overload\ndef f(x):\n  return x\n"
        "@typing.overload\ndef g(x):\n  pass\n  return x\n"
    )
    self.assertEqual([fn.qualname for fn in ov_non_stub], ["f", "g"])
    dup_fns = check_cl_complexity.analyze_functions(
        "def dup(x):\n  if x:\n    return 1\n  return 0\n"
        "def dup(x):\n  return x\n"
        "def __init__(self):\n  return None\n"
        "def _priv(x):\n  return x\n"
    )
    self.assertEqual(
        [fn.qualname for fn in dup_fns],
        ["dup", "dup@L5", "__init__", "_priv"],
    )
    self.assertEqual(dup_fns[0].delta_cyclomatic, 1)
    self.assertFalse(dup_fns[2].is_private)
    self.assertTrue(dup_fns[3].is_private)
    pub_mid_cg = check_cl_complexity.analyze_functions(
        "def a(x):\n  return b(x) + 1\n"
        "def b(x):\n  return _c(x) + 1\n"
        "def _c(x):\n  a1 = 1\n  a2 = 2\n  a3 = 3\n  a4 = 4\n  return _d(x)\n"
        "def _d(x):\n  return x + 1\n"
    )
    self.assertEqual(
        check_cl_complexity.compute_call_graph_metrics(pub_mid_cg, cfg),
        (2, ["b", "_c", "_d"], []),
    )

    f1 = check_cl_complexity.analyze_functions(
        "def compute_loss(x):\n  a = x + 1\n  b = a * 2\n  return b\n"
    )[0]
    f1_copy = dataclasses.replace(f1)
    check_cl_complexity.assign_baseline(f1_copy, f1)
    self.assertFalse(f1_copy.is_new_or_modified)
    f_empty1 = dataclasses.replace(
        f1, ast_digest="", shingles=collections.Counter()
    )
    f_empty2 = dataclasses.replace(
        f1, ast_digest="", shingles=collections.Counter()
    )
    self.assertEqual(
        check_cl_complexity.pair_move_score(f_empty1, f_empty2), 0.0
    )
    f_dig1 = dataclasses.replace(
        f1, ast_digest="same_dig", shingles=collections.Counter()
    )
    f_dig2 = dataclasses.replace(
        f1, ast_digest="same_dig", shingles=collections.Counter()
    )
    self.assertEqual(check_cl_complexity.pair_move_score(f_dig1, f_dig2), 2.0)
    f_hi1 = dataclasses.replace(
        f1,
        stripped_name="alpha",
        ast_digest="d1",
        shingles=collections.Counter(("a", "b", "c", "d", "e")),
    )
    f_hi2 = dataclasses.replace(
        f1,
        stripped_name="beta",
        ast_digest="d2",
        shingles=collections.Counter(("a", "b", "c", "d")),
    )
    self.assertEqual(check_cl_complexity.pair_move_score(f_hi1, f_hi2), 0.8)
    f2_diff = check_cl_complexity.analyze_functions(
        "def unrelated(x):\n  for i in x:\n    if i:\n      yield i\n"
    )[0]
    self.assertEqual(check_cl_complexity.pair_move_score(f1, f2_diff), 0.0)
    f_mid1 = check_cl_complexity.analyze_functions(
        "def compute_loss(x):\n  a = x + 1\n  b = a + 2\n  c = b + 3\n"
        "  d = c + 4\n  e = d * 5\n  return e\n"
    )[0]
    f_mid2 = check_cl_complexity.analyze_functions(
        "def compute_loss(x):\n  a = x + 1\n  b = a + 2\n  c = b + 3\n"
        "  d = c + 4\n  e = d - 5\n  return e\n"
    )[0]
    f_mid_renamed = check_cl_complexity.analyze_functions(
        "def other_name(x):\n  a = x + 1\n  b = a + 2\n  c = b + 3\n"
        "  d = c + 4\n  e = d - 5\n  return e\n"
    )[0]
    self.assertGreater(check_cl_complexity.pair_move_score(f_mid1, f_mid2), 0.0)
    self.assertEqual(
        check_cl_complexity.pair_move_score(f_mid1, f_mid_renamed), 0.0
    )
    nf_move = check_cl_complexity.analyze_functions(
        "def compute_loss(x):\n  a = x + 1\n  b = a + 2\n  c = b + 3\n"
        "  d = c + 4\n  e = d * 5\n  return e\n",
        "new.py",
    )[0]
    rf_exact = check_cl_complexity.analyze_functions(
        "def compute_loss(x):\n  a = x + 1\n  b = a + 2\n  c = b + 3\n"
        "  d = c + 4\n  e = d * 5\n  return e\n",
        "old_exact.py",
    )[0]
    rf_partial = check_cl_complexity.analyze_functions(
        "def compute_loss(x):\n  a = x + 1\n  b = a + 2\n  c = b + 3\n"
        "  d = c + 4\n  e = d - 5\n  return e\n",
        "old_partial.py",
    )[0]
    check_cl_complexity.match_baselines_tiered(
        {"old.py": [rf_partial, rf_exact]}, {"new.py": [nf_move]}
    )
    self.assertEqual(nf_move.matched_base_key, "old_exact.py:compute_loss")
    rf_tie0 = dataclasses.replace(rf_exact, file_path="tie0.py")
    rf_tie1 = dataclasses.replace(rf_exact, file_path="tie1.py")
    nf_tie = dataclasses.replace(nf_move, matched_base_key="")
    check_cl_complexity.match_baselines_tiered(
        {"old.py": [rf_tie0, rf_tie1]}, {"new.py": [nf_tie]}
    )
    self.assertEqual(nf_tie.matched_base_key, "tie0.py:compute_loss")
    bf_big = dataclasses.replace(
        rf_exact, file_path="same.py", qualname="fn", stmts=6
    )
    nf_pt_wrap = dataclasses.replace(
        f1,
        file_path="same.py",
        qualname="fn",
        is_pass_through=True,
        ast_digest="pt_dig",
        shingles=collections.Counter(),
    )
    nf_moved_body = dataclasses.replace(
        rf_exact,
        file_path="same.py",
        qualname="moved_fn",
        is_pass_through=False,
    )
    check_cl_complexity.match_baselines_tiered(
        {"same.py": [bf_big]}, {"same.py": [nf_pt_wrap, nf_moved_body]}
    )
    self.assertEqual(nf_pt_wrap.matched_base_key, "")
    self.assertEqual(nf_moved_body.matched_base_key, "same.py:fn")
    bf_small = dataclasses.replace(bf_big, stmts=2, ast_digest="other_dig")
    nf_small_pt = dataclasses.replace(
        rf_exact, file_path="same.py", qualname="fn", is_pass_through=True
    )
    check_cl_complexity.match_baselines_tiered(
        {"same.py": [bf_small, rf_exact]}, {"same.py": [nf_small_pt]}
    )
    self.assertEqual(nf_small_pt.matched_base_key, "same.py:fn")
    self.assertFalse(
        check_cl_complexity.is_fork_exempt(
            "tunix/sft/peft_trainer.py",
            "tunix/sft/peft_trainer.py",
            cfg.intentional_forks,
        )
    )
    self.assertTrue(
        check_cl_complexity.is_fork_exempt(
            "tunix/sft/peft_trainer.py",
            "tunix/rl/rl_learner.py",
            (("sft/*", "rl/*"),),
        )
    )
    self.assertTrue(
        check_cl_complexity.is_fork_exempt(
            "tunix/rl/rl_learner.py",
            "tunix/sft/peft_trainer.py",
            (("sft/*", "rl/*"),),
        )
    )
    self.assertEqual(check_cl_complexity.run_cmd_stdout(["false"], tmp_dir), "")
    self.assertEqual(
        check_cl_complexity.run_cmd_stdout(["echo", "ok"], tmp_dir), "ok\n"
    )
    fn_cyc = check_cl_complexity.analyze_functions("def f(x):\n  return x\n")[0]
    fn_cyc.cyclomatic = 16
    fn_cyc.base_cyclomatic = 12
    fn_cyc.delta_cyclomatic = 4
    self.assertNotEmpty(
        check_cl_complexity.func_local_violations("a.py", fn_cyc, cfg)
    )
    fn_cyc.base_cyclomatic = 16
    fn_cyc.cyclomatic = 21
    fn_cyc.delta_cyclomatic = 5
    fn_cyc.base_cognitive = 20
    fn_cyc.cognitive = 25
    fn_cyc.delta_cognitive = 5
    self.assertEmpty(
        check_cl_complexity.func_local_violations("a.py", fn_cyc, cfg)
    )
    fn_cyc.cognitive = 26
    fn_cyc.delta_cognitive = 6
    self.assertNotEmpty(
        check_cl_complexity.func_local_violations("a.py", fn_cyc, cfg)
    )
    fn_cyc.delta_cognitive = 0
    self.assertEmpty(
        check_cl_complexity.func_local_violations("a.py", fn_cyc, cfg)
    )
    fn_cyc.cyclomatic = 22
    fn_cyc.delta_cyclomatic = 6
    self.assertNotEmpty(
        check_cl_complexity.func_local_violations("a.py", fn_cyc, cfg)
    )
    gen_fn = check_cl_complexity.analyze_functions(
        "def _drain_items(q):\n  yield from q\n", "a.py"
    )[0]
    gen_fn.matched_base_key = "a.py:Cls._drain_old"
    self.assertFalse(check_cl_complexity.is_misnamed_generator(gen_fn))
    gen_fn.matched_base_key = "a.py:Cls._iter_old"
    self.assertTrue(check_cl_complexity.is_misnamed_generator(gen_fn))
    clean_gen = check_cl_complexity.analyze_functions(
        "def _iter_items(q):\n  yield from q\n", "a.py"
    )[0]
    self.assertFalse(check_cl_complexity.is_misnamed_generator(clean_gen))

    clone_src = (
        "def f1(x):\n  a = x + 1\n  b = a + 2\n  c = b + 3\n  d = c + 4\n"
        "  for i in range(2):\n    if a and b and i:\n      if c:\n"
        "        return d\n  return 0\n"
    )
    diff_src = (
        "def f2(x):\n  y = 1\n  z = 2\n  for i in x:\n"
        "    if i > 0 and i < 9:\n      while i:\n        i -= 1\n"
        "  return x + y + z\n"
    )
    cf1 = check_cl_complexity.analyze_functions(clone_src, "a.py")[0]
    cf2 = check_cl_complexity.analyze_functions(
        clone_src.replace("f1", "f2"), "b.py"
    )[0]
    short_cf1 = dataclasses.replace(cf1, stmts=4)
    short_cf2 = dataclasses.replace(cf2, stmts=4)
    self.assertEmpty(
        check_cl_complexity.detect_clones(
            {"a.py": [short_cf1], "b.py": [short_cf2]}, {}, {}, cfg
        )
    )
    unmod = check_cl_complexity.analyze_functions(clone_src, "a.py")[0]
    unmod.is_new_or_modified = False
    self.assertEmpty(
        check_cl_complexity.detect_clones(
            {"a.py": [unmod]}, {}, {"b.py": [cf2]}, cfg
        )
    )
    bf_diff = check_cl_complexity.analyze_functions(diff_src, "b.py")[0]
    cf1.matched_base_key = "a.py:f1"
    cf2.matched_base_key = "b.py:f2"
    self.assertEmpty(
        check_cl_complexity.detect_clones(
            {"a.py": [cf1], "b.py": [cf2]},
            {"a.py": [cf1], "b.py": [cf2]},
            {},
            cfg,
        )
    )
    cf2_ref = dataclasses.replace(
        cf2, is_new_or_modified=False, matched_base_key=""
    )
    self.assertEmpty(
        check_cl_complexity.detect_clones(
            {"a.py": [cf1]},
            {"a.py": [cf1], "b.py": [cf2]},
            {"b.py": [cf2_ref]},
            cfg,
        )
    )
    cf2_moved = dataclasses.replace(
        cf2, is_new_or_modified=False, matched_base_key="old_b.py:f2"
    )
    bf2_old = dataclasses.replace(cf2, file_path="old_b.py")
    self.assertEmpty(
        check_cl_complexity.detect_clones(
            {"a.py": [cf1], "b.py": [cf2_moved]},
            {"a.py": [cf1], "old_b.py": [bf2_old]},
            {},
            cfg,
        )
    )
    self.assertNotEmpty(
        check_cl_complexity.detect_clones(
            {"a.py": [cf1], "b.py": [cf2]},
            {"a.py": [cf1], "b.py": [bf_diff]},
            {},
            cfg,
        )
    )
    bf_low1 = check_cl_complexity.analyze_functions(clone_src, "a.py")[0]
    bf_low2 = check_cl_complexity.analyze_functions(
        clone_src.replace("f1", "f2"), "b.py"
    )[0]
    bf_low1.cognitive = bf_low2.cognitive = 1
    self.assertNotEmpty(
        check_cl_complexity.detect_clones(
            {"a.py": [cf1], "b.py": [cf2]},
            {"a.py": [bf_low1], "b.py": [bf_low2]},
            {},
            cfg,
        )
    )

    triv_funcs = check_cl_complexity.analyze_functions(
        "def a(x):\n  return _t1(x) + _t2(x) + _t3(x)\n"
        "def _t1(x):\n  return ext(x)\n"
        "def _t2(x):\n  return ext(x)\n"
        "def _t3(x):\n  return ext(x)\n",
        "t.py",
    )
    _, _, added_triv, viols = check_cl_complexity.check_file_violations(
        "t.py", triv_funcs, [], {}, cfg
    )
    self.assertLen(added_triv, 3)
    self.assertTrue(any("trivial_single_call_helpers=3" in v for v in viols))
    self.assertTrue(any("pass_through_ratio=0.75" in v for v in viols))
    matched_triv = [
        dataclasses.replace(tf, matched_base_key=f"t.py:{tf.qualname}")
        for tf in triv_funcs
    ]
    _, _, _, m_viols = check_cl_complexity.check_file_violations(
        "t.py", matched_triv, triv_funcs, {}, cfg
    )
    self.assertEmpty(m_viols)

    clos_funcs = check_cl_complexity.analyze_functions(
        "def a(x):\n  c1 = lambda y: y\n  c2 = lambda y: y\n"
        "  c3 = lambda y: y\n  c4 = lambda y: y\n  c5 = lambda y: y\n"
        "  return c1, c2, c3, c4, c5\n",
        "c.py",
    )
    _, _, _, c_viols = check_cl_complexity.check_file_violations(
        "c.py", clos_funcs, [], {}, cfg
    )
    self.assertTrue(any("nested_closures=5" in v for v in c_viols))
    _, _, _, cb_viols = check_cl_complexity.check_file_violations(
        "c.py", clos_funcs, clos_funcs, {}, cfg
    )
    self.assertEmpty(cb_viols)

    pub_sprawl = check_cl_complexity.analyze_functions(
        "\n".join(
            f"def pub_{i}(x):\n"
            + "  if x and (x > 1 or x < -1) and (x != 2 or x != 3):\n"
            + "    if x > 4 and x < 10:\n      return x\n  return 0\n"
            for i in range(6)
        ),
        "s.py",
    )
    base_one = check_cl_complexity.analyze_functions(
        "def pub_0(x):\n  return x\n", "s.py"
    )
    pub_sprawl[0].matched_base_key = "s.py:pub_0"
    _, _, _, s_viols = check_cl_complexity.check_file_violations(
        "s.py", pub_sprawl, base_one, {"s.py:pub_0": base_one[0]}, cfg
    )
    self.assertFalse(any("complexity_sprawl" in v for v in s_viols))
    priv_sprawl = [
        dataclasses.replace(
            base_one[0],
            qualname=f"_p{i}",
            is_private=True,
            cognitive=9 if i == 1 else 8,
        )
        for i in range(1, 6)
    ]
    _, _, _, ps_viols = check_cl_complexity.check_file_violations(
        "s.py",
        [base_one[0], *priv_sprawl],
        base_one,
        {"s.py:pub_0": base_one[0]},
        cfg,
    )
    self.assertTrue(any("complexity_sprawl=5 (+41 Cog)" in v for v in ps_viols))
    _, _, _, newfile_viols = check_cl_complexity.check_file_violations(
        "s.py", priv_sprawl, [], {}, cfg
    )
    self.assertEmpty(newfile_viols)

    with mock.patch.object(
        check_cl_complexity, "run_cmd_stdout", return_value="vcs.py\n"
    ):
      v_nsrc: dict[str, str] = {}
      v_files, v_bases = check_cl_complexity.resolve_changed_files(
          tmp_dir, "HEAD", None, v_nsrc
      )
      self.assertEqual(v_files, ["vcs.py"])
      self.assertEqual(v_bases["vcs.py"], "vcs.py\n")
      self.assertEqual(v_nsrc["vcs.py"], "vcs.py\n")

    pdir = tmp_dir / "piper_mut"
    (pdir / "client/GOOGLE_INTERNAL_PACKAGE_PATH").mkdir(parents=True)
    (pdir / "baseline/GOOGLE_INTERNAL_PACKAGE_PATH").mkdir(parents=True)
    (pdir / "request").write_text(
        'depot_path: "//depot/GOOGLE_INTERNAL_PACKAGE_PATH/m.py"\nSKIP_CL_COMPLEXITY=req_reason\n'
    )
    (pdir / "baseline/GOOGLE_INTERNAL_PACKAGE_PATH/m.py").write_text("def f(x):\n  return x\n")
    with mock.patch.dict(
        os.environ,
        {
            "SKIP_CL_COMPLEXITY": "env_reason",
            "PR_BODY": "SKIP_CL_COMPLEXITY=pr_reason",
        },
        clear=True,
    ):
      self.assertEqual(
          check_cl_complexity.resolve_skip_reason(pdir / "client/GOOGLE_INTERNAL_PACKAGE_PATH"),
          "env_reason",
      )
    with mock.patch.dict(
        os.environ,
        {"SKIP_CL_COMPLEXITY": "", "PR_BODY": "SKIP_CL_COMPLEXITY=pr_reason"},
        clear=True,
    ):
      self.assertEqual(
          check_cl_complexity.resolve_skip_reason(pdir / "client/GOOGLE_INTERNAL_PACKAGE_PATH"),
          "pr_reason",
      )
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertEqual(
          check_cl_complexity.resolve_skip_reason(pdir / "client/GOOGLE_INTERNAL_PACKAGE_PATH"),
          "req_reason",
      )
    with mock.patch.dict(
        os.environ, {"SKIP_CL_COMPLEXITY": "12345"}, clear=True
    ):
      self.assertEqual(check_cl_complexity.resolve_skip_reason(tmp_dir), "")
    with mock.patch.dict(
        os.environ, {"SKIP_CL_COMPLEXITY": "true"}, clear=True
    ):
      self.assertEqual(check_cl_complexity.resolve_skip_reason(tmp_dir), "")
    for vcs_dir, vcs_reason in (
        (".git", "git_skip_reason"),
    ):
      vcs_ws = tmp_dir / f"ws_{vcs_dir.lstrip('.')}"
      (vcs_ws / vcs_dir).mkdir(parents=True)
      with (
          mock.patch.dict(os.environ, {}, clear=True),
          mock.patch.object(
              check_cl_complexity,
              "run_cmd_stdout",
              return_value=f"Commit\nSKIP_CL_COMPLEXITY={vcs_reason}\n",
          ),
      ):
        self.assertEqual(
            check_cl_complexity.resolve_skip_reason(vcs_ws, "HEAD"),
            vcs_reason,
        )
    p_files, p_bases = check_cl_complexity.resolve_changed_files(
        pdir / "client/GOOGLE_INTERNAL_PACKAGE_PATH", None, None
    )
    self.assertEqual(p_files, ["m.py"])
    self.assertEqual(p_bases["m.py"], "def f(x):\n  return x\n")

    (tmp_dir / "leak.py").write_text("x = 1\n")
    sub = tmp_dir / "ws"
    sub.mkdir()
    (sub / "ok.py").write_text(
        "def f(x):\n  c1 = lambda y: y\n  c2 = lambda y: y\n  return c1, c2\n"
    )
    (sub / "ref_prod.py").write_text(clone_src.replace("f1", "f_ref"))
    (sub / "foo_test.py").write_text("def test_a():\n  assert True\n")
    (sub / "tests").mkdir()
    (sub / "tests/bar.py").write_text("def test_b():\n  assert True\n")
    self.assertEmpty(
        check_cl_complexity.discover_ref_sources(
            sub, [str(tmp_dir / "leak.py")]
        )
    )
    self.assertEqual(
        list(check_cl_complexity.discover_ref_sources(sub, ["ok.py"])),
        ["ref_prod.py"],
    )
    rep = check_cl_complexity.evaluate_gate(
        sub,
        ["ok.py", "foo_test.py", "tests/bar.py"],
        collections.defaultdict(str),
        cfg,
        ref_sources={},
    )
    self.assertEqual(rep.prod_files_changed, 1)
    self.assertEqual(rep.test_loc_added, 4)
    self.assertEqual(rep.score, 0.6)
    base_combo = (
        "def base_keep(x):\n"
        "  c0 = lambda y: y\n"
        "  if x and (x > 1 or x < -1) and (x != 2 or x != 3):\n"
        "    if x > 4 and x < 10 and (x != 5 or x != 6):\n"
        "      if x > 7 and (x != 8 or x != 9):\n"
        "        return c0(x)\n"
        "  return 0\n"
        "def base_pt(x):\n  return ext(x)\n"
    )
    new_combo = (
        base_combo
        + clone_src
        + "def a_top(x):\n"
        + "  c1 = lambda y: y\n  c2 = lambda y: y\n"
        + "  c3 = lambda y: y\n  c4 = lambda y: y\n"
        + "  return _h1(x) + _triv(x) + (c1(x) if c2 else c3(c4))\n"
        + "def _h1(x):\n  a = 1\n  b = 2\n  c = 3\n  d = 4\n  return _h2(x)\n"
        + "def _h2(x):\n  a = 1\n  b = 2\n  c = 3\n  d = 4\n  return _h3(x)\n"
        + "def _h3(x):\n  return x + 1\n"
        + "def _triv(x):\n  return ext(x)\n"
        + "def new_cog(x):\n"
        + "  if x and (x > 1 or x < -1) and (x != 2 or x != 3):\n"
        + "    if x > 4 and x < 10 and (x != 5 or x != 6):\n"
        + "      if x > 7 and (x != 8 or x != 9):\n"
        + "        return x\n"
        + "  return 0\n"
    )
    rep_combo = check_cl_complexity.evaluate_gate(
        sub,
        ["combo.py"],
        collections.defaultdict(str, {"combo.py": base_combo}),
        cfg,
        ref_sources={"ref_prod.py": clone_src.replace("f1", "f_ref")},
        new_sources={"combo.py": new_combo},
    )
    self.assertEqual(rep_combo.score, 72.7)
    self.assertNotIn(
        "Use SKIP_CL_COMPLEXITY", check_cl_complexity.format_report(rep)
    )
    self.assertNotIn(
        "[BYPASSED]",
        check_cl_complexity.format_report(
            dataclasses.replace(rep, skip_reason="migration")
        ),
    )
    rep_fail = dataclasses.replace(
        rep, passed=False, violations=["v1"], skip_reason=""
    )
    self.assertIn(
        "Use SKIP_CL_COMPLEXITY", check_cl_complexity.format_report(rep_fail)
    )
    self.assertNotIn("[BYPASSED]", check_cl_complexity.format_report(rep_fail))
    rep_skip = dataclasses.replace(
        rep, passed=True, violations=["v1"], skip_reason="migration"
    )
    self.assertIn("[BYPASSED]", check_cl_complexity.format_report(rep_skip))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      rc_json = check_cl_complexity.main([
          "--cwd",
          str(sub),
          "--config",
          str(_CONFIG),
          "--rev",
          "ok.py",
          "--files",
          "ok.py",
          "--json",
      ])
    self.assertEqual(rc_json, 0)
    self.assertIn('"passed": true', buf.getvalue())
    (sub / "bad.py").write_text(
        "def a(x):\n  return _b(x)\n"
        "def _b(x):\n  return _c(x)\n"
        "def _c(x):\n  return _d(x)\n"
        "def _d(x):\n  return x + 1\n"
    )
    self.assertEqual(
        check_cl_complexity.main(
            ["--cwd", str(sub), "--config", str(_CONFIG), "--files", "bad.py"]
        ),
        0,
    )

    # Async/await, vararg/kwarg, multiline assign-return, and lambda shims.
    shim_fns = check_cl_complexity.analyze_functions(
        "async def _async_pt(x):\n  return await ext(x)\n"
        "def _vararg_pt(*args, **kwargs):\n  return ext(*args, **kwargs)\n"
        'def _multiline_pt(x):\n  r = ext(x, "a\\nb")\n  return r\n'
        "_lam_pt = lambda x: ext(x)\n"
        "_lam_expr = lambda x: x + 1\n"
    )
    self.assertEqual(
        [fn.is_pass_through for fn in shim_fns],
        [True, True, True, True, False],
    )
    self.assertEqual([fn.cognitive for fn in shim_fns], [0, 0, 0, 0, 0])

    # Clone baseline-pair lookup preserves f1/f2 order when f1 > f2 lexically.
    z_moved = dataclasses.replace(
        cf1,
        file_path="z.py",
        matched_base_key="a_old.py:f1",
        is_new_or_modified=True,
    )
    a_old_base = dataclasses.replace(cf1, file_path="a_old.py")
    b_ref = dataclasses.replace(
        cf2, file_path="b.py", matched_base_key="", is_new_or_modified=False
    )
    self.assertEmpty(
        check_cl_complexity.detect_clones(
            {"z.py": [z_moved]},
            {"a_old.py": [a_old_base], "b.py": [cf2]},
            {"b.py": [b_ref]},
            cfg,
        )
    )

    # Syntax errors fail fast as Dim 2 violations without crashing helpers.
    self.assertEqual(check_cl_complexity.count_code_loc("def broken(\n"), set())
    self.assertEqual(check_cl_complexity.count_ast_stmts("def broken(\n"), 0)
    self.assertEmpty(check_cl_complexity.analyze_functions("def broken(\n"))
    syn_rep = check_cl_complexity.evaluate_gate(
        sub,
        ["broken.py"],
        collections.defaultdict(str),
        cfg,
        ref_sources={},
        new_sources={"broken.py": "def broken(\n"},
    )
    self.assertFalse(syn_rep.passed)
    self.assertTrue(
        any("Dim 2: broken.py syntax_error" in v for v in syn_rep.violations),
        syn_rep.violations,
    )

    # `/testdata/` fixtures are excluded from ref discovery and prod scope.
    (sub / "testdata").mkdir()
    (sub / "testdata/fixture.py").write_text(clone_src)
    self.assertEqual(
        list(check_cl_complexity.discover_ref_sources(sub, ["ok.py"])),
        ["bad.py", "ref_prod.py"],
    )

    self.assertEqual(
        check_cl_complexity.main([
            "--cwd",
            str(sub),
            "--config",
            str(_CONFIG),
            "--files",
            "bad.py",
            "--strict",
        ]),
        1,
    )

  def test_portable_config_and_g1_g2_ast_hygiene(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    other_repo = root / "other_repo"
    (other_repo / "sub").mkdir(parents=True)
    (other_repo / "pyproject.toml").write_text(
        '[project]\nname = "unrelated"\n[tool.pytest]\naddopts = "-q"\n',
        encoding="utf-8",
    )
    missing = other_repo / "missing.toml"
    with (
        mock.patch.object(check_cl_complexity, "_PKG_CONFIG", missing),
        mock.patch.object(check_cl_complexity, "_FALLBACK_CFG", missing),
        self.assertRaises(KeyError),
    ):
      check_cl_complexity.load_config(cwd=other_repo / "sub")
    default_cfg = check_cl_complexity.load_config(cwd=other_repo / "sub")
    self.assertEqual(default_cfg.enabled_dimensions, (1, 2, 3, 4))
    self.assertEqual(default_cfg.max_cognitive_fail, 15)

    anc_repo = root / "anc_repo"
    (anc_repo / "pkg/deep").mkdir(parents=True)
    (anc_repo / "pyproject.toml").write_text(
        _CONFIG.read_text(encoding="utf-8").replace(
            "max_cognitive_fail = 15", "max_cognitive_fail = 19"
        ),
        encoding="utf-8",
    )
    anc_cfg = check_cl_complexity.load_config(cwd=anc_repo / "pkg/deep")
    self.assertEqual(anc_cfg.max_cognitive_fail, 19)

    prod_bad = textwrap.dedent("""\
        from unittest import mock
        import unittest.mock as um

        def run(x: int) -> object:
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
    mock_hits = ast_gate_lib.scan_mock_ast(prod_tree)
    self.assertLen(mock_hits, 3)
    exc_hits = ast_gate_lib.scan_silent_except_ast(
        prod_tree, prod_bad.splitlines()
    )
    self.assertEqual(
        exc_hits,
        [
            (7, "[G1:silent-except] Silent broad except"),
            (9, "[G1:silent-except] Silent broad except"),
        ],
    )

    test_bad = textwrap.dedent("""\
        import unittest

        @unittest.skip("legacy")
        def test_skipped():
          assert 1 == 1

        def test_call_skip(self):
          self.skipTest("unconditional")

        def test_guarded_skip(self, n: int):
          if n < 2:
            self.skipTest("needs 2 devices")
          assert n >= 2

        def test_vacuous_doc():
          \"\"\"Only docstring and pass.\"\"\"
          pass

        async def test_vacuous_ellipsis():
          ...
    """)
    test_tree = ast.parse(test_bad)
    with mock.patch.dict(os.environ, {"ALLOW_TEST_SKIP_WITH_BUG": "true"}):
      self.assertLen(ast_gate_lib.scan_test_skip_ast(test_tree), 2)
    with mock.patch.dict(os.environ, {"ALLOW_TEST_SKIP_WITH_BUG": "#42"}):
      self.assertEmpty(ast_gate_lib.scan_test_skip_ast(test_tree))
    self.assertEqual(
        ast_gate_lib.scan_vacuous_test_ast(test_tree),
        [
            (15, "[G2:vacuous-test] Empty test_vacuous_doc"),
            (19, "[G2:vacuous-test] Empty test_vacuous_ellipsis"),
        ],
    )

    ws = root / "hygiene_ws"
    ws.mkdir()
    (ws / "prod.py").write_text(
        "def f(x: int) -> int:\n  try:\n    return x\n  except Exception:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (ws / "base_prod.py").write_text(
        (ws / "prod.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (ws / "prod_test.py").write_text(
        "def test_empty():\n  pass\n", encoding="utf-8"
    )
    # Ratcheted `--strict` passes unchanged baseline debt, while `--hygiene`
    # enforces full-file cleanliness unconditionally.
    self.assertEqual(
        check_cl_complexity.main([
            "--cwd",
            str(ws),
            "--config",
            str(_CONFIG),
            "--files",
            "prod.py",
            "--base",
            "base_prod.py",
            "--strict",
        ]),
        0,
    )
    self.assertEqual(
        check_cl_complexity.main([
            "--cwd",
            str(ws),
            "--config",
            str(_CONFIG),
            "--files",
            "prod.py",
            "--base",
            "base_prod.py",
            "--hygiene",
        ]),
        1,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      rc_test_hyg = check_cl_complexity.main([
          "--cwd",
          str(ws),
          "--config",
          str(_CONFIG),
          "--files",
          "prod_test.py",
          "--strict",
          "--json",
      ])
    self.assertEqual(rc_test_hyg, 1)
    rep_json = json.loads(buf.getvalue())
    self.assertFalse(rep_json["passed"])
    self.assertTrue(
        any("[G2:vacuous-test]" in v for v in rep_json["violations"])
    )

  def test_shard_plan_components_hub_cut_lpt_and_frozen_contracts(self):
    self.assertEqual(
        check_cl_complexity.profile_shard_file("b.py", "def broken(\n").stmts, 0
    )
    ws = pathlib.Path(self.create_tempdir().full_path)
    (ws / "pkg/testdata").mkdir(parents=True)
    (ws / "pkg/testdata/ignored.py").write_text(
        "def ignored() -> int:\n  return 1\n", encoding="utf-8"
    )
    (ws / "pkg/a_lib.py").write_text(
        "def compute_a(x: int) -> int:\n  return x + 1 if x > 0 else 0\n",
        encoding="utf-8",
    )
    (ws / "pkg/a_caller.py").write_text(
        "from pkg.a_lib import compute_a\n\ndef run_a(v: int) -> int:\n"
        "  return compute_a(v) + 2\n",
        encoding="utf-8",
    )
    (ws / "pkg/b_lib.py").write_text(
        "def compute_b(x: int) -> int:\n  return x * 2 if x > 0 else 1\n",
        encoding="utf-8",
    )
    (ws / "pkg/b_caller.py").write_text(
        "from . import b_lib as bl\n\ndef run_b(v: int) -> int:\n"
        "  return bl.compute_b(v) + 3\n",
        encoding="utf-8",
    )
    (ws / "pkg/standalone.py").write_text(
        "def solo(items: list[int]) -> int:\n  total = 0\n"
        "  for x in items:\n    if x > 0:\n      total += x\n  return total\n",
        encoding="utf-8",
    )
    pair_plan = check_cl_complexity.build_shard_plan(
        ws,
        [
            "pkg/a_lib.py",
            "pkg/a_caller.py",
            "pkg/b_lib.py",
            "pkg/b_caller.py",
            "pkg/standalone.py",
            "pkg/testdata/ignored.py",
        ],
        max_shards=2,
    )
    self.assertEqual(pair_plan["total_files"], 5)
    self.assertEqual(
        pair_plan["connected_components"],
        [
            ["pkg/a_caller.py", "pkg/a_lib.py"],
            ["pkg/b_caller.py", "pkg/b_lib.py"],
            ["pkg/standalone.py"],
        ],
    )
    self.assertEqual(pair_plan["hub_modules_cut"], [])
    self.assertEqual(pair_plan["frozen_contracts"], [])
    self.assertLen(pair_plan["shards"], 2)

    # Star graph: cutting oversized component on Hub module emits 1-line
    # frozen contracts across worker shards.
    star = ws / "star"
    star.mkdir()
    (star / "hub.py").write_text(
        textwrap.dedent("""\
            DEFAULT_TIMEOUT: int = 30

            class HubConfig:
              batch_size: int
              lr: float

            def build_spec(limit: int) -> dict[str, int]:
              return {"limit": limit} if limit > 0 else {"limit": 1}

            async def fetch_spec(name: str) -> str:
              return name.strip()
        """),
        encoding="utf-8",
    )
    for idx in range(1, 4):
      (star / f"worker_{idx}.py").write_text(
          textwrap.dedent(f"""\
              from hub import HubConfig, build_spec

              def execute_{idx}(items: list[int], cfg: HubConfig) -> int:
                spec = build_spec(cfg.batch_size)
                acc = 0
                for item in items:
                  if item > 0:
                    acc += item + spec["limit"]
                return acc
          """),
          encoding="utf-8",
      )
    out_json = star / "out/plan.json"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      rc = check_cl_complexity.main([
          "--cwd",
          str(star),
          "--shard-plan",
          "--max-shards",
          "3",
          "--files",
          "hub.py",
          "worker_1.py",
          "worker_2.py",
          "worker_3.py",
          "--out",
          str(out_json),
      ])
    self.assertEqual(rc, 0)
    plan = json.loads(out_json.read_text(encoding="utf-8"))
    self.assertEqual(plan, json.loads(buf.getvalue()))
    self.assertEqual(plan["hub_modules_cut"], ["hub.py"])
    self.assertLessEqual(len(plan["shards"]), 3)
    self.assertGreaterEqual(len(plan["shards"]), 2)
    for shard in plan["shards"]:
      self.assertEqual(
          shard["weight"], shard["stmts"] + 2 * shard["cognitive"]
      )
    contracts = plan["frozen_contracts"]
    self.assertTrue(
        any(
            "hub.py: class HubConfig: batch_size: int, lr: float" in c
            for c in contracts
        ),
        contracts,
    )
    self.assertTrue(
        any(
            "hub.py: def build_spec(limit: int) -> dict[str, int]" in c
            for c in contracts
        ),
        contracts,
    )

  def test_cli_help_documents_shard_plan_and_hygiene_usage(self):
    help_txt = check_cl_complexity.build_arg_parser().format_help()
    for expected in (
        "--shard-plan",
        "--max-shards",
        "--hygiene",
        "--out",
        "[G1:mock-branch]",
        "[G1:silent-except]",
        "[G2:test-skip]",
        "[G2:vacuous-test]",
        "frozen_contracts",
        "hub_modules_cut",
        "Usage Examples:",
    ):
      self.assertIn(expected, help_txt)
    buf = io.StringIO()
    with (
        contextlib.redirect_stdout(buf),
        self.assertRaises(SystemExit) as ctx,
    ):
      check_cl_complexity.main(["--help"])
    self.assertEqual(ctx.exception.code, 0)
    self.assertIn("--shard-plan", buf.getvalue())

  def test_check_test_coverage_gate_and_violations(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    self.assertEqual(
        ast_gate_lib.extract_coverage_targets("def broken(\n"),
        ([], set()),
    )
    stub_src = "def stub():\n  ...\ndef pstub():\n  pass\n"
    self.assertEqual(
        ast_gate_lib.extract_coverage_targets(stub_src), ([], set())
    )
    self.assertEqual(check_cl_complexity.evaluate_test_coverage(root, []), [])

    mod_p = root / "calc.py"
    mod_p.write_text(
        "def add(a: int, b: int) -> int:\n"
        "  x = a + b\n"
        "  return x\n"
        "def unused(a: int) -> int:\n"
        "  return a - 1\n",
        encoding="utf-8",
    )
    miss = check_cl_complexity.evaluate_test_coverage(root, ["calc.py"])
    self.assertTrue(any("G2_MISSING_UNIT_TEST" in v for v in miss))

    test_p = root / "calc_test.py"
    test_p.write_text(
        "import unittest\nimport calc\n"
        "class T(unittest.TestCase):\n"
        "  def test_add(self):\n"
        "    self.assertEqual(calc.add(1, 2), 3)\n",
        encoding="utf-8",
    )
    uncov = check_cl_complexity.evaluate_test_coverage(root, ["calc.py"])
    self.assertTrue(any("G2_UNCOVERED_FUNCTIONS" in v for v in uncov))
    self.assertTrue(any("unused" in v for v in uncov))

    low_p = root / "branchy.py"
    low_p.write_text(
        "def run(flag: bool) -> int:\n"
        "  if flag:\n"
        "    return 1\n"
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
    low_v = check_cl_complexity.evaluate_test_coverage(root, ["branchy.py"])
    self.assertTrue(any("G2_LOW_STMT_COVERAGE" in v for v in low_v))

    test_p.write_text(
        "import unittest\nimport calc\n"
        "class T(unittest.TestCase):\n"
        "  def test_all(self):\n"
        "    self.assertEqual(calc.add(1, 2), 3)\n"
        "    self.assertEqual(calc.unused(2), 1)\n",
        encoding="utf-8",
    )
    with contextlib.redirect_stdout(io.StringIO()):
      rc = check_cl_complexity.main([
          "--cwd",
          str(root),
          "--files",
          "calc.py",
          "--check-test-coverage",
      ])
    self.assertEqual(rc, 0)
    with (
        mock.patch.object(check_cl_complexity.sys, "executable", None),
        mock.patch.object(shutil, "which", return_value=None),
    ):
      self.assertEmpty(
          check_cl_complexity.evaluate_test_coverage(root, ["calc.py"])
      )

    test_p.write_text("raise RuntimeError('broken test')\n", encoding="utf-8")
    fail_v = check_cl_complexity.evaluate_test_coverage(root, ["calc.py"])
    self.assertTrue(any("G2_UNIT_TEST_FAILURE" in v for v in fail_v))
    with (
        mock.patch.object(check_cl_complexity.sys, "executable", None),
        mock.patch.object(shutil, "which", return_value=None),
    ):
      fail_inproc = check_cl_complexity.evaluate_test_coverage(
          root, ["calc.py"]
      )
      self.assertTrue(any("G2_UNIT_TEST_FAILURE" in v for v in fail_inproc))

  def test_v2_linear_router_cohesive_pipeline_and_contract_restraint(self):
    cfg = check_cl_complexity.load_config(_CONFIG)
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)

    # 1. Comprehension filter parity (`_comp` counts +1 cog per `if` clause).
    comp_src = (
        "def filtered(xs: list[int]) -> list[int]:\n"
        "  return [x for x in xs if x > 0 if x % 2 == 0]\n"
    )
    comp_fn = check_cl_complexity.analyze_functions(comp_src, "comp.py")[0]
    self.assertEqual(comp_fn.cognitive, 2)
    self.assertEqual(comp_fn.cyclomatic, 3)

    # 2. Flat linear dispatch tables (>= 5 branches, <= 2 simple stmts/branch).
    elif_arms = "\n".join(
        f"  {'if' if i == 0 else 'elif'} op == {i}:\n    res = x + {i}\n    return res"
        for i in range(16)
    )
    match_arms = "\n".join(
        f"    case {i}:\n      return x + {i}" for i in range(16)
    )
    router_src = (
        f"def route_elif(op: int, x: int) -> int:\n{elif_arms}\n  else:\n    return x\n\n"
        f"def route_match(op: int, x: int) -> int:\n  match op:\n{match_arms}\n"
    )
    (tmp_dir / "router.py").write_text(router_src, encoding="utf-8")
    router_rep = check_cl_complexity.evaluate_gate(
        tmp_dir, ["router.py"], collections.defaultdict(str), cfg
    )
    self.assertTrue(router_rep.passed, router_rep.violations)
    self.assertLessEqual(router_rep.max_cyclomatic, 5)
    self.assertLessEqual(router_rep.max_cognitive, 5)

    # Non-flat 16-branch router with nested `if` still fails cyclomatic > 15.
    nested_router = router_src.replace(
        "    case 15:\n      return x + 15",
        "    case 15:\n      if x > 0:\n        return x + 15\n      return 0",
    )
    (tmp_dir / "bad_router.py").write_text(nested_router, encoding="utf-8")
    bad_router_rep = check_cl_complexity.evaluate_gate(
        tmp_dir, ["bad_router.py"], collections.defaultdict(str), cfg
    )
    self.assertFalse(bad_router_rep.passed)
    self.assertGreater(bad_router_rep.max_cyclomatic, 15)

    # 3. Cohesive pure stage pipeline exemption (stmts >= 8, 4 <= cog <= 12).
    pipeline_src = textwrap.dedent("""\
        def run_pipeline(xs: list[int]) -> int:
          return _parse_stage(xs)

        def _parse_stage(xs: list[int]) -> int:
          if not xs:
            return 0
          out: list[int] = []
          step = len(xs)
          for val in xs:
            if val > 0:
              out.append(val + step)
          return _lower_stage(out)

        def _lower_stage(items: list[int]) -> int:
          if len(items) == 1:
            return items[0]
          scaled: dict[int, int] = {}
          idx = 0
          while idx < len(items):
            if items[idx] % 2 == 0:
              scaled[idx] = items[idx] * 3
            idx += 1
          return _compile_stage(scaled)

        def _compile_stage(table: dict[int, int]) -> int:
          if not table:
            return -1
          total = 1
          for key, weight in table.items():
            if weight > key:
              total = total * weight - key
            else:
              total += key
          return total
    """)
    (tmp_dir / "pipeline.py").write_text(pipeline_src, encoding="utf-8")
    pipe_rep = check_cl_complexity.evaluate_gate(
        tmp_dir, ["pipeline.py"], collections.defaultdict(str), cfg
    )
    self.assertTrue(pipe_rep.passed, pipe_rep.violations)
    self.assertEqual(pipe_rep.max_call_depth_hops, 0)

    # 4. Contract restraint flags 1-field @dataclass/TypedDict & 1-method Protocol.
    bloat_src = textwrap.dedent("""\
        import dataclasses
        from typing import Protocol, TypedDict

        @dataclasses.dataclass
        class SingleKnobConfig:
          lr: float = 1e-4

        class SingleKeySpec(TypedDict):
          lr: float

        class SingleStepHook(Protocol):
          def __call__(self, x: int) -> int:
            ...
    """)
    (tmp_dir / "bloat.py").write_text(bloat_src, encoding="utf-8")
    hyg_issues = ast_gate_lib.evaluate_hygiene(tmp_dir, ["bloat.py"])
    self.assertLen(hyg_issues, 3)
    self.assertTrue(
        all("[G1:trivial-contract]" in issue for issue in hyg_issues),
        hyg_issues,
    )

    # 5. Bounded helper sprawl & soft scope emit warnings, not hard violations;
    #    duck_typing_calls telemetry counts fallback getattr/hasattr/.get calls.
    sprawl_helpers = "\n\n".join(
        textwrap.dedent(f"""\
            def _helper_{i}(xs: list[int], d: dict[str, int]) -> int:
              total = d.get("k_{i}", 0) + getattr(xs, "bias_{i}", 0)
              for idx, val in enumerate(xs):
                if (val > {i} and idx % 2 == 0) or idx < 0:
                  if val < 100 or hasattr(xs, "cap_{i}"):
                    total += val
              return total""")
        for i in range(1, 6)
    )
    sprawl_src = (
        "def pub_entry(xs: list[int], d: dict[str, int]) -> int:\n"
        "  return (\n"
        "      _helper_1(xs, d)\n"
        "      + _helper_2(xs, d)\n"
        "      + _helper_3(xs, d)\n"
        "      + _helper_4(xs, d)\n"
        "      + _helper_5(xs, d)\n"
        "  )\n\n"
        + sprawl_helpers
        + "\n"
    )
    base_sprawl = "def pub_entry(xs: list[int], d: dict[str, int]) -> int:\n  return 0\n"
    (tmp_dir / "sprawl.py").write_text(sprawl_src, encoding="utf-8")
    sprawl_rep = check_cl_complexity.evaluate_gate(
        tmp_dir,
        ["sprawl.py"],
        collections.defaultdict(str, {"sprawl.py": base_sprawl}),
        dataclasses.replace(cfg, weight_prod_loc=4.0, base_score=100.0),
    )
    self.assertTrue(sprawl_rep.passed, sprawl_rep.violations)
    self.assertEqual(sprawl_rep.duck_typing_calls, 15)
    self.assertTrue(
        any("complexity_sprawl=5" in w for w in sprawl_rep.warnings),
        sprawl_rep.warnings,
    )
    self.assertTrue(
        any("Dim 1: score=" in w for w in sprawl_rep.warnings),
        sprawl_rep.warnings,
    )
    self.assertIn("[WARNING]", check_cl_complexity.format_report(sprawl_rep))

    # Unbounded sprawl (when a function exceeds Cog > 15) stays a hard violation.
    fn_over = dataclasses.replace(
        check_cl_complexity.analyze_functions(sprawl_src, "sprawl.py")[0],
        cognitive=16,
    )
    kept_v, kept_w = check_cl_complexity.partition_warnings(
        ["Dim 3: sprawl.py complexity_sprawl=5 (+45 Cog)"],
        [fn_over],
        0,
        50.0,
        cfg,
    )
    self.assertEqual(kept_v, ["Dim 3: sprawl.py complexity_sprawl=5 (+45 Cog)"])
    self.assertEmpty(kept_w)


if __name__ == "__main__":
  absltest.main()
