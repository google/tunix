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
"""Shared AST parsing, hygiene, and coverage utilities for CL gates."""

from __future__ import annotations

import ast
import collections
from collections.abc import Mapping, Sequence
import concurrent.futures
import contextlib
import io
import itertools
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_BLOCK_NODES = (ast.stmt, ast.ExceptHandler, ast.match_case)
_SKIP_BUG_RE = re.compile(r"(#[0-9]+|https?://|[A-Za-z]+[-/][0-9]+)")  # b/123
_TEST_PATH_RE = re.compile(
    r"(^|/)(test_[^/]*|[^/]*_(test|testing)|conftest)\.py$|/(tests|testing)/"
)  # pkg/foo_test.py -> ("/", "foo_test", "test", None)
_MOCK_NAMES = frozenset({"MagicMock", "NonCallableMagicMock"})
_BROAD_EXCEPTS = frozenset({None, "Exception", "BaseException"})
_SKIP_DECORATORS = frozenset({"skip", "xfail"})


def try_parse_ast(source: str) -> tuple[ast.Module | None, str]:
  """Parses `source` into an AST or returns `(None, error_detail)`."""
  try:
    return ast.parse(source), ""
  except (SyntaxError, ValueError) as exc:
    return None, f"{type(exc).__name__}: {exc}"


def iter_scope_defs(
    tree: ast.Module,
) -> list[tuple[str, ast.stmt | ast.expr, list[ast.stmt]]]:
  """Collects `(qualname, node, body)` across modules, classes, and blocks."""
  stack: list[tuple[str, ast.AST]] = [("", s) for s in reversed(tree.body)]
  out: list[tuple[str, ast.stmt | ast.expr, list[ast.stmt]]] = []
  while stack:
    pref, node = stack.pop()
    if isinstance(node, _FUNC_TYPES):
      has_doc = bool(ast.get_docstring(node, clean=False))
      out.append((f"{pref}{node.name}", node, list(node.body[has_doc:])))
      continue
    if (
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and node.value is not None
        and re.match(
            r"^(lambda\b|(?:functools\.)?partial\s*\()",
            ast.unparse(node.value),
        )
    ):
      lhs = ast.unparse(node).split("=")[0].split(":")[0].strip()
      fn_node, stmt = (
          (node.value, ast.Return(value=node.value.body))
          if isinstance(node.value, ast.Lambda)
          else (node, node)
      )
      body: list[ast.stmt] = [stmt]
      out.append((f"{pref}{lhs}", fn_node, body))
      continue
    kids = [
        c for c in ast.iter_child_nodes(node) if isinstance(c, _BLOCK_NODES)
    ]
    next_pref = (
        f"{pref}{node.name}." if isinstance(node, ast.ClassDef) else pref
    )
    stack.extend([(next_pref, c) for c in reversed(kids)])
  return out


def _format_class_signature(node: ast.ClassDef, name: str) -> str:
  """Formats a 1-line signature for a class definition."""
  bases = ", ".join(map(ast.unparse, node.bases))
  head = f"class {name}({bases})".removesuffix("()")
  fn_map = {
      s.name: f"__init__({ast.unparse(s.args)})"
      for s in node.body
      if isinstance(s, ast.FunctionDef)
  }
  ann_stmts = [s for s in node.body if isinstance(s, ast.AnnAssign)]
  fields = [
      f"{s.target.id}: {ast.unparse(s.annotation)}"
      for s in ann_stmts
      if isinstance(s.target, ast.Name)
  ]
  pub_fields = ", ".join(f for f in fields if not f.startswith("_"))
  detail = fn_map.get("__init__", pub_fields)
  full = f"{head}: {detail}".removesuffix(": ")
  return " ".join(full.split())


def format_public_signature(node: ast.stmt) -> tuple[str, str] | None:
  """Formats a 1-line public signature contract for top-level symbols."""
  t_id = getattr(getattr(node, "target", None), "id", "")
  name = getattr(node, "name", t_id)
  if name[:1] in ("", "_"):
    return None
  if isinstance(node, _FUNC_TYPES):
    kw = ("def", "async def")[isinstance(node, ast.AsyncFunctionDef)]
    ret = "" if node.returns is None else f" -> {ast.unparse(node.returns)}"
    return name, " ".join(f"{kw} {name}({ast.unparse(node.args)}){ret}".split())
  if isinstance(node, ast.AnnAssign):
    return name, " ".join(f"{name}: {ast.unparse(node.annotation)}".split())
  if isinstance(node, ast.ClassDef):
    return name, _format_class_signature(node, name)
  return None


def extract_public_signatures(source: str) -> dict[str, str]:
  """Extracts `{symbol_name: signature_str}` for all top-level public symbols."""
  tree, _ = try_parse_ast(source)
  if tree is None:
    return {}
  sigs: dict[str, str] = {}
  for stmt in tree.body:
    sig = format_public_signature(stmt)
    if sig is not None:
      sigs[sig[0]] = sig[1]
  return sigs


def _key_from_subscript(node: ast.Subscript) -> str | None:
  """Extracts string literal key from subscript if present."""
  if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
    return node.slice.value
  return None


def _key_from_call(node: ast.Call) -> str | None:
  """Extracts string literal key from dict method call if present."""
  if not isinstance(node.func, ast.Attribute) or not node.args:
    return None
  if node.func.attr not in ("get", "pop", "setdefault"):
    return None
  first = node.args[0]
  if isinstance(first, ast.Constant) and isinstance(first.value, str):
    return first.value
  return None


def extract_dict_key_accesses(source: str) -> set[str]:
  """Extracts string literal keys queried via subscript `d['k']` or `d.get('k')`."""
  tree, _ = try_parse_ast(source)
  if tree is None:
    return set()
  keys: set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Subscript):
      if (k := _key_from_subscript(node)) is not None:
        keys.add(k)
    elif isinstance(node, ast.Call):
      if (k := _key_from_call(node)) is not None:
        keys.add(k)
  return keys


def extract_typed_dict_keys(source: str) -> dict[str, set[str]]:
  """Extracts field names defined in TypedDict or dataclass definitions."""
  tree, _ = try_parse_ast(source)
  if tree is None:
    return {}
  type_defs: dict[str, set[str]] = {}
  for stmt in tree.body:
    if not isinstance(stmt, ast.ClassDef):
      continue
    fields: set[str] = set()
    for item in stmt.body:
      if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
        fields.add(item.target.id)
    if fields:
      type_defs[stmt.name] = fields
  return type_defs


def extract_schema_properties(
    schema_input: Mapping[str, object] | str,
) -> set[str]:
  """Extracts property names from a JSON schema object or JSON file path."""
  data: object = schema_input
  if isinstance(schema_input, str):
    p = pathlib.Path(schema_input)
    try:
      content = p.read_text(encoding="utf-8") if p.is_file() else schema_input
      data = json.loads(content)
    except (json.JSONDecodeError, OSError):
      return set()
  if not isinstance(data, dict):
    return set()
  keys: set[str] = set()
  _collect_schema_keys(data, keys)
  return keys


def _collect_schema_keys(data: Mapping[str, object], keys: set[str]) -> None:
  """Recursively collects property names declared in JSON Schema dicts."""
  props = data.get("properties")
  if isinstance(props, dict):
    for k in props:
      keys.add(str(k))
      v = props[k]
      if isinstance(v, dict):
        _collect_schema_keys(v, keys)
  items = data.get("items")
  if isinstance(items, dict):
    _collect_schema_keys(items, keys)


def count_test_assertions(source: str) -> int:
  """Counts executable assertions (`assert` statements and `self.assert*` calls)."""
  tree, _ = try_parse_ast(source)
  if tree is None:
    return 0
  count = 0
  for node in ast.walk(tree):
    if isinstance(node, ast.Assert):
      count += 1
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
      if node.func.attr.startswith("assert"):
        count += 1
  return count


def scan_mock_ast(tree: ast.AST) -> list[tuple[int, str]]:
  """Flags `unittest.mock` / `MagicMock` usage inside production modules."""
  hits: list[tuple[int, str]] = []
  for node in ast.walk(tree):
    mod = getattr(node, "module", "") or ""
    names = {getattr(a, "name", a) for a in getattr(node, "names", ())}
    sym = getattr(node, "id", "") or getattr(node, "attr", "")
    is_mock_imp = "unittest.mock" in {mod, *names} or (
        mod == "unittest" and "mock" in names
    )
    if is_mock_imp:
      msg = "[G1:mock-branch] unittest.mock in prod file"
      hits.append((getattr(node, "lineno", 0), msg))
    elif sym in _MOCK_NAMES:
      hits.append(
          (getattr(node, "lineno", 0), f"[G1:mock-branch] {sym} in prod file")
      )
  return hits


def scan_silent_except_ast(
    tree: ast.AST, lines: Sequence[str]
) -> list[tuple[int, str]]:
  """Flags bare/broad `except: pass/continue` handlers without `# noqa`."""
  hits: list[tuple[int, str]] = []
  for node in ast.walk(tree):
    if not isinstance(node, ast.ExceptHandler) or len(node.body) != 1:
      continue
    tp = node.type and getattr(node.type, "id", "")
    line = (lines[node.lineno - 1 : node.lineno] or [""])[0]
    if (
        tp in _BROAD_EXCEPTS
        and "noqa" not in line
        and isinstance(node.body[0], (ast.Pass, ast.Continue))
    ):
      hits.append((node.lineno, "[G1:silent-except] Silent broad except"))
  return hits


def scan_test_skip_ast(tree: ast.AST) -> list[tuple[int, str]]:
  """Flags unconditional `@skip`/`@xfail` or top-level `skipTest` in tests."""
  if _SKIP_BUG_RE.search(os.environ.get("ALLOW_TEST_SKIP_WITH_BUG", "")):
    return []
  targets = [
      n for n in ast.walk(tree) if isinstance(n, (*_FUNC_TYPES, ast.ClassDef))
  ]
  hits: list[tuple[int, str]] = []
  for node in targets:
    dec_names = {
        getattr(getattr(d, "func", d), "attr", "")
        or getattr(getattr(d, "func", d), "id", "")
        for d in node.decorator_list
    }
    if dec_names & _SKIP_DECORATORS:
      hits.append((node.lineno, "[G2:test-skip] Unconditional @skip/@xfail"))
    skip_stmts = [
        s.lineno
        for s in node.body
        if isinstance(s, ast.Expr)
        and getattr(getattr(s.value, "func", None), "attr", "")
        in {"skipTest", "skip"}
    ]
    hits.extend(
        (ln, "[G2:test-skip] Unconditional skip call") for ln in skip_stmts
    )
  return hits


def scan_vacuous_test_ast(tree: ast.AST) -> list[tuple[int, str]]:
  """Flags `def test_*` functions whose body is only docstring/`pass`/`...`."""
  test_fns = [
      n
      for n in ast.walk(tree)
      if isinstance(n, _FUNC_TYPES) and n.name.startswith("test_")
  ]
  hits: list[tuple[int, str]] = []
  for node in test_fns:
    real = [
        s
        for idx, s in enumerate(node.body)
        if not isinstance(s, ast.Pass)
        and not (
            isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Constant)
            and (idx == 0 or s.value.value is Ellipsis)
        )
    ]
    if not real:
      hits.append((node.lineno, f"[G2:vacuous-test] Empty {node.name}"))
  return hits


def scan_trivial_contract_ast(tree: ast.AST) -> list[tuple[int, str]]:
  """Flags 1-field `@dataclass`/`TypedDict` wrappers and 1-method `Protocol`s."""
  classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
  hits: list[tuple[int, str]] = []
  for cls in classes:
    dec_src = " ".join(ast.unparse(d) for d in cls.decorator_list)
    base_src = " ".join(ast.unparse(b) for b in cls.bases)
    fields = sum(isinstance(s, ast.AnnAssign) for s in cls.body)
    methods = sum(isinstance(s, _FUNC_TYPES) for s in cls.body)
    is_1field = (
        (bool(re.search(r"\bdataclass\b", dec_src)) and not cls.bases)
        or bool(re.search(r"\bTypedDict\b", base_src))
    ) and (fields == 1 and methods == 0)
    if is_1field:
      msg = f"[G1:trivial-contract] 1-field {cls.name}; prefer plain typed arg"
      hits.append((cls.lineno, msg))
    elif re.search(r"\bProtocol\b", base_src) and methods == 1 and fields == 0:
      msg = (
          f"[G1:trivial-contract] 1-method Protocol {cls.name}; prefer Callable"
      )
      hits.append((cls.lineno, msg))
  return hits


def scan_source_hygiene(
    rel: str, source: str, base_source: str = ""
) -> list[tuple[int, str]]:
  """Scans `source` for G1/G2 issues, optionally ratcheting `base_source`."""
  if re.search(r"/(testdata|test_data)/", f"/{rel}") or not source.strip():
    return []
  tree, _ = try_parse_ast(source)
  if tree is None:
    return []
  if _TEST_PATH_RE.search(rel):
    raw = [*scan_test_skip_ast(tree), *scan_vacuous_test_ast(tree)]
  else:
    raw = [
        *scan_mock_ast(tree),
        *scan_silent_except_ast(tree, source.splitlines()),
        *scan_trivial_contract_ast(tree),
    ]
  ordered = sorted(set(raw))
  if not base_source:
    return ordered
  base_cnt = collections.Counter(
      m for _, m in scan_source_hygiene(rel, base_source)
  )
  kept: list[tuple[int, str]] = []
  for ln, msg in ordered:
    if base_cnt[msg] > 0:
      base_cnt[msg] -= 1
    else:
      kept.append((ln, msg))
  return kept


def read_candidate_source(
    cwd: pathlib.Path, raw: str, new_sources: Mapping[str, str]
) -> tuple[str, str]:
  """Resolves normalized relative path and UTF-8 source for `raw`."""
  c_res = cwd.resolve()
  path = pathlib.Path(raw)
  abs_p = path if path.is_absolute() else c_res / path
  rel = (
      abs_p.relative_to(c_res).as_posix()
      if abs_p.is_relative_to(c_res)
      else raw
  )
  if raw in new_sources:
    return rel, new_sources[raw]
  if rel in new_sources:
    return rel, new_sources[rel]
  if not abs_p.is_file():
    return rel, ""
  return rel, abs_p.read_text(encoding="utf-8", errors="replace")


def evaluate_hygiene(
    cwd: pathlib.Path,
    files: Sequence[str],
    bases: Mapping[str, str] | None = None,
    new_sources: Mapping[str, str] | None = None,
    ratchet_base: bool = False,
) -> list[str]:
  """Evaluates G1/G2 AST hygiene across candidate `.py` files."""
  nsrc, bmap = new_sources or {}, bases or {}
  issues: list[str] = []
  for raw in sorted(set(files)):
    rel, src = read_candidate_source(cwd, raw, nsrc)
    base_src = (bmap.get(raw) or bmap.get(rel, "")) * ratchet_base
    hits = scan_source_hygiene(rel, src, base_src)
    issues.extend(f"Hygiene {rel}:{ln}: {msg}" for ln, msg in hits)
  return issues


def count_duck_typing_calls(source: str) -> int:
  """Counts defensive fallback calls (`getattr(..., d)`, `hasattr`, `.get(..., d)`)."""
  tree, _ = try_parse_ast(source)
  if tree is None:
    return 0
  total = 0
  for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
      continue
    fn, nargs = node.func, len(node.args) + len(node.keywords)
    is_getattr = isinstance(fn, ast.Name) and fn.id == "getattr" and nargs >= 3
    is_hasattr = isinstance(fn, ast.Name) and fn.id == "hasattr" and nargs >= 2
    is_get = isinstance(fn, ast.Attribute) and fn.attr == "get" and nargs >= 2
    total += is_getattr or is_hasattr or is_get
  return total


def sum_duck_typing_delta(
    raw_sources: Mapping[str, tuple[str, str]], prod_rels: Sequence[str]
) -> int:
  """Computes net delta of defensive duck-typing calls across `prod_rels`."""
  return sum(
      count_duck_typing_calls(raw_sources[r][0])
      - count_duck_typing_calls(raw_sources[r][1])
      for r in prod_rels
  )


_COVERAGE_RUNNER_CODE = """\
import importlib.util, io, json, os, pathlib, sys, tempfile, threading, types, unittest
target_abs = os.path.realpath(sys.argv[1])
test_path = pathlib.Path(sys.argv[2]).resolve()
sys.path[:0] = [str(test_path.parent), os.getcwd()]
sys.modules.pop(pathlib.Path(target_abs).stem, None)
sys.modules.pop(test_path.stem, None)
for pkg in ("tunix", "tunix.oss", "tunix.oss.scripts"):
  if pkg not in sys.modules:
    m = types.ModuleType(pkg)
    m.__path__ = [str(test_path.parent)]
    sys.modules[pkg] = m
try:
  import absl.testing.absltest  # noqa: F401
except ImportError:
  class _TestCase(unittest.TestCase):
    def create_tempdir(self):
      d = self.enterContext(tempfile.TemporaryDirectory())
      return types.SimpleNamespace(full_path=d)
    def assertLen(self, c, n, msg=None): self.assertEqual(len(c), n, msg)
    def assertEmpty(self, c, msg=None): self.assertEqual(len(c), 0, msg)
    def assertNotEmpty(self, c, msg=None): self.assertTrue(len(c) > 0, msg)
  absl_m, test_m = types.ModuleType("absl"), types.ModuleType("absl.testing")
  at_m = types.ModuleType("absl.testing.absltest")
  at_m.TestCase, at_m.main = _TestCase, unittest.main
  test_m.absltest, absl_m.testing = at_m, test_m
  sys.modules.update({"absl": absl_m, "absl.testing": test_m, "absl.testing.absltest": at_m})
executed, cache = set(), {}
prev_tr, prev_th = sys.gettrace(), getattr(threading, "gettrace", lambda: None)()
def line_tracer(frame, event, arg):
  if prev_tr is not None:
    prev_tr(frame, event, arg)
  if event == "line":
    executed.add(frame.f_lineno)
  return line_tracer
def call_tracer(frame, event, arg):
  if prev_tr is not None:
    prev_tr(frame, event, arg)
  fn = frame.f_code.co_filename
  if fn not in cache:
    cache[fn] = (os.path.realpath(fn) == target_abs)
  if cache[fn]:
    executed.add(frame.f_lineno)
    return line_tracer
  return None
sys.settrace(call_tracer)
threading.settrace(call_tracer)
try:
  spec = importlib.util.spec_from_file_location(test_path.stem, test_path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  suite = unittest.defaultTestLoader.loadTestsFromModule(mod)
  res = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
finally:
  sys.settrace(prev_tr)
  threading.settrace(prev_th)
ok = res.wasSuccessful() and res.testsRun > 0
print("COV_JSON:" + json.dumps({"ok": ok, "lines": sorted(executed)}))
"""


def extract_coverage_targets(
    source: str,
) -> tuple[list[tuple[str, set[int]]], set[int]]:
  """Extracts `(func_spans, executable_stmt_lines)` for coverage evaluation."""
  tree, _ = try_parse_ast(source)
  if tree is None:
    return [], set()
  fn_spans: list[tuple[str, set[int]]] = []
  stmt_lines: set[int] = set()
  skip_stmt = (*_FUNC_TYPES, ast.ClassDef, ast.Pass)
  for qual, node, body in iter_scope_defs(tree):
    if not isinstance(node, _FUNC_TYPES) or not body:
      continue
    if len(body) == 1 and ast.unparse(body[0]).strip() in ("pass", "..."):
      continue
    end_ln = node.end_lineno or body[-1].lineno
    fn_spans.append((qual, set(range(body[0].lineno, end_ln + 1))))
    sub_nodes = ast.walk(ast.Module(body=body, type_ignores=[]))
    stmt_lines.update(
        s.lineno
        for s in sub_nodes
        if isinstance(s, ast.stmt) and not isinstance(s, skip_stmt)
    )
  return fn_spans, stmt_lines


def run_traced_unittest(
    target_file: pathlib.Path, test_file: pathlib.Path, cwd: pathlib.Path
) -> tuple[bool, set[int]]:
  """Runs `test_file` under `sys.settrace` and returns `(ok, executed_lines)`."""
  argv = ["", str(target_file), str(test_file)]
  py_bin = sys.executable or shutil.which("python3")
  if py_bin:
    proc = subprocess.run(
        [py_bin, "-c", _COVERAGE_RUNNER_CODE, *argv[1:]],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=120,
    )
    out = proc.stdout if proc.returncode == 0 else ""
  else:
    buf, prev_argv = io.StringIO(), sys.argv
    try:
      sys.argv = argv
      with contextlib.redirect_stdout(buf):
        exec(_COVERAGE_RUNNER_CODE, {"__name__": "__cov__"})  # pylint: disable=exec-used
      out = buf.getvalue()
    except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
      out = ""
    finally:
      sys.argv = prev_argv
  cov_lines = [
      ln.removeprefix("COV_JSON:")
      for ln in out.splitlines()
      if ln.startswith("COV_JSON:")
  ]
  if not cov_lines:
    return False, set()
  payload = json.loads(cov_lines[-1])
  return bool(payload["ok"]), set(payload["lines"])


def check_single_file_coverage(
    c_res: pathlib.Path, rel: str, abs_p: pathlib.Path
) -> list[str]:
  """Verifies co-located unit test existence, pass status, and AST coverage."""
  cands = (
      abs_p.with_name(f"{abs_p.stem}_test.py"),
      abs_p.with_name(f"test_{abs_p.stem}.py"),
  )
  test_p = next((p for p in cands if p.is_file()), None)
  if test_p is None:
    return [
        f"Coverage {rel}: [G2_MISSING_UNIT_TEST] Missing {abs_p.stem}_test.py"
    ]
  fn_spans, stmt_lines = extract_coverage_targets(
      abs_p.read_text(encoding="utf-8", errors="replace")
  )
  ok, executed = run_traced_unittest(abs_p, test_p, c_res)
  if not ok:
    return [f"Coverage {rel}: [G2_UNIT_TEST_FAILURE] {test_p.name} failed"]
  uncov = [name for name, span in fn_spans if not (span & executed)]
  cov_cnt = len(stmt_lines & executed)
  stmt_pct = (cov_cnt / len(stmt_lines) * 100.0) if stmt_lines else 100.0
  viols: list[str] = []
  if uncov:
    names = ", ".join(uncov)
    viols.append(
        f"Coverage {rel}: [G2_UNCOVERED_FUNCTIONS] {len(uncov)}/{len(fn_spans)}"
        f" uncovered: {names}"
    )
  if stmt_pct < 90.0:
    viols.append(
        f"Coverage {rel}: [G2_LOW_STMT_COVERAGE] {cov_cnt}/{len(stmt_lines)}"
        f" ({stmt_pct:.1f}% < 90.0%)"
    )
  return viols


def evaluate_test_coverage(
    cwd: pathlib.Path, files: Sequence[str]
) -> list[str]:
  """Verifies co-located unit tests, 100% fn coverage, and >=90% stmt coverage."""
  c_res = cwd.resolve()
  targets: list[tuple[str, pathlib.Path]] = []
  for raw in sorted(set(files)):
    abs_p = (c_res / raw).resolve()
    rel = (
        abs_p.relative_to(c_res).as_posix()
        if abs_p.is_relative_to(c_res)
        else raw
    )
    if (
        abs_p.is_file()
        and abs_p.name != "__init__.py"
        and not _TEST_PATH_RE.search(rel)
        and not re.search(r"/(testdata|test_data)/", f"/{rel}")
    ):
      targets.append((rel, abs_p))
  if not targets:
    return []
  workers = 8 if (sys.executable or shutil.which("python3")) else 1
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    batches = pool.map(
        lambda it: check_single_file_coverage(c_res, *it), targets
    )
    return list(itertools.chain.from_iterable(batches))
