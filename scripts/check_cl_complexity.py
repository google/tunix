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
r"""4D CL Complexity Gate, AST Hygiene Checks, and Dependency Shard Planner.

Evaluates Python changes across four orthogonal understandability dimensions,
optional G1/G2 AST hygiene checks, and parallel AST dependency shard planning:
  - Dim 1 (Scope): Formatting-invariant AST statement delta (`net_stmts`).
  - Dim 2 (Local Control Flow): Cognitive, cyclomatic, and nesting ratchets.
  - Dim 3 (Anti-Layering): Single-caller hops, trivial helpers, pass-through
    wrappers, nested closures, and generator verb-to-execution naming.
  - Dim 4 (Clone Hygiene): Position-independent 2-gram AST Jaccard clones.
  - G1/G2 AST Hygiene (`--strict` / `--hygiene`):
      * `[G1:mock-branch]`: Blocks `unittest.mock` / `MagicMock` in prod `.py`.
      * `[G1:silent-except]`: Blocks broad `except: pass/continue` sans `noqa`.
      * `[G1:trivial-contract]`: Blocks 1-field `@dataclass`/`TypedDict` knob
        wrappers and 1-method `Protocol`s in prod `.py`.
      * `[G2:test-skip]`: Blocks unconditional `@skip`/`@xfail`/`skipTest` in
        test files unless `ALLOW_TEST_SKIP_WITH_BUG=<issue-id>` is set.
      * `[G2:vacuous-test]`: Blocks empty `def test_*` stubs (`pass`/`...`).
      * `--check-test-coverage`: Requires co-located `<stem>_test.py`
        (`G2_MISSING_UNIT_TEST`), 100% function coverage
        (`G2_UNCOVERED_FUNCTIONS`), and `>= 90%` statement line coverage
        (`G2_LOW_STMT_COVERAGE`).
  - Parallel Shard Planner (`--shard-plan`): Parses candidate `.py` files with
    `ThreadPoolExecutor(16)`, extracts cross-file import edges and top-level
    public 1-line signatures, weights files by `stmts + 2 * total_cognitive`,
    groups Union-Find connected components, cuts oversized components on Hub
    modules, packs `<= --max-shards` bins via Longest Processing Time (LPT),
    and emits 1-line `frozen_contracts` for cross-shard edges.

Usage Examples:
  # 1. Standard strict 4D gate + ratcheted G1/G2 hygiene on active VCS changes:
  python3 scripts/check_cl_complexity.py --strict

  # 2. Check changes against a base revision or explicit file list:
  python3 scripts/check_cl_complexity.py --rev main --strict
  python3 scripts/check_cl_complexity.py --files a.py a_test.py --hygiene

  # 3. Build a balanced AST dependency shard plan (stdout + optional JSON file):
  python3 scripts/check_cl_complexity.py --shard-plan --max-shards 4 \
      --files pkg/hub.py pkg/worker_a.py pkg/worker_b.py --out /tmp/plan.json

`--shard-plan` Output JSON Schema:
  {
    "max_shards": int,
    "total_files": int,
    "total_weight": int,
    "connected_components": list[list[str]],
    "hub_modules_cut": list[str],
    "shards": [{
      "shard_id": int,
      "files": list[str],
      "stmts": int,
      "cognitive": int,
      "weight": int,
      "frozen_contracts": list[str]
    }],
    "cross_shard_edges": [{
      "caller": str,
      "callee": str,
      "caller_shard": int,
      "callee_shard": int,
      "symbols": list[str]
    }],
    "frozen_contracts": list[str]
  }
"""

from __future__ import annotations

import argparse
import ast
import collections
from collections.abc import Mapping, Sequence
import concurrent.futures
import dataclasses
import fnmatch
import hashlib
import io
import itertools
import json
import os
import pathlib
import re
import subprocess
import sys
import tokenize
import tomllib

if str(pathlib.Path(__file__).resolve().parent) not in sys.path:
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
  from scripts.ast_gate_lib import (  # pylint: disable=g-import-not-at-top,g-importing-member,g-multiple-import
      count_duck_typing_calls,
      evaluate_hygiene,
      evaluate_test_coverage,
      format_public_signature,
      iter_scope_defs,
      read_candidate_source,
      sum_duck_typing_delta,
      try_parse_ast,
  )
except ImportError:
  from ast_gate_lib import (  # pylint: disable=g-import-not-at-top,g-importing-member,g-multiple-import  # pyrefly: ignore[missing-import]
      count_duck_typing_calls,
      evaluate_hygiene,
      evaluate_test_coverage,
      format_public_signature,
      iter_scope_defs,
      read_candidate_source,
      sum_duck_typing_delta,
      try_parse_ast,
  )

_DEFAULT_CONFIG = pathlib.Path("pyproject.toml")
_PKG_CONFIG = pathlib.Path(__file__).absolute().parent.parent / _DEFAULT_CONFIG
_FALLBACK_CFG = pathlib.Path(__file__).with_name("cl_complexity_config.toml")
_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_CALLABLE_TYPES = (*_FUNC_TYPES, ast.Lambda)
_SKIP_RE = re.compile(r"SKIP_CL_COMPLEXITY\s*[=:]\s*([^\s\"'\\][^\n\"'\\]*)")
_BLOCK_NODES = (ast.stmt, ast.ExceptHandler, ast.match_case)
_SHINGLE_OPS = (ast.operator, ast.unaryop, ast.boolop, ast.cmpop)
_FnNode = ast.FunctionDef | ast.AsyncFunctionDef
_LoopNode = ast.For | ast.AsyncFor | ast.While
_CompNode = ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp
_COMPOUND_BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Match,
    ast.Try,
    ast.TryStar,
    ast.IfExp,
    *_CALLABLE_TYPES,
)
_SKIP_TOKS = frozenset({
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENCODING,
    tokenize.ENDMARKER,
})
_NON_SEMANTIC_DECS = frozenset({
    "staticmethod",
    "classmethod",
    "no_type_check",
    "final",
    "override",
    "overload",
})
_GENERIC_NAMES = frozenset({
    "main",
    "run",
    "step",
    "train",
    "eval",
    "evaluate",
    "forward",
    "call",
    "close",
    "reset",
    "setup",
    "init",
    "get",
    "set",
    "parse",
    "build",
    "create",
    "process",
    "update",
    "validate",
    "helper",
    "wrapper",
    "inner",
})
_EAGER_STREAM_VERBS = (
    "drain_",
    "flush_",
    "collect_",
    "materialize_",
    "gather_",
    "load_all_",
)


@dataclasses.dataclass
class GateConfig:
  """Configuration thresholds, weights, and fork rules for the gate."""

  enabled_dimensions: tuple[int, ...]
  intentional_forks: tuple[tuple[str, str], ...]
  weight_prod_loc: float
  weight_cognitive_excess: float
  weight_call_depth_excess: float
  weight_trivial_helper: float
  weight_pass_through_func: float
  weight_nested_closure: float
  weight_duplicate_clone: float
  max_cognitive_warn: int
  max_cognitive_fail: int
  max_cyclomatic_fail: int
  max_nesting_fail: int
  max_legacy_cog_delta_fail: int
  max_legacy_cyc_delta_fail: int
  max_call_depth_warn: int
  max_call_depth_fail: int
  max_trivial_helpers_fail: int
  max_pass_through_ratio_fail: float
  max_nested_closures_warn: int
  max_nested_closures_fail: int
  trivial_helper_max_stmts: int
  trivial_helper_max_cognitive: int
  max_sprawl_helpers_fail: int
  max_file_cog_delta_sprawl: int
  min_clone_stmts: int
  min_clone_cognitive: int
  min_clone_jaccard: float
  base_score: float = 0.0


@dataclasses.dataclass
class FunctionMetrics:
  """Per-function complexity, layering, and clone-matching metrics."""

  file_path: str
  qualname: str
  stripped_name: str
  stmts: int
  cognitive: int
  base_cognitive: int
  delta_cognitive: int
  cyclomatic: int
  base_cyclomatic: int
  delta_cyclomatic: int
  max_nesting: int
  base_nesting: int
  delta_nesting: int
  is_private: bool
  semantic_decorated: bool
  is_pass_through: bool
  is_generator: bool
  nested_closures: tuple[str, ...]
  calls: tuple[str, ...]
  ast_digest: str
  shingles: collections.Counter[str]
  matched_base_key: str = ""
  is_new_or_modified: bool = True


@dataclasses.dataclass
class GateReport:
  """Aggregated CL Complexity Gate evaluation report."""

  passed: bool
  score: float
  enabled_dimensions: list[int]
  prod_loc_added: int
  net_loc_delta: int
  net_cog_delta: int
  test_loc_added: int
  prod_files_changed: int
  max_cognitive: int
  max_cyclomatic: int
  max_nesting: int
  max_call_depth_hops: int
  max_call_chain: list[str]
  trivial_single_call_helpers: list[str]
  nested_closures: list[str]
  pass_through_funcs: list[str]
  duplicate_clones: list[str]
  violations: list[str]
  skip_reason: str
  warnings: list[str] = dataclasses.field(default_factory=list)
  duck_typing_calls: int = 0


def load_config(
    config_path: pathlib.Path = _DEFAULT_CONFIG,
    cwd: pathlib.Path | None = None,
) -> GateConfig:
  """Loads `[tool.cl_complexity_gate]` thresholds, weights, and fork rules."""
  base = (cwd or pathlib.Path.cwd()).resolve()
  ancs = (p / _DEFAULT_CONFIG for p in (base, *base.parents))
  pool = (
      (base / config_path, config_path, *ancs, _PKG_CONFIG, _FALLBACK_CFG)
      if config_path == _DEFAULT_CONFIG
      else (config_path,)
  )
  cands = [p for p in pool if p.is_file()]
  valid = [
      p
      for p in cands
      if re.search(
          r"cl_complexity_gate|\[thresholds\]",
          p.read_text(encoding="utf-8"),
      )
  ]
  resolved = (valid or cands or [config_path])[0]
  raw = tomllib.loads(resolved.read_text(encoding="utf-8"))
  cfg = raw.get("tool", {}).get("cl_complexity_gate") or raw
  weights_tbl = cfg["weights"]
  thresh_tbl = cfg["thresholds"]
  assert isinstance(weights_tbl, Mapping) and isinstance(thresh_tbl, Mapping)
  weights = {f"weight_{k}": v for k, v in weights_tbl.items()}
  raw_forks = cfg.get("intentional_forks", ())
  assert isinstance(raw_forks, Sequence)
  forks = tuple((str(a), str(b)) for a, b in raw_forks)
  raw_dims = cfg.get("enabled_dimensions", (1, 2, 3, 4))
  assert isinstance(raw_dims, Sequence)
  dims = tuple(int(d) for d in raw_dims)
  merged = dict(thresh_tbl) | weights
  merged["intentional_forks"] = forks
  merged["enabled_dimensions"] = dims
  valid_fields = {f.name for f in dataclasses.fields(GateConfig)}
  return GateConfig(**{k: v for k, v in merged.items() if k in valid_fields})


def count_code_loc(source: str) -> set[int]:
  """Returns 1-based line numbers containing executable code."""
  if not source.strip():
    return set()
  tree, _ = try_parse_ast(source)
  if tree is None:
    return set()
  doc_lines = {
      n.body[0].lineno
      for n in ast.walk(tree)
      if isinstance(n, (ast.Module, ast.ClassDef, *_FUNC_TYPES))
      and ast.get_docstring(n, clean=False) is not None
  }
  toks = [
      range(t.start[0], t.end[0] + 1)
      for t in tokenize.generate_tokens(io.StringIO(source).readline)
      if t.type not in _SKIP_TOKS and t.start[0] not in doc_lines
  ]
  return set(itertools.chain.from_iterable(toks))


def count_ast_stmts(source: str) -> int:
  """Counts non-docstring executable AST statements (formatting-invariant)."""
  if not source.strip():
    return 0
  tree, _ = try_parse_ast(source)
  if tree is None:
    return 0
  docs = {
      id(n.body[0])
      for n in ast.walk(tree)
      if isinstance(n, (ast.Module, ast.ClassDef, *_FUNC_TYPES))
      and ast.get_docstring(n, clean=False) is not None
  }
  return sum(
      isinstance(n, ast.stmt) and id(n) not in docs for n in ast.walk(tree)
  )


def decorator_info(node: ast.AST) -> tuple[set[str], str]:
  """Returns `(decorator_names, accessor_suffix)` for a function AST node."""
  decs = node.decorator_list if isinstance(node, _FUNC_TYPES) else []
  targets = [d.func if isinstance(d, ast.Call) else d for d in decs]
  names = {
      t.id
      if isinstance(t, ast.Name)
      else (t.attr if isinstance(t, ast.Attribute) else ast.unparse(t))
      for t in targets
  }
  acc = [
      f":{t.attr}"
      for t in targets
      if isinstance(t, ast.Attribute) and t.attr in ("setter", "deleter")
  ]
  return names, (acc[-1] if acc else "")


def is_early_exit_guard(node: ast.If) -> bool:
  """Returns True when `if cond: return/raise/continue/break` has no `else`."""
  return (
      not node.orelse
      and len(node.body) == 1
      and isinstance(
          node.body[0], (ast.Return, ast.Raise, ast.Continue, ast.Break)
      )
  )


def is_linear_dispatch(branches: Sequence[Sequence[ast.stmt]]) -> bool:
  """Returns True for `>= 5`-arm flat dispatch tables (`<= 2` simple stmts/arm)."""
  if len(branches) < 5 or any(len(b) > 2 for b in branches):
    return False
  sub_nodes = itertools.chain.from_iterable(
      ast.walk(s) for b in branches for s in b
  )
  return not any(isinstance(n, _COMPOUND_BRANCH_NODES) for n in sub_nodes)


def stmt_shingles(stmts: list[ast.stmt]) -> collections.Counter[str]:
  """Computes position-independent DFS statement 2-gram multiset shingles."""
  seq: list[str] = []
  stack: list[tuple[ast.AST, int]] = [(s, 0) for s in reversed(stmts)]
  while stack:
    node, depth = stack.pop()
    exprs = [
        c for c in ast.iter_child_nodes(node) if not isinstance(c, _BLOCK_NODES)
    ]
    expr_nodes = itertools.chain.from_iterable(ast.walk(e) for e in exprs)
    ops = [
        type(sub).__name__
        for sub in expr_nodes
        if isinstance(sub, _SHINGLE_OPS)
    ]
    expr_src = "\n".join(
        ast.unparse(e) for e in exprs if isinstance(e, ast.expr)
    )
    calls = re.findall(r"([A-Za-z_]\w*)\s*\(", expr_src)[:2]
    seq.append(f"d{depth}:{type(node).__name__}:{ops}:{calls}")
    kids = [
        c for c in ast.iter_child_nodes(node) if isinstance(c, _BLOCK_NODES)
    ]
    step = 0 if isinstance(node, ast.Try) else 1
    stack.extend((c, depth + step) for c in reversed(kids))
  grams = (
      seq
      if len(seq) <= 1
      else [f"{seq[i]}=>{seq[i + 1]}" for i in range(len(seq) - 1)]
  )
  return collections.Counter(grams)


def multiset_jaccard(
    a: collections.Counter[str], b: collections.Counter[str]
) -> float:
  """Computes exact multiset Jaccard similarity `sum(min) / sum(max)`."""
  union = a | b
  return (a & b).total() / union.total() if union else 0.0


def canonical_ast_digest(stmts: list[ast.stmt]) -> str:
  """Computes alpha-normalized AST SHA-256 digest for move matching."""
  if not stmts:
    return ""
  tree = ast.parse("\n".join(ast.unparse(s) for s in stmts))
  for sub in ast.walk(tree):
    if isinstance(sub, ast.Name):
      sub.id = "_ID"
    elif isinstance(sub, ast.arg):
      sub.arg = "_ARG"
  raw = ast.dump(tree, include_attributes=False).encode()
  return hashlib.sha256(raw).hexdigest()[:16]


def extract_callee_name(
    expr: ast.expr, qualname: str, aliases: dict[str, str]
) -> str | None:
  """Extracts intra-file function, alias, or class/instance method callee."""
  if isinstance(expr, ast.Name):
    return aliases[expr.id] if expr.id in aliases else expr.id
  base_cls = (
      qualname.rsplit(".", 1)[0].split(":", 1)[0] if "." in qualname else ""
  )
  if (
      isinstance(expr, ast.Attribute)
      and isinstance(expr.value, ast.Name)
      and expr.value.id in ("self", "cls", base_cls)
  ):
    return f"{base_cls}.{expr.attr}" if base_cls else expr.attr
  return None


def is_pass_through(node: ast.AST, stmts: list[ast.stmt]) -> bool:
  """Detects 1-line `return f(x)` and 2-line `r = f(x); return r` shims."""
  if not isinstance(node, _CALLABLE_TYPES) or not stmts:
    return not isinstance(node, _CALLABLE_TYPES)
  first = stmts[0]
  if not isinstance(first, (ast.Return, ast.Expr, ast.Assign)):
    return False
  call = (
      first.value.value if isinstance(first.value, ast.Await) else first.value
  )
  if not isinstance(call, ast.Call):
    return False
  unparsed = "\n".join(ast.unparse(s) for s in stmts)
  valid_shape = (
      len(stmts) == 1 and isinstance(first, (ast.Return, ast.Expr))
  ) or (
      len(stmts) == 2
      and isinstance(first, ast.Assign)
      and bool(
          re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*[\s\S]+\nreturn\s+\1", unparsed)
      )
  )
  if not valid_shape:
    return False
  var_args = filter(None, (node.args.vararg, node.args.kwarg))
  all_args = [
      *node.args.posonlyargs,
      *node.args.args,
      *node.args.kwonlyargs,
      *var_args,
  ]
  params = {a.arg for a in all_args} - {"self", "cls"}
  unstarred = [a.value for a in call.args if isinstance(a, ast.Starred)]
  call_exprs = (*call.args, *unstarred, *(kw.value for kw in call.keywords))
  fwd = {sub.id for sub in call_exprs if isinstance(sub, ast.Name)}
  return bool(params & fwd)


class _ComplexityVisitor(ast.NodeVisitor):
  """Computes Cognitive, Cyclomatic, Nesting, Closure, and Call metrics."""

  # pylint: disable=invalid-name,missing-function-docstring

  def __init__(
      self,
      fn: ast.AST,
      qualname: str,
      body: list[ast.stmt],
      local_defs: set[str],
  ):
    self.qualname, self.local_defs = qualname, local_defs
    self.cognitive, self.cyclomatic, self.nesting, self.max_nesting = 0, 1, 0, 0
    self.closures, self.calls, self.aliases = [], [], {}
    decs, _ = decorator_info(fn)
    self.semantic_decorated = bool(decs - _NON_SEMANTIC_DECS - local_defs)
    mod = ast.Module(body=body, type_ignores=[])
    calls_in_mod = [sub for sub in ast.walk(mod) if isinstance(sub, ast.Call)]
    call_args = itertools.chain.from_iterable(
        (*c.args, *(kw.value for kw in c.keywords)) for c in calls_in_mod
    )
    self.arg_lambdas = {id(a) for a in call_args if isinstance(a, ast.Lambda)}
    self.visit_block(body, 0)
    self.pass_through = (
        not self.semantic_decorated
        and is_pass_through(fn, body)
        and self.cognitive == 0
    )

  def _bump(self, guard_exempt: bool = False) -> None:
    self.cognitive += 1 + (0 if guard_exempt else self.nesting)
    self.cyclomatic += 1
    if not guard_exempt:
      self.max_nesting = max(self.max_nesting, self.nesting + 1)

  def visit_block(
      self, nodes: collections.abc.Sequence[ast.AST], delta: int = 1
  ) -> None:
    self.nesting += delta
    for node in nodes:
      self.visit(node)
    self.nesting -= delta

  def visit_Assign(self, node: ast.Assign) -> None:
    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
      if callee := extract_callee_name(node.value, self.qualname, self.aliases):
        self.aliases[node.targets[0].id] = callee
        self.calls.append(callee)
    seq_elts = (
        list(node.value.elts)
        if isinstance(node.value, (ast.List, ast.Tuple, ast.Set))
        else []
    )
    elts = (
        list(node.value.values)
        if isinstance(node.value, ast.Dict)
        else seq_elts
    )
    for elem in elts:
      if elem and (
          callee := extract_callee_name(elem, self.qualname, self.aliases)
      ):
        self.calls.append(callee)
    self.generic_visit(node)

  def visit_If(self, node: ast.If) -> None:
    cap_cyc, cap_cog = self.cyclomatic + 4, self.cognitive + 4 + self.nesting
    branches: list[Sequence[ast.stmt]] = [node.body]
    is_flat_guard = is_early_exit_guard(node) and (
        self.nesting == 0
        or (
            self.nesting == 1
            and isinstance(node.body[0], (ast.Continue, ast.Break))
        )
    )
    self._bump(guard_exempt=is_flat_guard)
    self.visit(node.test)
    self.visit_block(node.body, 1)
    curr = node.orelse
    while len(curr) == 1 and isinstance(curr[0], ast.If):
      branches.append(curr[0].body)
      self.cognitive, self.cyclomatic = self.cognitive + 1, self.cyclomatic + 1
      self.visit(curr[0].test)
      self.visit_block(curr[0].body, 1)
      curr = curr[0].orelse
    if curr:
      branches.append(curr)
      self.cognitive += 1
      self.visit_block(curr, 1)
    if is_linear_dispatch(branches):
      self.cyclomatic = min(self.cyclomatic, cap_cyc)
      self.cognitive = min(self.cognitive, cap_cog)

  def visit_Match(self, node: ast.Match) -> None:
    self.visit(node.subject)
    cap_cyc = self.cyclomatic + 4
    cap_cog = self.cognitive + 4 * (1 + self.nesting)
    for case in node.cases:
      self._bump()
      if case.guard:
        self.visit(case.guard)
      self.visit_block(case.body, 1)
    if is_linear_dispatch([c.body for c in node.cases]):
      self.cyclomatic = min(self.cyclomatic, cap_cyc)
      self.cognitive = min(self.cognitive, cap_cog)

  def visit_For(self, node: _LoopNode) -> None:
    self.visit(node.test if isinstance(node, ast.While) else node.iter)
    self._bump()
    self.visit_block(node.body, 1)
    if node.orelse:
      self.cognitive += 1
      self.visit_block(node.orelse, 1)

  visit_AsyncFor = visit_While = visit_For

  def visit_Try(self, node: ast.Try | ast.TryStar) -> None:
    self.visit_block(node.body + node.finalbody, 0)
    for handler in node.handlers:
      self._bump()
      self.visit_block(handler.body, 1)
    if node.orelse:
      self.cognitive += 1
      self.visit_block(node.orelse, 1)

  visit_TryStar = visit_Try

  def visit_IfExp(self, node: ast.IfExp) -> None:
    self._bump()
    self.visit_block(list(ast.iter_child_nodes(node)), 1)

  def visit_BoolOp(self, node: ast.BoolOp) -> None:
    self.cognitive += 1
    self.cyclomatic += len(node.values) - 1
    for val in node.values:
      if not (
          isinstance(val, ast.BoolOp) and isinstance(val.op, type(node.op))
      ):
        self.visit(val)

  def visit_ListComp(self, node: _CompNode) -> None:
    extra = max(0, len(node.generators) - 1)
    self.cognitive += extra
    self.cyclomatic += extra
    for gen in node.generators:
      self.cyclomatic += len(gen.ifs)
      self.cognitive += len(gen.ifs)
    self.visit_block(list(ast.iter_child_nodes(node)), 1)

  visit_SetComp = visit_GeneratorExp = visit_DictComp = _comp = visit_ListComp

  def visit_Call(self, node: ast.Call) -> None:
    for expr in (node.func, *node.args, *(kw.value for kw in node.keywords)):
      if callee := extract_callee_name(expr, self.qualname, self.aliases):
        if callee == self.qualname and expr is node.func:
          self.cognitive += 1
        self.calls.append(callee)
    self.generic_visit(node)

  def visit_FunctionDef(self, node: _FnNode | ast.Lambda) -> None:
    if isinstance(node, ast.Lambda) and id(node) in self.arg_lambdas:
      self.visit_block([node.body], 0)
      return
    decs, _ = (
        decorator_info(node) if isinstance(node, _FUNC_TYPES) else (set(), "")
    )
    body = [node.body] if isinstance(node, ast.Lambda) else node.body
    if decs - _NON_SEMANTIC_DECS - self.local_defs:
      self.visit_block(body, 0)
      return
    self._bump()
    name = node.name if isinstance(node, _FUNC_TYPES) else "<lambda>"
    self.closures.append(f"{self.qualname}.{name}")
    self.visit_block(body, 1)

  visit_AsyncFunctionDef = visit_Lambda = visit_FunctionDef
  # pylint: enable=invalid-name,missing-function-docstring


def has_own_yield(stmts: list[ast.stmt]) -> bool:
  """Returns True if `stmts` yields directly outside nested function scopes."""
  stack: list[ast.AST] = list(stmts)
  while stack:
    node = stack.pop()
    if isinstance(node, (ast.Yield, ast.YieldFrom)):
      return True
    if not isinstance(node, (*_FUNC_TYPES, ast.Lambda)):
      stack.extend(ast.iter_child_nodes(node))
  return False


def analyze_functions(
    source: str, file_path: str = ""
) -> list[FunctionMetrics]:
  """Extracts function metrics across top-level, classes, and nested scopes."""
  if not source.strip():
    return []
  tree, _ = try_parse_ast(source)
  if tree is None:
    return []
  local_defs = {n.name for n in ast.walk(tree) if isinstance(n, _FUNC_TYPES)}
  seen, out = set(), []
  for base_qual, node, body in iter_scope_defs(tree):
    decs, suffix = decorator_info(node)
    is_stub = not body or (
        len(body) == 1 and isinstance(body[0], (ast.Pass, ast.Expr))
    )
    if "overload" in decs and is_stub:
      continue
    qual = f"{base_qual}{suffix}"
    qual = f"{qual}@L{node.lineno}" if qual in seen else qual
    seen.add(qual)
    v = _ComplexityVisitor(node, qual, body, local_defs)
    raw = qual.split(".")[-1].split(":", 1)[0].split("@", 1)[0]
    body_mod = ast.Module(body=body, type_ignores=[])
    stmts = sum(isinstance(sub, ast.stmt) for sub in ast.walk(body_mod))
    out.append(
        FunctionMetrics(
            file_path=file_path,
            qualname=qual,
            stripped_name=raw.lstrip("_"),
            stmts=stmts,
            cognitive=v.cognitive,
            base_cognitive=0,
            delta_cognitive=v.cognitive,
            cyclomatic=v.cyclomatic,
            base_cyclomatic=1,
            delta_cyclomatic=v.cyclomatic - 1,
            max_nesting=v.max_nesting,
            base_nesting=0,
            delta_nesting=v.max_nesting,
            is_private=raw.startswith("_") and not raw.startswith("__"),
            semantic_decorated=v.semantic_decorated,
            is_pass_through=v.pass_through,
            is_generator=has_own_yield(body),
            nested_closures=tuple(v.closures),
            calls=tuple(v.calls),
            ast_digest=canonical_ast_digest(body),
            shingles=stmt_shingles(body),
        )
    )
  return out


def is_sc_candidate(f: FunctionMetrics, callers: set[str]) -> bool:
  """Returns True for non-stage private helpers with one intra-file caller."""
  cohesive_stage = (
      f.stmts >= 8 and 4 <= f.cognitive <= 12 and not f.is_pass_through
  )
  return (
      f.is_private
      and not f.semantic_decorated
      and not cohesive_stage
      and len(callers) == 1
  )


def compute_call_graph_metrics(
    funcs: list[FunctionMetrics], cfg: GateConfig
) -> tuple[int, list[str], list[str]]:
  """Computes max intra-file single-caller chain hops and trivial helpers."""
  leaf_cnt = collections.Counter([f.qualname.split(".")[-1] for f in funcs])
  fmap = {
      f.qualname.split(".")[-1]: f
      for f in funcs
      if leaf_cnt[f.qualname.split(".")[-1]] == 1
  } | {f.qualname: f for f in funcs}
  adj: dict[str, set[str]] = {f.qualname: set() for f in funcs}
  callers: dict[str, set[str]] = {f.qualname: set() for f in funcs}
  for f in funcs:
    targets = {fmap[c].qualname for c in f.calls if c in fmap} - {f.qualname}
    adj[f.qualname] = targets
    for t in targets:
      callers[t].add(f.qualname)
  eligible = {
      f.qualname for f in funcs if is_sc_candidate(f, callers[f.qualname])
  }
  sc_adj = {u: tgts & eligible for u, tgts in adj.items()}
  paths = {n: [n] for n in sorted(sc_adj)}
  for _ in range(len(sc_adj)):
    paths = {
        u: max(
            [[u]]
            + [[u] + paths[v] for v in sorted(sc_adj[u]) if u not in paths[v]],
            key=len,
        )
        for u in paths
    }
  best = max(list(paths.values()), key=len, default=[])
  trivial = [
      q
      for q in sorted(eligible)
      if fmap[q].is_pass_through
      or (
          adj[q]
          and fmap[q].stmts <= cfg.trivial_helper_max_stmts
          and fmap[q].cognitive <= cfg.trivial_helper_max_cognitive
      )
  ]
  return max(0, len(best) - 1), best, trivial


def assign_baseline(nf: FunctionMetrics, bf: FunctionMetrics) -> None:
  """Transfers 1-to-1 baseline metrics and modification identity."""
  nf.base_cognitive = bf.cognitive
  nf.delta_cognitive = nf.cognitive - bf.cognitive
  nf.base_cyclomatic = bf.cyclomatic
  nf.delta_cyclomatic = nf.cyclomatic - bf.cyclomatic
  nf.base_nesting = bf.max_nesting
  nf.delta_nesting = nf.max_nesting - bf.max_nesting
  nf.matched_base_key = f"{bf.file_path}:{bf.qualname}"
  nf.is_new_or_modified = (
      nf.ast_digest != bf.ast_digest
      or nf.cognitive != bf.cognitive
      or nf.cyclomatic != bf.cyclomatic
      or nf.max_nesting != bf.max_nesting
  )


def pair_move_score(nf: FunctionMetrics, rf: FunctionMetrics) -> float:
  """Computes 1-to-1 move similarity if eligible, else `0.0`."""
  sc = (
      2.0
      if (nf.ast_digest and rf.ast_digest == nf.ast_digest)
      else multiset_jaccard(nf.shingles, rf.shingles)
  )
  same_domain_name = rf.stripped_name == nf.stripped_name not in _GENERIC_NAMES
  return sc if (sc >= 0.65 or (sc >= 0.35 and same_domain_name)) else 0.0


def match_baselines_tiered(
    base_by_file: dict[str, list[FunctionMetrics]],
    new_by_file: dict[str, list[FunctionMetrics]],
) -> None:
  """Performs global 1-to-1 baseline matching across edits and moves."""
  all_base = itertools.chain.from_iterable(base_by_file.values())
  bmaps = {(bf.file_path, bf.qualname): bf for bf in all_base}
  exact_locks: dict[tuple[str, str], FunctionMetrics] = {}
  unmatched: list[FunctionMetrics] = []
  for nf in itertools.chain.from_iterable(new_by_file.values()):
    bf = bmaps.get((nf.file_path, nf.qualname))
    if bf and (bf.stmts < 4 or not nf.is_pass_through):
      assign_baseline(nf, bf)
      exact_locks[(nf.file_path, nf.qualname)] = bf
      continue
    unmatched.append(nf)
  removed = [bf for k, bf in bmaps.items() if k not in exact_locks]
  cand_grid = itertools.product(enumerate(unmatched), enumerate(removed))
  pairs = sorted(
      [
          (s, -ni, -ri, ni, ri)
          for (ni, nf), (ri, rf) in cand_grid
          if (s := pair_move_score(nf, rf))
      ],
      reverse=True,
  )
  used: set[int | tuple[str, int]] = set()
  for _, _, _, ni, ri in pairs:
    if {ni, ("r", ri)}.isdisjoint(used):
      assign_baseline(unmatched[ni], removed[ri])
      used.update((ni, ("r", ri)))


def is_fork_exempt(
    p1: str, p2: str, forks: tuple[tuple[str, str], ...]
) -> bool:
  """Returns True when `(p1, p2)` is a sanctioned cross-file fork pair."""
  if p1 == p2:
    return False
  q1, q2 = p1.split("tunix/")[-1], p2.split("tunix/")[-1]
  return any(
      (fnmatch.fnmatch(q1, a) and fnmatch.fnmatch(q2, b))
      or (fnmatch.fnmatch(q1, b) and fnmatch.fnmatch(q2, a))
      for a, b in forks
  )


def detect_clones(
    new_by_file: dict[str, list[FunctionMetrics]],
    base_by_file: dict[str, list[FunctionMetrics]],
    ref_by_file: dict[str, list[FunctionMetrics]],
    cfg: GateConfig,
) -> list[str]:
  """Detects newly introduced intra-file and cross-file structural clones."""
  all_new_and_ref = [*new_by_file.values(), *ref_by_file.values()]
  pool = [
      f
      for f in itertools.chain.from_iterable(all_new_and_ref)
      if f.stmts >= cfg.min_clone_stmts
      and f.cognitive >= cfg.min_clone_cognitive
  ]
  changed = [
      f for f in pool if f.is_new_or_modified and f.file_path in new_by_file
  ]
  bcands = [
      bf
      for bf in itertools.chain.from_iterable(base_by_file.values())
      if bf.stmts >= cfg.min_clone_stmts
      and bf.cognitive >= cfg.min_clone_cognitive
  ]
  bpairs = {
      tuple(
          sorted((f"{a.file_path}:{a.qualname}", f"{b.file_path}:{b.qualname}"))
      )
      for a, b in itertools.combinations(bcands, 2)
      if multiset_jaccard(a.shingles, b.shingles) >= cfg.min_clone_jaccard
  }
  seen = {
      tuple(
          sorted(
              (f"{f1.file_path}:{f1.qualname}", f"{f2.file_path}:{f2.qualname}")
          )
      ): (f1, f2)
      for f1, f2 in itertools.product(changed, pool)
      if (f1.file_path, f1.qualname) != (f2.file_path, f2.qualname)
      and not is_fork_exempt(f1.file_path, f2.file_path, cfg.intentional_forks)
      and multiset_jaccard(f1.shingles, f2.shingles) >= cfg.min_clone_jaccard
  }
  return [
      f"{pair[0]} <-> {pair[1]}"
      for pair, (f1, f2) in seen.items()
      if tuple(
          sorted((
              f1.matched_base_key or f"{f1.file_path}:{f1.qualname}",
              f2.matched_base_key or f"{f2.file_path}:{f2.qualname}",
          ))
      )
      not in bpairs
  ]


def run_cmd_stdout(cmd: list[str], cwd: pathlib.Path) -> str:
  """Runs `cmd` in `cwd` and returns stdout on exit code 0, else empty str."""
  res = subprocess.run(
      cmd, cwd=cwd, capture_output=True, text=True, check=False, timeout=30
  )
  return res.stdout if res.returncode == 0 else ""


def resolve_changed_files(
    cwd: pathlib.Path,
    rev: str | None,
    explicit_files: list[str] | None,
    new_sources: dict[str, str] | None = None,
) -> tuple[list[str], collections.defaultdict[str, str]]:
  """Resolves changed `.py` files and their baseline source contents."""
  bases: collections.defaultdict[str, str] = collections.defaultdict(str)
  nsrc = new_sources if new_sources is not None else {}
  if explicit_files:
    bpath = cwd / (rev or "__none__")
    cands = [
        (
            f,
            (
                bpath,
                bpath
                / re.sub(r"^.*?/(?:\.gemini/config|_agents|GOOGLE_INTERNAL_PACKAGE_PATH)/", "", f),
            )[bpath.is_dir()],
        )
        for f in explicit_files
    ]
    bases.update(
        {f: p.read_text(encoding="utf-8") for f, p in cands if p.is_file()}
    )
    nsrc.update([(os.path.relpath(bpath, cwd), "")] * bpath.is_file())
    return explicit_files, bases
  gdiff = "git diff --name-only --diff-filter=AMDR"
  git_rev = [(f"{gdiff} {rev}", f"git show {rev}:{{}}", "")]
  git_def = [
      (f"{gdiff} HEAD", "git show HEAD:{}", ""),
      (f"{gdiff} HEAD~1", "git show HEAD~1:{}", ""),
  ]
  git_specs = git_rev if rev else git_def
  for lcmd, bfmt, ncmd in [
      *git_specs,
  ]:
    if files := re.findall(
        r"(?m)^(\S+\.py)$", run_cmd_stdout(lcmd.split(), cwd)
    ):
      bases.update(
          {f: run_cmd_stdout(bfmt.format(f).split(), cwd) for f in files}
      )
      nsrc.update({
          f: run_cmd_stdout([*ncmd.split(), f], cwd) for f in files * bool(ncmd)
      })
      return files, bases
  return [], bases


def func_local_violations(
    rel: str, fn: FunctionMetrics, cfg: GateConfig
) -> list[str]:
  """Checks per-function Dim 2 ratchets (Cognitive, Cyclomatic, Nesting)."""
  cog_lim = (
      cfg.max_cognitive_fail
      if fn.base_cognitive <= cfg.max_cognitive_fail
      else fn.base_cognitive + cfg.max_legacy_cog_delta_fail
  )
  cyc_lim = (
      cfg.max_cyclomatic_fail
      if fn.base_cyclomatic <= cfg.max_cyclomatic_fail
      else fn.base_cyclomatic + cfg.max_legacy_cyc_delta_fail
  )
  checks = (
      (
          fn.delta_cognitive > 0 and fn.cognitive > cog_lim,
          (
              f"Dim 2: {rel}:{fn.qualname} cognitive={fn.cognitive}"
              f" (base={fn.base_cognitive}, delta=+{fn.delta_cognitive})"
          ),
      ),
      (
          fn.delta_cyclomatic > 0 and fn.cyclomatic > cyc_lim,
          (
              f"Dim 2: {rel}:{fn.qualname} cyclomatic={fn.cyclomatic}"
              f" (delta=+{fn.delta_cyclomatic})"
          ),
      ),
      (
          fn.delta_nesting > 0 and fn.max_nesting > cfg.max_nesting_fail,
          (
              f"Dim 2: {rel}:{fn.qualname} nesting={fn.max_nesting}"
              f" > {cfg.max_nesting_fail}"
          ),
      ),
  )
  return [msg for bad, msg in checks if bad]


def resolve_skip_reason(cwd: pathlib.Path, rev: str | None = None) -> str:
  """Extracts a substantive `SKIP_CL_COMPLEXITY=<reason>` justification."""
  c_res = cwd.resolve()
  ancs = (c_res, *c_res.parents)
  vcs_cmds = [
      (".git", ["git", "log", "-1", "--format=%B", *([rev] if rev else [])]),
  ]
  sources = [
      os.environ.get("PR_BODY", ""),
      *(
          run_cmd_stdout(cmd, cwd)
          for marker, cmd in vcs_cmds
          if any((p / marker).exists() for p in ancs)
      ),
  ]
  matches = [m.group(1).strip() for s in sources if (m := _SKIP_RE.search(s))]
  raw = os.environ.get("SKIP_CL_COMPLEXITY", "").strip() or next(
      iter(matches), ""
  )
  is_valid = bool(re.search(r"[A-Za-z]{4,}", raw)) and raw.lower() not in {
      "true",
      "yes",
      "skip",
  }
  return raw if is_valid else ""


def is_misnamed_generator(fn: FunctionMetrics) -> bool:
  """Returns True when a new/renamed generator uses an eager stream verb."""
  base_raw = fn.matched_base_key.split(":")[-1].split(".")[-1].lstrip("_")
  return (
      fn.is_generator
      and fn.stripped_name.startswith(_EAGER_STREAM_VERBS)
      and not base_raw.startswith(_EAGER_STREAM_VERBS)
  )


def check_file_violations(
    rel: str,
    nfuncs: list[FunctionMetrics],
    bfuncs: list[FunctionMetrics],
    bmap_all: dict[str, FunctionMetrics],
    cfg: GateConfig,
) -> tuple[int, list[str], list[str], list[str]]:
  """Evaluates per-file Dim 2 & Dim 3 ratchets and returns violations."""
  dims = set(cfg.enabled_dimensions)
  eff_map = {f"{b.file_path}:{b.qualname}": b for b in bfuncs} | {
      f.matched_base_key: bmap_all[f.matched_base_key]
      for f in nfuncs
      if f.matched_base_key in bmap_all
  }
  eff_bfuncs = list(eff_map.values())
  bhops, _, _ = compute_call_graph_metrics(eff_bfuncs, cfg)
  hops, chain, triv = compute_call_graph_metrics(nfuncs, cfg)
  matched_quals = {bf.qualname for bf in eff_bfuncs} | {
      f.qualname for f in nfuncs if f.matched_base_key
  }
  added_triv = [q for q in triv if q not in matched_quals]
  fcog = sum([f.cognitive for f in nfuncs]) - sum(
      [bf.cognitive for bf in eff_bfuncs]
  )
  new_privs = [
      f.qualname for f in nfuncs if f.is_private and not f.matched_base_key
  ]
  misnamed = [f.qualname for f in nfuncs if is_misnamed_generator(f)]
  pt_ratio = sum(
      [f.is_pass_through and not f.matched_base_key for f in nfuncs]
  ) / max(len(nfuncs), 4)
  nclos = list(
      itertools.chain.from_iterable([f.nested_closures for f in nfuncs])
  )
  bclos = list(
      itertools.chain.from_iterable([bf.nested_closures for bf in eff_bfuncs])
  )
  checks = (
      (
          hops > cfg.max_call_depth_fail and hops > bhops,
          (
              f"Dim 3: {rel} max_call_depth_hops={hops} ({' -> '.join(chain)};"
              " preserve outer domain names and inline single-caller inner"
              " callees upward)"
          ),
      ),
      (
          len(added_triv) >= cfg.max_trivial_helpers_fail,
          (
              f"Dim 3: {rel} trivial_single_call_helpers={len(added_triv)}"
              f" ({added_triv}; preserve outer domain name when callee has 1"
              " caller, inline inner callee upward)"
          ),
      ),
      (
          pt_ratio > cfg.max_pass_through_ratio_fail,
          (
              f"Dim 3: {rel} pass_through_ratio={pt_ratio:.2f}"
              f" > {cfg.max_pass_through_ratio_fail}"
          ),
      ),
      (
          len(nclos) > cfg.max_nested_closures_fail and len(nclos) > len(bclos),
          (
              f"Dim 3: {rel} nested_closures={len(nclos)}"
              f" > {cfg.max_nested_closures_fail}"
          ),
      ),
      (
          bool(eff_bfuncs)
          and len(new_privs) >= cfg.max_sprawl_helpers_fail
          and fcog > cfg.max_file_cog_delta_sprawl,
          f"Dim 3: {rel} complexity_sprawl={len(new_privs)} (+{fcog} Cog)",
      ),
      (
          bool(misnamed),
          (
              f"Dim 3: {rel} semantic_verb_mismatch={misnamed} (lazy generator"
              " yields items; prefer stdlib iter(callable, sentinel) or"
              " _iter_*/_stream_* over eager drain/flush/collect verbs)"
          ),
      ),
  )
  dim3_v = [msg for failed, msg in checks if failed] * (3 in dims)
  dim2_v = list(
      itertools.chain.from_iterable(
          [func_local_violations(rel, fn, cfg) for fn in nfuncs]
      )
  ) * (2 in dims)
  return hops, chain, added_triv, dim3_v + dim2_v


def discover_ref_sources(
    cwd: pathlib.Path,
    prod_rels: list[str],
    excl: set[str] | frozenset[str] = frozenset(),
) -> dict[str, str]:
  """Discovers unmodified sibling/package `.py` files for clone detection."""
  c_res = cwd.resolve()
  skip = set(prod_rels) | excl
  roots = {
      (
          c_res / r[: r.index("tunix/") + 5]
          if "tunix/" in r
          else (c_res / r).parent
      )
      for r in prod_rels
  }
  cands = itertools.chain.from_iterable(
      [*rt.glob("*.py"), *rt.glob("*/*.py")]
      for rt in roots
      if rt.is_dir() and rt.is_relative_to(c_res)
  )
  selected = [
      (rel, p)
      for p in sorted(set(cands))
      if (rel := p.relative_to(c_res).as_posix()) not in skip
      and not rel.endswith("_test.py")
      and "/tests/" not in f"/{rel}"
      and "/testdata/" not in f"/{rel}"
  ][:60]
  if not selected:
    return {}
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    texts = pool.map(lambda item: item[1].read_text(encoding="utf-8"), selected)
    return dict(zip((rel for rel, _ in selected), texts, strict=True))


def partition_warnings(
    viols: Sequence[str],
    funcs: Sequence[FunctionMetrics],
    max_hops: int,
    score: float,
    cfg: GateConfig,
) -> tuple[list[str], list[str]]:
  """Splits bounded helper-sprawl and soft-scope advisories into `warnings`."""
  max_cog = max((f.cognitive for f in funcs), default=0)
  bounded = (
      max_cog <= cfg.max_cognitive_fail and max_hops <= cfg.max_call_depth_fail
  )
  kept = [v for v in viols if not (bounded and "complexity_sprawl=" in v)]
  warns = [v for v in viols if bounded and "complexity_sprawl=" in v]
  cap = (
      round(max(100.0, 1.2 * cfg.base_score), 1)
      if cfg.base_score >= 95.0
      else 100.0
  )
  if score > cap:
    warns.append(
        f"Dim 1: score={score} > {cap} (soft split signal; never"
        " statement-golf)"
    )
  return kept, warns


def evaluate_gate(
    cwd: pathlib.Path,
    files: list[str],
    bases: collections.defaultdict[str, str],
    cfg: GateConfig,
    ref_sources: dict[str, str] | None = None,
    new_sources: dict[str, str] | None = None,
) -> GateReport:
  """Evaluates active dimensions across changed Python files and baselines."""
  dims = set(cfg.enabled_dimensions)
  all_rels = sorted(set(files) | set(bases))
  disk_src = {
      r: (cwd / r).read_text(encoding="utf-8")
      for r in all_rels
      if (cwd / r).is_file()
  }
  nsrc = disk_src | (new_sources or {})
  raw_sources = {rel: (nsrc.get(rel, ""), bases[rel]) for rel in all_rels}
  test_rels = {
      r
      for r in raw_sources
      if re.search(r"(?:_test\.py$|/(?:tests|testdata)/)", f"/{r}")
  }
  test_loc = sum([
      max(
          0,
          len(count_code_loc(raw_sources[r][0]))
          - len(count_code_loc(raw_sources[r][1])),
      )
      for r in test_rels
  ])
  prod_rels = sorted(set(raw_sources) - test_rels)
  net_stmts = sum([
      count_ast_stmts(raw_sources[r][0]) - count_ast_stmts(raw_sources[r][1])
      for r in prod_rels
  ])
  base_by_file = collections.defaultdict(
      list,
      {
          r: analyze_functions(raw_sources[r][1], r)
          for r in prod_rels
          if raw_sources[r][1]
      },
  )
  new_by_file = {
      r: analyze_functions(raw_sources[r][0], r)
      for r in prod_rels
      if raw_sources[r][0]
  }
  match_baselines_tiered(base_by_file, new_by_file)
  funcs = list(itertools.chain.from_iterable(new_by_file.values()))
  bfuncs_all = list(itertools.chain.from_iterable(base_by_file.values()))
  bmap_all = {f"{bf.file_path}:{bf.qualname}": bf for bf in bfuncs_all}
  net_cog = sum([f.cognitive for f in funcs]) - sum(
      [bf.cognitive for bf in bfuncs_all]
  )
  per_file = [
      check_file_violations(r, nfuncs, base_by_file[r], bmap_all, cfg)
      for r, nfuncs in new_by_file.items()
  ]
  max_hops, max_chain = max(
      [(h, ch) for h, ch, _, _ in per_file], default=(0, [])
  )
  trivial_new = list(
      itertools.chain.from_iterable([tr for _, _, tr, _ in per_file])
  )
  refs = (
      discover_ref_sources(cwd, prod_rels * (4 in dims), set(nsrc))
      if ref_sources is None
      else ref_sources
  )
  ref_by_file = {
      p: analyze_functions(s, p)
      for p, s in refs.items()
      if p not in new_by_file
  }
  clones = detect_clones(new_by_file, base_by_file, ref_by_file, cfg) * (
      4 in dims
  )
  syntax_viols = [
      f"Dim 2: {r} syntax_error ({err})"
      for r in all_rels * (2 in dims)
      if (err := try_parse_ast(raw_sources[r][0])[1])
  ]
  viols = (
      syntax_viols
      + list(itertools.chain.from_iterable([fv for _, _, _, fv in per_file]))
      + (
          [f"Dim 4: duplicate_logic_clones={len(clones)} ({clones[:3]})"]
          * bool(clones)
      )
  )
  closures = list(
      itertools.chain.from_iterable([f.nested_closures for f in funcs])
  )
  bclos_all = list(
      itertools.chain.from_iterable([bf.nested_closures for bf in bfuncs_all])
  )
  added_clos = max(0, len(closures) - len(bclos_all))
  pass_through = [f.qualname for f in funcs if f.is_pass_through]
  added_pt = sum(not f.matched_base_key for f in funcs if f.is_pass_through)
  s1 = (1 in dims) * cfg.weight_prod_loc * max(0, net_stmts)
  s2 = (
      (2 in dims)
      * cfg.weight_cognitive_excess
      * sum([
          max(0, f.cognitive - cfg.max_cognitive_warn)
          for f in funcs
          if f.delta_cognitive > 0
      ])
  )
  s3 = (3 in dims) * (
      cfg.weight_call_depth_excess * max(0, max_hops - cfg.max_call_depth_warn)
      + cfg.weight_trivial_helper * len(trivial_new)
      + cfg.weight_pass_through_func * added_pt
      + cfg.weight_nested_closure
      * max(0, added_clos - cfg.max_nested_closures_warn)
  )
  s4 = (4 in dims) * cfg.weight_duplicate_clone * len(clones)
  score = round(s1 + s2 + s3 + s4, 1)
  viols, warns = partition_warnings(viols, funcs, max_hops, score, cfg)
  skip_reason = resolve_skip_reason(cwd)
  passed = not viols or bool(skip_reason)
  return GateReport(
      passed=passed,
      score=score,
      enabled_dimensions=list(cfg.enabled_dimensions),
      prod_loc_added=max(0, net_stmts),
      net_loc_delta=net_stmts,
      net_cog_delta=net_cog,
      test_loc_added=test_loc,
      prod_files_changed=len(new_by_file),
      max_cognitive=max([f.cognitive for f in funcs], default=0),
      max_cyclomatic=max([f.cyclomatic for f in funcs], default=0),
      max_nesting=max([f.max_nesting for f in funcs], default=0),
      max_call_depth_hops=max_hops,
      max_call_chain=max_chain,
      trivial_single_call_helpers=trivial_new,
      nested_closures=closures,
      pass_through_funcs=pass_through,
      duplicate_clones=clones,
      violations=viols,
      skip_reason=skip_reason,
      warnings=warns,
      duck_typing_calls=sum_duck_typing_delta(raw_sources, prod_rels),
  )


@dataclasses.dataclass(frozen=True)
class FileShardProfile:
  """Per-file AST metrics, imports, and public signatures for shard planning."""

  path: str
  stmts: int
  cognitive: int
  weight: int
  imports: tuple[tuple[int, str, tuple[str, ...]], ...]
  attr_refs: dict[str, tuple[str, ...]]
  public_signatures: dict[str, str]


def profile_shard_file(rel: str, source: str) -> FileShardProfile:
  """Profiles AST statements, cognitive weight, imports, and signatures."""
  tree, _ = try_parse_ast(source)
  if tree is None:
    return FileShardProfile(rel, 0, 0, 0, (), {}, {})
  stmts = count_ast_stmts(source)
  cog = sum(fn.cognitive for fn in analyze_functions(source, rel))
  imports: list[tuple[int, str, tuple[str, ...]]] = []
  aliases: list[tuple[str | None, str]] = []
  raw_attrs: dict[str, set[str]] = collections.defaultdict(set)
  for node in ast.walk(tree):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
      lvl = getattr(node, "level", 0)
      mod = getattr(node, "module", None)
      imports.append((lvl, mod or "", tuple(a.name for a in node.names)))
      aliases.extend((a.asname, a.name.split(".")[-1]) for a in node.names)
    elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
      keep = not node.attr.startswith("_")
      raw_attrs[node.value.id].update((node.attr,) * keep)
  for asc, orig in aliases:
    if asc:
      raw_attrs[orig].update(raw_attrs.get(asc, ()))
  attr_refs = {k: tuple(sorted(v)) for k, v in raw_attrs.items() if v}
  sigs = dict(filter(None, (format_public_signature(s) for s in tree.body)))
  return FileShardProfile(
      rel, stmts, cog, stmts + 2 * cog, tuple(imports), attr_refs, sigs
  )


def build_module_index(paths: Sequence[str]) -> dict[str, str]:
  """Maps unambiguous dotted module suffixes to candidate `.py` paths."""
  buckets: dict[str, set[str]] = collections.defaultdict(set)
  for rel in paths:
    stem = rel.removesuffix(".py")
    for cand in filter(None, {stem, stem.removesuffix("/__init__")}):
      parts = cand.split("/")
      for key in (".".join(parts[i:]) for i in range(len(parts))):
        buckets[key].add(rel)
  return {k: next(iter(v)) for k, v in buckets.items() if len(v) == 1}


def match_single_import(
    caller: str,
    imp: tuple[int, str, tuple[str, ...]],
    attr_refs: Mapping[str, Sequence[str]],
    mod_index: Mapping[str, str],
    known_paths: set[str],
) -> list[tuple[str, tuple[str, ...]]]:
  """Resolves one AST import tuple to `(callee_path, symbols)` pairs."""
  level, mod, names = imp
  if level == 0:
    direct = [(mod_index[mod], names)] if mod in mod_index else []
    dotted = [
        (mod_index[c], tuple(attr_refs.get(nm.split(".")[-1], ())))
        for nm in names
        if (c := ".".join(filter(None, (mod, nm)))) in mod_index
    ]
    return direct + dotted
  parents = list(pathlib.PurePosixPath(caller).parents)
  anc = parents[min(level - 1, len(parents) - 1)]
  raw_base = str(anc / mod.replace(".", "/")) if mod else str(anc)
  rel_dir = raw_base.removeprefix("./").removesuffix(".")
  hits = [
      (c, names)
      for c in (f"{rel_dir}.py", f"{rel_dir}/__init__.py")
      if rel_dir and c in known_paths
  ]
  sub_cands = [
      (nm, f"{rel_dir}/{nm}".lstrip("/") + sfx)
      for nm, sfx in itertools.product(names, (".py", "/__init__.py"))
  ]
  hits.extend(
      (c, tuple(attr_refs.get(nm, ())))
      for nm, c in sub_cands
      if c in known_paths
  )
  return hits


def resolve_import_edges(
    profiles: Sequence[FileShardProfile],
) -> list[tuple[str, str, tuple[str, ...]]]:
  """Extracts deduplicated cross-file `(caller, callee, symbols)` edges."""
  known = {p.path for p in profiles}
  by_path = {p.path: p for p in profiles}
  mod_index = build_module_index([p.path for p in profiles])
  merged: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
  for prof in profiles:
    found = itertools.chain.from_iterable(
        match_single_import(prof.path, imp, prof.attr_refs, mod_index, known)
        for imp in prof.imports
    )
    valid = [(c, s) for c, s in found if c != prof.path]
    for callee, syms in valid:
      merged[(prof.path, callee)].update(set(syms) - {"*"})
  edges: list[tuple[str, str, tuple[str, ...]]] = []
  for (caller, callee), syms in sorted(merged.items()):
    pub = by_path[callee].public_signatures
    resolved = sorted(syms & set(pub)) or sorted(pub)
    edges.append((caller, callee, tuple(resolved)))
  return edges


def connected_file_components(
    paths: Sequence[str],
    edges: Sequence[tuple[str, str, tuple[str, ...]]],
) -> list[list[str]]:
  """Groups `paths` into deterministic Union-Find connected components."""
  parent = {p: p for p in sorted(paths)}
  for src, dst, _ in edges:
    if src not in parent or dst not in parent:
      continue
    while parent[src] != src:
      parent[src] = parent[parent[src]]
      src = parent[src]
    while parent[dst] != dst:
      parent[dst] = parent[parent[dst]]
      dst = parent[dst]
    lo, hi = (src, dst) if src < dst else (dst, src)
    parent[hi] = lo
  groups: dict[str, list[str]] = collections.defaultdict(list)
  for p in sorted(parent):
    root = p
    while parent[root] != root:
      root = parent[root]
    groups[root].append(p)
  return sorted(groups.values(), key=lambda g: g[0])


def split_oversized_components(
    components: Sequence[list[str]],
    weights: Mapping[str, int],
    edges: Sequence[tuple[str, str, tuple[str, ...]]],
    max_shards: int,
) -> tuple[list[list[str]], list[str]]:
  """Cuts oversized connected components on highest-degree Hub modules."""
  eff = min(max(1, max_shards), max(1, sum(len(c) for c in components)))
  if eff <= 1:
    return [list(c) for c in components], []
  total_w = max(1, sum(weights.values()))
  caps = {2: total_w * 0.65}
  default_cap = (total_w / eff) * 1.25
  units, hubs_cut = [], []
  queue = collections.deque(list(c) for c in components)
  while queue:
    comp = queue.popleft()
    if len(comp) <= 1 or sum(weights[p] for p in comp) <= caps.get(
        len(comp), default_cap
    ):
      units.append(comp)
      continue
    cset = set(comp)
    in_e = [(s, d) for s, d, _ in edges if s in cset and d in cset]
    deg = collections.Counter(itertools.chain.from_iterable(in_e))
    indeg = collections.Counter(d for _, d in in_e)
    hub = max(comp, key=lambda p: (deg[p], indeg[p], weights[p], p))
    hubs_cut.append(hub)
    units.append([hub])
    rem = [p for p in comp if p != hub]
    sub_e = [e for e in edges if hub not in e[:2]]
    queue.extend(connected_file_components(rem, sub_e))
  return units, hubs_cut


def pack_units_lpt(
    units: Sequence[list[str]], weights: Mapping[str, int], max_shards: int
) -> list[list[str]]:
  """Packs file units into `<= max_shards` balanced bins via LPT."""
  if not units:
    return []
  k = min(max(1, max_shards), len(units))
  ordered = sorted(units, key=lambda u: (-sum(weights[p] for p in u), u[0]))
  bins: list[list[str]] = [[] for _ in range(k)]
  loads = [0] * k
  for unit in ordered:
    idx = min(range(k), key=lambda i: (loads[i], i))
    bins[idx].extend(unit)
    loads[idx] += sum(weights[p] for p in unit)
  return [sorted(b) for b in bins if b]


def collect_frozen_contracts(
    shards: Sequence[Sequence[str]],
    profiles: Mapping[str, FileShardProfile],
    edges: Sequence[tuple[str, str, tuple[str, ...]]],
) -> tuple[list[dict[str, object]], list[str], dict[int, list[str]]]:
  """Emits cross-shard edge metadata and 1-line `frozen_contracts` strings."""
  shard_of: dict[str, int] = {}
  for idx, s in enumerate(shards):
    shard_of.update(dict.fromkeys(s, idx))
  cross = [
      (c, d, syms, shard_of[c], shard_of[d])
      for c, d, syms in edges
      if shard_of.get(c, -1) != shard_of.get(d, -1)
  ]
  cross_edges: list[dict[str, object]] = []
  callers_by_sig: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
  shards_by_sig: dict[tuple[str, str], set[int]] = collections.defaultdict(set)
  for caller, callee, syms, cs, ds in cross:
    cross_edges.append({
        "caller": caller,
        "callee": callee,
        "caller_shard": cs,
        "callee_shard": ds,
        "symbols": list(syms),
    })
    pub = profiles[callee].public_signatures
    for sig in [pub[s] for s in syms if s in pub] or [f"module {callee}"]:
      callers_by_sig[(callee, sig)].add(caller)
      shards_by_sig[(callee, sig)].update((cs, ds))
  contracts: list[str] = []
  per_shard: dict[int, list[str]] = collections.defaultdict(list)
  for (callee, sig), callers in sorted(callers_by_sig.items()):
    line = f"{callee}: {sig}  # callers: {', '.join(sorted(callers))}"
    contracts.append(line)
    for sid in sorted(shards_by_sig[(callee, sig)]):
      per_shard[sid].append(line)
  return cross_edges, contracts, per_shard


def build_shard_plan(
    cwd: pathlib.Path,
    files: Sequence[str],
    max_shards: int = 4,
    new_sources: Mapping[str, str] | None = None,
) -> dict[str, object]:
  """Builds a parallel AST dependency & LPT shard plan (`<= max_shards`)."""
  nsrc, td_re = new_sources or {}, re.compile(r"/(testdata|test_data)/")
  raw_items = [read_candidate_source(cwd, r, nsrc) for r in sorted(set(files))]
  items = [
      (r, s)
      for r, s in raw_items
      if (s or r in nsrc) and not td_re.search(f"/{r}")
  ]
  with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    prof_list = list(pool.map(lambda it: profile_shard_file(*it), items))
  profs = {p.path: p for p in prof_list}
  weights = {p.path: p.weight for p in prof_list}
  edges = resolve_import_edges(prof_list)
  comps = connected_file_components(list(profs), edges)
  units, hubs = split_oversized_components(comps, weights, edges, max_shards)
  packed = pack_units_lpt(units, weights, max_shards)
  cross, contracts, per_s = collect_frozen_contracts(packed, profs, edges)
  shards = [
      {
          "shard_id": i,
          "files": fs,
          "frozen_contracts": per_s.get(i, []),
          "stmts": sum(profs[f].stmts for f in fs),
          "cognitive": sum(profs[f].cognitive for f in fs),
          "weight": sum(profs[f].weight for f in fs),
      }
      for i, fs in enumerate(packed)
  ]
  return {
      "max_shards": max(1, max_shards),
      "total_files": len(prof_list),
      "total_weight": sum(weights.values()),
      "connected_components": comps,
      "hub_modules_cut": hubs,
      "shards": shards,
      "cross_shard_edges": cross,
      "frozen_contracts": contracts,
  }


def format_report(r: GateReport) -> str:
  """Formats the human-readable CL Complexity Gate summary."""
  status = "PASS" if r.passed else "FAIL"
  lines = [
      f"=== CL Complexity Gate: {status} (Score: {r.score}/100) ===",
      (
          f"  Scope: net_stmts={r.net_loc_delta:+d},"
          f" net_cog={r.net_cog_delta:+d} | Local: cog={r.max_cognitive},"
          f" cyc={r.max_cyclomatic} | Layer: hops={r.max_call_depth_hops} |"
          f" Clones: {len(r.duplicate_clones)}"
      ),
      *[f"  [WARNING] {w}" for w in r.warnings],
      *[f"  [VIOLATION] {v}" for v in r.violations],
      *(
          [f"  [BYPASSED] SKIP_CL_COMPLEXITY={r.skip_reason}"]
          if r.skip_reason and r.violations
          else []
      ),
      *(
          ["  Use SKIP_CL_COMPLEXITY=<reason> in CL/PR description to skip."]
          if not r.passed
          else []
      ),
  ]
  return "\n".join(lines)


def emit_output(rendered: str, out_path: pathlib.Path | None) -> None:
  """Prints `rendered` to stdout and optionally writes it to `out_path`."""
  if out_path is not None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered + "\n", encoding="utf-8")
  print(rendered)


def build_arg_parser() -> argparse.ArgumentParser:
  """Constructs the CLI `ArgumentParser` with usage examples and schema."""
  p = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("--cwd", default=".", help="Workspace root directory.")
  p.add_argument("--rev", help="Base VCS revision or baseline file path.")
  p.add_argument("--base", dest="rev", help="Alias for --rev.")
  p.add_argument("--files", nargs="*", help="Explicit .py files to analyze.")
  p.add_argument("--dimensions", nargs="+", type=int, help="Dimensions 1..4.")
  p.add_argument(
      "--config", default=_DEFAULT_CONFIG, type=pathlib.Path, help="TOML path."
  )
  p.add_argument("--json", action="store_true", help="Emit GateReport JSON.")
  p.add_argument(
      "--strict", action="store_true", help="Exit 1 on gate/hygiene violations."
  )
  p.add_argument(
      "--hygiene", action="store_true", help="Enforce full-file G1/G2 hygiene."
  )
  p.add_argument(
      "--check-test-coverage",
      action="store_true",
      help="Require unit tests with 100%% fn and >=90%% stmt coverage.",
  )
  p.add_argument(
      "--shard-plan", action="store_true", help="Emit LPT AST shard plan JSON."
  )
  p.add_argument(
      "--max-shards", type=int, default=4, help="Max shards (default: 4)."
  )
  p.add_argument(
      "--base-score",
      type=float,
      default=0.0,
      help="Baseline CL score for soft-scope advisory scaling.",
  )
  p.add_argument("--out", type=pathlib.Path, help="Optional output file path.")
  return p


def main(argv: list[str] | None = None) -> int:
  """CLI entrypoint for the Tunix CL Understandability & Complexity Gate."""
  args = build_arg_parser().parse_args(argv)
  cwd, new_sources = pathlib.Path(args.cwd), {}
  files, bases = resolve_changed_files(cwd, args.rev, args.files, new_sources)
  if args.shard_plan:
    plan = build_shard_plan(cwd, files, args.max_shards, new_sources)
    emit_output(json.dumps(plan, indent=2), args.out)
    return 0
  cfg = load_config(args.config, cwd)
  cfg.enabled_dimensions = tuple(args.dimensions or cfg.enabled_dimensions)
  cfg.base_score = args.base_score
  r = evaluate_gate(cwd, files, bases, cfg, new_sources=new_sources)
  hyg = args.hygiene if args.dimensions else (args.strict or args.hygiene)
  hyg_v = (
      evaluate_hygiene(cwd, files, bases, new_sources, not args.hygiene)
      if hyg
      else []
  )
  cov_v = evaluate_test_coverage(cwd, files) if args.check_test_coverage else []
  r.violations.extend(hyg_v + cov_v)
  r.passed = r.passed and not (hyg_v or cov_v)
  rendered = (
      json.dumps(dataclasses.asdict(r), indent=2)
      if args.json
      else format_report(r)
  )
  emit_output(rendered, args.out)
  enforce = args.strict or args.hygiene or args.check_test_coverage
  return 1 * (enforce and not r.passed)


if __name__ == "__main__":
  sys.exit(main())
