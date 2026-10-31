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
"""3D CL Understandability & Complexity Gate for Tunix."""

from __future__ import annotations

import argparse
import ast
import collections
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
import types

_DEFAULT_CONFIG = pathlib.Path("pyproject.toml")
_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_FnNode = ast.FunctionDef | ast.AsyncFunctionDef
_LoopNode = ast.For | ast.AsyncFor | ast.While
_CompNode = ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp
_SKIP_TOKS = frozenset({
    tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
    tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER,
})
GateConfig = GateReport = FunctionMetrics = types.SimpleNamespace


def load_config(config_path: pathlib.Path = _DEFAULT_CONFIG) -> GateConfig:
  """Loads `[tool.cl_complexity_gate]` thresholds and weights from TOML."""
  raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
  cfg = raw.get("tool", {}).get("cl_complexity_gate", raw)
  weights = {f"weight_{k}": v for k, v in cfg["weights"].items()}
  return GateConfig(**(dict(cfg["thresholds"]) | weights))


def count_code_loc(source: str) -> set[int]:
  """Returns 1-based line numbers containing executable code."""
  if not source.strip():
    return set()
  doc_lines = {
      n.body[0].lineno
      for n in ast.walk(ast.parse(source))
      if isinstance(n, (ast.Module, ast.ClassDef, *_FUNC_TYPES))
      and ast.get_docstring(n, clean=False) is not None
  }
  toks = [
      range(t.start[0], t.end[0] + 1)
      for t in tokenize.generate_tokens(io.StringIO(source).readline)
      if t.type not in _SKIP_TOKS and t.start[0] not in doc_lines
  ]
  return set(itertools.chain.from_iterable(toks))


class _ComplexityVisitor(ast.NodeVisitor):
  """Computes Cognitive, Cyclomatic, Nesting, and Call metrics on an AST."""

  def __init__(self, fn: ast.stmt, qualname: str, body: list[ast.stmt]):
    self.func_name, self.qualname = qualname.split(".")[-1], qualname
    self.cognitive, self.cyclomatic, self.nesting, self.max_nesting = 0, 1, 0, 0
    self.closures, self.calls = [], []
    self.visit_block(body, 0)
    self.pass_through = is_pass_through(fn, body) and self.cognitive == 0

  def _bump(self) -> None:
    self.cognitive += 1 + self.nesting
    self.cyclomatic += 1
    self.max_nesting = max(self.max_nesting, self.nesting + 1)

  def visit_block(self, nodes: collections.abc.Sequence[ast.AST], nest_delta: int = 1) -> None:  # pylint: disable=line-too-long
    self.nesting += nest_delta
    for node in nodes:
      self.visit(node)
    self.nesting -= nest_delta

  def visit_If(self, node: ast.If) -> None:  # pylint: disable=invalid-name
    """Visits an `ast.If` node and its `elif`/`else` chain."""
    self._bump()
    self.visit(node.test)
    self.visit_block(node.body, 1)
    curr = node.orelse
    while len(curr) == 1 and isinstance(curr[0], ast.If):
      self.cognitive, self.cyclomatic = self.cognitive + 1, self.cyclomatic + 1
      self.visit(curr[0].test)
      self.visit_block(curr[0].body, 1)
      curr = curr[0].orelse
    if curr:
      self.cognitive += 1
      self.visit_block(curr, 1)

  def visit_For(self, node: _LoopNode) -> None:  # pylint: disable=invalid-name
    self._bump()
    self.visit_block(node.body, 1)
    if node.orelse:
      self.cognitive += 1
      self.visit_block(node.orelse, 1)

  visit_AsyncFor = visit_While = visit_For  # pylint: disable=invalid-name

  def visit_Try(self, node: ast.Try | ast.TryStar) -> None:  # pylint: disable=invalid-name
    self.visit_block(node.body, 0)
    for handler in node.handlers:
      self._bump()
      self.visit_block(handler.body, 1)
    self.visit_block(node.orelse + node.finalbody, 0)

  visit_TryStar = visit_Try  # pylint: disable=invalid-name

  def visit_IfExp(self, node: ast.IfExp) -> None:  # pylint: disable=invalid-name
    self._bump()
    self.visit_block(list(ast.iter_child_nodes(node)), 1)

  def visit_BoolOp(self, node: ast.BoolOp) -> None:  # pylint: disable=invalid-name
    self.cognitive += 1
    self.cyclomatic += len(node.values) - 1
    for val in node.values:
      if not (isinstance(val, ast.BoolOp) and isinstance(val.op, type(node.op))):  # pylint: disable=line-too-long
        self.visit(val)

  def visit_ListComp(self, node: _CompNode) -> None:  # pylint: disable=invalid-name
    for gen in node.generators:
      self.cyclomatic, self.cognitive = self.cyclomatic + len(gen.ifs), self.cognitive + len(gen.ifs)  # pylint: disable=line-too-long
    self.visit_block(list(ast.iter_child_nodes(node)), 1)

  visit_SetComp = visit_GeneratorExp = visit_DictComp = visit_ListComp  # pylint: disable=invalid-name

  def visit_Call(self, node: ast.Call) -> None:  # pylint: disable=invalid-name
    for expr in (node.func, *node.args[: "partial" in ast.unparse(node.func)]):
      if callee := _extract_callee_name(expr, self.qualname):
        self.calls.append(callee)
        self.cognitive += callee in (self.func_name, self.qualname)
    self.generic_visit(node)

  def visit_FunctionDef(self, node: _FnNode | ast.Lambda) -> None:  # pylint: disable=invalid-name
    self._bump()
    self.closures.append(f"{self.qualname}.{getattr(node, 'name', '<lambda>')}")
    self.visit_block([node.body] if isinstance(node, ast.Lambda) else node.body, 1)  # pylint: disable=line-too-long

  visit_AsyncFunctionDef = visit_Lambda = visit_FunctionDef  # pylint: disable=invalid-name


def _extract_callee_name(func_expr: ast.expr, qualname: str) -> str | None:
  """Extracts intra-file function or `self.method` callee name from Call AST."""
  if isinstance(func_expr, ast.Name):
    return func_expr.id
  if isinstance(func_expr, ast.Attribute) and isinstance(func_expr.value, ast.Name):  # pylint: disable=line-too-long
    if func_expr.value.id in ("self", "cls"):
      return f"{qualname.split('.')[0]}.{func_expr.attr}"
  return None


def is_pass_through(node: ast.stmt, stmts: list[ast.stmt]) -> bool:
  """Returns True if a function has 1 return/expr statement forwarding params."""
  if not isinstance(node, _FUNC_TYPES):
    return True
  if len(stmts) != 1 or not isinstance(stmts[0], (ast.Return, ast.Expr)):
    return False
  if not isinstance(call := stmts[0].value, ast.Call):
    return False
  all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
  params = {a.arg for a in all_args} - {"self", "cls"}
  forwarded = {a.id for a in call.args if isinstance(a, ast.Name)} | {
      kw.value.id for kw in call.keywords if isinstance(kw.value, ast.Name)
  }
  return bool(params and (forwarded & params))


def analyze_functions(source: str, base_cognitives: collections.Counter[str]) -> list[FunctionMetrics]:  # pylint: disable=line-too-long
  """Extracts function metrics for all top-level functions and class methods."""
  if not source.strip():
    return []
  tree = ast.parse(source)
  nodes = [("", n) for n in tree.body]
  for cls in [c for c in tree.body if isinstance(c, ast.ClassDef)]:
    nodes += [(f"{cls.name}.", m) for m in cls.body]
  defs: list[tuple[str, ast.stmt, list[ast.stmt]]] = [
      (f"{p}{n.name}", n, list(n.body[bool(ast.get_docstring(n, clean=False)):]))  # pylint: disable=line-too-long
      for p, n in nodes if isinstance(n, _FUNC_TYPES)
  ]
  for p, n in [(p, n) for p, n in nodes if isinstance(n, (ast.Assign, ast.AnnAssign))]:  # pylint: disable=line-too-long
    if any(isinstance(x, ast.Lambda) for x in ast.walk(n)) or "partial(" in ast.unparse(n):  # pylint: disable=line-too-long
      defs.append((f"{p}{ast.unparse(n).split('=')[0].split(':')[0].strip()}", n, [n]))  # pylint: disable=line-too-long
  results: list[FunctionMetrics] = []
  for qual, fn, body in defs:
    v = _ComplexityVisitor(fn, qual, body)
    name = qual.split(".")[-1]
    priv = name.startswith("_") and not name.startswith("__")
    results.append(FunctionMetrics(
        qualname=qual, stmts=len(body), cognitive=v.cognitive,
        delta_cognitive=v.cognitive - base_cognitives[qual],
        cyclomatic=v.cyclomatic, max_nesting=v.max_nesting, is_private=priv,
        nested_closures=tuple(v.closures), is_pass_through=v.pass_through,
        calls=tuple(v.calls),
    ))
  return results


def compute_call_graph_metrics(funcs: list[FunctionMetrics], cfg: GateConfig) -> tuple[int, list[str], list[str]]:  # pylint: disable=line-too-long
  """Computes max intra-file call depth (hops) and trivial 1-call helpers."""
  name_map = {f.qualname.split(".")[-1]: f.qualname for f in funcs} | {f.qualname: f.qualname for f in funcs}  # pylint: disable=line-too-long
  adj: dict[str, set[str]] = {f.qualname: set() for f in funcs}
  call_sites: collections.Counter[str] = collections.Counter()
  for f in funcs:
    for raw in [c for c in f.calls if c in name_map]:
      if (target := name_map[raw]) != f.qualname:
        adj[f.qualname].add(target)
        call_sites[target] += 1

  best_chain, frontier = [], [[node] for node in adj]
  while frontier:
    best_chain = frontier[0]
    frontier = list(itertools.chain.from_iterable((p + [nxt] for nxt in adj[p[-1]] - set(p)) for p in frontier))  # pylint: disable=line-too-long
  trivial = [
      f.qualname for f in funcs
      if f.is_private and call_sites[f.qualname] == 1
      and f.stmts <= cfg.trivial_helper_max_stmts and f.cognitive <= cfg.trivial_helper_max_cognitive  # pylint: disable=line-too-long
  ]
  return max(0, len(best_chain) - 1), best_chain, trivial


def _run_cmd_stdout(cmd: list[str], cwd: pathlib.Path) -> str:
  """Runs `cmd` in `cwd` and returns stdout on exit code 0, else empty str."""
  res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)  # pylint: disable=line-too-long
  return res.stdout if res.returncode == 0 else ""


def resolve_changed_files(cwd: pathlib.Path, rev: str | None, explicit_files: list[str] | None) -> tuple[list[str], collections.defaultdict[str, str]]:  # pylint: disable=line-too-long
  """Resolves changed `.py` files and their baseline source contents."""
  bases: collections.defaultdict[str, str] = collections.defaultdict(str)
  if explicit_files:
    return explicit_files, bases
  git_base = rev or "HEAD"
  git_out = _run_cmd_stdout(["git", "diff", "--name-only", "--diff-filter=AM", git_base], cwd)  # pylint: disable=line-too-long
  files = re.findall(r"(?m)^(\S+\.py)$", git_out)
  bases.update({f: _run_cmd_stdout(["git", "show", f"{git_base}:{f}"], cwd) for f in files})  # pylint: disable=line-too-long
  return files, bases


def collect_violations(
    prod_loc: int, prod_files: int, funcs: list[FunctionMetrics],
    max_hops: int, max_chain: list[str], trivial: list[str],
    closures: list[str], pt_ratio: float, cfg: GateConfig,
) -> list[str]:
  """Evaluates hard-fail thresholds across all 3 dimensions."""
  checks = [
      (prod_loc > cfg.max_prod_loc_fail, f"Dim 1: prod_loc_added={prod_loc}"),
      (prod_files > cfg.max_prod_files_fail, f"Dim 1: prod_files={prod_files}"),
      (max_hops > cfg.max_call_depth_fail,
       f"Dim 3: max_call_depth_hops={max_hops} ({' -> '.join(max_chain)})"),
      (len(trivial) >= cfg.max_trivial_helpers_fail,
       f"Dim 3: trivial_single_call_helpers={len(trivial)}"),
      (len(closures) > cfg.max_nested_closures_fail,
       f"Dim 3: nested_closures_count={len(closures)}"),
      (len(funcs) >= 4 and pt_ratio > cfg.max_pass_through_ratio_fail,
       f"Dim 3: pass_through_ratio={pt_ratio:.0%}"),
  ]
  violations = [msg for failed, msg in checks if failed]
  for fn in [f for f in funcs if f.delta_cognitive > 0]:
    for val, lim, label in (
        (fn.cognitive, cfg.max_cognitive_fail, "cognitive"),
        (fn.cyclomatic, cfg.max_cyclomatic_fail, "cyclomatic"),
        (fn.max_nesting, cfg.max_nesting_fail, "nesting"),
    ):
      if val > lim:
        violations.append(f"Dim 2: {fn.qualname} {label}={val} > {lim}")
  return violations


def evaluate_gate(cwd: pathlib.Path, files: list[str], bases: collections.defaultdict[str, str], cfg: GateConfig) -> GateReport:  # pylint: disable=line-too-long
  """Evaluates all 3 dimensions across changed Python files."""
  prod_files, prod_loc, test_loc, max_hops = 0, 0, 0, 0
  funcs, max_chain, trivial_all = [], [], []
  for rel in files:
    if not (path := cwd / rel).is_file():
      continue
    src, base_src = path.read_text(encoding="utf-8"), bases[rel]
    added_loc = max(0, len(count_code_loc(src)) - len(count_code_loc(base_src)))
    if rel.endswith("_test.py") or "/tests/" in f"/{rel}":
      test_loc += added_loc
      continue
    prod_files, prod_loc = prod_files + 1, prod_loc + added_loc
    base_cogs = collections.Counter({f.qualname: f.cognitive for f in analyze_functions(base_src, collections.Counter())})  # pylint: disable=line-too-long
    file_funcs = analyze_functions(src, base_cogs)
    hops, chain, trivial = compute_call_graph_metrics(file_funcs, cfg)
    if hops > max_hops:
      max_hops, max_chain = hops, chain
    trivial_all.extend(trivial)
    funcs.extend(file_funcs)

  closures = list(itertools.chain.from_iterable(f.nested_closures for f in funcs))  # pylint: disable=line-too-long
  pass_through = [f.qualname for f in funcs if f.is_pass_through]
  pt_ratio = len(pass_through) / len(funcs) if funcs else 0.0
  cog_ex = sum(max(0, f.cognitive - cfg.max_cognitive_warn) for f in funcs if f.delta_cognitive > 0)  # pylint: disable=line-too-long
  score = round(
      cfg.weight_prod_loc * prod_loc + cfg.weight_cognitive_excess * cog_ex
      + cfg.weight_call_depth_excess * max(0, max_hops - cfg.max_call_depth_warn)  # pylint: disable=line-too-long
      + cfg.weight_trivial_helper * len(trivial_all) + cfg.weight_pass_through_func * len(pass_through)  # pylint: disable=line-too-long
      + cfg.weight_nested_closure * len(closures), 1,
  )
  viols = collect_violations(prod_loc, prod_files, funcs, max_hops, max_chain, trivial_all, closures, pt_ratio, cfg)  # pylint: disable=line-too-long
  passed = not viols or bool(os.environ.get("SKIP_CL_COMPLEXITY"))
  return GateReport(
      passed=passed, score=score, prod_loc_added=prod_loc, test_loc_added=test_loc,  # pylint: disable=line-too-long
      prod_files_changed=prod_files, max_cognitive=max((f.cognitive for f in funcs), default=0),  # pylint: disable=line-too-long
      max_cyclomatic=max((f.cyclomatic for f in funcs), default=0),
      max_nesting=max((f.max_nesting for f in funcs), default=0),
      max_call_depth_hops=max_hops, max_call_chain=max_chain,
      trivial_single_call_helpers=trivial_all, nested_closures=closures,
      pass_through_funcs=pass_through, violations=viols,
  )


def format_report(r: GateReport) -> str:
  """Formats the human-readable 3D Complexity Gate summary."""
  st = "PASS" if r.passed else "FAIL"
  lines = [
      f"=== 3D CL Complexity Gate: {st} (Score: {r.score}/100) ===",
      f"  Dim 1: prod_loc={r.prod_loc_added}, files={r.prod_files_changed}",
      f"  Dim 2: cog={r.max_cognitive}, cyc={r.max_cyclomatic}",
      f"  Dim 3: hops={r.max_call_depth_hops}, nest={r.max_nesting}",
      *[f"  [VIOLATION] {v}" for v in r.violations],
  ]
  if not r.passed:
    lines.append("  Use SKIP_CL_COMPLEXITY=<reason> in CL/PR description to skip.")  # pylint: disable=line-too-long
  return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
  """CLI entrypoint for the 3D CL Understandability & Complexity Gate."""
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--cwd", default=".", help="Repository root directory.")
  p.add_argument("--rev", default=None, help="VCS revision to inspect.")
  p.add_argument("--files", nargs="*", default=None, help="File paths.")
  p.add_argument("--config", default=_DEFAULT_CONFIG, type=pathlib.Path)
  p.add_argument("--json", action="store_true", help="Output JSON report.")
  p.add_argument("--strict", action="store_true", help="Exit 1 on failure.")
  args = p.parse_args(argv)
  cfg, cwd = load_config(args.config), pathlib.Path(args.cwd)
  files, bases = resolve_changed_files(cwd, args.rev, args.files)
  r = evaluate_gate(cwd, files, bases, cfg)
  print(json.dumps(vars(r), indent=2) if args.json else format_report(r))
  return 1 if (args.strict and not r.passed) else 0


if __name__ == "__main__":
  sys.exit(main())
