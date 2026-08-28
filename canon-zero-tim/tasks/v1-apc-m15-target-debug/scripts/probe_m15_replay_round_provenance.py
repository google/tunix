#!/usr/bin/env python3
"""Verify that the installed M15 replay producer binds every row to a round."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys


class ReplayRoundProvenanceError(RuntimeError):
  """The installed replay-envelope producer does not bind its round."""


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ReplayRoundProvenanceError(message)


def verify_runner_source(path: Path) -> None:
  tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
  functions = [
      node for node in tree.body
      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
      and node.name == "_p38_m15_replay_ledger"
  ]
  _require(len(functions) == 1, "expected one M15 replay producer")

  records = []
  for node in ast.walk(functions[0]):
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
      continue
    target = node.targets[0]
    if not isinstance(target, ast.Name) or target.id != "record":
      continue
    if not isinstance(node.value, ast.Dict):
      continue
    values = {
        key.value: value
        for key, value in zip(node.value.keys, node.value.values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    schema = values.get("schema")
    if isinstance(schema, ast.Constant) and schema.value == "m15-apc-serving-envelope-v1":
      records.append(values)
  _require(len(records) == 1, "expected one M15 replay-envelope record")

  diagnostic_round = records[0].get("diagnostic_round")
  _require(isinstance(diagnostic_round, ast.Call),
           "M15 replay envelope diagnostic_round is absent or not live-bound")
  call = diagnostic_round
  _require(
      isinstance(call.func, ast.Name)
      and call.func.id == "int"
      and len(call.args) == 1
      and not call.keywords,
      "M15 replay envelope diagnostic_round must be int(_p38_seam_round())",
  )
  round_call = call.args[0]
  _require(
      isinstance(round_call, ast.Call)
      and isinstance(round_call.func, ast.Name)
      and round_call.func.id == "_p38_seam_round"
      and not round_call.args
      and not round_call.keywords,
      "M15 replay envelope diagnostic_round must read the live round file",
  )


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("runner", type=Path)
  args = parser.parse_args()
  try:
    verify_runner_source(args.runner)
  except (OSError, SyntaxError, ReplayRoundProvenanceError) as exc:
    print(f"M15_REPLAY_ROUND_PROVENANCE_RED reason={exc}", file=sys.stderr)
    return 2
  print(
      "M15_REPLAY_ROUND_PROVENANCE_PASS "
      "schema=m15-apc-serving-envelope-v1 binding=int(_p38_seam_round())"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
