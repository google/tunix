#!/usr/bin/env python3
"""Execute the installed P57 runner witness helper against bounded inputs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import stat
import types

import numpy as np


def _load_helpers(runner_path: Path, output_dir: Path) -> dict:
  tree = ast.parse(runner_path.read_text(encoding="utf-8"), filename=str(runner_path))
  wanted = {"_p38_token_history_sha256", "_p57_tito_runner_prompt_witness"}
  functions = [
      node
      for node in tree.body
      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
      and node.name in wanted
  ]
  if {node.name for node in functions} != wanted:
    raise AssertionError("installed runner lacks P57 witness helper dependencies")
  namespace = {
      "np": np,
      "_P57_TITO_RUNNER_WITNESS_DIR": str(output_dir),
      "_P57_TITO_RUNNER_WITNESS_STATE": {"records": 0},
      "_P57_TITO_RUNNER_WITNESS_MAX_RECORDS": 8192,
  }
  code = compile(ast.Module(body=functions, type_ignores=[]), str(runner_path), "exec")
  exec(code, namespace)  # pylint: disable=exec-used
  return namespace


def _fixtures():
  input_batch = types.SimpleNamespace(
      num_prompt_logprobs={"score": 1},
      req_id_to_index={"a0": 0, "a1": 1, "score": 2},
      num_prompt_tokens=np.asarray([3, 2, 2], dtype=np.int32),
      token_ids_cpu=np.asarray(
          [[101, 102, 103, 0], [201, 202, 0, 0], [9, 9, 0, 0]],
          dtype=np.int32,
      ),
  )
  runner = types.SimpleNamespace(input_batch=input_batch)
  scheduler = types.SimpleNamespace(
      scheduled_new_reqs=[
          types.SimpleNamespace(req_id="a0"),
          types.SimpleNamespace(req_id="score"),
          types.SimpleNamespace(req_id="a1"),
      ]
  )
  return runner, scheduler


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--overlay", type=Path, required=True)
  args = parser.parse_args()
  runner_path = args.overlay / "tpu_runner_p21_l30.py"
  output_dir = args.overlay / "p57-tito-runner-witness-test"
  output_dir.mkdir(mode=0o700)
  namespace = _load_helpers(runner_path, output_dir)
  witness = namespace["_p57_tito_runner_prompt_witness"]
  runner, scheduler = _fixtures()

  witness(runner, scheduler, {0: ["a0", "score"], 1: ["a1"]})
  paths = sorted(output_dir.glob("runner-input-*.json"))
  if len(paths) != 2:
    raise AssertionError(f"expected two A-arm runner witnesses, got {len(paths)}")
  records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
  if {record["request_id"] for record in records} != {"a0", "a1"}:
    raise AssertionError("runner witness included B or lost an A request")
  if any(stat.S_IMODE(path.stat().st_mode) != 0o600 for path in paths):
    raise AssertionError("runner witness is not mode 0600")
  expected = {
      "a0": hashlib.sha256(np.asarray([101, 102, 103], dtype="<i8").tobytes()).hexdigest(),
      "a1": hashlib.sha256(np.asarray([201, 202], dtype="<i8").tobytes()).hexdigest(),
  }
  if any(record["prompt_sha256"] != expected[record["request_id"]] for record in records):
    raise AssertionError("runner witness did not hash input_batch.token_ids_cpu")

  runner, scheduler = _fixtures()
  try:
    witness(runner, scheduler, {0: ["a0"], 1: ["a0", "a1"]})
  except RuntimeError as error:
    if "two DP ranks" not in str(error):
      raise
  else:
    raise AssertionError("duplicate DP mapping was not rejected")

  runner, scheduler = _fixtures()
  try:
    witness(runner, scheduler, {0: ["a0"]})
  except RuntimeError as error:
    if "no DP-rank mapping" not in str(error):
      raise
  else:
    raise AssertionError("missing DP mapping was not rejected")

  namespace["_P57_TITO_RUNNER_WITNESS_STATE"]["records"] = namespace[
      "_P57_TITO_RUNNER_WITNESS_MAX_RECORDS"
  ]
  runner, scheduler = _fixtures()
  scheduler.scheduled_new_reqs = [types.SimpleNamespace(req_id="a0")]
  try:
    witness(runner, scheduler, {0: ["a0"]})
  except RuntimeError as error:
    if "record bound exceeded" not in str(error):
      raise
  else:
    raise AssertionError("runner witness cap was not enforced")

  print(
      "P57_TITO_RUNNER_WITNESS_OVERLAY_PASS "
      "a_requests=2 b_excluded=1 negatives=3 mode0600=1"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
