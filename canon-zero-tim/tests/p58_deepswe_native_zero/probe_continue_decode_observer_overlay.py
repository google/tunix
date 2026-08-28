"""Fail-closed contract probe for the installed P58 mixed-path observer."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import os
from pathlib import Path


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
  matches = [
      node
      for node in tree.body
      if isinstance(node, ast.FunctionDef) and node.name == name
  ]
  if len(matches) != 1:
    raise AssertionError(f"expected one {name}, found {len(matches)}")
  return matches[0]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--runner", type=Path, required=True)
  args = parser.parse_args()

  source = args.runner.read_text(encoding="utf-8")
  tree = ast.parse(source, filename=str(args.runner))
  predicate = _function(tree, "_p38_p58_continue_program_path")
  begin = _function(tree, "_p38_serving_begin")
  round_budget = _function(tree, "_p38_begin_observer_round")

  namespace = {}
  exec(
      compile(
          ast.Module(body=[predicate], type_ignores=[]),
          filename=str(args.runner),
          mode="exec",
      ),
      namespace,
  )
  check = namespace["_p38_p58_continue_program_path"]
  cases = (
      ("p58-seam-v1", "standard", "continue_decode", True),
      ("other-profile", "standard", "continue_decode", False),
      ("p58-seam-v1", "continue_decode", "continue_decode", False),
      ("p58-seam-v1", "standard", "standard", False),
      ("p58-seam-v1", "standard", "unknown", False),
  )
  for profile, expected, actual, want in cases:
    namespace["_P38_DURABILITY_PROFILE"] = profile
    namespace["_P38_SERVING_CAPTURE_EXPECTED_PATH"] = expected
    got = check(actual)
    if got is not want:
      raise AssertionError(
          f"predicate drift: {(profile, expected, actual)} got={got} want={want}"
      )

  begin_source = ast.get_source_segment(source, begin)
  if begin_source is None:
    raise AssertionError("could not recover _p38_serving_begin source")
  required = (
      "not m15_continue_path and not p58_continue_path",
      "[CANON_P58_CONTINUE_DECODE_OBSERVER_BYPASS]",
      "actual=continue_decode tensor_capture=0",
      "if p58_continue_path:",
      "return None",
      'candidate_strata = []',
  )
  for token in required:
    if token not in begin_source:
      raise AssertionError(f"missing installed observer contract: {token}")
  bypass = begin_source.index("if p58_continue_path:")
  capture = begin_source.index("candidate_strata = []")
  if bypass >= capture:
    raise AssertionError("P58 continue-decode bypass occurs after tensor capture")
  bypass_body = begin_source[bypass:capture]
  if "return None" not in bypass_body:
    raise AssertionError("P58 continue-decode bypass does not return before capture")

  exec(
      compile(
          ast.Module(body=[begin], type_ignores=[]),
          filename=str(args.runner),
          mode="exec",
      ),
      namespace,
  )
  events = []
  namespace.update({
      "_P38_SERVING_CAPTURE_DIR": "/capture",
      "_P38_M15_TARGET_DEBUG": "",
      "_P38_DURABILITY_PROFILE": "p58-seam-v1",
      "_P38_SERVING_CAPTURE_EXPECTED_PATH": "standard",
      "_P38_P58_CONTINUE_BYPASS_STATE": {"calls": 0, "reported": False},
      "_p38_m15_continue_program_path": lambda _: False,
      "_p38_scheduled_decode_prefixes": (
          lambda *_: events.append("schedule") or [{"num_computed_tokens": 8}]
      ),
      "_p38_observe_scheduled_prefixes": (
          lambda prefixes, path: events.append(("observe", prefixes, path))
      ),
      "_p38_request_journal": (
          lambda *values: events.append(("journal", values[3], values[4])) or 7
      ),
      "_p38_m15_replay_ledger": (
          lambda *values: events.append(("replay", values[3], values[4]))
      ),
      "_p38_incident_ledger": (
          lambda *_: (_ for _ in ()).throw(
              AssertionError("continue-decode reached incident capture")
          )
      ),
  })
  serving_begin = namespace["_p38_serving_begin"]
  stdout = io.StringIO()
  with contextlib.redirect_stdout(stdout):
    for _ in range(2):
      result = serving_begin(*([None] * 12), program_path="continue_decode")
      if result is not None:
        raise AssertionError(f"continue-decode returned capture id: {result}")
  marker = (
      "[CANON_P58_CONTINUE_DECODE_OBSERVER_BYPASS] "
      "profile=p58-seam-v1 expected=standard "
      "actual=continue_decode tensor_capture=0"
  )
  if stdout.getvalue().splitlines() != [marker]:
    raise AssertionError(f"unexpected bypass receipts: {stdout.getvalue()!r}")
  if namespace["_P38_P58_CONTINUE_BYPASS_STATE"] != {
      "calls": 2,
      "reported": True,
  }:
    raise AssertionError("P58 bypass counter/receipt state drifted")
  if len(events) != 8:
    raise AssertionError(f"chronology hook count drifted: {events!r}")
  for offset in (0, 4):
    if events[offset] != "schedule":
      raise AssertionError(f"scheduler chronology missing: {events!r}")
    if events[offset + 2] != ("journal", [], "continue_decode"):
      raise AssertionError(f"continue journal included tensor prefixes: {events!r}")
    if events[offset + 3] != ("replay", 7, "continue_decode"):
      raise AssertionError(f"chronology replay hook drifted: {events!r}")

  round_namespace = {"os": os}
  exec(
      compile(
          ast.Module(body=[round_budget], type_ignores=[]),
          filename=str(args.runner),
          mode="exec",
      ),
      round_namespace,
  )
  begin_round = round_namespace["_p38_begin_observer_round"]
  prior_profile = os.environ.get("CANON_P38_DURABILITY_PROFILE")
  try:
    for profile in ("m15-wide-v1", "p58-seam-v1"):
      os.environ["CANON_P38_DURABILITY_PROFILE"] = profile
      state = {"records": 7, "bytes": 0}
      receipts = io.StringIO()
      with contextlib.redirect_stdout(receipts):
        begin_round(state, 0, "seam")
        state["bytes"] = 123
        begin_round(state, 1, "seam")
      if state != {"records": 7, "bytes": 0, "diagnostic_round": 1}:
        raise AssertionError(f"round budget did not reset for {profile}: {state}")
      if receipts.getvalue().count("bytes=0") != 2:
        raise AssertionError(
            f"round budget receipts drifted for {profile}: {receipts.getvalue()!r}"
        )
      try:
        begin_round(state, 3, "seam")
      except ValueError as error:
        if "P38 observer diagnostic rounds must increase by one" not in str(error):
          raise
      else:
        raise AssertionError(f"round jump was admitted for {profile}")

    os.environ["CANON_P38_DURABILITY_PROFILE"] = "foreign-profile"
    foreign = {"records": 7, "bytes": 123}
    begin_round(foreign, 1, "seam")
    if foreign != {"records": 7, "bytes": 123}:
      raise AssertionError(f"foreign profile received a round budget: {foreign}")
  finally:
    if prior_profile is None:
      os.environ.pop("CANON_P38_DURABILITY_PROFILE", None)
    else:
      os.environ["CANON_P38_DURABILITY_PROFILE"] = prior_profile

  print(
      "P58_CONTINUE_DECODE_OVERLAY_PASS "
      f"cases={len(cases)} tensor_capture=standard-only "
      "round_budget=p58+m15"
  )


if __name__ == "__main__":
  main()
