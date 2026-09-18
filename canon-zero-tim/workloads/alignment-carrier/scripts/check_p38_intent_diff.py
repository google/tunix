#!/usr/bin/env python3
"""Fail closed unless a P38s12b manifest changes only max concurrency."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shlex
from typing import Any

import yaml


def _main_container(document: dict[str, Any]) -> dict[str, Any]:
  matches = []

  def visit(value: Any) -> None:
    if isinstance(value, dict):
      if value.get("name") == "jax-tpu" and isinstance(value.get("env"), list):
        matches.append(value)
      for child in value.values():
        visit(child)
    elif isinstance(value, list):
      for child in value:
        visit(child)

  visit(document)
  if len(matches) != 1:
    raise ValueError(f"expected one jax-tpu container, observed {len(matches)}")
  return matches[0]


def _env_entry(document: dict[str, Any], name: str) -> dict[str, Any]:
  matches = [
      item for item in _main_container(document)["env"]
      if item.get("name") == name
  ]
  if len(matches) != 1 or "value" not in matches[0]:
    raise ValueError(f"expected one literal environment value for {name}")
  return matches[0]


def _normalize_command(command: str, expected: int) -> str:
  tokens = shlex.split(command)
  prefix = "--max_concurrency="
  matches = [index for index, token in enumerate(tokens) if token.startswith(prefix)]
  if len(matches) != 1:
    raise ValueError("CANON_RUN_CMD must contain one --max_concurrency argument")
  index = matches[0]
  observed = int(tokens[index].removeprefix(prefix))
  if observed != expected:
    raise ValueError(
        f"unexpected max concurrency: observed={observed} expected={expected}"
    )
  tokens[index] = "--max_concurrency=<INTENT>"
  return shlex.join(tokens)


def classify(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
  reasons = []
  left = copy.deepcopy(baseline)
  right = copy.deepcopy(candidate)
  try:
    left_command = _normalize_command(
        str(_env_entry(left, "CANON_RUN_CMD")["value"]), 256
    )
    right_command = _normalize_command(
        str(_env_entry(right, "CANON_RUN_CMD")["value"]), 32
    )
    _env_entry(left, "CANON_RUN_CMD")["value"] = left_command
    _env_entry(right, "CANON_RUN_CMD")["value"] = right_command
    for document, expected in ((left, "256"), (right, "32")):
      labels = document.get("metadata", {}).get("labels", {})
      if labels.get("canon.zero-tim/max-concurrency") != expected:
        raise ValueError(
            "max-concurrency attestation label does not match the arm"
        )
      labels["canon.zero-tim/max-concurrency"] = "<INTENT>"
  except (KeyError, TypeError, ValueError) as exc:
    reasons.append(str(exc))
  if not reasons and left != right:
    reasons.append("manifests differ outside max concurrency and its label")
  return {
      "verdict": "PASS" if not reasons else "FAIL",
      "baseline_max_concurrency": 256,
      "candidate_max_concurrency": 32,
      "allowed_changes": [
          "CANON_RUN_CMD.--max_concurrency",
          "metadata.labels.canon.zero-tim/max-concurrency",
      ],
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--baseline", type=Path, required=True)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  result = classify(
      yaml.safe_load(args.baseline.read_text(encoding="utf-8")),
      yaml.safe_load(args.candidate.read_text(encoding="utf-8")),
  )
  payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
  if args.output is not None:
    if args.output.exists():
      raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(payload, encoding="utf-8")
  print(payload, end="")
  return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
