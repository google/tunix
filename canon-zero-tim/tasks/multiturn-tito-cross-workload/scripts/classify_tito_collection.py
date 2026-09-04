#!/usr/bin/env python3
"""Classify one bounded P57 TiTO rollout-only evidence directory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


_REPO = Path(__file__).resolve().parents[4]
_TOKEN_PATH = _REPO / "tunix/rl/agentic/token_continuity.py"
_TOKEN_SPEC = importlib.util.spec_from_file_location(
    "p57_token_continuity_classifier", _TOKEN_PATH
)
if _TOKEN_SPEC is None or _TOKEN_SPEC.loader is None:
  raise RuntimeError(f"cannot load token-continuity module: {_TOKEN_PATH}")
token_continuity = importlib.util.module_from_spec(_TOKEN_SPEC)
sys.modules[_TOKEN_SPEC.name] = token_continuity
_TOKEN_SPEC.loader.exec_module(token_continuity)

_SHA_RE = re.compile(r"[0-9a-f]{64}")
_HOST_FIELDS = {
    "schema",
    "request_id",
    "trajectory_id",
    "workload",
    "turn",
    "pair_index",
    "group_id",
    "submitted_tokens",
    "submitted_sha256",
    "engine_echo_tokens",
    "engine_echo_sha256",
    "submitted_equals_engine_echo",
}
_RUNNER_FIELDS = {
    "schema",
    "record_index",
    "request_id",
    "dp_rank",
    "input_batch_index",
    "prompt_tokens",
    "prompt_sha256",
}
_CAPSULE_HEADER_FIELDS = {
    "schema",
    "record",
    "capsule_id",
    "workload",
    "trajectory_id",
    "policy_step",
    "turn",
    "pair_index",
    "group_id",
    "trajectory_steps",
    "first_mismatch",
    "actual_tokens",
    "expected_tokens",
    "actual_sha256",
    "expected_sha256",
    "segments",
    "token_chunk_records",
    "records_metadata_sha256",
}
_CAPSULE_STREAM_FIELDS = {
    "stream",
    "segment_index",
    "kind",
    "turn_index",
    "length",
    "sha256",
    "tokens",
}


def _load_json(path: Path) -> dict[str, Any]:
  if not path.is_file():
    raise ValueError(f"evidence file is absent: {path}")
  if path.stat().st_mode & 0o077:
    raise ValueError(f"raw TiTO evidence is not mode 0600: {path}")
  value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
  if not isinstance(value, dict):
    raise ValueError(f"evidence file is not a JSON object: {path}")
  return value


def _positive_int(record: dict[str, Any], name: str) -> int:
  value = record.get(name)
  if type(value) is not int or value <= 0:
    raise ValueError(f"{name} must be a positive integer")
  return value


def _nonnegative_int(record: dict[str, Any], name: str) -> int:
  value = record.get(name)
  if type(value) is not int or value < 0:
    raise ValueError(f"{name} must be a nonnegative integer")
  return value


def _sha(record: dict[str, Any], name: str) -> str:
  value = record.get(name)
  if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
    raise ValueError(f"{name} is not a lowercase SHA256")
  return value


def _host_record(path: Path) -> dict[str, Any]:
  record = _load_json(path)
  if set(record) != _HOST_FIELDS:
    raise ValueError(f"host witness fields differ: {path}")
  if record.get("schema") != "canon.p57-tito-host-witness.v1":
    raise ValueError(f"host witness schema differs: {path}")
  request_id = record.get("request_id")
  trajectory_id = record.get("trajectory_id")
  if not isinstance(request_id, str) or not request_id:
    raise ValueError(f"host request ID is absent: {path}")
  if (
      not isinstance(trajectory_id, str)
      or re.fullmatch(r"[0-9a-f]{32}", trajectory_id) is None
  ):
    raise ValueError(f"host trajectory ID is invalid: {path}")
  if record.get("workload") not in ("p45", "m15"):
    raise ValueError(f"host workload is invalid: {path}")
  _nonnegative_int(record, "turn")
  submitted_tokens = _positive_int(record, "submitted_tokens")
  echo_tokens = _positive_int(record, "engine_echo_tokens")
  submitted_sha = _sha(record, "submitted_sha256")
  echo_sha = _sha(record, "engine_echo_sha256")
  equality = submitted_tokens == echo_tokens and submitted_sha == echo_sha
  if type(record.get("submitted_equals_engine_echo")) is not bool:
    raise ValueError(f"host equality marker is not boolean: {path}")
  if record["submitted_equals_engine_echo"] != equality or not equality:
    raise ValueError(f"submitted prompt and engine echo differ: {request_id}")
  return record


def _runner_record(path: Path) -> dict[str, Any]:
  record = _load_json(path)
  if set(record) != _RUNNER_FIELDS:
    raise ValueError(f"runner witness fields differ: {path}")
  if record.get("schema") != "canon.p57-tito-runner-input.v1":
    raise ValueError(f"runner witness schema differs: {path}")
  if not isinstance(record.get("request_id"), str) or not record["request_id"]:
    raise ValueError(f"runner request ID is absent: {path}")
  _positive_int(record, "record_index")
  dp_rank = _nonnegative_int(record, "dp_rank")
  if dp_rank >= 8:
    raise ValueError(f"runner DP rank is outside DP8: {path}")
  _nonnegative_int(record, "input_batch_index")
  _positive_int(record, "prompt_tokens")
  _sha(record, "prompt_sha256")
  return record


def _unique_by_request(
    paths: list[Path], loader
) -> dict[str, dict[str, Any]]:
  result = {}
  for path in paths:
    record = loader(path)
    request_id = record["request_id"]
    if request_id in result:
      raise ValueError(f"duplicate request witness: {request_id}")
    result[request_id] = record
  if not result:
    raise ValueError("request witness set is empty")
  return result


def _validate_capsule(
    capsule: dict[str, Any], *, workload: str, path: Path
) -> str:
  if set(capsule) != {"schema", "header", "actual", "expected_segments"}:
    raise ValueError(f"persisted capsule fields differ: {path}")
  if capsule.get("schema") != "p57-token-first-diff-capsule-v1":
    raise ValueError(f"persisted capsule schema differs: {path}")
  header = capsule.get("header")
  actual_record = capsule.get("actual")
  segments = capsule.get("expected_segments")
  if (
      not isinstance(header, dict)
      or not isinstance(actual_record, dict)
      or not isinstance(segments, list)
      or not segments
  ):
    raise ValueError(f"persisted capsule structure differs: {path}")
  if header.get("workload") != workload:
    raise ValueError(f"persisted capsule workload differs: {path}")
  if set(header) != _CAPSULE_HEADER_FIELDS:
    raise ValueError(f"persisted capsule header fields differ: {path}")
  if (
      header.get("schema") != "p57-token-first-diff-v1"
      or header.get("record") != "header"
  ):
    raise ValueError(f"persisted capsule header identity differs: {path}")
  capsule_id = header.get("capsule_id")
  if not isinstance(capsule_id, str) or re.fullmatch(
      r"[0-9a-f]{32}", capsule_id
  ) is None:
    raise ValueError(f"persisted capsule ID differs: {path}")
  if (
      not isinstance(header.get("trajectory_id"), str)
      or re.fullmatch(r"[0-9a-f]{32}", header["trajectory_id"]) is None
      or type(header.get("policy_step")) is not int
      or header["policy_step"] < 0
  ):
    raise ValueError(f"persisted capsule join identity differs: {path}")

  for name in (
      "actual_sha256",
      "expected_sha256",
      "records_metadata_sha256",
  ):
    _sha(header, name)
  turn = _positive_int(header, "turn")
  trajectory_steps = _positive_int(header, "trajectory_steps")
  if trajectory_steps != turn:
    raise ValueError(f"persisted capsule turn metadata differs: {path}")
  if _positive_int(header, "segments") != len(segments):
    raise ValueError(f"persisted capsule segment count differs: {path}")
  _positive_int(header, "token_chunk_records")
  _nonnegative_int(header, "first_mismatch")
  for name in ("pair_index", "group_id"):
    if header.get(name) is not None and not isinstance(header[name], str):
      raise ValueError(f"persisted capsule {name} differs: {path}")

  def _tokens(
      record: dict[str, Any], label: str, *, allow_done: bool
  ) -> np.ndarray:
    expected_fields = _CAPSULE_STREAM_FIELDS | ({"done"} if allow_done else set())
    if set(record) != expected_fields:
      raise ValueError(f"persisted capsule {label} fields differ: {path}")
    raw = record.get("tokens")
    array = np.asarray(raw)
    if (
        array.ndim != 1
        or array.dtype.kind not in "iu"
        or np.any(array < 0)
        or np.any(array > np.iinfo(np.int32).max)
    ):
      raise ValueError(f"persisted capsule {label} tokens differ: {path}")
    tokens = np.asarray(array, dtype=np.int32)
    if record.get("length") != int(tokens.size):
      raise ValueError(f"persisted capsule {label} length differs: {path}")
    if record.get("sha256") != hashlib.sha256(
        np.ascontiguousarray(tokens).tobytes()
    ).hexdigest():
      raise ValueError(f"persisted capsule {label} hash differs: {path}")
    return tokens

  actual = _tokens(actual_record, "actual", allow_done=False)
  if (
      actual_record.get("stream") != "actual"
      or actual_record.get("segment_index") != 0
      or actual_record.get("kind") != "serving_prompt"
      or actual_record.get("turn_index") != turn
  ):
    raise ValueError(f"persisted capsule actual attribution differs: {path}")
  expected_parts = []
  for index, segment in enumerate(segments):
    if not isinstance(segment, dict) or segment.get("segment_index") != index:
      raise ValueError(f"persisted capsule segment order differs: {path}")
    expected_parts.append(
        _tokens(segment, f"segment-{index}", allow_done=index > 0)
    )
    if segment.get("stream") != "expected":
      raise ValueError(f"persisted capsule segment stream differs: {path}")
  first = segments[0]
  if first.get("kind") != "initial_prompt" or first.get("turn_index") != -1:
    raise ValueError(f"persisted capsule initial prompt differs: {path}")
  cursor = 1
  for step_index in range(trajectory_steps):
    if cursor >= len(segments):
      raise ValueError(f"persisted capsule trajectory is incomplete: {path}")
    assistant = segments[cursor]
    if (
        assistant.get("kind") != "assistant"
        or assistant.get("turn_index") != step_index
        or type(assistant.get("done")) is not bool
    ):
      raise ValueError(f"persisted capsule assistant attribution differs: {path}")
    cursor += 1
    has_environment = (
        cursor < len(segments)
        and segments[cursor].get("kind") == "environment"
        and segments[cursor].get("turn_index") == step_index
    )
    if has_environment:
      environment = segments[cursor]
      if (
          type(environment.get("done")) is not bool
          or environment["done"] != assistant["done"]
      ):
        raise ValueError(f"persisted capsule environment attribution differs: {path}")
      cursor += 1
    elif not assistant["done"]:
      raise ValueError(f"persisted capsule nonterminal environment is absent: {path}")
    if assistant["done"] and step_index != trajectory_steps - 1:
      raise ValueError(f"persisted capsule terminal turn is not final: {path}")
  if cursor != len(segments):
    raise ValueError(f"persisted capsule has extra segments: {path}")
  expected = np.concatenate(expected_parts)
  digest = lambda value: hashlib.sha256(  # noqa: E731
      np.ascontiguousarray(value).tobytes()
  ).hexdigest()
  if (
      actual.size != header.get("actual_tokens")
      or expected.size != header.get("expected_tokens")
      or digest(actual) != header.get("actual_sha256")
      or digest(expected) != header.get("expected_sha256")
  ):
    raise ValueError(f"persisted capsule content hash differs: {path}")
  common = min(actual.size, expected.size)
  unequal = np.flatnonzero(actual[:common] != expected[:common])
  first_mismatch = (
      int(unequal[0])
      if unequal.size
      else common
      if actual.size != expected.size
      else -1
  )
  if first_mismatch < 0 or first_mismatch != header.get("first_mismatch"):
    raise ValueError(f"persisted capsule is not a verified difference: {path}")
  return capsule_id


def classify(state_dir: Path) -> dict[str, Any]:
  """Validates three-way witness joins and bounded diff accounting."""
  witness_root = state_dir / "p57_tito_witness"
  summary = _load_json(witness_root / "diagnostic-summary.json")
  if summary.get("schema") != "canon.p57-tito-diagnostic.v1":
    raise ValueError("TiTO diagnostic summary schema differs")
  if summary.get("workload") not in ("p45", "m15"):
    raise ValueError("TiTO diagnostic summary workload differs")
  for name in (
      "backward_calls",
      "optimizer_commits",
      "checkpoint_writes",
  ):
    if summary.get(name) != 0:
      raise ValueError(f"TiTO diagnostic executed forbidden {name}")
  if (
      summary.get("train_steps_before") != summary.get("train_steps_after")
      or summary.get("global_steps_before")
      != summary.get("global_steps_after")
  ):
    raise ValueError("TiTO diagnostic mutated training step state")

  collection = summary.get("collection")
  rollout = summary.get("rollout")
  if not isinstance(collection, dict) or not isinstance(rollout, dict):
    raise ValueError("TiTO diagnostic summary lacks collection/rollout")
  if collection.get("active") is not True:
    raise ValueError("TiTO collection was not active")
  if collection.get("mode") != "collect-64":
    raise ValueError("TiTO collection mode differs")
  counters = {
      name: _nonnegative_int(collection, name)
      for name in (
          "trajectories",
          "compared_trajectories",
          "unexercised_single_turn_trajectories",
          "equal_trajectories",
          "different_trajectories",
          "later_turn_comparisons",
          "engine_echo_comparisons",
          "engine_echo_differences",
          "capsules_reserved",
          "capsules_emitted",
          "capsules_omitted",
          "emission_failures",
          "backward_transactions",
          "gradient_microbatches",
          "optimizer_commits",
          "alignment_updates",
      )
  }
  if counters["capsules_reserved"] > 64:
    raise ValueError("TiTO capsule reservation exceeded its fixed bound")
  if counters["trajectories"] != (
      counters["compared_trajectories"]
      + counters["unexercised_single_turn_trajectories"]
  ) or counters["compared_trajectories"] != (
      counters["equal_trajectories"] + counters["different_trajectories"]
  ):
    raise ValueError("TiTO trajectory accounting differs")
  if counters["different_trajectories"] != (
      counters["capsules_reserved"] + counters["capsules_omitted"]
  ):
    raise ValueError("TiTO difference accounting differs")
  if counters["capsules_reserved"] != (
      counters["capsules_emitted"] + counters["emission_failures"]
  ):
    raise ValueError("TiTO capsule emission accounting differs")
  if counters["emission_failures"]:
    raise ValueError("TiTO capsule emission failed")
  if any(
      counters[name]
      for name in (
          "backward_transactions",
          "gradient_microbatches",
          "optimizer_commits",
          "alignment_updates",
      )
  ):
    raise ValueError("TiTO rollout-only collection executed training work")
  if rollout.get("trajectories") != counters["trajectories"]:
    raise ValueError("TiTO rollout/collection trajectory counts differ")
  records = rollout.get("records")
  if not isinstance(records, list) or len(records) != counters["trajectories"]:
    raise ValueError("TiTO rollout scalar record coverage differs")
  different_statuses = sum(
      isinstance(record, dict)
      and record.get("status") == "TOKEN_CONTINUITY_DIFFERENT"
      for record in records
  )
  if different_statuses != counters["different_trajectories"]:
    raise ValueError("TiTO rollout status/difference counts differ")

  host_paths = sorted((witness_root / "host").glob("host-request-*.json"))
  runner_paths = sorted(
      (witness_root / "runner").glob("runner-input-*.json")
  )
  hosts = _unique_by_request(host_paths, _host_record)
  runners = _unique_by_request(runner_paths, _runner_record)
  if set(hosts) != set(runners):
    missing = sorted(set(hosts) - set(runners))
    foreign = sorted(set(runners) - set(hosts))
    raise ValueError(
        "host/runner request sets differ: "
        f"missing_runner={missing} foreign_runner={foreign}"
    )
  ordered_runner_indices = sorted(
      record["record_index"] for record in runners.values()
  )
  if ordered_runner_indices != list(range(1, len(runners) + 1)):
    raise ValueError("runner witness record indices are not contiguous")
  by_trajectory: dict[str, list[int]] = {}
  for request_id, host in hosts.items():
    if host["workload"] != summary["workload"]:
      raise ValueError(f"host witness workload differs: {request_id}")
    by_trajectory.setdefault(host["trajectory_id"], []).append(host["turn"])
    runner = runners[request_id]
    for host_count in ("submitted_tokens", "engine_echo_tokens"):
      if host[host_count] != runner["prompt_tokens"]:
        raise ValueError(f"three-way prompt length differs: {request_id}")
    for host_sha in ("submitted_sha256", "engine_echo_sha256"):
      if host[host_sha] != runner["prompt_sha256"]:
        raise ValueError(f"three-way prompt SHA differs: {request_id}")
  if len(by_trajectory) != counters["trajectories"]:
    raise ValueError("host witness trajectory coverage differs")
  for trajectory_id, turns in by_trajectory.items():
    ordered_turns = sorted(turns)
    if ordered_turns != list(range(len(ordered_turns))):
      raise ValueError(
          f"host witness turns are not contiguous: {trajectory_id}"
      )
  if len(hosts) != counters["engine_echo_comparisons"]:
    raise ValueError("host witness/engine echo comparison counts differ")

  capsule_paths = sorted(
      (state_dir / "token-continuity-first-diff").glob("*.json")
  )
  if len(capsule_paths) != counters["capsules_emitted"]:
    raise ValueError("persisted capsule count differs from its summary")
  capsule_ids = set()
  for path in capsule_paths:
    capsule = _load_json(path)
    capsule_id = _validate_capsule(
        capsule, workload=summary["workload"], path=path
    )
    if capsule_id in capsule_ids:
      raise ValueError("persisted capsules have absent/duplicate IDs")
    capsule_ids.add(capsule_id)
    if capsule["header"]["trajectory_id"] not in by_trajectory:
      raise ValueError("persisted capsule trajectory is not in host witnesses")

  return {
      "schema": "canon.p57-tito-collection-classification.v1",
      "verdict": "PASS",
      "witness_verdict": "PASS",
      "token_verdict": (
          "DIFFERENT" if counters["different_trajectories"]
          else "EQUAL" if counters["compared_trajectories"]
          else "UNEXERCISED"
      ),
      "workload": summary["workload"],
      "requests": len(hosts),
      **counters,
      "backward_calls": 0,
      "optimizer_commits": 0,
      "checkpoint_writes": 0,
  }


def _exclusive_json(path: Path, record: dict[str, Any]) -> None:
  if path.exists():
    raise FileExistsError(f"refusing to overwrite classification: {path}")
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
  descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
  with os.fdopen(descriptor, "wb") as output:
    output.write(payload)
    output.flush()
    os.fsync(output.fileno())


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--state", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  try:
    result = classify(args.state)
  except Exception as error:  # pylint: disable=broad-exception-caught
    result = {
        "schema": "canon.p57-tito-collection-classification.v1",
        "verdict": "FAIL",
        "reason": str(error),
    }
    _exclusive_json(args.output, result)
    print(f"P57_TITO_COLLECTION_FAIL reason={error}")
    return 1
  _exclusive_json(args.output, result)
  digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
  print(
      "P57_TITO_COLLECTION_PASS "
      f"workload={result['workload']} requests={result['requests']} "
      f"token_verdict={result['token_verdict']} "
      f"classification_sha256={digest}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
