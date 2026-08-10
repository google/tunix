#!/usr/bin/env python3
"""Classify one P35.3c first-record infrastructure stage probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_STAGES = (
    "model",
    "logits",
    "sample",
    "logprobs",
    "target_gathers",
    "record_outputs",
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _shape_list_valid(value: Any) -> bool:
  return bool(
      isinstance(value, list)
      and value
      and all(
          isinstance(shape, list)
          and all(isinstance(dimension, int) and dimension >= 0 for dimension in shape)
          for shape in value
      )
  )


def classify(events: list[dict[str, Any]]) -> dict[str, Any]:
  """Returns a fail-closed non-numerical stage-probe classification."""
  reasons: list[str] = []
  ready_prefix: list[str] = []
  prefix_intact = True
  expected_record_count = events[0].get("record_count") if events else None
  if len(events) != len(REQUIRED_STAGES):
    reasons.append(
        f"event_count:{len(events)}!={len(REQUIRED_STAGES)}"
    )
  for index, stage in enumerate(REQUIRED_STAGES, start=1):
    if index > len(events):
      reasons.append(f"missing_stage:{stage}")
      continue
    event = events[index - 1]
    checks = {
        "schema_version": event.get("schema_version") == 1,
        "event": event.get("event") == "ready",
        "replay": event.get("replay") == "R0_live_first",
        "record_index": event.get("record_index") == 1,
        "record_count": (
            isinstance(event.get("record_count"), int)
            and event["record_count"] >= 1
            and event["record_count"] == expected_record_count
        ),
        "stage": event.get("stage") == stage,
        "ordinal": event.get("ordinal") == index,
        "stage_count": event.get("stage_count") == len(REQUIRED_STAGES),
        "leaf_shapes": _shape_list_valid(event.get("leaf_shapes")),
    }
    reasons.extend(
        f"{stage}.{name}" for name, passed in checks.items() if not passed
    )
    if prefix_intact and all(checks.values()):
      ready_prefix.append(stage)
    else:
      prefix_intact = False
  progress = {
      "ready_stages": ready_prefix,
      "last_ready_stage": ready_prefix[-1] if ready_prefix else None,
      "first_missing_stage": (
          REQUIRED_STAGES[len(ready_prefix)]
          if len(ready_prefix) < len(REQUIRED_STAGES)
          else None
      ),
  }
  if reasons:
    return {
        "measurement_verdict": "INCONCLUSIVE",
        "classification": None,
        "numerical_verdict": False,
        "reasons": reasons,
        **progress,
    }
  return {
      "measurement_verdict": "COMPLETE",
      "classification": "first_record_stage_probe_complete",
      "numerical_verdict": False,
      "reasons": [],
      **progress,
      "replay": "R0_live_first",
      "record_index": 1,
      "record_count": events[0]["record_count"],
  }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  events = []
  with path.open(encoding="utf-8") as source:
    for line_number, line in enumerate(source, start=1):
      try:
        value = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL at line {line_number}") from exc
      if not isinstance(value, dict):
        raise ValueError(f"stage event at line {line_number} is not an object")
      events.append(value)
  return events


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--report", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  result = classify(_load_jsonl(args.report))
  result["report_sha256"] = _sha256(args.report)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write("\n")
  print(json.dumps(result, sort_keys=True))
  return 0 if result["measurement_verdict"] == "COMPLETE" else 1


if __name__ == "__main__":
  raise SystemExit(main())
