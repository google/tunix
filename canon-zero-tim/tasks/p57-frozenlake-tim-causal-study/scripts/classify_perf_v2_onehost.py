#!/usr/bin/env python3
"""Classify the P57.1c three-update one-host Perf v2 target gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


_TRACER_RED_PATTERNS = (
    "Purging uncompleted span",
    "no more spans to end",
    "cannot commit PerfTracer timelines with active host spans",
    "cannot commit step with active spans",
)


def _json_lines(path: Path) -> list[dict[str, Any]]:
  rows = []
  for line in path.read_text(encoding="utf-8").splitlines():
    if line.strip():
      value = json.loads(line)
      if not isinstance(value, dict):
        raise ValueError(f"expected JSON objects in {path}")
      rows.append(value)
  return rows


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def classify(
    *,
    raw_path: Path,
    alignment_path: Path,
    update_path: Path,
    semantic_path: Path,
    docker_exit: int,
) -> dict[str, Any]:
  reasons: list[str] = []

  def require(condition: bool, reason: str) -> None:
    if not condition:
      reasons.append(reason)

  raw = (
      raw_path.read_text(encoding="utf-8", errors="replace")
      if raw_path.is_file()
      else ""
  )
  alignments = _json_lines(alignment_path) if alignment_path.is_file() else []
  updates = _json_lines(update_path) if update_path.is_file() else []
  semantic = (
      json.loads(semantic_path.read_text(encoding="utf-8"))
      if semantic_path.is_file()
      else {}
  )

  require(raw_path.is_file(), "missing_raw")
  require(alignment_path.is_file(), "missing_alignment")
  require(update_path.is_file(), "missing_updates")
  require(semantic_path.is_file(), "missing_semantic")

  require(docker_exit == 0, f"docker_exit={docker_exit}")
  require(
      raw.count("[V1.PERFETTO] captured training_step=2") == 1,
      "perfetto_target_step_2",
  )
  require(
      len(re.findall(r"Global step [0-9]+ completed in", raw)) == 3,
      "global_step_completions",
  )
  require(
      raw.count("[CANON_FROZENLAKE_P27] update_step_committed") == 3,
      "optimizer_commit_markers",
  )
  for pattern in _TRACER_RED_PATTERNS:
    require(pattern not in raw, f"tracer_red:{pattern}")
  require("verdict=FAIL" not in raw, "raw_alignment_fail")

  require(len(alignments) == 12, f"alignment_rows={len(alignments)}")
  for index, row in enumerate(alignments):
    require(row.get("verdict") == "PASS", f"alignment[{index}].verdict")
    require(not row.get("blocking_reds"), f"alignment[{index}].blocking_reds")
    require(
        isinstance(row.get("N_action"), int) and row["N_action"] > 0,
        f"alignment[{index}].N_action",
    )
    boundaries = row.get("boundaries", {})
    require(bool(boundaries), f"alignment[{index}].boundaries")
    for name, boundary in boundaries.items():
      require(
          boundary.get("differing_bytes") == 0,
          f"alignment[{index}].{name}.differing_bytes",
      )
      require(boundary.get("finite") is True, f"alignment[{index}].{name}.finite")

  require(len(updates) == 3, f"update_rows={len(updates)}")
  for index, row in enumerate(updates):
    require(row.get("verdict") == "PASS", f"update[{index}].verdict")
    require(row.get("commits") == 1, f"update[{index}].commits")
    require(
        row.get("train_steps_before") == index,
        f"update[{index}].train_steps_before",
    )
    require(
        row.get("train_steps_after") == index + 1,
        f"update[{index}].train_steps_after",
    )
    require(row.get("gradient_finite") is True, f"update[{index}].gradient_finite")
    norm = row.get("commit_gradient_norm")
    require(
        isinstance(norm, (int, float)) and math.isfinite(norm) and norm > 0,
        f"update[{index}].commit_gradient_norm",
    )
    evidence = row.get("commit_evidence", {})
    require(
        isinstance(evidence.get("gradient_nonzero_elements"), int)
        and evidence["gradient_nonzero_elements"] > 0,
        f"update[{index}].gradient_nonzero_elements",
    )
    require(
        isinstance(evidence.get("parameter_changed_elements"), int)
        and evidence["parameter_changed_elements"] > 0,
        f"update[{index}].parameter_changed_elements",
    )
    require(
        evidence.get("parameter_delta_finite") is True,
        f"update[{index}].parameter_delta_finite",
    )
    require(
        row.get("optimizer_transaction_valid") is True,
        f"update[{index}].optimizer_transaction_valid",
    )

  required_semantic = {
      "data_loading",
      "rollout",
      "advantage_computation",
      "peft_train",
      "weight_sync",
  }
  require(semantic.get("verdict") == "PASS", "semantic.verdict")
  require(semantic.get("files") == 1, "semantic.files")
  require(
      semantic.get("reference_inference_contract") == "disabled",
      "semantic.reference_inference_contract",
  )
  counts = semantic.get("event_counts", {})
  for name in sorted(required_semantic):
    require(
        isinstance(counts.get(name), int) and counts[name] > 0,
        f"semantic.{name}",
    )
  require(
      counts.get("reference_inference", 0) == 0,
      "semantic.reference_inference_unexpected",
  )

  return {
      "schema": "p57-perf-v2-onehost-g4-v1",
      "verdict": "PASS" if not reasons else "FAIL",
      "scope": (
          "one-host Qwen3-8B DP1xTP4 FrozenLake three-update Perf-v2 "
          "step-boundary target gate"
      ),
      "optimizer_commits": len(updates),
      "strict_alignment_rows": len(alignments),
      "strict_alignment_failures": sum(
          row.get("verdict") != "PASS" for row in alignments
      ),
      "semantic_event_counts": counts,
      "sha256": {
          name: _sha256(path)
          for name, path in (
              ("raw", raw_path),
              ("alignment", alignment_path),
              ("updates", update_path),
              ("semantic", semantic_path),
          )
          if path.is_file()
      },
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw", required=True, type=Path)
  parser.add_argument("--alignment", required=True, type=Path)
  parser.add_argument("--updates", required=True, type=Path)
  parser.add_argument("--semantic", required=True, type=Path)
  parser.add_argument("--docker-exit", required=True, type=int)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  result = classify(
      raw_path=args.raw,
      alignment_path=args.alignment,
      update_path=args.updates,
      semantic_path=args.semantic,
      docker_exit=args.docker_exit,
  )
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print("P57_PERF_V2_ONEHOST_JSON " + json.dumps(result, sort_keys=True))
  return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
