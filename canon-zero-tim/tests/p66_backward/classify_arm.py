#!/usr/bin/env python3
"""Fail-closed classifier for one P66 backward-no-commit arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CHUNK_ELEMENTS = 1_048_576


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def data_sha256(array: np.ndarray) -> str:
  digest = hashlib.sha256()
  flattened = array.reshape(-1)
  for start in range(0, flattened.size, CHUNK_ELEMENTS):
    stop = min(start + CHUNK_ELEMENTS, flattened.size)
    digest.update(
        np.ascontiguousarray(flattened[start:stop]).tobytes(order="C")
    )
  return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected a JSON object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = [
      json.loads(line)
      for line in path.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]
  if not all(isinstance(row, dict) for row in rows):
    raise ValueError(f"{path}: expected JSON objects")
  return rows


def load_capture(root: Path, name: str) -> dict[str, Any]:
  capture_dir = root / name
  manifest_path = capture_dir / "manifest.json"
  if capture_dir.is_symlink() or not capture_dir.is_dir():
    raise ValueError(f"invalid capture directory: {capture_dir}")
  manifest = load_json(manifest_path)
  leaves = manifest.get("leaves")
  if (
      manifest.get("schema") != "canon-p61-full-tree-capture-v1"
      or manifest.get("capture") != name
      or not isinstance(leaves, list)
      or not leaves
      or manifest.get("leaf_count") != len(leaves)
  ):
    raise ValueError(f"invalid full-tree manifest: {manifest_path}")
  total_bytes = 0
  for index, leaf in enumerate(leaves):
    if not isinstance(leaf, dict) or leaf.get("index") != index:
      raise ValueError(f"non-contiguous capture leaf at {index}")
    path = capture_dir / f"leaf_{index:05d}.npy"
    if (
        leaf.get("file") != path.name
        or path.is_symlink()
        or not path.is_file()
        or sha256(path) != leaf.get("file_sha256")
    ):
      raise ValueError(f"invalid capture leaf: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        list(array.shape) != leaf.get("shape")
        or str(array.dtype) != leaf.get("dtype")
        or int(array.size) != leaf.get("elements")
        or int(array.nbytes) != leaf.get("data_bytes")
        or data_sha256(array) != leaf.get("data_sha256")
    ):
      raise ValueError(f"capture metadata mismatch: {path}")
    total_bytes += int(array.nbytes)
  if total_bytes != manifest.get("total_data_bytes"):
    raise ValueError(f"capture byte total mismatch: {manifest_path}")
  return manifest


def classify(
    *,
    arm: str,
    run_log: Path,
    pre_alignment_report: Path,
    alignment_report: Path,
    update_report: Path,
    capture_root: Path,
) -> dict[str, Any]:
  reasons = []
  raw = run_log.read_text(encoding="utf-8", errors="replace")
  pre = load_jsonl(pre_alignment_report)
  align = load_jsonl(alignment_report)
  update = load_json(update_report)
  captures = {
      name: load_capture(capture_root, name)
      for name in ("model_before", "gradient")
  }
  expected_marker = f"[P66.BACKWARD] arm={arm} verdict=PASS commits=0"
  if raw.count(expected_marker) != 1:
    reasons.append("terminal_marker")
  if "verdict=FAIL" in raw:
    reasons.append("raw_failure")
  if len(pre) != 1 or pre[0].get("verdict") != "PASS":
    reasons.append("pre_alignment")
  if len(align) != 16 or any(row.get("verdict") != "PASS" for row in align):
    reasons.append("alignment")
  if (
      update.get("schema") != "canon-p66-backward-gate-v1"
      or update.get("arm") != arm
      or update.get("verdict") != "PASS"
      or update.get("commits") != 0
      or (update.get("dp_size"), update.get("tp_size")) != (4, 1)
      or update.get("global_trajectories") != 64
      or update.get("gradient_groups") != 16
      or update.get("alignment_verdicts") != ["PASS"] * 16
      or update.get("train_steps_before") != update.get("train_steps_after")
      or any(update.get("state_changed_paths", {}).values())
  ):
    reasons.append("update_contract")
  gradient = update.get("gradient", {})
  if (
      not isinstance(gradient, dict)
      or gradient.get("all_finite") is not True
      or gradient.get("any_nonzero") is not True
      or not isinstance(gradient.get("stable_norm"), (int, float))
      or not np.isfinite(gradient["stable_norm"])
      or gradient["stable_norm"] <= 0.0
  ):
    reasons.append("gradient_contract")
  return {
      "schema": "canon-p66-backward-arm-classification-v1",
      "verdict": "PASS" if not reasons else "FAIL",
      "arm": arm,
      "zero_tim": {
          "expected_pass": 17,
          "observed_pass": sum(
              row.get("verdict") == "PASS" for row in pre + align
          ),
          "observed_fail": sum(
              row.get("verdict") == "FAIL" for row in pre + align
          ),
      },
      "captures": {
          name: {
              "leaf_count": value["leaf_count"],
              "total_data_bytes": value["total_data_bytes"],
          }
          for name, value in captures.items()
      },
      "evidence_sha256": {
          "run_log": sha256(run_log),
          "pre_alignment_report": sha256(pre_alignment_report),
          "alignment_report": sha256(alignment_report),
          "update_report": sha256(update_report),
      },
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--arm", choices=("ordinary", "segmented"), required=True)
  parser.add_argument("--run-log", type=Path, required=True)
  parser.add_argument("--pre-alignment-report", type=Path, required=True)
  parser.add_argument("--alignment-report", type=Path, required=True)
  parser.add_argument("--update-report", type=Path, required=True)
  parser.add_argument("--capture-root", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  result = classify(
      arm=args.arm,
      run_log=args.run_log,
      pre_alignment_report=args.pre_alignment_report,
      alignment_report=args.alignment_report,
      update_report=args.update_report,
      capture_root=args.capture_root,
  )
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      f"P66_BACKWARD_ARM verdict={result['verdict']} arm={args.arm} "
      f"zero_tim={result['zero_tim']['observed_pass']}/17 "
      f"fail={result['zero_tim']['observed_fail']}"
  )
  return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
